import torch
import numpy as np
import scipy
import scanpy as sc

from ndreamer.single_cell_utils import split_and_cluster,multi_resolution_cluster
from torch.utils.data import DataLoader, TensorDataset,Dataset

def process_adata(adata, condition_key, control_name, input_dim, require_batch=False, batch_key=None,
                  resolution_low=0.5, resolution_high=7, cluster_method="leiden",
                  library_size_normalize_adata=True):
    """
    Processes AnnData object by performing clustering, converting batch and condition values to integers,
    and creating a group column.

    Parameters:
    - adata: AnnData object to process.
    - condition_key: str, the column name in adata.obs representing conditions.
    - require_batch: bool, whether batch information is required.
    - batch_key: str or None, the column name in adata.obs representing batches.
    - resolution_low: float, low resolution for clustering.
    - resolution_high: float, high resolution for clustering.
    - cluster_method: str, clustering method ("leiden" or "louvain").

    Returns:
    - adata: Processed AnnData object.
    - batch_mapping: dict, mapping of original batch values to integers.
    - condition_mapping: dict, mapping of original condition values to integers.
    """
    if require_batch and batch_key is None:
        raise ValueError("If you require batch, then you must provide the batch key in the adata.")

    batch_names=None

    if batch_key is not None:
        if isinstance(batch_key, str):
            adata.obs["batch_backup"] = adata.obs[batch_key].copy()
            print(f"Original adata.obs[{batch_key}] back up to adata.obs['batch_backup']")
            adata.obsm["batch"] = np.expand_dims(np.array(adata.obs[batch_key].copy()),axis=-1)
            adata.obs["batch"]=adata.obs[batch_key].copy()
            batch_names = [batch_key]
        elif isinstance(batch_key, list):
            for batch_keyi in batch_key:
                adata.obs[f"{batch_keyi}_backup"] = adata.obs[batch_keyi].copy()
                print(f"Original adata.obs[{batch_keyi}] back up to adata.obs['{batch_keyi}_backup']")
            batch_names=[]
            for i in range(adata.shape[0]):
                batch_namei="--".join([str(adata.obs[batch_keyi][i]) for batch_keyi in batch_key])
                batch_names.append(batch_namei)
            adata.obs["batch"]=batch_names
            adata.obsm["batch"]=np.array(adata.obs[batch_key].copy())
            batch_names=batch_key
        else:
            raise ValueError("You must provide a batch_key in the form of list of strings or string")
    adata.obs["condition_backup"] = adata.obs[condition_key].copy()
    print(f"Original adata.obs[{condition_key}] back up to adata.obs['condition_backup']")

    # Handle batch information
    if not require_batch:
        adata.obs["batch"] = np.zeros(adata.shape[0], dtype=int)
        batch_names = ["batch"]
        adata.obsm["batch"]=np.zeros((adata.shape[0],1), dtype=int)

        # Convert batch and condition values to integers
    batch_mappings={}
    batch_data=[]
    for i in range(adata.obsm["batch"].shape[-1]):
        batch_namei=batch_names[i]
        batch_mappingi = {val: i for i, val in enumerate(adata.obs[batch_namei].unique())}
        batch_mappings[batch_namei]=batch_mappingi
        batch_data.append(adata.obs[batch_namei].map(batch_mappingi))
    batch_data=np.stack(batch_data,axis=-1)
    #print(batch_data.shape,adata.shape)
    adata.obsm["batch"]=batch_data

    unique_conditions = adata.obs[condition_key].unique()

    assert control_name in unique_conditions,f"{control_name} is not in {unique_conditions}"
    condition_mapping = {control_name: 0} if control_name in unique_conditions else {}

    next_index = 1
    for val in unique_conditions:
        if val != control_name:
            condition_mapping[val] = next_index
            next_index += 1

    print("Condition mapping to adata.obs['condition']:", condition_mapping)
    print("Condition mapping to adata.obs['batch']:", batch_mappings)

    adata.obs["condition"] = adata.obs[condition_key].map(condition_mapping)

    # Create a combined group column
    unique_batch_mapping={val: i for i, val in enumerate(adata.obs["batch"].unique())}
    print("Unique batch mapping:", unique_batch_mapping)
    adata.obs["batch"]=adata.obs["batch"].map(unique_batch_mapping)

    adata.obs["group"] = adata.obs.apply(lambda row: f"{row['batch']}_{row['condition']}", axis=1)

    if library_size_normalize_adata:
        sc.pp.filter_cells(adata, min_genes=300)
        sc.pp.filter_genes(adata, min_cells=10)
        sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
        sc.pp.log1p(adata)

    if adata.shape[1]>input_dim:
        sc.pp.highly_variable_genes(adata, n_top_genes=input_dim, batch_key="group")
        adata = adata[:, adata.var['highly_variable']].copy()

    # Perform clustering
    print("Calculating the clusters")
    adata = split_and_cluster(
        adata=adata, condition_key=condition_key,
        batch_key="batch", resolution1=resolution_low,
        resolution2=resolution_high, method=cluster_method
    )

    print("Finished clustering")
    return adata, batch_mappings, condition_mapping


def generate_balanced_dataloader(adata, batch_size, batch_key="batch", condition_key="condition"):
    if not adata.obs_names.is_unique:
        print("Indices are not unique. Adding condition information to adata.obs_names")
        names=list(adata.obs_names.copy())
        names=[names[i]+adata.obs["group"][i] for i in range(len(names))]
        adata.obs_names=np.array(names)

    if not adata.obs_names.is_unique:
        print("Indices are not unique even within each group. Using`adata.obs_names=np.array(range(adata.shape[0]))`.")
        adata.obs_names=np.array(range(adata.shape[0]))

    # Map unique batch-condition combinations to integers
    unique_groups = adata.obs["group"].unique()
    group_to_int = {group: i for i, group in enumerate(unique_groups)}

    # Extract unsupervised cluster labels
    unsupervised_labels1 = adata.obs['leiden1'].cat.codes.values
    unsupervised_labels2 = adata.obs['leiden2'].cat.codes.values

    # Separate the dataset by groups and sample indices
    group_indices = []
    condition_labels_list = []
    batch_labels_list = []
    group_labels_list = []

    for group in unique_groups:
        # Find the indices for the current group
        group_indices_in_adata = adata.obs[adata.obs["group"] == group].index

        # Sample indices from the current group
        if len(group_indices_in_adata) >= batch_size:
            sampled_indices = np.random.choice(group_indices_in_adata, batch_size, replace=False)
        else:
            # If not enough cells, sample with replacement
            sampled_indices = np.random.choice(group_indices_in_adata, batch_size, replace=True)

        # Get the integer positions of the sampled indices
        sampled_indices_pos = [adata.obs_names.get_loc(idx) for idx in sampled_indices]
        group_indices.extend(sampled_indices_pos)

        # Map the group keys to integers and add to the label lists
        group_labels_list.extend([group_to_int[group]] * batch_size)
        condition_labels_list.extend(adata.obs.loc[sampled_indices, condition_key].values)
        batch_labels_list.extend(adata.obs.loc[sampled_indices, batch_key].values)

    # Extract the feature data
    X_sampled = adata.X[group_indices, :]

    # Convert features to tensor
    if isinstance(X_sampled, np.ndarray):
        X_tensor = torch.tensor(X_sampled, dtype=torch.float32)
    else:  # if it's a sparse matrix
        X_tensor = torch.tensor(X_sampled.toarray(), dtype=torch.float32)

    # Convert labels to tensors
    condition_tensor = torch.tensor(condition_labels_list, dtype=torch.int64)
    batch_tensor = torch.tensor(adata.obsm["batch"][group_indices, :], dtype=torch.int64)#torch.tensor(batch_labels_list, dtype=torch.int64)
    group_tensor = torch.tensor(group_labels_list, dtype=torch.int64)
    label_tensor1 = torch.tensor(unsupervised_labels1[group_indices], dtype=torch.int64)
    label_tensor2 = torch.tensor(unsupervised_labels2[group_indices], dtype=torch.int64)

    # Create a TensorDataset and DataLoader
    combined_dataset = TensorDataset(X_tensor, condition_tensor, batch_tensor, label_tensor1, label_tensor2)
    dataloader = DataLoader(combined_dataset, batch_size=batch_size*2, shuffle=True)
    return dataloader


# Loader only for Visualization
def generate_adata_to_dataloader(adata, batch_size=2048):
    if isinstance(adata.X, scipy.sparse.spmatrix):
        X_dense = adata.X.toarray()
    else:
        X_dense = adata.X

    X_tensor = torch.tensor(X_dense, dtype=torch.float32)

    batch_tensor=torch.tensor(np.array(adata.obsm["batch"]), dtype=torch.int64)
    condition_tensor=torch.tensor(np.array(adata.obs["condition"]), dtype=torch.int64)

    # Create a DataLoader for batch-wise processing
    dataset = torch.utils.data.TensorDataset(X_tensor, torch.arange(len(X_tensor)), condition_tensor, batch_tensor)  # include indices
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return data_loader

if __name__=="__main__":
    pass

