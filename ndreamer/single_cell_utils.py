import scanpy as sc
import torch
import scipy
import numpy as np

# Acknowledgement to scCRAFT from Chuan He (https://www.biorxiv.org/content/10.1101/2024.10.22.619682v1)

def multi_resolution_cluster(adata, resolution1=0.5, resolution2=7, method="Leiden"):
    """
    Performs PCA, neighbors calculation, and clustering with specified resolutions and method.

    Parameters:
    - adata: AnnData object containing single-cell data.
    - resolution1: float, the resolution parameter for the first clustering.
    - resolution2: float, the resolution parameter for the second clustering.
    - method: str, clustering method to use ("Louvain" or "Leiden").

    The function updates `adata` in place, adding two new columns to `adata.obs`:
    - 'leiden1': contains cluster labels from the first clustering.
    - 'leiden2': contains cluster labels from the second clustering.
    """
    assert method in ["Louvain", "Leiden"], "method must be one of ['Louvain', 'Leiden']"

    # Perform PCA
    sc.tl.pca(adata, n_comps=50)
    # Compute neighbors using the PCA representation
    sc.pp.neighbors(adata, use_rep="X_pca")

    # Determine the clustering function based on the method
    if method.lower() == "louvain":
        clustering_function = sc.tl.louvain
    elif method.lower() == "leiden":
        clustering_function = sc.tl.leiden
    else:
        raise ValueError("Method should be 'Louvain' or 'Leiden'")

    # Perform the first round of clustering
    clustering_function(adata, resolution=resolution1)
    adata.obs['leiden1'] = adata.obs[method.lower()]

    # Perform the second round of clustering with a different resolution
    clustering_function(adata, resolution=resolution2)
    adata.obs['leiden2'] = adata.obs[method.lower()]
    return adata


def split_and_cluster(adata, condition_key, batch_key=None, resolution1=0.5, resolution2=7, method="Leiden"):
    """
    Splits the AnnData object by batch and/or condition, applies multi-resolution clustering
    to each subset, and returns a merged AnnData with new clustering results.

    Parameters:
    - adata: AnnData object containing single-cell data.
    - batch_key: str or None, column in `adata.obs` representing batch information (optional).
    - condition_key: str, column in `adata.obs` representing condition information.
    - resolution1: float, resolution for the first clustering step.
    - resolution2: float, resolution for the second clustering step.
    - method: str, clustering method to use ("Louvain" or "Leiden").

    Returns:
    - AnnData: The updated AnnData object with clustering results added to `adata.obs`.
    """
    assert condition_key is not None, "condition_key must be provided"
    assert condition_key in adata.obs, f"{condition_key} not found in adata.obs"

    # Handle splitting key
    if batch_key is not None:
        assert batch_key in adata.obs, f"{batch_key} not found in adata.obs"
        # Create group key using both batch and condition
        adata.obs['group_key'] = adata.obs[batch_key].astype(str) + "_" + adata.obs[condition_key].astype(str)
    else:
        # Create group key using only condition
        adata.obs['group_key'] = adata.obs[condition_key].astype(str)

    print("Unique groups:",np.unique(adata.obs['group_key']))
    # Initialize an empty list to store clustered subsets
    clustered_adatas = []

    group_num_cells=np.unique(adata.obs['group_key'],return_counts=True)[1]
    if np.mean(np.array(group_num_cells)) < 2000:
        print("Mean number of cells in this dataset less than 2000, cluster using all cells instead of one group (unique batch and condition combination) by one group")
        return multi_resolution_cluster(adata=adata, resolution1=resolution1, resolution2=resolution2, method=method)

    # Iterate over unique groups
    for group in adata.obs['group_key'].unique():
        # Subset the AnnData object for the current group
        subset = adata[adata.obs['group_key'] == group].copy()

        if subset.shape[0]<100:
            print("Subset",group,"have too few of number of cells (less than 100), discard it")
            continue

        # Apply the clustering function to the subset
        subset=multi_resolution_cluster(subset, resolution1=resolution1, resolution2=resolution2, method=method)

        # Append the subset with clustering results to the list
        clustered_adatas.append(subset)

    # Concatenate all subsets back into a single AnnData object
    combined_adata = clustered_adatas[0].concatenate(*clustered_adatas[1:], batch_key="batch_condition",
                                                     index_unique=None)

    if np.mean(np.array(group_num_cells)) < 2000:
        print("Mean number of cells in this dataset less than 2000, cluster using all cells instead of one group (unique batch and condition combination) by one group")

    # Clean up the temporary group_key column
    del combined_adata.obs['group_key']

    return combined_adata

# Loader only for Visualization
def generate_adata_to_dataloader(adata, batch_size=2048):
    if isinstance(adata.X, scipy.sparse.spmatrix):
        X_dense = adata.X.toarray()
    else:
        X_dense = adata.X

    X_tensor = torch.tensor(X_dense, dtype=torch.float32)

    # Create a DataLoader for batch-wise processing
    dataset = torch.utils.data.TensorDataset(X_tensor, torch.arange(len(X_tensor)))  # include indices
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return data_loader