'''import scanpy as sc

adata=sc.read_h5ad("../data/PBMC.h5ad")
print(adata.X[:10,:10])

sc.pp.highly_variable_genes(adata, n_top_genes=2000, batch_key="condition")
adata = adata[:, adata.var['highly_variable']]

#Visualization
sc.pp.neighbors(adata)
sc.tl.umap(adata, min_dist=0.5)
sc.pl.umap(adata, color=['condition', 'cell_type'], frameon=False, ncols=1, save="PBMC_raw.png")
sc.pl.umap(adata, color='cell_type', frameon=False, ncols=1)

sc.pl.umap(adata, color='condition', frameon=False, ncols=1)
'''

import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc

'''def run_cell_anova(adata, batch_key, condition_key, control_name, cell_type_label, dataset_name):
    integrate_key=batch_key
    sc.pp.filter_cells(adata, min_genes=300)
    sc.pp.filter_genes(adata, min_cells=10)
    adata.obs['dataidx'] = adata.obs[batch_key].copy()
    adata_prep = cnova.model.preprocess_data(adata, integrate_key='dataidx')
    control_batches = list(set(adata_prep[adata_prep.obs[condition_key] == control_name,].obs[batch_key]))
    control_dict = {
        'g1': control_batches,
    }
    adata_prep = cnova.model.calc_ME(adata_prep, integrate_key='dataidx')
    adata_prep = cnova.model.calc_BE(adata_prep, integrate_key, control_dict)
    adata_prep = cnova.model.calc_TE(adata_prep, integrate_key)
    print(control_dict)
    adata_prep.obs['PseudoState'] = adata_prep.obs[condition_key].copy()
    adata_prep.write_h5ad("./cellanova/"+dataset_name+"_results.h5ad")

    integrated = ad.AnnData(adata_prep.layers['denoised'], dtype=np.float32)
    integrated.obs = adata_prep.obs.copy()
    integrated.var_names = adata_prep.var_names

    sc.pp.neighbors(integrated, n_neighbors=15, n_pcs=30)
    sc.tl.umap(integrated)
    if cell_type_label is not None:
        sc.pl.umap(integrated, color=['dataidx', cell_type_label])
    else:
        sc.pl.umap(integrated, color='dataidx')

    res = cnova.utils.calc_oobNN(integrated, batch_key='dataidx', condition_key='PseudoState')
    df = res.obsm['knn_prop']
    df['condition'] = res.obs['PseudoState']
    df = df.reset_index()
    df = pd.melt(df, id_vars=['index', 'condition'], var_name='neighbor', value_name='proportion')
    df = df.rename(columns={'index': 'obs_name'})
    df.to_csv("./cellanova/"+dataset_name+"_oobNN.csv")

    g = sea.FacetGrid(df, col='neighbor', hue='condition')
    g.map(sea.kdeplot, 'proportion', bw_adjust=2, alpha=1)
    g.set(xlabel='NN proportion', ylabel='Density')
    g.add_legend()
    plt.suptitle('CellANOVA integration')
    sea.set_style('white')
    plt.show()'''
import os
import torch


'''def run_scdisinfact(adata, batch_key, condition_key, dataset_name, cell_type_label=None):
    import warnings

    # Suppress all warnings
    warnings.filterwarnings("ignore")

    result_dir = "./scd/" + dataset_name + "/"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    if isinstance(adata.X, scipy.sparse.spmatrix):
        adata.X = adata.X.toarray()

    counts = adata.X
    adata.obs["batch"] = adata.obs[batch_key].copy()
    meta_cells = adata.obs.copy()

    if isinstance(condition_key, str):
        condition_key = [condition_key]

    if not isinstance(condition_key, list):
        print("Wrong condition_key, must be string or list of string")

    data_dict = create_scdisinfact_dataset(counts, meta_cells, condition_key=condition_key, batch_key=batch_key)

    # default setting of hyper-parameters
    reg_mmd_comm = 1e-4
    reg_mmd_diff = 1e-4
    reg_kl_comm = 1e-5
    reg_kl_diff = 1e-2
    reg_class = 1
    reg_gl = 1

    Ks = [8, 4]

    batch_size = 64
    nepochs = 100
    interval = 10
    lr = 5e-4
    lambs = [reg_mmd_comm, reg_mmd_diff, reg_kl_comm, reg_kl_diff, reg_class, reg_gl]
    model = scdisinfact(data_dict=data_dict, Ks=Ks, batch_size=batch_size, interval=interval, lr=lr,
                        reg_mmd_comm=reg_mmd_comm, reg_mmd_diff=reg_mmd_diff, reg_gl=reg_gl, reg_class=reg_class,
                        reg_kl_comm=reg_kl_comm, reg_kl_diff=reg_kl_diff, seed=0, device=device)
    model.train()
    losses = model.train_model(nepochs=nepochs, recon_loss="NB")
    torch.save(model.state_dict(), result_dir + f"model_{Ks}_{lambs}_{batch_size}_{nepochs}_{lr}.pth")
    model.load_state_dict(
        torch.load(result_dir + f"model_{Ks}_{lambs}_{batch_size}_{nepochs}_{lr}.pth", map_location=device))
    _ = model.eval()

    # one forward pass
    z_cs = []
    z_ds = []
    zs = []

    for dataset in data_dict["datasets"]:
        with torch.no_grad():
            # pass through the encoders
            dict_inf = model.inference(counts=dataset.counts_norm.to(model.device),
                                       batch_ids=dataset.batch_id[:, None].to(model.device), print_stat=True)
            # pass through the decoder
            dict_gen = model.generative(z_c=dict_inf["mu_c"], z_d=dict_inf["mu_d"],
                                        batch_ids=dataset.batch_id[:, None].to(model.device))
            z_c = dict_inf["mu_c"]
            z_d = dict_inf["mu_d"]
            z = torch.cat([z_c] + z_d, dim=1)
            mu = dict_gen["mu"]
            z_ds.append([x.cpu().detach().numpy() for x in z_d])
            z_cs.append(z_c.cpu().detach().numpy())
            zs.append(np.concatenate([z_cs[-1]] + z_ds[-1], axis=1))

    latent = np.concatenate(z_cs, axis=0)
    meta_dict = {}
    for namei in meta_cells.columns.tolist():
        meta_dict[namei] = np.concatenate([x[namei].values for x in data_dict["meta_cells"]])

    meta_df = pd.DataFrame(meta_dict)
    adata_latent = ad.AnnData(X=latent)
    adata_latent.obs = meta_df

    denoised_counts = model.predict_counts(input_counts=counts, meta_cells=meta_cells, condition_keys=condition_key,
                                           batch_key=batch_key, predict_conds=None, predict_batch=None)
    # adata.obsm["denoised"] = denoised_counts
    adata.layers["denoised"] = denoised_counts
    adata.layers['main_effect'] = latent

    adata.write_h5ad("./scd/" + dataset_name + "_latent.h5ad")

    sc.pp.neighbors(adata_latent, n_neighbors=15, n_pcs=50)
    sc.tl.umap(adata_latent)
    sc.pl.umap(adata_latent, color=condition_key, ncols=1)
    sc.pl.umap(adata_latent, color=batch_key, ncols=1)
    if cell_type_label is not None:
        sc.pl.umap(adata_latent, color=cell_type_label)'''

def plot_distribution_with_binary_zscore(vector, gene_name):
    """
    Plots the distribution of the values in the vector, calculates the z-score
    based on the proportion of values > 0 using CLT, and uses the gene name and z-score
    as the title of the plot.

    Args:
    - vector (numpy.ndarray or list): The input vector of values.
    - gene_name (str): The gene name to include in the title.

    Returns:
    - None: Displays the plot.
    """
    # Convert to numpy array if not already
    vector = np.array(vector)

    # Binary transformation: count the number of values > 0
    count_positive = np.sum(vector > 0)
    n = len(vector)
    proportion_positive = count_positive / n  # Proportion of values > 0

    # Calculate z-score using the CLT
    p_null = 0.5  # Null hypothesis: Proportion of values > 0 is 0.5
    std_error = np.sqrt(p_null * (1 - p_null) / n)
    z_score = (proportion_positive - p_null) / std_error

    # Plot the distribution of the vector
    plt.figure(figsize=(8, 6))
    plt.hist(vector, bins=30, alpha=0.7, edgecolor='k', color='blue')
    plt.axvline(0, color='red', linestyle='--', label='Reference: 0')
    plt.title(f"{gene_name} | Z-score: {z_score:.2f} (Proportion > 0: {proportion_positive:.2f})", fontsize=14)
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()

    # Show the plot
    plt.show()


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_distribution(vector, bins=30, kde=True, title="Distribution Plot"):
    """
    Plots the distribution of a vector using a histogram and optionally overlays a KDE fitted line.

    Parameters:
    - vector (array-like): The input data vector.
    - bins (int): Number of bins for the histogram. Default is 30.
    - kde (bool): Whether to add a KDE fitted line. Default is True.
    - title (str): Title for the plot. Default is "Distribution Plot".
    """
    plt.figure(figsize=(8, 6))

    # Plot histogram and optionally the KDE
    sns.histplot(vector, bins=bins, kde=kde, color='blue', stat='density', edgecolor='black')

    # Add titles and labels
    plt.title(title, fontsize=14)
    plt.xlabel("Value", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Show the plot
    plt.tight_layout()
    plt.show()

import pandas as pd


def calculate_mean_proportion_matrix(df):
    """
    Calculates the mean proportion for each combination of condition and neighbor
    and summarizes the result in a square matrix dataframe.

    Parameters:
    df (pd.DataFrame): Input dataframe with columns ['condition', 'neighbor', 'proportion'].

    Returns:
    pd.DataFrame: A square matrix dataframe where rows represent 'condition' and columns represent 'neighbor'.
    """
    # Use a pivot table to calculate the mean proportions
    mean_matrix = df.pivot_table(
        index='condition',
        columns='neighbor',
        values='proportion',
        aggfunc='mean',
        fill_value=0  # Replace NaN with 0 if there are missing combinations
    )

    return mean_matrix


def calculate_rowwise_correlation(adata1, adata2, batch_key="batch_all_with_condition"):
    # Ensure the obs index and batch_key match
    assert adata1.obs.index.equals(adata2.obs.index), "obs indices do not match between the two AnnData objects"
    assert batch_key in adata1.obs.columns, f"{batch_key} not found in adata1.obs"
    assert batch_key in adata2.obs.columns, f"{batch_key} not found in adata2.obs"

    results = []

    # Iterate through unique batches
    unique_batches = adata1.obs[batch_key].unique()
    for batch in unique_batches:
        # Subset the data for the current batch
        batch_mask = adata1.obs[batch_key] == batch
        data1 = adata1[batch_mask].X
        data2 = adata2[batch_mask].X

        # Ensure the data is in dense format if sparse
        if not isinstance(data1, np.ndarray):
            data1 = data1.toarray()
        if not isinstance(data2, np.ndarray):
            data2 = data2.toarray()

        # Compute correlation for each row
        for i in range(data1.shape[0]):
            row_corr = np.corrcoef(data1[i, :], data2[i, :])[0, 1]
            results.append({"correlation": row_corr, batch_key: batch})

    # Convert results to DataFrame
    result_df = pd.DataFrame(results)
    return result_df

if __name__=="__main__":
    '''# Example usage:
    vector = np.random.normal(loc=0, scale=1, size=100)  # Example data
    plot_distribution_with_binary_zscore(vector, "GeneXYZ")'''

    '''# Example usage:
    import numpy as np

    sample_data = np.random.normal(loc=0, scale=1, size=1000)  # Generate sample data
    plot_distribution(sample_data, bins=40, kde=True, title="Sample Data Distribution")'''

    '''# Create a sample dataframe
    data = {
        'obs_name': ['AAACCTGAGGCGCTCT-12', 'AAACCTGAGTTACCCA-12', 'AAACCTGAGTTCGCAT-12',
                     'TTTGGTTCAAGATCCT-19', 'TTTGGTTGGTGCCCACA-19'],
        'condition': ['AAB', 'AAB', 'AAB', 'T1D', 'T1D'],
        'neighbor': ['AAB', 'AAB', 'T1D', 'AAB', 'T1D'],
        'proportion': [0.142857, 0.428571, 0.142857, 0.0, 0.214286]
    }

    df = pd.DataFrame(data)
    print(calculate_mean_proportion_matrix(df))'''

    import numpy as np
    import pandas as pd
    import anndata

    # Create sample data
    n_obs = 4  # number of observations
    n_vars = 5  # number of variables

    # Generate random data for two AnnData objects
    adata1_X = np.random.randn(n_obs, n_vars)
    adata2_X = np.random.randn(n_obs, n_vars)
    print(adata1_X)
    print(adata2_X)

    # Create a shared observation dataframe with a 'batch_all_with_condition' column
    obs_data = pd.DataFrame({
        "batch_all_with_condition": ["batch1"] * 2 + ["batch2"] * 2
    })

    # Create AnnData objects
    adata1 = anndata.AnnData(X=adata1_X, obs=obs_data)
    adata2 = anndata.AnnData(X=adata2_X, obs=obs_data)

    print(calculate_rowwise_correlation(adata1, adata2))