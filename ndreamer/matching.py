import numpy as np
import scipy.spatial
import torch

import matplotlib.pyplot as plt
from scipy.stats import norm

from ndreamer.plot import plot_distribution_with_binary_zscore


def weighted_mean_neighbors(
    adata,
    reference_batch,
    reference_condition,
    bandwidth=1.0,
    nearest_neighbor=5,
    save_path="results_weighting.pth"
):
    """
    Compute the weighted mean matrix for distinct condition and batch combinations
    for cells in a reference batch and condition.

    Parameters:
    - adata: AnnData object containing obsm 'X_effect_modifier_space_PCA',
             obs 'batch', and 'condition', and X (gene expression matrix).
    - reference_batch: The batch to use as the reference.
    - reference_condition: The condition to use as the reference.
    - bandwidth: The bandwidth for the softmax function (default: 1.0).
    - nearest_neighbor: The number of nearest neighbors to consider (default: 10).

    Returns:
    - A dictionary with keys as tuples of (batch, condition) and values as matrices
      of the weighted mean for the reference points.
    """
    if nearest_neighbor==1:
        bandwidth=0.00001
        nearest_neighbor=2

    if "X_effect_modifier_space_PCA" not in adata.obsm:
        print("Please run .get_modifier_space() first")
        raise ValueError("adata must have obsm 'X_effect_modifier_space_PCA'")
    if "batch" not in adata.obs or "condition" not in adata.obs:
        raise ValueError("adata must have obs 'batch' and 'condition'")
    if adata.X is None:
        raise ValueError("adata must have a gene expression matrix in X")

    # Extract relevant data
    pca_space = adata.obsm["X_effect_modifier_space_PCA"]
    # Check if adata.X is a sparse matrix
    if scipy.sparse.issparse(adata.X):
        # Convert sparse matrix to dense matrix
        adata.X = adata.X.toarray()
    expression = adata.X
    batches = adata.obs["batch"]
    conditions = adata.obs["condition"]

    # Filter reference indices
    reference_indices = (batches == reference_batch) & (conditions == reference_condition)
    reference_points = pca_space[reference_indices]
    reference_expression = expression[reference_indices]

    # Store results
    results = {}

    # Iterate over each unique batch and condition
    unique_batches = batches.unique()
    unique_conditions = conditions.unique()
    for batch in unique_batches:
        for condition in unique_conditions:
            # Skip reference batch and condition
            if batch == reference_batch and condition == reference_condition:
                results[(batch, condition)] = reference_expression
                continue

            # Get indices of points for the current batch and condition
            target_indices = (batches == batch) & (conditions == condition)
            target_points = pca_space[target_indices]
            target_expression = expression[target_indices]

            if target_points.shape[0] == 0:
                continue

            # Build a KDTree specifically for the target batch/condition
            target_tree = scipy.spatial.cKDTree(target_points)

            # Compute neighbors and weights
            weighted_expression = np.zeros(reference_expression.shape)

            for i, ref_point in enumerate(reference_points):
                distances, indices = target_tree.query(
                    ref_point, k=min(nearest_neighbor, len(target_points))
                )

                if isinstance(distances,float):
                    distances=np.array([distances])
                if isinstance(indices,float):
                    indices=np.array([indices])

                valid_distances = distances[~np.isinf(distances)]
                valid_indices = indices[~np.isinf(distances)]

                if len(valid_distances) == 0:
                    # If no neighbors are found (shouldn't happen now), fill with NaN
                    weighted_expression[i] = np.nan
                    print("No neighbors are found?")
                    continue

                # Apply softmax weights
                valid_distances=valid_distances-np.min(valid_distances)
                weights = np.exp(-valid_distances / bandwidth)
                weights /= np.sum(weights)  # Normalize

                # Compute weighted mean expression for the valid neighbors
                neighbors_expression = target_expression[valid_indices]
                weighted_expression[i] = np.sum(neighbors_expression*np.expand_dims(weights,axis=-1),axis=0)

            results[(batch, condition)] = weighted_expression

    if save_path is not None:
        torch.save(results, save_path)
    return results

def Estimate_ITE(
    adata,
    reference_batch,
    reference_condition,
    bandwidth=1.0,
    nearest_neighbor=10,
    save_path_counterfactual="results_counterfactual.pth",
    save_path_ITE="results_ITE.pth"
):
    """
    Estimate the Individual Treatment Effect (ITE) by comparing the mean expression
    across conditions to the reference condition.

    Parameters:
    - adata: AnnData object containing obsm 'X_effect_modifier_space_PCA',
             obs 'batch', and 'condition', and X (gene expression matrix).
    - reference_batch: The batch to use as the reference.
    - reference_condition: The condition to use as the reference.
    - bandwidth: The bandwidth for the softmax function (default: 1.0).
    - nearest_neighbor: The number of nearest neighbors to consider (default: 10).
    - save_path: Path to save the intermediate results using PyTorch (default: "results.pth").

    Returns:
    - A dictionary with keys as distinct conditions (other than the reference) and values
      as matrices representing the cell-by-gene differential expression.
    """
    # Use the previous function to calculate weighted means for all (batch, condition) pairs
    weighted_results = weighted_mean_neighbors(
        adata,
        reference_batch,
        reference_condition,
        bandwidth,
        nearest_neighbor,
        save_path_counterfactual,
    )

    # Extract relevant data for calculating ITE
    batches = adata.obs["batch"]
    conditions = adata.obs["condition"]

    # Compute the mean expression for the reference condition across all batches
    reference_batches = batches[conditions == reference_condition].unique()
    reference_means = []
    #print(reference_batches,reference_condition)
    for batch in reference_batches:
        if (batch, reference_condition) in weighted_results:
            reference_means.append(weighted_results[(batch, reference_condition)])

    if not reference_means:
        raise ValueError("No valid batches found for the reference condition.")

    # Compute the mean cell-by-gene matrix for the reference condition
    reference_mean_matrix = np.mean(reference_means, axis=0)

    # Dictionary to store the ITE results
    ITE_results = {}

    # Iterate over conditions (excluding the reference condition)
    unique_conditions = conditions.unique()
    for condition in unique_conditions:
        if condition == reference_condition:
            continue

        # Collect weighted means across all batches for the current condition
        condition_means = []
        for batch in batches[conditions == condition].unique():
            if (batch, condition) in weighted_results:
                condition_means.append(weighted_results[(batch, condition)])
            else:
                print("Wrong, (batch, condition) not calculated in matching results?")

        if not condition_means:
            continue  # Skip if there are no valid batches for the condition

        # Compute the mean cell-by-gene matrix for the current condition
        condition_mean_matrix = np.mean(condition_means, axis=0)

        # Compute the cell-by-gene differential expression matrix
        differential_matrix = condition_mean_matrix - reference_mean_matrix

        # Store the result in the dictionary
        ITE_results[condition] = differential_matrix
    #print(save_path_ITE)
    if save_path_ITE is not None:
        torch.save(ITE_results, save_path_ITE)
    return ITE_results


def Plot_and_Estimate_CATE(
    adata,
    ITE_results,
    target_condition,
    reference_condition,
    reference_batch,
    genes=None,
    topk=20,
    indices=None,
    cell_type=None,
    cell_type_key=None,
    plot=True
):
    """
    Estimate and plot the Conditional Average Treatment Effect (CATE) for specific genes
    and cells.

    Parameters:
    - adata: AnnData object containing gene expression data.
    - ITE_results: Output dictionary from the `Estimate_ITE` function.
    - target_condition: The condition to compare against the reference condition.
    - reference_condition: The reference condition.
    - reference_batch: The reference batch.
    - genes: List of genes to consider (default: None).
    - topk: Number of top significant genes to select if `genes` is None (default: 20).
    - indices: Indices of cells to analyze (default: None).
    - cell_type: Target cell type to select (used if `indices` is None, default: None).
    - cell_type_key: Key in `adata.obs` for cell type annotation (used if `indices` is None, default: None).

    Returns:
    - A dictionary of z-scores for the selected genes.
    """
    # Validate input
    if indices is None and (cell_type is None or cell_type_key is None):
        raise ValueError("Either `indices` or both `cell_type` and `cell_type_key` must be provided.")

    # Determine indices if not provided
    if indices is None:
        ref_indices = (adata.obs["condition"] == reference_condition) & \
                      (adata.obs["batch"] == reference_batch) & \
                      (adata.obs[cell_type_key] == cell_type)
        indices = np.where(ref_indices)[0]

    # Extract the ITE matrix for the target condition
    if target_condition not in ITE_results:
        raise ValueError(f"Target condition '{target_condition}' not found in ITE_results.")
    ITE_matrix = ITE_results[target_condition]

    # Subset the ITE matrix using the selected indices
    ITE_subset = ITE_matrix[indices, :]

    # Initialize z-scores for genes
    z_scores = []

    # Calculate z-scores for each gene (column in the ITE matrix)
    for gene_idx in range(ITE_subset.shape[1]):
        gene_ite = ITE_subset[:, gene_idx]
        # Test if ITE > 0 using non-parametric test (binomial distribution assumption)
        n = len(gene_ite)
        successes = np.sum(gene_ite > 0) + 0.5*np.sum(gene_ite==0)
        # z-score for binomial test
        z_score = (successes - n * 0.5) / np.sqrt(n * 0.25)
        z_scores.append(z_score)

    # Select top-k significant genes if `genes` is not provided
    z_scores = np.array(z_scores)
    if genes is None:
        # Get indices of top-k absolute z-scores
        topk_indices = np.argsort(-np.abs(z_scores))[:topk]
        genes = adata.var_names[topk_indices]

    # Print the selected most significant genes
    print("Selected most significant genes:")
    print(genes)

    if plot:
        # Plot the distribution of ITE values for the selected genes
        plt.figure(figsize=(10, 6))
        for gene in genes:
            print("Plotting:", gene)
            gene_idx = np.where(adata.var_names == gene)[0][0]
            ite_values = ITE_subset[:, gene_idx]

            plot_distribution_with_binary_zscore(vector=ite_values, gene_name=gene)

    # Return only the z-score
    return z_scores


def Plot_and_Estimate_CATE_adata(
    adata,
    genes=None,
    topk=20,
    indices=None,
    plot=True
):
    # Subset the ITE matrix using the selected indices
    ITE_subset = adata.X[indices, :]

    # Initialize z-scores for genes
    z_scores = []

    # Calculate z-scores for each gene (column in the ITE matrix)
    for gene_idx in range(ITE_subset.shape[1]):
        gene_ite = ITE_subset[:, gene_idx]
        # Test if ITE > 0 using non-parametric test (binomial distribution assumption)
        n = len(gene_ite)
        successes = np.sum(gene_ite > 0) + 0.5*np.sum(gene_ite==0)
        # z-score for binomial test
        z_score = (successes - n * 0.5) / np.sqrt(n * 0.25)
        z_scores.append(z_score)

    # Select top-k significant genes if `genes` is not provided
    z_scores = np.array(z_scores)
    if genes is None:
        # Get indices of top-k absolute z-scores
        topk_indices = np.argsort(-np.abs(z_scores))[:topk]
        genes = adata.var_names[topk_indices]

    # Print the selected most significant genes
    print("Selected most significant genes:")
    print(genes)

    if plot:
        # Plot the distribution of ITE values for the selected genes
        plt.figure(figsize=(10, 6))
        for gene in genes:
            print("Plotting:", gene)
            gene_idx = np.where(adata.var_names == gene)[0][0]
            ite_values = ITE_subset[:, gene_idx]

            plot_distribution_with_binary_zscore(vector=ite_values, gene_name=gene)

    # Return only the z-score
    return z_scores


