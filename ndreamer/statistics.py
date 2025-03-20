import os
import numpy as np
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

import warnings

def run_kbet(df, do_PCA=False):
    """
    Run kBET analysis using R from Python.

    Parameters:
    df (pd.DataFrame): Input dataframe with a 'condition' column and columns containing 'latent' in their names.

    Returns:
    R object: The kBET result from R.
    """
    warnings.warn(
        "Running kBET (https://github.com/theislab/kBET/) may take a significant amount of time and CPU memory. Please be patient.",
        UserWarning
    )

    if "R_HOME" not in os.environ:
        print("R_HOME was not set, please set this environment variable first!, e.g. os.environ[\"R_HOME\"]=r\"D:/opt/R/R-4.3.1\"")
    else:
        print("R_HOME is already set to:", os.environ["R_HOME"])

    import pandas as pd
    pd.DataFrame.iteritems = pd.DataFrame.items

    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr
    from rpy2.robjects import r

    # Activate the automatic pandas to R conversion
    pandas2ri.activate()

    import anndata2ri
    anndata2ri.activate()

    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri
    pandas2ri.activate()

    # Perhaps require the library relocation (to the library contains kBET and lisi library)
    robjects.r('.libPaths(c("current path", "library location"))')

    rscript = '''
    library(kBET)
    library(lisi)
    '''
    robjects.r(rscript)
    kbet = robjects.r('kBET')
    lisi = robjects.r['compute_lisi']

    # Ensure the required R library is installed
    kBET = importr('kBET')

    # Pass the dataframe to R
    r_df = pandas2ri.py2rpy(df)
    r.assign("df", r_df)

    # Extract the batch and data columns in R
    r('batch <- df[,"group"]')
    r('data <- as.matrix(df[,grep("latent", colnames(df))])')

    str_do_PCA=",do.pca = "
    if do_PCA:
        str_do_PCA=str_do_PCA+"TRUE"
    else:
        str_do_PCA=str_do_PCA+"FALSE"

    # Perform kBET
    kbet_result = r('kbet_result <- kBET(data, batch'+str_do_PCA+')')

    # Print results in R
    print(r('print(kbet_result)'))

    # Extract the average p-value
    average_pval = r('kbet_result$average.pval')[0]
    if average_pval >= 0.05:
        print("P-value larger than 0.05, indicating good mixing performance")

    return kbet_result

def get_significant_names(z_scores, names, alpha=0.05, method='fdr_bh'):
    """
    Calculate two-sided p-values from z-scores, adjust p-values,
    and return significant names sorted by descending z-scores.

    Parameters:
        z_scores (np.array): Array of z-scores.
        names (list): List of names corresponding to the z-scores.
        alpha (float): Significance level for p-value adjustment.
        method (str): Method for p-value adjustment (default: 'fdr_bh').

    Returns:
        list: Significant names after p-value adjustment, sorted by descending z-scores.
    """
    # Calculate two-sided p-values
    p_values = 2 * norm.sf(np.abs(z_scores))

    # Adjust p-values
    _, p_adj, _, _ = multipletests(p_values, alpha=alpha, method=method)

    # Get significant indices
    significant_indices = [i for i in range(len(names)) if p_adj[i] < alpha]

    # Extract significant z-scores and names
    significant_z_scores = z_scores[significant_indices]
    significant_names = [names[i] for i in significant_indices]

    # Sort by descending absolute z-scores
    sorted_indices = np.argsort(-np.abs(significant_z_scores))  # Negative sign for descending order
    sorted_names = [significant_names[i] for i in sorted_indices]

    return sorted_names

if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    # Example dataframe
    # Randomly generate conditions (batches)
    # Create a larger dataframe with simulated data
    n_samples = 1000
    n_latent_features = 10

    conditions = np.random.choice(['batch1', 'batch2', 'batch3'], n_samples)

    # Randomly generate latent features
    latent_features = {f'latent_{i + 1}': np.random.randn(n_samples) for i in range(n_latent_features)}

    # Combine into a dataframe
    data = {
        'group': conditions,
        **latent_features
    }

    df = pd.DataFrame(data)

    # Run kBET analysis
    result = run_kbet(df)