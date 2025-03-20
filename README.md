# Decomposing condition-related signals, effect modifiers, and batch effect of complex forms in scRNA-seq studies with neural discrete representation learning

by: Xiao Xiao (xiao.xiao.xx244@yale.edu)



## Abstract

Advances in single-cell sequencing techniques and the growing volume of single-cell data have created unprecedented opportunities for uncovering statistically significant gene expression patterns induced by perturbations or associated with specific disease conditions. However, existing methods struggle to effectively denoise batch-effect-free expressions and estimate condition-related responses due to neglecting data non-linearity and introducing unwanted biases. To address them, we developed NDreamer, a novel approach that combines neural discrete representation learning with counterfactual causal matching. NDreamer can be used to denoise batch-effect-free and condition-related-signal-preserved expression data while accurately estimating single-cell-level condition-induced signals. Benchmarked on datasets across platforms, organs, and species, NDreamer robustly outperformed previous single-cell-level perturbation effect estimation methods and batch effect denoising methods. Finally, we applied NDreamer to a large Alzheimer’s disease cohort and uncovered meaningful gene expression patterns between the dementia patients and health controls for each cell type.

---

In a word, denoise the batch effect to generate the batch-effect-free and condition-related-signal-preserved expression data, and calculate the effect related to or induced by different conditions (e.g. health v.s. disease) or perturbations (e.g. control v.s. stimulated) in the original gene expression space.

For more details, please refere to our paper at https://www.biorxiv.org/content/10.1101/2025.03.05.641743v1



## Data you need

An anndata object that includes a cell by gene matrix and the condition, batch information of each cell stored in the same anndata object.

## Installation

```bash
git clone https://github.com/lugia-xiao/NDreamer.git
cd ndreamer
pip install .
```

## Quick start

### Import the package

```python
import scanpy as sc
import numpy as np
from ndreamer import NDreamer
```

### Load data

```python
adata = sc.read_h5ad('./data/xxx.h5ad')
```

### Run NDreamer

#### Necessary inputs

- **adata** (`AnnData`):
  Single-cell gene expression data in `AnnData` format, typically preprocessed and stored as an `h5ad` file.
- **condition_key** (`str`):
  The key in `adata.obs` that contains condition labels. Here, it is 'condition' since the cell's condition meta data is stored in adata.obs\['condition'\].
- **control_name** (`str`):
  The name of the control condition in `adata.obs[condition_key]`. We will then calculate the individual treatment effect for all other conditions compared to the `control_name` condition. Here, the control name is "control".
- **num_hvg** (`int`, default: `2000`):
  Number of highly variable genes to select for analysis. If the input adata have number of genes more than `num_hvg`, then NDreamer will automatically select `num_hvg` number of highly variable genes.
- **require_batch** (`bool`, default: `False`):
  Whether batch correction is required. Here, we do not need it.
- **batch_key** (`str` or `None`, default: `None`):
  The key in `adata.obs` that contains batch labels (if batch correction is needed). Multiple batch categories are supported. For example, you can input `batch_key="patient_id"`, or if you have patient id and sequencing technique that both make up the batch effect, then you can do `batch_key=["sequencing_tech",patient_id"]`
- **save_pth** (`str`, default: `"./PBMC/"`):
  Directory path for saving model outputs.
- **save_preprocessed_adata_path** (`str`, here we set it to: `"./PBMC/preprocessed.h5ad"`):
  Path to save the preprocessed `adata` file. If `None`, then the preprocessed results will not be saved
- **Keep all other inputs similar to the values below:**

#### When we do not consider the batch effect

```python
# define the model
model = NDreamer(adata, condition_key="condition", contorl_name='control', num_hvg=2000, require_batch=False,batch_key=None, save_pth="./PBMC/",save_preprocessed_adata_path="./PBMC/preprocessed.h5ad",library_size_normalize_adata=False,resolution_low=0.5, resolution_high=7, cluster_method="Leiden", embedding_dim=512,codebooks=[1024 for i in range(32)],codebook_dim=8, encoder_hidden=[1024, 512], decoder_hidden=[512, 1024], z_dim=256,cos_loss_scaler=20, random_seed=123, batch_size=1024, epoches=100, lr=1e-3,triplet_margin=5,independent_loss_scaler=1000,developer_test_mode=False,KL_scaler=5e-3,reconstruct_scaler=1, triplet_scaler=5,num_triplets_per_label=15,tau=0.01,commitment_loss_scaler=1,cluster_correlation_scaler=50,reset_threshold=1/1024,reset_interval=30,try_identify_cb_specific_subtypes=False,local_neighborhood_loss_scaler=1,local_neighbor_sigma=1,try_identify_perturb_escaped_cell=False,n_neighbors=20,local_neighbor_across_cluster_scaler=20)

# train the model
model.train_model()

# get the effect modifier space
model.get_modifier_space()

# denoise batch effect (if present)
model.decompose_true_expression_batch_effect_all(nearest_neighbor=1,bandwidth=1)

# Estimate ITE for conditions other than the control condition
model.Estmiate_ITE_all(nearest_neighbor=1,bandwidth=1)
```

#### When we do consider the batch effect

```python
# define the model
model = NDreamer(adata, condition_key="condition", contorl_name='control', num_hvg=2000, require_batch=True,batch_key="patient_id", save_pth="./PBMC/",save_preprocessed_adata_path="./PBMC/preprocessed.h5ad",library_size_normalize_adata=False,resolution_low=0.5, resolution_high=7, cluster_method="Leiden", embedding_dim=512,codebooks=[1024 for i in range(32)],codebook_dim=8, encoder_hidden=[1024, 512], decoder_hidden=[512, 1024], z_dim=256,cos_loss_scaler=20, random_seed=123, batch_size=1024, epoches=100, lr=1e-3,triplet_margin=5,independent_loss_scaler=1000,developer_test_mode=False,KL_scaler=5e-3,reconstruct_scaler=1, triplet_scaler=5,num_triplets_per_label=15,tau=0.01,commitment_loss_scaler=1,cluster_correlation_scaler=50,reset_threshold=1/1024,reset_interval=30,try_identify_cb_specific_subtypes=False,local_neighborhood_loss_scaler=1,local_neighbor_sigma=1,try_identify_perturb_escaped_cell=False,n_neighbors=20,local_neighbor_across_cluster_scaler=20)

# train the model
model.train_model()

# get the effect modifier space
model.get_modifier_space()

# denoise batch effect (if present)
model.decompose_true_expression_batch_effect_all(nearest_neighbor=1,bandwidth=1)

# Estimate ITE for conditions other than the control condition
model.Estmiate_ITE_all(nearest_neighbor=1,bandwidth=1)
```

If `nearest_neighbor>1` (must be integer), then you are doing kernel weighting. If `nearest_neighbor=1`, then you are doing causal matching. Keep `nearest_neighbor` equals to 1 unless you know what you are doing.

### Results

After the calculation, all results would be stored in `save_pth`, including:

1. `adata.h5ad`
    - `AnnData` object containing effect modifier embeddings:
        - `model.DL_model.adata.obsm['X_effect_modifier_space']`: Raw effect modifier embeddings.
        - `model.DL_model.adata.obsm['X_effect_modifier_space_PCA']`: PCA-transformed effect modifier embeddings.
    - Can be retrieved using: `model.DL_model.adata.copy()`.
    - There is also `adata.obs["original_order"]` that store the original order of each cell in the original input adata
2. `batch2num_mapping.pth`
    - Mapping of batch identifiers to numerical indices for batch encoding.
3. `ndreamer.pth`
    - Trained **NDreamer** model parameters.
4. `b-x--c-y__ITE.h5ad`
    - `AnnData` object storing **Individual Treatment Effects (ITE)** for condition `c=y` and batch `b=x`, the number here are number encoding for the condition and batch, if we do not consider batch effect, all data belongs to the `0` batch. The mapping between condition and the number is available at `condition2num_mapping.pth`. Though we split the ITE for each unique batch-condition combination, the original order of each cell in the orginal input adata is stored at the `.obs["original_order"]`. Please note that $c\ne 0$ since that is the control condition.
5. `c-y--b-x__expression.h5ad` The denoised expression of the x batch and y condition. x and y are numbers with their mapping to the original batch/condition is available at the two num_mapping.pth files. Though we split the denoised expression for each unique batch-condition combination, the original order of each cell in the orginal input adata is stored at the `.obs["original_order"]`.
6. `c-y--b-x__batch.h5ad`: The estimated batch-effect-induced expression of the x batch and y condition. x and y are numbers with their mapping to the original batch/condition is available at the two num_mapping.pth files. Though we split the denoised expression for each unique batch-condition combination, the original order of each cell in the orginal input adata is stored at the `.obs["original_order"]`.
7. `condition2num_mapping.pth`
    - Mapping of condition labels to numerical indices.
8. `preprocessed.h5ad`
    - Preprocessed `AnnData` object used as input for model training.



For more detailed tutorial, please refer to `tutorial_experimental_PBMC.ipynb` and `tutorial_case_contorl_T1D.ipynb`.



## Reproducibility

All codes used to generate data and figures provided in the manuscript are available at https://github.com/lugia-xiao/NDreamer_reproducible

## License

This project is licensed under the terms of GNU GENERAL PUBLIC LICENSE Version 3.