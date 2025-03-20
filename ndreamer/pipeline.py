import os
import torch
import warnings

from ndreamer.model import NDreamer_DL
from ndreamer.matching import *
from ndreamer.statistics import get_significant_names
import time
import anndata as ad


class NDreamer(torch.nn.Module):
    def __init__(self, adata, condition_key, contorl_name, num_hvg, require_batch=False, batch_key=None,
                 resolution_low=0.5, resolution_high=7, cluster_method="Leiden", embedding_dim=512 ,codebooks=None,
                 codebook_dim=8, encoder_hidden=None, decoder_hidden=None, z_dim=256,
                 cos_loss_scaler=20, random_seed=123, batch_size=2048,epoches=10,lr=1e-3,triplet_margin=5,
                 independent_loss_scaler=1000,save_pth="./model/",developer_test_mode=False,
                 library_size_normalize_adata=False,save_preprocessed_adata_path="./model/preprocessed.h5ad",
                 KL_scaler=5e-3,reconstruct_scaler=1,triplet_scaler=5,num_triplets_per_label=15,
                 tau=0.01,commitment_loss_scaler=1,cluster_correlation_scaler=50,
                 try_identify_perturb_escaped_cell=False,reset_threshold=1/1024, reset_interval=30,
                 try_identify_cb_specific_subtypes=False,local_neighborhood_loss_scaler=1,
                 local_neighbor_sigma=1,n_neighbors=20,local_neighbor_across_cluster_scaler=20,
                 have_negative_data=False):
        super(NDreamer, self).__init__()

        start_time = time.time()

        self.DL_model = NDreamer_DL(adata=adata, condition_key=condition_key, contorl_name=contorl_name,
                                    num_hvg=num_hvg, require_batch=require_batch, batch_key=batch_key,
                                    resolution_low=resolution_low, resolution_high=resolution_high,
                                    cluster_method=cluster_method, embedding_dim=embedding_dim,
                                    codebooks=codebooks, codebook_dim=codebook_dim,
                                    encoder_hidden=encoder_hidden, decoder_hidden=decoder_hidden, z_dim=z_dim,
                                    cos_loss_scaler=cos_loss_scaler, random_seed=random_seed,
                                    batch_size=batch_size, epoches=epoches, lr=lr, triplet_margin=triplet_margin,
                                    independent_loss_scaler=independent_loss_scaler,
                                    save_pth=save_pth, developer_test_mode=developer_test_mode,
                                    library_size_normalize_adata=library_size_normalize_adata,
                                    save_preprocessed_adata_path=save_preprocessed_adata_path,
                                    KL_scaler=KL_scaler, reconstruct_scaler=reconstruct_scaler,
                                    triplet_scaler=triplet_scaler, num_triplets_per_label=num_triplets_per_label,
                                    tau=tau, commitment_loss_scaler=commitment_loss_scaler,
                                    cluster_correlation_scaler=cluster_correlation_scaler,
                                    try_identify_perturb_escaped_cell=try_identify_perturb_escaped_cell,
                                    try_identify_cb_specific_subtypes=try_identify_cb_specific_subtypes,
                                    reset_threshold=reset_threshold, reset_interval=reset_interval,
                                    local_neighborhood_loss_scaler=local_neighborhood_loss_scaler,
                                    local_neighbor_sigma=local_neighbor_sigma, n_neighbors=n_neighbors,
                                    local_neighbor_across_cluster_scaler=local_neighbor_across_cluster_scaler,
                                    have_negative_data=have_negative_data)
        self.save_pth = save_pth
        self.ITE = None
        self.reference_condition = None
        self.reference_batch = None

        self.true_exp_adatas = {}
        self.batch_effect_adatas = {}
        self.have_run_decompose_true_expression_batch_effect_all = False
        self.condition_effect_adatas = {}

        end_time = time.time()

        if not developer_test_mode:
            torch.save(self.DL_model.batch_dict, os.path.join(self.save_pth, "batch2num_mapping.pth"))
            torch.save(self.DL_model.condition_dict, os.path.join(self.save_pth, "condition2num_mapping.pth"))

            print("Batch name to number mapping (may be used for search for saved adata):\n", self.DL_model.batch_dict)
            print("Condition name to number mapping (may be used for search for saved adata):\n",
                  self.DL_model.condition_dict)

        print(f"Preprocessing time: {end_time - start_time:.5f} seconds")

    def train_model(self):
        start_time = time.time()
        self.DL_model.train_model()
        end_time = time.time()
        print(f"Model training time: {end_time - start_time:.5f} seconds")

    def get_modifier_space(self, PCA_dim=50, save_path_adata=None, save_path_latent_df=None,
                           print_codebook_statisitics=False, kBET_test=False):
        start_time = time.time()
        self.DL_model.get_modifier_space(dim=PCA_dim, save_path_adata=save_path_adata,
                                         save_path_latent_df=save_path_latent_df,
                                         print_codebook_statisitics=print_codebook_statisitics,
                                         kBET_test=kBET_test)
        end_time = time.time()
        print(f"Model evaluating time: {end_time - start_time:.5f} seconds")

    '''def get_effect_modifier_space_criteria(self, celltype_key, batch_key="group",
                                           all=True, n_neighbors=5, use_PCA_embed=False):
        warnings.warn(
            "To run this, make sure you have set Environmental variable `R_Home`, if not, use os.environ['R_HOME'] = r'D:/opt/R/R-4.3.1' (change to your path)",
            UserWarning
        )

        import pandas as pd
        pd.DataFrame.iteritems = pd.DataFrame.items
        import anndata2ri
        anndata2ri.activate()

        import rpy2.robjects as robjects
        from rpy2.robjects import pandas2ri
        pandas2ri.activate()

        # Perhaps require the library relocation (to the library contains kBET and lisi library)
        robjects.r('.libPaths(c("current path", "library location"))')

        rscript = "library(kBET); library(lisi)"

        robjects.r(rscript)
        kbet = robjects.r('kBET')
        lisi = robjects.r['compute_lisi']

        embed='X_effect_modifier_space_PCA' if use_PCA_embed else 'X_effect_modifier_space'
        calculate_metrics(adata=self.DL_model.adata,celltype_key=celltype_key,
                          batch_key=batch_key,all=all,n_neighbors=n_neighbors, embed=embed)'''

    def load_effect_modifier_space_adata(self, adata):
        self.DL_model.adata = adata

    def Estimate_ITE(self, this_condition, this_batch=None, bandwidth=1.0, nearest_neighbor=1,
                     save_path_counterfactual=None, save_path_ITE=None, raw_name=True):
        reference_condition = this_condition
        reference_batch = this_batch

        start_time = time.time()
        if self.DL_model.require_batch == True:
            if reference_batch is None:
                print("Wrong, in datasets that have batches within condition, you must specify batch")
                return

        if not self.DL_model.require_batch:
            reference_batch = 0

        self.reference_condition = reference_condition
        self.reference_batch = reference_batch

        name = "b-" + str(self.reference_batch) + "--c-" + str(self.reference_condition) + "__"

        if raw_name:
            reference_condition = self.DL_model.condition_dict[reference_condition]
            if self.DL_model.require_batch:
                reference_batch = self.DL_model.batch_dict[reference_batch]
            else:
                reference_batch = 0

        valid_indices = np.logical_and(self.DL_model.adata.obs["batch"] == reference_batch,
                                       self.DL_model.adata.obs["condition"] == reference_condition)
        adata_part = self.DL_model.adata[valid_indices].copy()

        self.ITE = Estimate_ITE(adata=self.DL_model.adata, reference_batch=reference_batch,
                                reference_condition=reference_condition, bandwidth=bandwidth,
                                nearest_neighbor=nearest_neighbor, save_path_counterfactual=save_path_counterfactual,
                                save_path_ITE=save_path_ITE)
        end_time = time.time()
        if reference_condition != 0:
            ITE_i = self.ITE[0]
            adata_ITE = ad.AnnData(X=-ITE_i, obs=adata_part.obs, var=adata_part.var)# Now, it is f(X_i,c_i=1)-f(X_i,c_i=0)
            adata_ITE.write(os.path.join(self.save_pth, name + "ITE.h5ad"))

        print(f"ITE evaluating: {end_time - start_time:.5f} seconds")
        return self.ITE

    def Estmiate_ITE_all(self, nearest_neighbor=1, bandwidth=1):
        unique_group = np.unique(self.DL_model.adata.obs["group"])
        for groupi in unique_group:
            batchi, conditioni = groupi.split("_")
            batchi = int(batchi)  # self.DL_model.batch_dict[batchi]
            conditioni = int(conditioni)  # self.DL_model.condition_dict[conditioni]
            self.Estimate_ITE(this_condition=conditioni, this_batch=batchi,
                              nearest_neighbor=nearest_neighbor, bandwidth=bandwidth,
                              raw_name=False)

    def decompose_true_expression__batch_effect(self, this_condition, this_batch=None, bandwidth=1.0,
                                                nearest_neighbor=1, save_path_exp=None,
                                                save_path_batch=None, use_raw_name=True):

        if use_raw_name:
            this_condition1 = self.DL_model.condition_dict[this_condition]
            if self.DL_model.require_batch:
                this_batch1 = self.DL_model.batch_dict[this_batch]
            else:
                this_batch = "0"
                this_batch1 = 0
        else:
            this_condition1 = this_condition
            if not self.DL_model.require_batch:
                this_batch = 0
            this_batch1 = this_batch

        if save_path_exp is None:
            save_path_exp = os.path.join(self.save_pth,
                                         "c-" + str(this_condition) + "--" + "b-" + str(this_batch) + "__expression.h5ad")
        if save_path_batch is None:
            save_path_batch = os.path.join(self.save_pth,
                                           "c-" + str(this_condition) + "--" + "b-" + str(this_batch) + "__batch.h5ad")

        valid_indices = np.logical_and(self.DL_model.adata.obs["batch"] == this_batch1,
                                       self.DL_model.adata.obs["condition"] == this_condition1)
        adata_part = self.DL_model.adata[valid_indices].copy()

        if self.DL_model.require_batch == False:
            print("There is no batch in your dataset, return the expression")
            exp = self.DL_model.adata.X[valid_indices, :]
            adata_exp = ad.AnnData(X=exp, obs=adata_part.obs, var=adata_part.var)
            adata_exp.write(save_path_exp)
            adata_batch_effect = ad.AnnData(X=np.zeros(exp.shape), obs=adata_part.obs, var=adata_part.var)

            #self.true_exp_adatas[(str(this_condition1), str(this_batch1))] = adata_exp
            #self.batch_effect_adatas[(str(this_condition1), str(this_batch1))] = adata_batch_effect

            return adata_exp, adata_batch_effect

        mathing_dict = weighted_mean_neighbors(adata=self.DL_model.adata, reference_batch=this_batch1,
                                               reference_condition=this_condition1, bandwidth=bandwidth,
                                               nearest_neighbor=nearest_neighbor)

        results = []
        for keyi in mathing_dict.keys():
            batch, condition = keyi
            if condition == this_condition1:
                results.append(mathing_dict[keyi])
        batch_effect_free_expression = np.mean(np.stack(results, axis=0), axis=0)
        batch_effect = mathing_dict[(this_batch1, this_condition1)] - batch_effect_free_expression
        adata_exp = ad.AnnData(X=batch_effect_free_expression, obs=adata_part.obs, var=adata_part.var)
        adata_batch_effect = ad.AnnData(X=batch_effect, obs=adata_part.obs, var=adata_part.var)

        '''rs=np.array([
            np.corrcoef(batch_effect_free_expression[i, :], adata_part.X.toarray()[i, :])[0,1]
            for i in range(batch_effect_free_expression.shape[0])
        ])
        print(rs.tolist())
        print(np.mean(rs),np.max(rs),np.min(rs),np.median(rs))'''

        adata_exp.write_h5ad(save_path_exp)
        adata_batch_effect.write_h5ad(save_path_batch)

        #self.true_exp_adatas[(str(this_condition1), str(this_batch1))] = adata_exp
        #self.batch_effect_adatas[(str(this_condition1), str(this_batch1))] = adata_batch_effect
        return adata_exp, adata_batch_effect

    def decompose_true_expression_batch_effect_all(self, bandwidth=1.0, nearest_neighbor=1):
        self.have_run_decompose_true_expression_batch_effect_all = True
        unique_group = np.unique(self.DL_model.adata.obs["group"])
        for groupi in unique_group:
            batchi, conditioni = groupi.split("_")
            batchi = int(batchi)  # self.DL_model.batch_dict[batchi]
            conditioni = int(conditioni)  # self.DL_model.condition_dict[conditioni]
            self.decompose_true_expression__batch_effect(this_condition=conditioni, this_batch=batchi,
                                                         nearest_neighbor=nearest_neighbor,
                                                         bandwidth=bandwidth, use_raw_name=False)

    def Plot_and_Estimate_CATE(self, this_condition, target_condition, reference_batch=None,
                               genes=None, topk=20, indices=None, cell_type=None, cell_type_key=None,
                               bandwidth=1.0, nearest_neighbor=10, plot=True, p_adj_cutoff=0.05):
        if not self.DL_model.require_batch:
            reference_batch = 0

        reference_condition = this_condition
        if self.DL_model.require_batch == True:
            if reference_batch is None:
                print("Wrong, in datasets that have batches within condition, you must specify batch")
                return

        if self.reference_condition == reference_condition and self.reference_batch == reference_batch and self.ITE is not None:
            ITE = self.ITE
        else:
            print(
                "ITE has not been estimated for this input, first estimate ITE using default parameters:\n ITE=self.Estimate_ITE(reference_batch,reference_condition,bandwidth=1.0,nearest_neighbor=10)")
            ITE = self.Estimate_ITE(this_condition=reference_condition, this_batch=reference_batch,
                                    bandwidth=bandwidth, nearest_neighbor=nearest_neighbor)
            print("Finish estimating ITE")

        if reference_batch is None:
            reference_batch = 0

        z_scores = Plot_and_Estimate_CATE(adata=self.DL_model.adata, ITE_results=ITE,
                                          target_condition=self.DL_model.condition_dict[target_condition],
                                          reference_condition=self.DL_model.condition_dict[reference_condition],
                                          reference_batch=self.DL_model.batch_dict[reference_batch],
                                          genes=genes, topk=topk, indices=indices, cell_type=cell_type,
                                          cell_type_key=cell_type_key, plot=plot)

        print("z_scores", np.mean(z_scores), np.std(z_scores), np.min(z_scores), np.max(z_scores),
              np.sum(z_scores > 1.96), np.sum(z_scores < -1.96))
        signficant_genes = get_significant_names(np.array(z_scores), self.DL_model.adata.var_names.tolist(),
                                                 alpha=p_adj_cutoff)
        return signficant_genes


if __name__ == "__main__":
    import scanpy as sc

    '''adata = sc.read_h5ad("../data/PBMC.h5ad")
    print(np.unique(adata.obs["condition"]))
    print(np.unique(adata.obs["cell_type"]))
    contorl_name = 'control'
    model = NDreamer(adata, condition_key="condition", contorl_name="control", num_hvg=3608,
                            require_batch=False,
                            batch_key=None,
                            resolution_low=0.5, resolution_high=7, cluster_method="Leiden", embedding_dim=512,
                            codebooks=[1024 for i in range(32)],
                            codebook_dim=8, encoder_hidden=[1024, 512], decoder_hidden=[512, 1024], z_dim=256,
                            cos_loss_scaler=20, random_seed=123, batch_size=1024, epoches=100, lr=1e-3,
                            triplet_margin=5,independent_loss_scaler=1000, save_pth="./model/",
                            developer_test_mode=False,
                            library_size_normalize_adata=False,
                            save_preprocessed_adata_path="../PBMC_preprocessed.h5ad",
                            KL_scaler=5e-2, reconstruct_scaler=1, triplet_scaler=1, num_triplets_per_label=15,
                            tau=1, commitment_loss_scaler=0.25, cluster_correlation_scaler=50,reset_threshold=1/1024,
                            reset_interval=30,try_identify_cb_specific_subtypes=True,
                            local_neighborhood_loss_scaler=500,local_neighbor_sigma=1,
                            try_identify_perturb_escaped_cell=True,n_neighbors=15,
                            local_neighbor_across_cluster_scaler=10)

    model.train_model()
    model.get_modifier_space()

    adata1 = model.DL_model.adata.copy()
    sc.pp.neighbors(adata1, use_rep='X_effect_modifier_space_PCA', n_neighbors=15)
    sc.tl.umap(adata1)
    sc.pl.umap(adata1, color=['condition', 'cell_type'], frameon=False, ncols=1)

    sc.pp.neighbors(adata1, use_rep='X_effect_modifier_space', n_neighbors=15)
    sc.tl.umap(adata1)
    sc.pl.umap(adata1, color=['condition', 'cell_type'], frameon=False, ncols=1)

    if model.DL_model.developer_test_mode:
        model.DL_model.batch_dict = {0: 0}
        model.DL_model.condition_dict = {'control': 0, 'stimulated': 1}

    #model.load_effect_modifier_space_adata(sc.read_h5ad("../PBMC_results.h5ad"))
    model.decompose_true_expression_batch_effect_all(nearest_neighbor=1, bandwidth=1)
    model.Estmiate_ITE_all(nearest_neighbor=1, bandwidth=1)
    ITE=model.Estimate_ITE(this_condition="stimulated",nearest_neighbor=1)

    CATE_CD8=model.Plot_and_Estimate_CATE(this_condition="stimulated",target_condition="control",cell_type="CD8T", cell_type_key="cell_type")
    print(len(CATE_CD8),CATE_CD8)'''
    adata=sc.read_h5ad("../data/t1d_results.h5ad")

    model = NDreamer(adata, condition_key="disease_state", contorl_name='Control', num_hvg=2000, require_batch=True,
                     batch_key='donor_id',
                     resolution_low=0.5, resolution_high=7, cluster_method="Leiden", embedding_dim=512,
                     codebooks=[1024 for i in range(32)],
                     codebook_dim=8, encoder_hidden=[1024, 512], decoder_hidden=[512, 1024], z_dim=256,
                     cos_loss_scaler=20, random_seed=123, batch_size=1024, epoches=10, lr=1e-3,
                     triplet_margin=5, independent_loss_scaler=1000, save_pth="./t1d/",
                     developer_test_mode=True,
                     library_size_normalize_adata=False,
                     save_preprocessed_adata_path="./t1d/preprocessed.h5ad",
                     KL_scaler=5e-3, reconstruct_scaler=1, triplet_scaler=5, num_triplets_per_label=15,
                     tau=0.01, commitment_loss_scaler=1, cluster_correlation_scaler=50, reset_threshold=1 / 1024,
                     reset_interval=30, try_identify_cb_specific_subtypes=False,
                     local_neighborhood_loss_scaler=1, local_neighbor_sigma=1,
                     try_identify_perturb_escaped_cell=False, n_neighbors=20,
                     local_neighbor_across_cluster_scaler=20)
    print(torch.load("../data/batch2num_mapping.pth"),torch.load("../data/condition2num_mapping.pth"))
    model.DL_model.batch_dict=torch.load("../data/batch2num_mapping.pth")
    model.DL_model.condition_dict = torch.load("../data/condition2num_mapping.pth")
    model.load_effect_modifier_space_adata(adata)
    model.decompose_true_expression__batch_effect(this_condition=0, this_batch=0, nearest_neighbor=1, bandwidth=1,use_raw_name=False)