import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import device

from ndreamer.model_DL import NDreamer_generator,Discriminator,set_seed
from ndreamer.DL_loss_func import CrossEntropy, create_triplets_within_groups, IndependenceLoss, kl_divergence_loss, \
    IndependenceLoss_label, OrthogonalityRegularization, EntropyPenalty, create_triplets_within_groups_logits, \
    IndependenceLoss_between_matrix, compute_mmd, compute_local_neighborhood_loss, create_triplets_within_groups_tensor
from ndreamer.data_preprocess import process_adata,generate_balanced_dataloader,generate_adata_to_dataloader
from ndreamer.DL_loss_func import reconstruction_error
from ndreamer.statistics import *

from sklearn.decomposition import PCA
# Dynamic import of tqdm based on the environment
import sys
if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

from ndreamer.plot import plot_distribution

class NDreamer_DL(nn.Module):
    def __init__(self, adata, condition_key, contorl_name, num_hvg, require_batch=False, batch_key=None,
                 resolution_low=0.5, resolution_high=7, cluster_method="Leiden", embedding_dim=512, codebooks=None,
                 codebook_dim=8, encoder_hidden=None, decoder_hidden=None, z_dim=256,
                 cos_loss_scaler=20, random_seed=123, batch_size=2048, epoches=10, lr=1e-3, triplet_margin=5,
                 independent_loss_scaler=1000, save_pth="./model/", developer_test_mode=False,
                 library_size_normalize_adata=False, save_preprocessed_adata_path="./model/preprocessed.h5ad",
                 KL_scaler=5e-3, reconstruct_scaler=1, triplet_scaler=5, num_triplets_per_label=15,
                 tau=0.01, commitment_loss_scaler=1, cluster_correlation_scaler=50,
                 try_identify_perturb_escaped_cell=False, reset_threshold=1 / 1024, reset_interval=30,
                 try_identify_cb_specific_subtypes=False, local_neighborhood_loss_scaler=1,
                 local_neighbor_sigma=1, n_neighbors=20, local_neighbor_across_cluster_scaler=20,
                 have_negative_data=False):
        super(NDreamer_DL, self).__init__()
        input_dim=num_hvg

        if codebooks is None:
            codebooks = [1024 for i in range(32)]
        if encoder_hidden is None:
            encoder_hidden = [2048, 1024]
        if decoder_hidden is None:
            decoder_hidden = [512, 1024]

        self.developer_test_mode=developer_test_mode
        self.require_batch=require_batch
        self.batch_key = batch_key
        self.condition_key = condition_key
        self.triplet_margin=triplet_margin
        if not os.path.exists(save_pth):
            os.mkdir(save_pth)
        self.save_pth=save_pth
        self.random_seed = random_seed
        self.batch_size = batch_size
        self.epoches = epoches
        self.lr = lr
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using device:",self.device)
        self.kl_scaler = KL_scaler
        self.reconstruct_scaler = reconstruct_scaler
        self.triplet_scaler = triplet_scaler
        self.num_triplets_per_label=num_triplets_per_label
        self.independent_loss_scaler = independent_loss_scaler
        self.codebooks=codebooks
        self.local_neighborhood_loss_scaler=local_neighborhood_loss_scaler
        print(local_neighborhood_loss_scaler)
        self.local_neighbor_sigma=local_neighbor_sigma
        self.try_identify_cb_specific_subtypes=try_identify_cb_specific_subtypes
        self.try_identify_perturb_escaped_cell=try_identify_perturb_escaped_cell
        self.n_neighbors=n_neighbors
        self.local_neighbor_across_cluster_scaler=local_neighbor_across_cluster_scaler

        #data preprocessing
        if not developer_test_mode:
            print("Start data preprocessing")
            adata, batch_dict, condition_dict = process_adata(adata=adata, condition_key=condition_key,
                                                              input_dim=input_dim, control_name=contorl_name,
                                                              require_batch=require_batch, batch_key=batch_key,
                                                              resolution_low=resolution_low,
                                                              resolution_high=resolution_high,
                                                              cluster_method=cluster_method,
                                                              library_size_normalize_adata=library_size_normalize_adata)


            self.adata = adata
            self.batch_dict = batch_dict
            self.condition_dict = condition_dict
            print("Data preprocessing done")
            if save_preprocessed_adata_path is not None:
                adata.write(save_preprocessed_adata_path)
        else:
            self.adata=adata

        print("Remaining number of cells:",self.adata.shape[0])

        calcluated_epoches=15*self.adata.shape[0]//(self.batch_size*len(np.unique(self.adata.obs["group"])))+1
        if calcluated_epoches>self.epoches:
            if len(np.unique(self.adata.obs["group"]))<=10:
                calcluated_epoches = calcluated_epoches + reset_interval - calcluated_epoches % reset_interval - 2
            self.epoches=calcluated_epoches
            print("Too few epoches (steps, if rigorously speaking). Changing epoch to", self.epoches, "to adjust for number of cells")

        num_batches = []#np.unique(adata.obs["batch"]).shape[0]
        if not self.developer_test_mode and self.require_batch:
            if isinstance(batch_key, str):
                num_batches = [max(self.batch_dict[batch_key].values()) + 1]
            elif isinstance(batch_key, list):
                num_batches=[]
                for batch_keyi in batch_key:
                    num_batches.append(max(self.batch_dict[batch_keyi].values()) + 1)
            else:
                num_batches=[]
        self.num_batches=num_batches
        num_treatments = np.unique(adata.obs["condition"]).shape[0]
        if not developer_test_mode:
            num_treatments=max(num_treatments,max(self.condition_dict.values())+1)

        # define the model
        self.VQ_VAE = NDreamer_generator(input_dim=input_dim, num_treatments=num_treatments,z_dim=z_dim,
                                         num_batches=num_batches,embedding_dim=embedding_dim,tau=tau,
                                         codebooks=codebooks,codebook_dim=codebook_dim,encoder_hidden=encoder_hidden,
                                         decoder_hidden=decoder_hidden,commitment_loss_scaler=commitment_loss_scaler,
                                         reset_threshold=reset_threshold,reset_interval=reset_interval,
                                         try_identify_cb_specific_subtypes=try_identify_cb_specific_subtypes,
                                         try_identify_perturb_escaped_cell=try_identify_perturb_escaped_cell,
                                         have_negative_data=have_negative_data)

        self.require_batch = require_batch
        print("Require batch:",self.require_batch)

        self.independence_loss=IndependenceLoss(scaler=independent_loss_scaler)
        self.independence_loss_label=IndependenceLoss_label(scaler=cluster_correlation_scaler/num_treatments)
        self.independence_loss_between_codebook=IndependenceLoss_between_matrix(scaler=independent_loss_scaler/100)
        self.cross_entropy = CrossEntropy()
        self.cos_loss_scaler = cos_loss_scaler
        #self.MINE=MINE(x_dim=input_dim,z_dim=z_dim,hidden_dim=1024)

        # init the model
        self.VQ_VAE.to(self.device)
        self.independence_loss.to(self.device)
        self.independence_loss_label.to(self.device)
        #self.cross_entropy.to(self.device)
        #self.orthogonality_regularization.to(self.device)
        #self.entropy_penalty.to(self.device)
        self.independence_loss_between_codebook.to(self.device)
        #self.MINE.to(self.device)

        self.logits=None
        self.df_latent=None

    def train_model(self):
        optimizer_G = torch.optim.AdamW(self.VQ_VAE.parameters(), lr=self.lr)

        progress_bar = tqdm(total=self.epoches, desc="Overall Progress", leave=True, miniters=1, mininterval=0)
        for epoch in range(self.epoches):
            set_seed(self.random_seed+epoch)
            data_loader = generate_balanced_dataloader(self.adata, batch_size=self.batch_size)
            self.VQ_VAE.train()
            all_losses = 0
            T_loss = 0
            V_loss = 0
            I_loss = 0
            K_loss = 0
            N_loss = 0
            Commitment_loss=0
            Dependent_loss=0
            for i, (exp, condition, batch, labels_low, labels_high) in enumerate(data_loader):
                #print(exp.shape, condition.shape, batch.shape, labels_low.shape, labels_high.shape)
                # convert to cuda
                exp = exp.to(self.device)
                condition = condition.to(self.device)
                batch = batch.to(self.device)
                labels_low = labels_low.to(self.device)
                labels_high = labels_high.to(self.device)
                # run VQ-VAE
                z, variance, reconstructed, logits, commitment_loss,escape_judger_choice, cb_specifc_embedding=self.VQ_VAE(exp=exp, treatment=condition, batch=batch)
                commitment_loss=commitment_loss/len(self.codebooks)

                if self.VQ_VAE.encoder.codebooks[0].just_reset_codebook:
                    print("Finish resetting codebook embeddings, current step (epoch):",epoch)
                    continue

                # calculate the reconstruction loss
                reconst_loss_mse = reconstruction_error(exp, reconstructed)*self.reconstruct_scaler
                reconst_loss_cos = (1 - torch.sum(F.normalize(reconstructed, p=2) * F.normalize(exp, p=2), 1)).mean()
                reconst_loss_cos = self.cos_loss_scaler * reconst_loss_cos
                reconst_loss = reconst_loss_mse + reconst_loss_cos
                reconst_loss = torch.clamp(reconst_loss, max=1e5)

                KL_loss=kl_divergence_loss(z_mean=z,z_var=variance)*self.kl_scaler
                KL_loss = torch.clamp(KL_loss, max=1e5)

                # independence loss for VAE
                independent_loss=0
                independent_loss=independent_loss+self.independence_loss(condition, logits)
                for i in range(len(self.num_batches)):
                    independent_loss = independent_loss + self.independence_loss(batch[:,i], logits)
                independent_loss = torch.clamp(independent_loss, max=1e5)

                # convert matrix batch to unique int number vector
                base = batch.max() + 1  # Base larger than the largest number in the matrix
                powers = base ** torch.arange(batch.size(1), device=exp.device).flip(0).float()  # Positional encoding powers
                batch_vector = torch.matmul(batch.float(), powers).long()  # Encoded as unique integers

                dependent_loss = 0
                dependent_loss=dependent_loss-self.independence_loss_label(assignments=labels_high,
                                                                                   probabilities=logits,
                                                                                   condition=condition,
                                                                                   batch=batch_vector)

                dependent_loss=dependent_loss-self.independence_loss_label(assignments=labels_low,
                                                                                   probabilities=logits,
                                                                                   condition=condition,
                                                                                   batch=batch_vector)

                # triplet loss
                triplet_loss = create_triplets_within_groups_tensor(z, labels_low, labels_high, batch_vector,
                                                             condition, margin=self.triplet_margin,
                                                             num_triplets=self.num_triplets_per_label)
                triplet_loss=triplet_loss*self.triplet_scaler

                neighbor_loss = compute_local_neighborhood_loss(latent=z, inputs=exp,
                                                                condition=condition, batch=batch_vector,
                                                                sigma=self.local_neighbor_sigma,
                                                                n_neighbors=self.n_neighbors,
                                                                label_low=labels_low,
                                                                label_high=labels_high,
                                                                local_neighbor_across_cluster_scaler=self.local_neighbor_across_cluster_scaler
                                                                ) * self.local_neighborhood_loss_scaler

                all_loss = reconst_loss  + independent_loss +KL_loss + triplet_loss  + neighbor_loss  + commitment_loss + dependent_loss# + embedding_loss +entropy_loss

                all_loss = torch.clamp(all_loss, max=1e5)

                all_loss.backward()
                optimizer_G.step()
                optimizer_G.zero_grad()
                all_losses += all_loss.item()/len(data_loader)
                N_loss += neighbor_loss.item()/len(data_loader)
                T_loss += triplet_loss.item()/len(data_loader)
                V_loss += reconst_loss.item()/len(data_loader)
                I_loss +=independent_loss.item()/len(data_loader)
                K_loss +=KL_loss.item()/len(data_loader)
                Dependent_loss+=dependent_loss.item()/len(data_loader)
                #Entropy_loss+=entropy_loss.item()
                #Embedding_loss += 0#embedding_loss.item()
                Commitment_loss+=commitment_loss.item()/len(data_loader)

            print(f"Epoch: {epoch + 1}/{self.epoches} | "
                  f"All Loss: {all_losses:.4f} | "
                  f"Neighborhood Loss: {N_loss:.4f} | "
                  f"Triplet Loss: {T_loss:.4f} | "
                  f"Reconstruction Loss: {V_loss:.4f} | "
                  f"Independent Loss: {I_loss:.4f} | "
                  f"KL Loss: {K_loss:.4f} | "
                  f"Commitment Loss: {Commitment_loss:.4f} | "
                  f"Dependent Loss: {Dependent_loss:.4f}")
            progress_bar.update(1)  # Increment the progress bar by one for each batch processed
            progress_bar.set_postfix(epoch=f"{epoch + 1}/{self.epoches}", all_loss=all_losses, neigh_loss=N_loss, triplet_loss=T_loss, reconst_loss=V_loss, independent_loss=I_loss, KL_loss=K_loss, Commitment_loss=Commitment_loss, Dependent_loss=Dependent_loss)

        progress_bar.close()
        torch.save(self.state_dict(), os.path.join(self.save_pth, 'ndreamer'  + '.pth'))

    def get_modifier_space(self,dim=50, save_path_adata=None, save_path_latent_df=None,
                           print_codebook_statisitics=False, kBET_test=False, save_others=False):
        self.VQ_VAE.eval()
        data_loader = generate_adata_to_dataloader(self.adata)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        all_z = []
        all_indices = []
        logits=[]
        cb_specific_subtypes=[]
        escape_probs=[]
        with torch.no_grad():
            for i, (x, indices, condition, batch) in enumerate(data_loader):
                x = x.to(device)
                condition = condition.to(device)
                batch = batch.to(device)

                z, variance, reconstructed, logit, commitment_loss, escape_judger_choice, cb_specifc_embedding = self.VQ_VAE(
                    exp=x, treatment=condition, batch=batch)
                all_z.append(z.cpu().detach())
                all_indices.extend(indices.tolist())

                if self.try_identify_perturb_escaped_cell:
                    escape_probs.append(escape_judger_choice.detach().cpu())

                if self.try_identify_cb_specific_subtypes:
                    cb_specific_subtypes.append(cb_specifc_embedding.detach().cpu())

                if len(logits) == 0:
                    logits = [i.detach().cpu() for i in logit]
                else:
                    logits = [torch.concatenate([logits[i], logit[i].detach().cpu()], dim=0) for i in
                              range(len(logits))]

        all_z_combined = torch.cat(all_z, dim=0)
        all_indices_tensor = torch.tensor(all_indices)
        all_z_reordered = all_z_combined[all_indices_tensor.argsort()]
        all_z_np = all_z_reordered.numpy()

        # Create anndata object with reordered embeddings
        self.adata.obsm['X_effect_modifier_space'] = all_z_np
        pca = PCA(n_components=dim)
        # Fit and transform the data
        X_pca = pca.fit_transform(self.adata.obsm['X_effect_modifier_space'])

        # Store the PCA-reduced data back into adata.obsm
        self.adata.obsm['X_effect_modifier_space_PCA'] = X_pca
        if save_path_adata is not None:
            if save_path_adata.find(".h5ad")<0:
                save_path_adata = save_path_adata+".h5ad"
            self.adata.write(save_path_adata)
        else:
            self.adata.write(os.path.join(self.save_pth, 'adata'  + '.h5ad'))
        print("Effect modifier space saved.")

        if self.try_identify_cb_specific_subtypes:
            print(cb_specific_subtypes)
            cb_specific_subtypes = torch.concat(cb_specific_subtypes, dim=0)
            plot_distribution(cb_specific_subtypes.reshape(-1).numpy())

        self.adata.obs["original_order"]=np.array(range(self.adata.shape[0]))
        if not save_others:
            return

        torch.save(logits, os.path.join(self.save_pth, "logits.pth"))
        self.logits = logits

        if self.try_identify_perturb_escaped_cell:
            escape_probs = torch.concat(escape_probs, dim=0)
            torch.save(escape_probs, os.path.join(self.save_pth, "escape_probs.pth"))

        if self.try_identify_cb_specific_subtypes:
            cb_specific_subtypes = torch.concat(cb_specific_subtypes, dim=0)
            torch.save(cb_specific_subtypes, os.path.join(self.save_pth, "cb_specific_subtypes.pth"))

        df_latent=pd.DataFrame(data=self.adata.obsm['X_effect_modifier_space_PCA'],columns=['latent_'+str(i) for i in range(X_pca.shape[1])])
        df_latent["batch"]=np.array(self.adata.obs["batch"])
        df_latent["condition"]=np.array(self.adata.obs["condition"])
        df_latent["group"]=np.array(self.adata.obs["group"])
        if save_path_latent_df is not None:
            if save_path_latent_df.find(".csv")<0:
                save_path_latent_df = save_path_latent_df+".csv"
            df_latent.to_csv(save_path_latent_df)
        else:
            df_latent.to_csv(os.path.join(self.save_pth, 'latent.csv'))
        self.df_latent=df_latent

        if print_codebook_statisitics:
            print(torch.max(self.logits[0],dim=-1))
            print("Statistic values of the codebook selection")
            print("mean:\n",[torch.mean(logits[i], dim=0) for i in range(len(logits))])
            print("variance:\n",[torch.std(logits[i], dim=0) for i in range(len(logits))])

        if kBET_test:
            run_kbet(df_latent,do_PCA=False)

    def run_kBET_test(self):
        run_kbet(self.df_latent, do_PCA=False)


if __name__ == '__main__':
    test_run_mode=False
    train_model=True
    if test_run_mode:
        import scanpy as sc
        import anndata as ad

        '''exp = torch.abs(torch.randn((2048, 2000)))
        condition = torch.randint(low=0, high=3, size=(2048,), dtype=torch.long)
        batch1 = torch.randint(low=0, high=4, size=(2048,), dtype=torch.long)  # np.zeros(2048)
        batch2 = torch.randint(low=0, high=2, size=(2048,), dtype=torch.long)

        adata = ad.AnnData(X=exp.numpy())
        adata.obs["condition"] = condition.numpy()
        adata.obs["group"] = condition.numpy()
        adata.obs["batch1"] = batch1
        adata.obs["batch"] = batch1
        adata.obs["batch2"] = batch2
        adata.obs["leiden1"] = torch.randint(low=0, high=3, size=(2048,), dtype=torch.long).numpy()
        adata.obs["leiden1"] = adata.obs["leiden1"].astype("category")
        adata.obs["leiden2"] = torch.randint(low=0, high=9, size=(2048,), dtype=torch.long).numpy()
        adata.obs["leiden2"] = adata.obs["leiden2"].astype("category")

        model = NDreamer_DL(adata=adata, condition_key="condition", contorl_name=0, num_hvg=2000,
                            developer_test_mode=False, require_batch=True, batch_key="batch1",#["batch1","batch2"],
                            batch_size=512)
        model.train_model()'''

        adata = sc.read("../ECCITE_preprocessed.h5ad")
        adata.obsm["batch"]=np.expand_dims(adata.obs["batch"].copy(),axis=-1)
        print(adata)
        print(np.unique(adata.obs["condition"]))
        model = NDreamer_DL(adata, condition_key='perturbation', contorl_name='NT', num_hvg=2000, require_batch=True,
                            batch_key='replicate',
                            resolution_low=0.5, resolution_high=7, cluster_method="Leiden", embedding_dim=512,
                            codebooks=[1024 for i in range(32)],
                            codebook_dim=8, encoder_hidden=[1024, 512], decoder_hidden=[512, 1024], z_dim=256,
                            cos_loss_scaler=20, random_seed=123, batch_size=1024, epoches=1, lr=1e-3,
                            triplet_margin=5, independent_loss_scaler=1000, save_pth="./model/",
                            developer_test_mode=True,
                            library_size_normalize_adata=False,
                            save_preprocessed_adata_path=None,
                            KL_scaler=5e-3, reconstruct_scaler=1, triplet_scaler=5, num_triplets_per_label=15,
                            tau=0.01, commitment_loss_scaler=1, cluster_correlation_scaler=50, reset_threshold=1 / 1024,
                            reset_interval=30, try_identify_cb_specific_subtypes=False,
                            local_neighborhood_loss_scaler=1, local_neighbor_sigma=1,
                            try_identify_perturb_escaped_cell=False, n_neighbors=20,
                            local_neighbor_across_cluster_scaler=20)
        model.train_model()
        print()
        model.get_modifier_space(save_path_adata="../ECCITE_results.h5ad")
        adata1 = model.adata.copy()
        sc.pp.neighbors(adata1, use_rep='X_effect_modifier_space_PCA', n_neighbors=15)
        sc.tl.umap(adata1)
        sc.pl.umap(adata1, color=["MULTI_ID","HTO_classification",
                                  "gene_target","perturbation",
                                  "replicate","Phase"], frameon=False, ncols=1)
    else:
        import os

        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        import scanpy as sc

        #adata=sc.read_h5ad("../data/PBMC.h5ad")
        #adata = sc.read("../PBMC_preprocessed.h5ad")
        adata = sc.read("../virus_preprocessed.h5ad")
        #adata=sc.read_h5ad("../PBMC_imbalance_preprocessed.h5ad")
        #adata=sc.read_h5ad("../common1.h5ad")
        if "condition" not in adata.obsm.keys():
            adata.obsm["batch"] = np.ones((adata.shape[0],1))#np.expand_dims(adata.obs["condition"].copy(), axis=1)
        print(adata)
        print(np.unique(adata.obs["condition"]))
        model = NDreamer_DL(adata, condition_key="condition", contorl_name="control", num_hvg=min(adata.shape[1],3608),
                            require_batch=False,
                            batch_key=None,
                            resolution_low=0.5, resolution_high=7, cluster_method="Leiden", embedding_dim=512,
                            codebooks=[1024 for i in range(32)],
                            codebook_dim=8, encoder_hidden=[2048, 1024], decoder_hidden=[512, 1024], z_dim=256,
                            cos_loss_scaler=20, random_seed=123, batch_size=1024, epoches=100, lr=1e-3,
                            triplet_margin=5,independent_loss_scaler=1000, save_pth="./model/",
                            developer_test_mode=True,
                            library_size_normalize_adata=False,
                            save_preprocessed_adata_path=None,
                            KL_scaler=5e-3, reconstruct_scaler=1, triplet_scaler=5, num_triplets_per_label=15,
                            tau=0.01, commitment_loss_scaler=1, cluster_correlation_scaler=50,reset_threshold=1/1024,
                            reset_interval=30,try_identify_cb_specific_subtypes=False,
                            local_neighborhood_loss_scaler=1,local_neighbor_sigma=1,
                            try_identify_perturb_escaped_cell=False,n_neighbors=20,
                            local_neighbor_across_cluster_scaler=20, have_negative_data=True)
        '''model = NDreamer_DL(adata, condition_key="condition", contorl_name="control", num_hvg=min(adata.shape[1], 3608),
                            require_batch=False,
                            batch_key=None,have_negative_data=True,developer_test_mode=True)'''
        if train_model:
            model.train_model()
        print()
        model.load_state_dict(torch.load(os.path.join(model.save_pth, 'ndreamer' + '.pth')))
        model.get_modifier_space(save_path_adata="../PBMC_results.h5ad")

        adata1 = model.adata.copy()
        sc.pp.neighbors(adata1, use_rep='X_effect_modifier_space_PCA',n_neighbors=15)
        sc.tl.umap(adata1)
        try:
            sc.pl.umap(adata1, color=['condition', 'cell_type'], frameon=False, ncols=1)
        except:
            sc.pl.umap(adata1, color=['condition', 'cell_type1021'], frameon=False, ncols=1)

        sc.pp.neighbors(adata1, use_rep='X_effect_modifier_space',n_neighbors=15)
        sc.tl.umap(adata1)
        try:
            sc.pl.umap(adata1, color=['condition', 'cell_type'], frameon=False, ncols=1)
        except:
            sc.pl.umap(adata1, color=['condition', 'cell_type1021'], frameon=False, ncols=1)

        #model.run_kBET_test()
