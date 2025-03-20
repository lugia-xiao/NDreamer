import torch
import random
from itertools import combinations

import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence as kl


# Acknowledgement to scCRAFT from Chuan He (https://www.biorxiv.org/content/10.1101/2024.10.22.619682v1)

def count_labels_per_group(labels, batch_ids, condition_ids):
    unique_groups = torch.unique(torch.stack((batch_ids, condition_ids), dim=1), dim=0)
    label_counts_per_group = {}

    for batch, condition in unique_groups:
        mask = (batch_ids == batch) & (condition_ids == condition)
        group_labels = labels[mask]
        unique_labels = torch.unique(group_labels)
        label_counts_per_group[(batch.item(), condition.item())] = (unique_labels, mask)

    return label_counts_per_group

def create_triplets_within_groups(embeddings, labels, labels_high, batch_ids, condition_ids,
                                  margin=1.0, num_triplets_per_label=15):
    label_counts_per_group = count_labels_per_group(labels, batch_ids, condition_ids)
    all_triplet_losses = []

    for (batch_id, condition_id), (unique_labels, group_mask) in label_counts_per_group.items():
        #print(batch_id, condition_id)
        for label in unique_labels:
            label_mask = (labels == label) & group_mask
            non_label_mask = ~label_mask & group_mask

            label_indices = torch.where(label_mask)[0]
            non_label_indices = torch.where(non_label_mask)[0]

            if len(label_indices) < 2 or len(non_label_indices) == 0:
                continue

            # Generate positive pairs and sample negatives
            positive_pairs = torch.combinations(label_indices, r=2)
            num_pairs = min(num_triplets_per_label, positive_pairs.size(0))
            selected_pairs = positive_pairs[torch.randperm(positive_pairs.size(0))[:num_pairs]]

            num_negatives = min(num_pairs, len(non_label_indices))
            selected_negatives = non_label_indices[torch.randperm(len(non_label_indices))[:num_negatives]]

            # Vectorized loss computation
            anchor_indices = selected_pairs[:, 0]
            positive_indices = selected_pairs[:, 1]

            # Filter valid triplets
            valid_pairs_mask = labels_high[anchor_indices] == labels_high[positive_indices]
            valid_anchor_indices = anchor_indices[valid_pairs_mask]
            valid_positive_indices = positive_indices[valid_pairs_mask]

            if valid_anchor_indices.numel() > 0:
                num_pairi=min(valid_anchor_indices.shape[0],selected_negatives.shape[0])
                valid_anchor_indices=valid_anchor_indices[:num_pairi]
                valid_anchors = embeddings[valid_anchor_indices]
                valid_positives = embeddings[valid_positive_indices[:num_pairi]]
                #print(selected_negatives[:valid_anchor_indices.size(0)].shape,selected_negatives.shape,valid_anchor_indices.shape,valid_positives.shape)
                valid_negatives = embeddings[selected_negatives[:valid_anchor_indices.size(0)]]  # Match count

                # Compute distances and triplet losses
                positive_distances = torch.norm(valid_anchors - valid_positives, dim=1)
                negative_distances = torch.norm(valid_anchors - valid_negatives, dim=1)

                triplet_losses = torch.relu(positive_distances - negative_distances + margin)
                all_triplet_losses.extend(triplet_losses)

    # Aggregate all triplet losses
    if all_triplet_losses:
        return torch.mean(torch.stack(all_triplet_losses))
    else:
        return torch.tensor(0.0, device=embeddings.device)

def select_indices(bool_matrix: torch.Tensor, num_triplet: torch.Tensor) -> torch.Tensor:
    """
    Selects a specified number of `True` indices for each row in a boolean matrix.

    Args:
        bool_matrix (torch.Tensor): A boolean tensor of shape (b, b).
        num_triplet (torch.Tensor): A 1D tensor of shape (b,) specifying the number of
                                    indices to select from each row.

    Returns:
        torch.Tensor: A 1D tensor containing the selected indices concatenated row-wise.
    """
    # Convert the boolean matrix to float for ranking
    float_matrix = bool_matrix.float()

    # Assign a very large negative value to `False` entries
    float_matrix[~bool_matrix] = -float('inf')

    # Get the top max(num_triplet) indices for each row
    _, top_indices = torch.topk(float_matrix, num_triplet.max().item(), dim=1)

    # Create a mask to retain only the required number of indices for each row
    mask = torch.arange(num_triplet.max().item(), device=num_triplet.device)[None, :] < num_triplet[:, None]

    # Select the valid indices based on the mask
    final_indices = top_indices[mask]

    return final_indices

def create_triplets_within_groups_tensor(embeddings, labels, labels_high, batch_ids, condition_ids,
                                         margin=1.0, num_triplets=15):
    """
    Create triplets (anchor, positive, negative) within groups defined by labels and other parameters and compute triplet loss.

    Args:
        embeddings (torch.Tensor): Embeddings of shape (N, D), where N is the number of samples and D is the embedding dimension.
        labels (torch.Tensor): Tensor of shape (N,) containing fine-grained labels.
        labels_high (torch.Tensor): Tensor of shape (N,) containing high-level labels.
        batch_ids (torch.Tensor): Tensor of shape (N,) indicating batch IDs for each sample.
        condition_ids (torch.Tensor): Tensor of shape (N,) indicating condition IDs for each sample.
        margin (float): Margin for the triplet loss.
        num_triplets (int): Number of triplets to generate.

    Returns:
        triplet_loss (float): Mean triplet loss over the generated triplets.
    """
    # Ensure tensors are on the same device
    device = embeddings.device
    n, c=embeddings.shape

    # Mask for positive samples
    positive_mask = (
        (labels.unsqueeze(1) == labels.unsqueeze(0)) &
        (labels_high.unsqueeze(1) == labels_high.unsqueeze(0)) &
        (batch_ids.unsqueeze(1) == batch_ids.unsqueeze(0)) &
        (condition_ids.unsqueeze(1) == condition_ids.unsqueeze(0)) &
        (~torch.eye(len(labels), device=device, dtype=torch.bool))
    )

    # Mask for negative samples
    negative_mask = (
        (batch_ids.unsqueeze(1) == batch_ids.unsqueeze(0)) &
        (condition_ids.unsqueeze(1) == condition_ids.unsqueeze(0)) &
        (labels.unsqueeze(1) != labels.unsqueeze(0)) &
        (labels_high.unsqueeze(1) != labels_high.unsqueeze(0))
    )

    # Random sampling for anchors, positives, and negatives
    num_triplets_filtered=torch.min(torch.sum(positive_mask,dim=-1),torch.sum(negative_mask,dim=-1))
    num_triplets_filtered=torch.min(torch.ones_like(num_triplets_filtered,device=device)*num_triplets,num_triplets_filtered)

    positive_indices=select_indices(positive_mask, num_triplets_filtered)
    negative_indices=select_indices(negative_mask, num_triplets_filtered)
    anchor_indices=torch.repeat_interleave(torch.arange(len(num_triplets_filtered),device=device), num_triplets_filtered)

    anchor_embeddings=embeddings[anchor_indices]
    positive_embeddings=embeddings[positive_indices]
    negative_embeddings=embeddings[negative_indices]

    positive_distances = torch.sum(torch.square(anchor_embeddings - positive_embeddings),dim=-1)
    negative_distances = torch.sum(torch.square(anchor_embeddings - negative_embeddings),dim=-1)

    losses = torch.relu(positive_distances - negative_distances + margin)
    triplet_loss = losses.mean()

    return triplet_loss


def create_triplets_within_groups_logits(logit_groups, labels, labels_high,
                                         batch_ids, condition_ids,margin=0.15,
                                         num_triplets_per_label=15):
    embeddings=torch.stack(logit_groups, dim=-1)

    label_counts_per_group = count_labels_per_group(labels, batch_ids, condition_ids)
    triplets = []

    for (batch_id, condition_id), (unique_labels, counts) in label_counts_per_group.items():
        for label in unique_labels:
            indices_with_label = (labels == label) & (batch_ids == batch_id) & (condition_ids == condition_id)
            indices_without_label = (labels != label) & (batch_ids == batch_id) & (condition_ids == condition_id)

            positive_pairs = list(combinations(torch.where(indices_with_label)[0], 2))
            negative_indices = torch.where(indices_without_label)[0]

            # Randomly sample triplets
            sampled_positive_pairs = random.sample(positive_pairs, min(num_triplets_per_label, len(positive_pairs)))
            # Ensure the number of negative samples doesn't exceed the available negatives
            num_negative_samples = min(len(sampled_positive_pairs), len(negative_indices))
            sampled_negative_indices = random.sample(list(negative_indices), num_negative_samples)

            for (anchor_idx, positive_idx), negative_idx in zip(sampled_positive_pairs, sampled_negative_indices):
                anchor, positive, negative = embeddings[anchor_idx], embeddings[positive_idx], embeddings[negative_idx]
                if labels_high[anchor_idx] != labels_high[positive_idx]:
                    # Skip if anchor and positive don't share the same higher-level label
                    continue
                else:
                    # Standard triplet loss calculation
                    triplet_loss = torch.relu(
                        torch.norm(anchor-positive,p=1) - torch.norm(anchor-negative,p=1) + margin
                    )
                    triplets.append(triplet_loss)

    if triplets:
        return torch.mean(torch.stack(triplets))
    else:
        return torch.tensor(0.0)

def reconstruction_error(x,mu):
    """Log likelihood (scalar) of a minibatch according to a nb model.

    Parameters
    ----------
    x
        data
    mu
        mean of the negative binomial (has to be positive support) (shape: minibatch x vars)
    theta
        inverse dispersion parameter (has to be positive support) (shape: minibatch x vars)
    eps
        numerical stability constant
    log_fn
        log function
        lgamma_fn
        log gamma function
    """
    #x, mu, theta=x.view(-1),mu.view(-1),theta.view(-1)
    '''log_theta_mu_eps = torch.log(theta + mu + eps)
    res = (
        theta * (torch.log(theta + eps) - log_theta_mu_eps)
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1)
    )

    return -res.mean()'''
    return torch.mean(torch.square(x - mu))


def compute_mmd(z, sigma=1.0):
    """Compute the Maximum Mean Discrepancy (MMD) loss."""
    z_prior=torch.randn_like(z,device=z.device)
    # Pairwise distances
    z_z = torch.cdist(z, z, p=2) ** 2
    z_prior_prior = torch.cdist(z_prior, z_prior, p=2) ** 2
    z_z_prior = torch.cdist(z, z_prior, p=2) ** 2

    # Gaussian kernel
    kernel_z = torch.exp(-z_z / (2 * sigma ** 2))
    kernel_prior = torch.exp(-z_prior_prior / (2 * sigma ** 2))
    kernel_cross = torch.exp(-z_z_prior / (2 * sigma ** 2))

    # MMD loss
    mmd = kernel_z.mean() - 2 * kernel_cross.mean() + kernel_prior.mean()
    return mmd


def correlation_matrix_difference(data):
    """
    Calculate the difference between the features' correlation matrix
    and the identity matrix for a given data matrix.

    Args:
        data (torch.Tensor): A data matrix of shape (n, c), where n is the number of samples
                             and c is the number of features.

    Returns:
        torch.Tensor: The difference between the correlation matrix and the identity matrix.
    """
    # Center the data by subtracting the mean
    data_centered = data - data.mean(dim=0, keepdim=True)

    # Compute the covariance matrix
    covariance_matrix = data_centered.T @ data_centered / (data.shape[0] - 1)

    # Compute the variance for each feature
    variance = torch.diag(covariance_matrix)

    # Avoid division by zero for variance
    variance = variance + 1e-8  # Add a small value to avoid numerical issues

    # Compute the correlation matrix
    std_dev = torch.sqrt(variance)
    correlation_matrix = covariance_matrix / (std_dev[:, None] @ std_dev[None, :])

    # Identity matrix of the same size as correlation matrix
    identity_matrix = torch.eye(correlation_matrix.shape[0], device=data.device)

    # Compute the difference
    difference = correlation_matrix - identity_matrix

    return torch.mean(torch.square(difference))

def kl_divergence_loss(z_mean, z_var):
    '''# KL divergence for Gaussian
    kl_loss = -0.5 * torch.sum(1 + torch.log(z_var) - z_mean.pow(2) - z_var)
    return kl_loss'''
    mean = torch.zeros_like(z_mean)
    scale = torch.ones_like(z_var)
    kl_divergence = kl(Normal(z_mean, torch.sqrt(z_var)), Normal(mean, scale)).sum(dim=1)

    covariance_loss=correlation_matrix_difference(z_mean)+correlation_matrix_difference(z_var)
    return kl_divergence.mean()+covariance_loss

class CrossEntropy(nn.Module):
    def __init__(self, reduction='mean'):
        super(CrossEntropy, self).__init__()
        self.reduction = reduction

    def forward(self, output, target):
        # Apply log softmax to the output
        log_preds = F.log_softmax(output, dim=-1)

        # Compute the negative log likelihood loss
        loss = F.nll_loss(log_preds, target, reduction=self.reduction)

        return loss


class IndependenceLoss(torch.nn.Module):
    """
    Mutual information loss function to encourage independence between
    assignments and probabilities. This implementation uses sparse masking
    for computational efficiency.

    Args:
        scaler (float): A scaling factor applied to the mutual information loss.
    """
    def __init__(self, scaler=1.0):
        super(IndependenceLoss, self).__init__()
        self.scaler = scaler

    def forward(self, assignments, probabilities):
        """
        Calculate mutual information as a loss function.

        Args:
            assignments (torch.LongTensor): Assignment vector of shape (n,), values in range [0, b].
            probabilities (torch.Tensor): Probability tensor of shape (n, c).
            weights (torch.Tensor): weight vector with shape n, sum up to 1, all positive values

        Returns:
            torch.Tensor: Scaled mutual information loss (scalar).
        """
        #weight=-torch.cdist(latent, latent, p=2)
        #weight=F.softmax(weight, dim=-1).unsqueeze(dim=0)

        probabilities=torch.stack(probabilities, dim=0)
        #probabilities=torch.matmul(weight, probabilities)
        #probabilities=probabilities/torch.sum(probabilities,dim=(-1,-2),keepdim=True)

        n = probabilities.shape[1]  # Number of samples
        b = assignments.max().item() + 1  # Number of unique assignment classes (0 to b)
        c = probabilities.size(-1)  # Number of probability classes

        # One-hot encode the assignments
        assignments_onehot = F.one_hot(assignments, num_classes=b).float()
        #weights = weights/torch.sum(weights)
        #assignments_onehot = assignments_onehot*weights.view(-1,1)#
        assignments_onehot=assignments_onehot.unsqueeze(dim=0)
        #print(assignments_onehot.shape,probabilities.shape)

        #s=torch.matmul((assignments_onehot*weights.view(1,-1,1)).permute(0,2,1), probabilities*weights.view(1,-1,1))
        #s=s/torch.sum(s,dim=(-1,-2),keepdim=True)

        # Compute the joint probability matrix P(A, B)
        joint_prob = torch.matmul(assignments_onehot.permute(0,2,1), probabilities)#*s
        joint_prob=joint_prob/torch.sum(joint_prob,dim=(-1,-2),keepdim=True)

        # Marginal probabilities P(A) and P(B)
        p_a = torch.sum(assignments_onehot,dim=-2)  # Marginals over rows
        p_a=p_a/torch.sum(p_a, dim=-1, keepdim=True)
        p_b = torch.sum(probabilities,dim=-2)
        p_b=p_b/torch.sum(p_b, dim=-1, keepdim=True)#joint_prob.sum(dim=1)#torch.sum(probabilities,dim=0)  # Marginals over columns
        #print(p_a.shape,p_b.shape,joint_prob.shape)
        # Compute the entropies using only non-zero probabilities
        epsilon = 1e-10
        h_a = -torch.sum(p_a * torch.log(p_a + epsilon), dim=-1)
        h_b = -torch.sum(p_b * torch.log(p_b + epsilon), dim=-1)
        h_ab = -torch.sum(joint_prob * torch.log(joint_prob + epsilon), dim=(-1,-2))
        # Mutual information
        #print(h_a,h_b,h_ab)
        mi = torch.mean(2*(h_a + h_b - h_ab)/(h_a+h_b))
        # Return scaled negative mutual information (as loss)
        return self.scaler * mi


class IndependenceLoss_label(torch.nn.Module):
    """
    Mutual information loss function to encourage independence between
    assignments and probabilities within subsets defined by condition and batch.

    Args:
        scaler (float): A scaling factor applied to the mutual information loss.
    """

    def __init__(self, scaler=1.0):
        super(IndependenceLoss_label, self).__init__()
        self.scaler = scaler

    def forward(self, assignments, probabilities, condition, batch):
        """
        Calculate mutual information as a loss function within subsets defined by
        unique combinations of condition and batch.

        Args:
            assignments (torch.LongTensor): Assignment vector of shape (n,), values in range [0, b].
            probabilities (torch.Tensor): Probability tensor of shape (n, c).
            condition (torch.LongTensor): Condition vector of shape (n,), values representing conditions.
            batch (torch.LongTensor): Batch vector of shape (n,), values representing batches.

        Returns:
            torch.Tensor: Scaled mutual information loss (scalar).
        """
        probabilities=torch.stack(probabilities,dim=0)
        unique_keys = torch.unique(torch.stack([condition, batch], dim=1), dim=0)
        total_loss = 0.0

        for key in unique_keys:
            # Identify the subset of data for the current key
            mask = (condition == key[0]) & (batch == key[1])
            sub_assignments = assignments[mask]
            sub_probabilities = probabilities[:,mask]

            # Skip subsets with no data
            if sub_assignments.numel() == 0:
                continue

            n = sub_probabilities.shape[1]  # Number of samples
            b = sub_assignments.max().item() + 1  # Number of unique assignment classes (0 to b)
            c = sub_probabilities.size(-1)  # Number of probability classes

            # One-hot encode the assignments
            assignments_onehot = F.one_hot(sub_assignments, num_classes=b).float()
            # weights = weights/torch.sum(weights)
            assignments_onehot = assignments_onehot.unsqueeze(dim=0)
            # print(assignments_onehot.shape,probabilities.shape)
            # Compute the joint probability matrix P(A, B)
            joint_prob = torch.matmul(assignments_onehot.permute(0, 2, 1), sub_probabilities)
            joint_prob = joint_prob / torch.sum(joint_prob,dim=(-1,-2),keepdim=True)

            # Marginal probabilities P(A) and P(B)
            p_a = torch.sum(assignments_onehot, dim=-2)  # Marginals over rows
            p_a = p_a / torch.sum(p_a, dim=-1, keepdim=True)
            p_b = torch.sum(sub_probabilities, dim=-2)
            p_b = p_b / torch.sum(p_b, dim=-1, keepdim=True)  # joint_prob.sum(dim=1)#torch.sum(probabilities,dim=0)  # Marginals over columns
            # print(p_a.shape,p_b.shape,joint_prob.shape)
            # Compute the entropies using only non-zero probabilities
            epsilon = 1e-10
            h_a = -torch.sum(p_a * torch.log(p_a + epsilon), dim=-1)
            h_b = -torch.sum(p_b * torch.log(p_b + epsilon), dim=-1)
            h_ab = -torch.sum(joint_prob * torch.log(joint_prob + epsilon), dim=(-1, -2))

            # Mutual information
            # print(h_a.shape,h_b.shape,h_ab.shape)
            mi = torch.mean(2*(h_a + h_b - h_ab)/(h_a+h_b))
            '''if mi>0.8*torch.min(h_a,h_b):
                total_loss += 0
                continue'''
            total_loss += mi

        # Return scaled negative mutual information (as loss)
        return self.scaler*total_loss/len(unique_keys)


def compute_local_neighborhood_loss_(latent, inputs, label_low,
                                     label_high, sigma=1, n_neighbors=15,
                                     local_neighbor_across_cluster_scaler=10):
    """
    Computes the local neighborhood loss to maintain continuity in the latent space.
    Args:
        latent (Tensor): Latent space embeddings of shape (N, D).
        inputs (Tensor): Input data of shape (N, F).
        sigma (float): Parameter for the Gaussian kernel.
        k_neighbors (int): Number of neighbors to consider.
    Returns:
        Tensor: Local neighborhood loss.
    """
    '''# Compute pairwise distances in input space
    input_distances = torch.cdist(inputs, inputs, p=2)  # Shape: (N, N)
    input_weights = torch.exp(-input_distances ** 2 / (2 * sigma ** 2))  # Gaussian kernel weights

    boundary_weight=(labels_high.unsqueeze(dim=-1)!=labels_high.unsqueeze(dim=0)).float()
    boundary_weight=boundary_weight+(labels_low.unsqueeze(dim=-1)!=labels_low.unsqueeze(dim=0))
    boundary_weight=boundary_weight#*10
    input_weights=(boundary_weight+torch.ones_like(boundary_weight,device=latent.device))*input_weights
    input_weights = input_weights / torch.sum(input_weights)

    # Mask to select only k-nearest neighbors
    topk_indices = torch.topk(input_weights, k_neighbors + 1, dim=1, largest=True).indices
    neighbor_mask = torch.zeros_like(input_weights)
    neighbor_mask.scatter_(1, topk_indices, 1)
    neighbor_mask = neighbor_mask.fill_diagonal_(0)  # Exclude self-connections

    # Compute pairwise distances in latent space
    latent_distances = torch.cdist(latent, latent, p=2)  # Shape: (N, N)

    # Apply the mask and compute loss
    weighted_latent_distances = input_weights * latent_distances ** 2 * neighbor_mask
    local_loss = torch.sum(weighted_latent_distances) / torch.sum(neighbor_mask)
    return local_loss'''
    input_distances = torch.cdist(inputs, inputs, p=2)
    input_distances = input_distances + 1e4*torch.eye(input_distances.shape[0],device=latent.device)

    #boundary_weight = (labels_low.unsqueeze(dim=-1) != labels_low.unsqueeze(dim=0)).float()
    #boundary_weight = boundary_weight*1000+torch.ones_like(boundary_weight,device=latent.device)
    k_neighbors=n_neighbors
    k_neighbors=min(input_distances.shape[0]-1,k_neighbors)

    topk_indices = torch.topk(-input_distances, k_neighbors + 1, dim=1, largest=True).indices
    neighbor_mask = torch.zeros_like(input_distances)
    neighbor_mask.scatter_(1, topk_indices, 1)
    neighbor_mask = neighbor_mask.fill_diagonal_(0)  # Exclude self-connections
    #mask=~torch.eye(input_distances.shape[0], dtype=torch.bool, device=input_distances.device)#neighbor_mask.bool()#
    #mask=neighbor_mask.bool()

    #boundary_weight = boundary_weight[mask]
    #print(torch.sum(mask,dim=-1))
    '''input_distances = input_distances[mask]
    input_distances = input_distances-torch.min(input_distances)
    input_distances = torch.exp(-input_distances / sigma)
    #input_distances = input_distances / torch.sum(input_distances)
    #input_distances = input_distances*boundary_weight'''

    latent_distances = torch.cdist(latent, latent, p=2)
    latent_distances= latent_distances + 1e4*torch.eye(latent_distances.shape[0],device=latent.device)
    #latent_distances = latent_distances.view(input_distances.shape[0], k_neighbors)
    topk_indices_latent = torch.topk(-latent_distances, k_neighbors + 1, dim=1, largest=True).indices
    neighbor_mask_latent = torch.zeros_like(input_distances)
    neighbor_mask_latent.scatter_(1, topk_indices_latent, 1)
    neighbor_mask_latent = neighbor_mask_latent.fill_diagonal_(0)

    mask=torch.logical_or(neighbor_mask.bool(), neighbor_mask_latent.bool())
    input_distances = torch.where(mask, input_distances, torch.ones_like(input_distances,device=latent.device)*1e5)
    input_distances = torch.softmax(-input_distances/sigma, dim=-1)

    #scaler=torch.mean(latent_distances[mask])

    latent_distances = torch.where(mask, latent_distances, torch.ones_like(latent_distances,device=latent.device)*1e5)
    latent_distances = torch.softmax(-latent_distances, dim=-1)
    #print(torch.mean(input_distances))

    across_cluster_flag=label_low.unsqueeze(dim=-1)!=label_low.unsqueeze(dim=0)
    across_cluster_flag=torch.logical_or(across_cluster_flag, label_low.unsqueeze(dim=-1)!=label_high.unsqueeze(dim=0))
    across_cluster_flag=across_cluster_flag.float()*local_neighbor_across_cluster_scaler+torch.ones_like(across_cluster_flag,device=latent.device)

    #input_distances=input_distances/torch.sqrt(torch.sum(input_distances**2, dim=-1, keepdim=True))
    #latent_distances=latent_distances/torch.sqrt(torch.sum(latent_distances**2, dim=-1, keepdim=True))
    '''latent_distances = latent_distances - torch.min(latent_distances)
    latent_distances = torch.exp(-latent_distances/sigma)'''
    #latent_distances = latent_distances/torch.sum(latent_distances)
    #latent_distances = latent_distances*boundary_weight
    return torch.mean(torch.sum(torch.square((input_distances - latent_distances)*across_cluster_flag),dim=-1))

def compute_local_neighborhood_loss(latent, inputs, condition, batch, label_low, label_high,
                                    sigma=1, n_neighbors=15, local_neighbor_across_cluster_scaler=10):
    """
    Computes the local neighborhood loss to maintain continuity in the latent space.
    Args:
        latent (Tensor): Latent space embeddings of shape (N, D).
        inputs (Tensor): Input data of shape (N, F).
        sigma (float): Parameter for the Gaussian kernel.
        k_neighbors (int): Number of neighbors to consider.
    Returns:
        Tensor: Local neighborhood loss.
    """
    unique_keys = torch.unique(torch.stack([condition, batch], dim=1), dim=0)
    total_loss = 0.0

    for key in unique_keys:
        # Identify the subset of data for the current key
        mask = (condition == key[0]) & (batch == key[1])
        total_loss=total_loss+compute_local_neighborhood_loss_(latent[mask], inputs[mask],
                                                               label_low[mask], label_high[mask],
                                                               sigma, n_neighbors,
                                                               local_neighbor_across_cluster_scaler)/len(unique_keys)
    return total_loss

class IndependenceLoss_between_matrix(torch.nn.Module):
    """
    Mutual information loss for two probability matrices to encourage independence.

    Args:
        scaler (float): A scaling factor applied to the mutual information loss.
    """

    def __init__(self, scaler=1.0):
        super(IndependenceLoss_between_matrix, self).__init__()
        self.scaler = scaler

    def forward(self, prob_a, prob_b):
        """
        Args:
            prob_a (torch.Tensor): Probability matrix of shape (n, b), rows sum to 1.
            prob_b (torch.Tensor): Probability matrix of shape (n, c), rows sum to 1.

        Returns:
            torch.Tensor: Scaled mutual information loss (scalar).
        """
        # Number of samples
        n = prob_a.shape[0]

        # Compute joint probability matrix
        joint_prob = torch.matmul(prob_a.T, prob_b) / n  # Shape: (b, c)

        # Marginal probabilities
        marginal_a = joint_prob.sum(dim=1, keepdim=True)  # Shape: (b, 1)
        marginal_b = joint_prob.sum(dim=0, keepdim=True)  # Shape: (1, c)

        # Apply small epsilon for numerical stability
        epsilon = 1e-10

        # Compute mutual information
        mi_matrix = joint_prob * torch.log(joint_prob / (marginal_a @ marginal_b + epsilon) + epsilon)
        mi = mi_matrix.sum()

        # Return positive mutual information as the loss
        return self.scaler * mi

class OrthogonalityRegularization(nn.Module):
    """
    Orthogonality Regularization Loss to enforce embeddings to be orthogonal.

    Loss Function:
    L_orthogonal = ||E^T E - I||_F

    Args:
        embedding_dim (int): Dimensionality of the embeddings (columns in E).
    """

    def __init__(self, scaler):
        super(OrthogonalityRegularization, self).__init__()
        self.scaler = scaler

    def forward(self, embeddings):
        """
        Args:
            embeddings (torch.Tensor): Matrix of embeddings of shape (N, D),
                                       where N = number of samples, D = embedding dimension.

        Returns:
            torch.Tensor: Scalar orthogonality loss.
        """
        # Compute E^T E
        gram_matrix = torch.matmul(embeddings.T, embeddings)  # Shape: (D, D)

        # Calculate ||E^T E - I||_F
        loss = torch.mean(gram_matrix - torch.eye(gram_matrix.shape[0]).to(embeddings.device))
        return loss*self.scaler


class EntropyPenalty(nn.Module):
    def __init__(self, weight: float = 0.1):
        """
        Initializes the Entropy Penalty Module.

        Args:
            weight (float): Weight for the entropy penalty term.
        """
        super(EntropyPenalty, self).__init__()
        self.weight = weight  # Weight for the penalty

    def forward(self, logits):
        """
        Computes the entropy penalty for the given logits.

        Args:
            logits (torch.Tensor): Raw logits output from the model (batch_size x num_classes).

        Returns:
            torch.Tensor: The entropy penalty term.
        """
        probabilities = logits
        eps = 1e-8  # Small constant to prevent log(0)
        log_probabilities = torch.log(probabilities + eps)  # Log of probabilities
        entropy = -torch.sum(probabilities * log_probabilities, dim=1)  # Entropy per sample
        mean_entropy = torch.mean(entropy)  # Mean entropy across the batch
        return self.weight * mean_entropy  # Scaled entropy penalty

if __name__=="__main__":
    '''print(torch.abs(torch.randn((16)).cuda()).lgamma())

    z=torch.randn((16,256)).cuda()
    mean=torch.exp(torch.randn((16,2000))).cuda()
    dispersion=torch.exp(torch.randn((16,2000))).cuda()
    logits=[torch.randn((16,32)).cuda(), torch.randn((16,32)).cuda()]

    exp=torch.abs(torch.randn((16,2000))).cuda()'''

    loss_fn=OrthogonalityRegularization(1).cuda()
    print(loss_fn(torch.randn((16,32)).cuda()))












