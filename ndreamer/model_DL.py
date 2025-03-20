import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import numpy as np
import sklearn
import os

class Encoder(nn.Module):
    def __init__(self,input_dim, output_dim, hidden_dims):
        super(Encoder, self).__init__()
        self.encoders = nn.ModuleList(
            [nn.Sequential(nn.Linear(input_dim, hidden_dims[0]),nn.BatchNorm1d(hidden_dims[0]))]+
            [nn.Sequential(nn.ReLU(),nn.Linear(hidden_dims[i], hidden_dims[i+1]),nn.BatchNorm1d(hidden_dims[i+1])) for i in range(len(hidden_dims)-1)]+
            [nn.Sequential(nn.ReLU(),nn.Linear(hidden_dims[-1], output_dim))]
        )
        self.num_hidden_dims = len(hidden_dims)

    def forward(self, x):
        for encoderi in self.encoders:
            x=encoderi(x)
        return x

class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, have_negative_data):
        super(Decoder, self).__init__()
        self.have_negative_data=have_negative_data

        self.decoders = nn.ModuleList(
            [nn.Linear(input_dim, hidden_dims[0])]+
            [nn.Sequential(nn.Linear(hidden_dims[i], hidden_dims[i + 1]), nn.ReLU()) for i in
             range(len(hidden_dims) - 1)]
        )
        self.mid_act=nn.ReLU()
        self.out=nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x, bias):
        cnt=0
        for decoderi in self.decoders:
            x = decoderi(x)
            if cnt==0:
                x=self.mid_act(x+bias)
            cnt+=1
        x=self.out(x)
        if not self.have_negative_data:
            x=F.relu(x)
        return x

class Codebook(nn.Module):
    def __init__(self, input_dim, num_embeddings, embedding_dim, tau=1, reset_threshold=0.01, reset_interval=20):
        super(Codebook, self).__init__()
        self.classifier = nn.Linear(input_dim, embedding_dim)
        self.bn=nn.BatchNorm1d(embedding_dim)
        self.tau = tau
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

        # Track usage of embeddings
        self.register_buffer("embedding_usage", torch.zeros(num_embeddings))

        # Reinitialization parameters
        self.reset_threshold = reset_threshold
        self.reset_interval = reset_interval
        self.step_count = 1  # To track steps for periodic reset

        self.just_reset_codebook=False

    def forward(self, x):
        self.just_reset_codebook=False
        # Compute logits (distances to embeddings)
        logits = torch.square(self.bn(self.classifier(x)).unsqueeze(dim=1) - self.embedding.weight.unsqueeze(dim=0))
        logits = -torch.sum(logits, dim=-1)#/self.tau

        # Gumbel-Softmax for assignments
        assignments = F.gumbel_softmax(logits, tau=self.tau, dim=-1)  # Shape: (batch_size, num_embeddings)
        #assignments=F.softmax(logits, dim=-1)

        # Update embedding usage
        self.embedding_usage += assignments.sum(dim=0).detach()

        # Quantized embedding
        z = torch.matmul(assignments, self.embedding.weight)  # Shape: (batch_size, embedding_dim)

        # Periodic Reset of Unused Embeddings
        if self.training and self.step_count % self.reset_interval == 0:
            self.reset_unused_embeddings(x)
            self.just_reset_codebook=True

        self.step_count += 1
        return F.softmax(logits, dim=-1), z

    def reset_unused_embeddings(self, x):
        """
        Reinitialize embeddings that are underused based on the reset threshold.
        """
        # Identify unused embeddings
        total_usage = self.embedding_usage.sum().item()
        usage_probs = self.embedding_usage / (total_usage + 1e-10)
        unused_indices = torch.where(usage_probs < self.reset_threshold)[0]

        if unused_indices.numel() > 0:
            print(f"Resetting {unused_indices.numel()} unused embeddings.")

            # Reinitialize embeddings using encoder outputs
            selected_samples = x[torch.randint(0, x.size(0), (unused_indices.numel(),))]
            new_embeddings = self.classifier(selected_samples).detach()

            # Update embedding weights for unused indices
            with torch.no_grad():
                self.embedding.weight[unused_indices] = new_embeddings

            # Reset usage counts for the reinitialized embeddings
            self.embedding_usage[unused_indices] = 0

def create_binary_matrix_with_max_values(matrix):
    """
    Converts a matrix into a binary matrix where only the largest value
    in each row is set to 1, and other values are set to 0.

    Args:
        matrix (torch.Tensor): Input matrix of shape (n, c) on CUDA or CPU.

    Returns:
        torch.Tensor: A binary matrix of the same shape as input.
    """
    if matrix.ndim != 2:
        raise ValueError("Input matrix must have exactly 2 dimensions (n, c).")

    # Find the indices of the maximum values in each row
    max_indices = torch.argmax(matrix, dim=1)

    # Create a zero matrix of the same shape
    binary_matrix = torch.zeros_like(matrix)

    # Set the largest value indices to 1
    binary_matrix[torch.arange(matrix.size(0), device=matrix.device), max_indices] = 1

    return binary_matrix

def compute_codebook_loss(logits, zs, codebooks):
    codebook_loss = 0
    for logiti, zi, codebook in zip(logits, zs, codebooks):
        # Compute the expected embedding
        #expected_embedding = torch.matmul(logiti, codebook.embedding.weight)  # Shape: (batch_size, embedding_dim)
        expected_embedding = torch.matmul(create_binary_matrix_with_max_values(logiti), codebook.embedding.weight)  # Shape: (batch_size, embedding_dim)

        # Stop-gradient to prevent updates to the encoder
        codebook_loss += torch.mean((zi.detach() - expected_embedding) ** 2)
    return codebook_loss


def compute_commitment_loss(logits, zs, codebooks, beta=0.25):
    commitment_loss = 0
    for logiti, zi, codebook in zip(logits, zs, codebooks):
        # Compute the expected embedding
        #expected_embedding = torch.matmul(logiti, codebook.embedding.weight)  # Shape: (batch_size, embedding_dim)
        expected_embedding = torch.matmul(create_binary_matrix_with_max_values(logiti), codebook.embedding.weight)  # Shape: (batch_size, embedding_dim)

        # Stop-gradient to prevent updates to the codebook
        commitment_loss += torch.mean((zi - expected_embedding.detach()) ** 2)
    return beta * commitment_loss

class FFN(nn.Module):
    def __init__(self,dim):
        super(FFN,self).__init__()
        self.fc1 = nn.Linear(dim,dim*4)
        self.fc2 = nn.Linear(dim*4,dim)
        self.act=nn.GELU()
        self.ln1=nn.BatchNorm1d(dim)
        self.ln2=nn.BatchNorm1d(dim)

    def forward(self, x):
        x=self.ln1(x)
        return x+self.ln2(self.fc2(self.act(self.fc1(x))))

class Encoder_With_Codebook(nn.Module):
    def __init__(self, input_dim, embedding_dim, codebook_dim, codebooks,
                 hidden_dims, z_dim, tau, commitment_loss_scaler, reset_threshold,
                 reset_interval):
        super(Encoder_With_Codebook, self).__init__()
        self.encoders = Encoder(input_dim=input_dim, output_dim=embedding_dim, hidden_dims=hidden_dims)
        self.codebooks = nn.ModuleList([
            Codebook(input_dim=embedding_dim, num_embeddings=codebooks[i], embedding_dim=codebook_dim,tau=tau,
                     reset_threshold=reset_threshold,reset_interval=reset_interval)
            for i in range(len(codebooks))
        ])
        self.linear=nn.Sequential(
            nn.Linear(codebook_dim * len(codebooks), z_dim*2),nn.ReLU()
        )
        self.FFN=FFN(z_dim*2)
        self.mean_encoder=nn.Sequential(nn.ReLU(),nn.Linear(z_dim*2, z_dim))#nn.Linear(codebook_dim * len(codebooks), z_dim)#
        self.variance_encoder=nn.Sequential(nn.ReLU(),nn.Linear(z_dim*2, z_dim))#nn.Linear(codebook_dim * len(codebooks), z_dim)#

        self.commitment_loss_scaler = commitment_loss_scaler

    def forward(self, exp):
        x = self.encoders(exp)
        logits=[]
        zs=[]
        for codebooki in self.codebooks:
            logiti,zi=codebooki(x)
            logits.append(logiti)
            zs.append(zi)

        codebook_loss=compute_codebook_loss(logits, zs, self.codebooks)
        commitment_loss=compute_commitment_loss(logits, zs, self.codebooks)

        embedding=torch.concatenate(zs,dim=-1)
        embedding=self.FFN(self.linear(embedding))

        zs=self.mean_encoder(embedding)
        variance=torch.exp(self.variance_encoder(embedding))
        variance=torch.where(variance>1e-6,variance,torch.ones_like(variance,device=variance.device)*1e-6)
        return logits, zs, variance, (codebook_loss+commitment_loss)*self.commitment_loss_scaler

class Discriminator(nn.Module):
    def __init__(self, input_dim, num_labels, hidden_dims):
        super(Discriminator, self).__init__()
        self.FFN = nn.ModuleList(
            [nn.Linear(input_dim, hidden_dims[0])] +
            [nn.Sequential(nn.ReLU(), nn.Linear(hidden_dims[i], hidden_dims[i + 1])) for i in
             range(len(hidden_dims) - 1)] +
            [nn.Sequential(nn.ReLU(), nn.Linear(hidden_dims[-1], num_labels))]
        )

    def forward(self, x):
        for encoderi in self.FFN:
            x=encoderi(x)
        return x


def reparameterize(mean, variance):
    """
    Reparameterization trick to sample z ~ N(mean, variance).

    Args:
        mean (torch.Tensor): The mean of the latent Gaussian distribution.
        variance (torch.Tensor): The variance of the latent Gaussian distribution (not log variance).

    Returns:
        torch.Tensor: A sampled latent vector z.
    """
    std = torch.sqrt(variance)  # Compute standard deviation from variance
    eps = torch.randn_like(std).to(mean.device)  # Sample epsilon ~ N(0, I)
    z = mean + eps * std  # Reparameterized z
    return z+torch.randn_like(z,device=z.device)

class NDreamer_generator(nn.Module):
    def __init__(self, input_dim, num_treatments, num_batches, embedding_dim ,codebooks,
                 codebook_dim, z_dim, encoder_hidden, decoder_hidden, tau,
                 commitment_loss_scaler, try_identify_perturb_escaped_cell,
                 reset_threshold, reset_interval,try_identify_cb_specific_subtypes,
                 have_negative_data):
        super(NDreamer_generator, self).__init__()
        self.require_batch = False
        if len(num_batches) > 0:
            self.require_batch = True
        self.num_batches = num_batches

        '''self.condition_embedding0 = nn.Embedding(num_treatments, z_dim//4)
        self.batch_embedding0 = nn.Embedding(num_batches, z_dim//4)
        input_dim1=input_dim+z_dim//4
        if self.require_batch:
            input_dim1=input_dim1+z_dim//4'''

        self.encoder = Encoder_With_Codebook(input_dim=input_dim, embedding_dim=embedding_dim,
                                             codebook_dim=codebook_dim,z_dim=z_dim,tau=tau,
                                             codebooks=codebooks, hidden_dims=encoder_hidden,
                                             commitment_loss_scaler=commitment_loss_scaler,
                                             reset_threshold=reset_threshold,
                                             reset_interval=reset_interval)

        self.condition_embedding = nn.Embedding(num_treatments, decoder_hidden[0])
        self.batch_embedding = nn.ModuleList([
            nn.Embedding(num_batches[i], decoder_hidden[0]) for i in range(len(num_batches))
        ])

        self.try_identify_perturb_escaped_cell = try_identify_perturb_escaped_cell
        if self.try_identify_perturb_escaped_cell:
            self.escape_judger1 = nn.Sequential(
                Encoder(input_dim=input_dim, hidden_dims=encoder_hidden[:-1],
                        output_dim=encoder_hidden[-1]),
                nn.ReLU()
            )
            input_judger2=encoder_hidden[-1] + decoder_hidden[0]
            if self.require_batch:
                input_judger2=input_judger2+decoder_hidden[0]
            self.escape_judger2 = nn.Sequential(nn.Linear(input_judger2, encoder_hidden[-1]),
                                                nn.ReLU(), nn.Linear(encoder_hidden[-1], 2))
            self.tau = tau

        self.try_identify_cb_specific_subtypes = try_identify_cb_specific_subtypes
        if self.try_identify_cb_specific_subtypes:
            out_dim=1 if not self.require_batch else 2
            '''self.cb_specific_judger1 = nn.Sequential(
                Encoder(input_dim=input_dim, hidden_dims=encoder_hidden[:-1],
                        output_dim=encoder_hidden[-1]),
                nn.ReLU()
            )
            input_judger2 = encoder_hidden[-1] + decoder_hidden[0]'''
            input_judger2=z_dim+decoder_hidden[0]
            if self.require_batch:
                input_judger2 = input_judger2 + decoder_hidden[0]
            self.cb_specific_judger2 = nn.Sequential(nn.Linear(input_judger2, encoder_hidden[-1]),
                                                nn.ReLU(), nn.Linear(encoder_hidden[-1], out_dim))
            self.tau = tau


        self.decoder = Decoder(input_dim=z_dim, output_dim=input_dim,
                               hidden_dims=decoder_hidden,
                               have_negative_data=have_negative_data)

    def forward(self, exp, treatment, batch):
        '''

        :param exp: b,input_dim
        :param treatment: b, id
        :param batch: b, id
        :return:
        '''
        '''exp1=torch.concatenate([exp,self.condition_embedding0(treatment)],dim=-1)
        if self.require_batch:
            exp1=torch.concatenate([exp1,self.batch_embedding0(batch)],dim=-1)'''

        logits, z, variance, commitment_loss = self.encoder(exp)

        treatment_effect = self.condition_embedding(treatment)

        batch_effect = torch.zeros_like(treatment_effect, device=treatment_effect.device)
        if self.require_batch:
            for i in range(len(self.num_batches)):
                batch_effect = batch_effect + self.batch_embedding[i](batch[:,i])

        if self.try_identify_perturb_escaped_cell:
            escape_judger_embedding=torch.concatenate([self.escape_judger1(exp), treatment_effect], dim=-1)
            if self.require_batch:
                escape_judger_embedding=torch.concatenate([escape_judger_embedding,batch_effect],dim=-1)
            escape_judger_embedding = self.escape_judger2(escape_judger_embedding)
            escape_judger_choice = F.gumbel_softmax(logits=escape_judger_embedding, tau=self.tau, hard=False)
            # print(self.condition_embedding.weight[0:1].unsqueeze(dim=0).expand(treatment_effect.shape[0],-1,-1).shape,treatment_effect.unsqueeze(dim=1).shape)
            escape_judger_embedding = torch.concatenate(
                [self.condition_embedding.weight[0:1].unsqueeze(dim=0).expand(treatment_effect.shape[0], -1, -1),
                 treatment_effect.unsqueeze(dim=1)], dim=1) * escape_judger_choice.unsqueeze(dim=-1)
            treatment_effect = torch.sum(escape_judger_embedding, dim=1)
        else:
            escape_judger_choice=None

        cb_specifc_embedding=None
        if self.try_identify_cb_specific_subtypes:
            cb_specifc_embedding = torch.concatenate([z, treatment_effect], dim=-1)
            if self.require_batch:
                cb_specifc_embedding = torch.concatenate([cb_specifc_embedding, batch_effect], dim=-1)
            cb_specifc_embedding = self.cb_specific_judger2(cb_specifc_embedding)
            cb_specifc_embedding=F.sigmoid(cb_specifc_embedding)#F.softmax(cb_specifc_embedding,dim=0)#+1
            #print(torch.max(cb_specifc_embedding),torch.min(cb_specifc_embedding))
            #cb_specifc_embedding=cb_specifc_embedding/torch.sum(cb_specifc_embedding, dim=1, keepdim=True)
        else:
            if self.require_batch:
                cb_specifc_embedding=torch.ones((exp.shape[0],2),device=exp.device)
            else:
                cb_specifc_embedding = torch.ones((exp.shape[0], 1),device=exp.device)
        cb_specifc_embedding=cb_specifc_embedding/torch.sum(cb_specifc_embedding, dim=0, keepdim=True)

        bias = treatment_effect + batch_effect

        reconstructed = self.decoder(reparameterize(z,variance), bias)#
        return z, variance, reconstructed, logits, commitment_loss, escape_judger_choice, cb_specifc_embedding

def set_seed(seed: int):
    # Set Python random seed
    random.seed(seed)

    # Set NumPy random seed
    np.random.seed(seed)

    # Set PyTorch random seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # If using multi-GPU.

        # Ensure deterministic behavior in PyTorch (can slow down computations)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Set sklearn random seed
    sklearn.utils.check_random_state(seed)

    # Set environment variable for reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)

if __name__=="__main__":
    pass



