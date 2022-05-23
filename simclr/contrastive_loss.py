# Assignment from CS231n, Stanford
# Completed by ** MYSELF **
# https://github.com/anyeZHY/2021-CS231n/blob/main/assignment%203/cs231n/simclr/contrastive_loss.py

import torch
import numpy as np


def sim(z_i, z_j):
    """Normalized dot product between two vectors.

    Inputs:
    - z_i: 1xD tensor.
    - z_j: 1xD tensor.
    
    Returns:
    - A scalar value that is the normalized dot product between z_i and z_j.
    """
    norm_dot_product = None
    z_i, z_j = z_i / torch.linalg.norm(z_i), z_j / torch.linalg.norm(z_j)
    norm_dot_product = torch.sum(z_i * z_j)
    return norm_dot_product


def sim_positive_pairs(out_left, out_right):
    """Normalized dot product between positive pairs.

    Inputs:
    - out_left: NxD tensor; output of the projection head g(), left branch in SimCLR model.
    - out_right: NxD tensor; output of the projection head g(), right branch in SimCLR model.
    Each row is a z-vector for an augmented sample in the batch.
    The same row in out_left and out_right form a positive pair.
    
    Returns:
    - A Nx1 tensor; each row k is the normalized dot product between out_left[k] and out_right[k].
    """
    z_i, z_j = out_left, out_right
    N, D = z_i.size()
    z_i = z_i / torch.linalg.norm(z_i, dim=1).reshape(N, 1)
    z_j = z_j / torch.linalg.norm(z_j, dim=1).reshape(N, 1)
    pos_pairs = torch.sum(z_i * z_j, dim=1).reshape(N, 1)
    return pos_pairs


def compute_sim_matrix(out):
    """Compute a 2N x 2N matrix of normalized dot products between all pairs of augmented examples in a batch.

    Inputs:
    - out: 2N x D tensor; each row is the z-vector (output of projection head) of a single augmented example.
    There are a total of 2N augmented examples in the batch.
    
    Returns:
    - sim_matrix: 2N x 2N tensor; each element i, j in the matrix is the normalized dot product between out[i] and out[j].
    """
    NN, D = out.size()
    out = out / torch.linalg.norm(out, dim=1).reshape(NN,1)
    sim_matrix = out.matmul(out.T)
    return sim_matrix


def simclr_loss(out_left, out_right, tau, device='cuda'):
    """Compute the contrastive loss L over a batch (vectorized version). No loops are allowed.
    
    Inputs and output are the same as in simclr_loss_naive.
    """
    N = out_left.shape[0]
    
    # Concatenate out_left and out_right into a 2*N x D tensor.
    out = torch.cat([out_left, out_right], dim=0)  # [2*N, D]
    
    # Compute similarity matrix between all pairs of augmented examples in the batch.
    sim_matrix = compute_sim_matrix(out)  # [2*N, 2*N]
    
    # Step 1: Use sim_matrix to compute the denominator value for all augmented samples.
    # Hint: Compute e^{sim / tau} and store into exponential, which should have shape 2N x 2N.
    exponential = torch.exp(sim_matrix/tau).to(device)
    
    # This binary mask zeros out terms where k=i.
    mask = (torch.ones_like(exponential, device=device) - torch.eye(2 * N, device=device)).to(device).bool()
    
    # We apply the binary mask.
    exponential = exponential.masked_select(mask).view(2 * N, -1)  # [2*N, 2*N-1]
    
    # Hint: Compute the denominator values for all augmented samples. This should be a 2N x 1 vector.
    denom = torch.sum(exponential, dim=1).reshape(2*N, 1)

    # Step 2: Compute similarity between positive pairs.
    # You can do this in two ways: 
    # Option 1: Extract the corresponding indices from sim_matrix. 
    # Option 2: Use sim_positive_pairs().
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    sim_pairs = sim_positive_pairs(out_left, out_right).to(device)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # Step 3: Compute the numerator value for all augmented samples.
    numerator = None
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    tmp = torch.exp(sim_pairs/tau).to(device)
    numerator = torch.cat([tmp, tmp], dim=0).to(device)

    # Step 4: Now that you have the numerator and denominator for all augmented samples, compute the total loss.
    loss = torch.sum(- torch.log(numerator / denom))/ (2 * N)
    
    return loss


def simclr_fe(f_i, l_i, r_i, f_j, l_j, r_j, tau=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, D = l_i.shape
    D *= 4
    sim_loss_fl = 0
    sim_loss_fl += simclr_loss(f_i[:, 0:D//4], l_i, tau=tau, device=device)
    sim_loss_fl += simclr_loss(f_i[:, D//4:D//2], r_i, tau=tau, device=device)
    sim_loss_fl += simclr_loss(f_j[:, 0:D//4], l_j, tau=tau, device=device)
    sim_loss_fl += simclr_loss(f_j[:, D//4:D//2], r_j, tau=tau, device=device)
    return sim_loss_fl
