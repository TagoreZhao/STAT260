import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def generate_covariance_matrix(d):
    """Generate the covariance matrix Σ with (i, j)th entry 2 × 0.5^|i−j|"""
    indices = np.arange(d)
    Sigma = 2 * 0.5 ** np.abs(indices[:, None] - indices[None, :])
    return Sigma

def generate_gaussian_A(n, d, seed=1234):
    """Generate A from multivariate normal N(1_d, Σ)"""
    rng = np.random.default_rng(seed)  # Set seed for reproducibility
    Sigma = generate_covariance_matrix(d)
    mean = np.ones(d)  # Mean vector of 1_d
    A = rng.multivariate_normal(mean, Sigma, size=n)
    return A

def generate_t_distribution_A(n, d, df, seed=1234):
    """Generate A from multivariate t-distribution with df degrees of freedom"""
    rng = np.random.default_rng(seed)  # Set seed for reproducibility
    Sigma = generate_covariance_matrix(d)
    mean = np.ones(d)  # Mean vector of 1_d
    z = rng.multivariate_normal(mean, Sigma, size=n)  # Gaussian samples
    chi2_samples = rng.chisquare(df, size=(n, 1))  # Chi-square samples for t-distribution
    A = z / np.sqrt(chi2_samples / df)  # Convert to t-distribution samples
    return A

def compute_matrices(A):
    """Compute A^T A and U_A^T U_A using SVD and QR"""
    A_T_A = A.T @ A  # Compute A^T A

    U_A, _, _ = np.linalg.svd(A, full_matrices=False)
    U_A_T_U_A_svd = U_A.T @ U_A

    Q, _ = np.linalg.qr(A)
    U_A_T_U_A_qr = Q.T @ Q

    return A_T_A, U_A_T_U_A_svd, U_A_T_U_A_qr

def compute_left_singular_vectors(A):
    """Compute the left singular vectors (U_A) from the SVD of A."""
    U_A, _, _ = np.linalg.svd(A, full_matrices=False)
    return U_A

def compute_sampling_probabilities(A, B, method="uniform"):
    """
    Compute a probability distribution p for sampling columns of A and rows of B.

    Parameters:
    A (numpy.ndarray): An (m x n) matrix.
    B (numpy.ndarray): An (n x p) matrix.
    method (str): The sampling method to use. Options:
                  - "uniform": Uniform probabilities.
                  - "norm_based": Probabilities based on ||A(:,i)|| * ||B(i,:)||
                  - "frobenius_based": Probabilities based on ||A(:,i)||^2 / ||A||_F^2 when B = A^T

    Returns:
    p (numpy.ndarray): A probability distribution vector of length n.
    """
    n = A.shape[1]
    
    if method == "uniform":
        # Uniform probabilities: each column/row is equally likely to be chosen
        p = np.full(n, 1 / n)
    
    elif method == "norm_based":
        # Compute column norms of A and row norms of B.
        A_norms = np.linalg.norm(A, axis=0)  # shape: (n,)
        B_norms = np.linalg.norm(B, axis=1)  # shape: (n,)
        norm_product = A_norms * B_norms
        p = norm_product / np.sum(norm_product)  # Normalize to sum to 1
    
    elif method == "frobenius_based":
        # Assume B = A^T, so use ||A(:,i)||^2 / ||A||_F^2
        A_norms_sq = np.linalg.norm(A, axis=0)**2  # Column-wise squared norms, shape: (n,)
        p = A_norms_sq / np.sum(A_norms_sq)  # Normalize to sum to 1
    
    else:
        raise ValueError("Invalid method. Choose from 'uniform', 'norm_based', or 'frobenius_based'.")

    return p

def basic_matrix_multiplication(A, B, c, p):
    """
    Implements Algorithm 3: Basic Matrix Multiplication.

    Parameters:
    A (numpy.ndarray): An (m x n) matrix.
    B (numpy.ndarray): An (n x p) matrix.
    c (int): The number of sampled columns/rows.
    p (numpy.ndarray): A probability distribution over the n columns/rows.

    Returns:
    C (numpy.ndarray): A (m x c) matrix sampled from A.
    R (numpy.ndarray): A (c x p) matrix sampled from B.
    """
    m, n = A.shape
    n_, p_dim = B.shape
    assert n == n_, "Matrix dimensions do not align for multiplication."
    assert len(p) == n, "Probability distribution length must match number of columns in A (rows in B)."
    assert np.isclose(np.sum(p), 1), "Probability distribution must sum to 1."

    # Initialize C and R
    C = np.zeros((m, c))
    R = np.zeros((c, p_dim))

    # Sampling process
    for t in range(c):
        # Sample index i_t according to probability p
        it = np.random.choice(n, p=p)
        
        # Scale and assign the selected columns/rows
        C[:, t] = A[:, it] / np.sqrt(c * p[it])
        R[t, :] = B[it, :] / np.sqrt(c * p[it])

    return C, R