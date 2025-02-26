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

    # SVD-based U_A
    U_A, _, _ = np.linalg.svd(A, full_matrices=False)
    U_A_T_U_A_svd = U_A.T @ U_A

    # QR-based U_A
    Q, _ = np.linalg.qr(A)
    U_A_T_U_A_qr = Q.T @ Q

    return A_T_A, U_A_T_U_A_svd, U_A_T_U_A_qr