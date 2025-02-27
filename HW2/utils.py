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
    rng = np.random.default_rng(seed) 
    Sigma = generate_covariance_matrix(d)
    mean = np.ones(d) 
    A = rng.multivariate_normal(mean, Sigma, size=n)
    return A

def generate_t_distribution_A(n, d, df, seed=1234):
    """Generate A from multivariate t-distribution with df degrees of freedom"""
    rng = np.random.default_rng(seed) 
    Sigma = generate_covariance_matrix(d)
    mean = np.ones(d)
    z = rng.multivariate_normal(mean, Sigma, size=n) 
    chi2_samples = rng.chisquare(df, size=(n, 1))  
    A = z / np.sqrt(chi2_samples / df)
    return A


def compute_matrices(A):
    """Compute A^T A and U_A^T U_A using SVD and QR"""
    A_T_A = A.T @ A 

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
                    - "frobenius_based": Probabilities based on ||A(:,i)||^2 / ||A||_F^2 when B = A^T.
                    - "leverage": Probabilities based on the leverage scores of A.
                               (i.e., p_i = ||U(i,:)||^2 / (sum_j ||U(j,:)||^2),
                                where U is the left singular vector matrix of A.)
    
    Returns:
      p (numpy.ndarray): A probability distribution vector.
         For "uniform", "norm_based", and "frobenius_based", the distribution is computed over columns (length = A.shape[1]).
         For "leverage", the distribution is computed over rows of A (length = A.shape[0]).
    """
    if method == "uniform":
        n = A.shape[1]
        p = np.full(n, 1 / n)
    
    elif method == "norm_based":
        A_norms = np.linalg.norm(A, axis=0)  # shape: (n,)
        B_norms = np.linalg.norm(B, axis=1)    # shape: (n,)
        norm_product = A_norms * B_norms
        p = norm_product / np.sum(norm_product)  # Normalize to sum to 1
    
    elif method == "frobenius_based":
        # Assume B = A^T, so use ||A(:,i)||^2 / ||A||_F^2
        A_norms_sq = np.linalg.norm(A, axis=0)**2  # Column-wise squared norms, shape: (n,)
        p = A_norms_sq / np.sum(A_norms_sq) 
    
    elif method == "leverage":
        U, _, _ = np.linalg.svd(B, full_matrices=False)
        leverage_scores = np.linalg.norm(U, axis=1)**2  # shape: (n,)
        p = leverage_scores / np.sum(leverage_scores)
    
    else:
        raise ValueError("Invalid method. Choose from 'uniform', 'norm_based', 'frobenius_based', or 'leverage'.")
    
    return p

def compute_sampling_probabilities_rows(A, method="uniform"):
    """
    Compute a probability distribution p for sampling rows of A.
    
    Parameters:
      A (numpy.ndarray): An (m x n) matrix.
      method (str): The sampling method to use. Options:
                    - "uniform": Uniform probabilities.
                    - "norm_based": Probabilities based on ||A(i,:)||
                    - "frobenius_based": Probabilities based on ||A(i,:)||^2 / ||A||_F^2.
                    - "leverage": Probabilities based on the leverage scores of A.
                      (i.e., p_i = ||U(i,:)||^2 / sum_j ||U(j,:)||^2, where U is the left singular vector matrix of A.)
    
    Returns:
      p (numpy.ndarray): A probability distribution vector over the rows of A (length = m).
    """
    m = A.shape[0]
    
    if method == "uniform":
        p = np.full(m, 1 / m)
    
    elif method == "norm_based":
        A_norms = np.linalg.norm(A, axis=1)  # shape: (m,)
        p = A_norms / np.sum(A_norms)
    
    elif method == "frobenius_based":
        A_norms_sq = np.linalg.norm(A, axis=1)**2  # Row-wise squared norms, shape: (m,)
        p = A_norms_sq / np.sum(A_norms_sq)
    
    elif method == "leverage":
        U, _, _ = np.linalg.svd(A, full_matrices=False)
        leverage_scores = np.linalg.norm(U, axis=1)**2  # shape: (m,)
        p = leverage_scores / np.sum(leverage_scores)
    
    else:
        raise ValueError("Invalid method. Choose from 'uniform', 'norm_based', 'frobenius_based', or 'leverage'.")
    
    return p

def slow_least_squares_sampling(A, b, c, p, seed=1234):
    """
    Implements Algorithm 4: A “slow” random sampling algorithm for the LS problem.
    
    Input:
      A (numpy.ndarray): An (m x d) matrix, with m ≫ d.
      b (numpy.ndarray): An (m,) vector.
      c (int): The number of sampled rows.
      p (numpy.ndarray): A probability distribution over the rows of A (length m).
      seed (int): Random seed for reproducibility.
      
    Output:
      x (numpy.ndarray): A (d,) vector that approximates the solution to the LS problem.
      
    The algorithm:
      1. Computes p_i (e.g. leverage scores from the SVD).
      2. Randomly samples c rows of A and corresponding entries of b, rescaling each by 1/sqrt(c * p_i).
      3. Solves the smaller LS problem and returns the approximate solution.
    """
    m, d = A.shape
    assert len(p) == m, "Probability distribution length must match number of rows in A."
    assert np.isclose(np.sum(p), 1), "Probability distribution must sum to 1."
    
    rng = np.random.default_rng(seed)

    it = rng.choice(m, size=c, p=p)

    scaling_factors = 1 / np.sqrt(c * p[it])  # shape: (c,)
    S_A = A[it, :] * scaling_factors[:, np.newaxis]  
    S_b = b[it] * scaling_factors  
    x = np.linalg.lstsq(S_A, S_b, rcond=None)[0]
    return x

def least_squares_projection(A, b, P):
    """
    Solve the least squares problem min_x ||A x - b||_2 using the projection matrix P.
    
    Parameters:
      A (numpy.ndarray): An (m x d) matrix.
      b (numpy.ndarray): An (m,) vector.
      P (numpy.ndarray): A (k x m) projection matrix.
    
    Returns:
      x (numpy.ndarray): A (k,) vector that approximates the solution to the LS problem.
    """
    return np.linalg.lstsq(P @ A, P @ b, rcond=None)[0]


def basic_matrix_multiplication(A, B, c, p, seed=1234):
    """
    Implements Algorithm 3: Basic Matrix Multiplication.

    Parameters:
      A (numpy.ndarray): An (m x n) matrix.
      B (numpy.ndarray): An (n x p) matrix.
      c (int): The number of sampled columns/rows.
      p (numpy.ndarray): A probability distribution over the n columns/rows.
      seed (int): Random seed for reproducibility.

    Returns:
      C (numpy.ndarray): A (m x c) matrix sampled from A.
      R (numpy.ndarray): A (c x p) matrix sampled from B.
    """
    m, n = A.shape
    n_B, p_dim = B.shape
    assert n == n_B, "Matrix dimensions do not align for multiplication."
    assert len(p) == n, "Probability distribution length must match number of columns in A (rows in B)."
    assert np.isclose(np.sum(p), 1), "Probability distribution must sum to 1."

    C = np.zeros((m, c))
    R = np.zeros((c, p_dim))
    
    rng = np.random.default_rng(seed)
    for t in range(c):
        it = rng.choice(n, p=p)

        C[:, t] = A[:, it] / np.sqrt(c * p[it])
        R[t, :] = B[it, :] / np.sqrt(c * p[it])

    return C, R

def projection_matrix_multiplication(A, B, P):
    return A @ P, P.T @ B

def compute_sample_approximation_errors(A,B, p, c_values, seed=1234):
    """
    For a given matrix X (to approximate X^T X), its sampling distribution p,
    and a list of sample sizes (c_values), compute the relative spectral and Frobenius errors.
    
    Returns:
      spectral_errors: list of relative spectral norm errors.
      fro_errors: list of relative Frobenius norm errors.
    """
    spectral_errors = []
    fro_errors = []
    exact = A @ B
    norm_exact_spec = np.linalg.norm(exact, 2)
    norm_exact_fro = np.linalg.norm(exact, 'fro')
    
    for c in c_values:
        C, R = basic_matrix_multiplication(A, B, c,p, seed=seed)
        approx = C @ R
        err_spec = np.linalg.norm(approx - exact, 2) / norm_exact_spec
        err_fro = np.linalg.norm(approx - exact, 'fro') / norm_exact_fro
        spectral_errors.append(err_spec)
        fro_errors.append(err_fro)
    return spectral_errors, fro_errors

def compute_leverage_scores_rows(U):
    """
    Given the left singular vectors U (m x d), compute the leverage scores for the rows.
    Since U has orthonormal columns, sum_i ||U(i,:)||^2 = d.
    Returns p as a probability vector of length m.
    """
    lev_scores = np.sum(U**2, axis=1)
    return lev_scores / np.sum(lev_scores)

def plot_probability_distribution(p, title):
    """
    Plot a bar chart for the probability distribution p.
    """
    plt.figure(figsize=(6, 4))
    plt.bar(np.arange(len(p)), p)
    plt.xlabel("Index")
    plt.ylabel("Probability")
    plt.title(title)
    plt.show()

def plot_errors(c_values, spectral_errors, fro_errors, title):
    """
    Plot the relative spectral and Frobenius norm errors versus the number of samples.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(c_values, spectral_errors, marker='o', label="Spectral Error")
    plt.plot(c_values, fro_errors, marker='s', label="Frobenius Error")
    plt.xlabel("Number of Samples (c)")
    plt.ylabel("Relative Error")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def generate_random_projection_matrix(d, k, sparsity=0, method="gaussian", seed=1234):
    """
    Generate a random projection matrix of size (d x k) with either scaled Gaussian or scaled ±1 entries,
    with an option to zero out a fraction of the entries.

    Parameters:
      d (int): The number of rows (original dimension).
      k (int): The number of columns (projected dimension).
      sparsity (float): Fraction of entries to zero out (between 0 and 1). 0 means fully dense.
      method (str): The type of random projection to use. Options are:
                    - "gaussian": Use scaled Gaussian entries.
                    - "sign": Use scaled {+1, -1} entries.
      seed (int): Seed for reproducibility.
                    
    Returns:
      P (numpy.ndarray): A (d x k) projection matrix.
    """
    rng = np.random.default_rng(seed)

    if method == "gaussian":
        P = rng.standard_normal((d, k)) / np.sqrt(k)
    elif method == "sign":
        P = rng.choice([1, -1], size=(d, k)) / np.sqrt(k)
    else:
        raise ValueError("Invalid method. Choose 'gaussian' or 'sign'.")

    if sparsity > 0:
        if not (0 <= sparsity < 1):
            raise ValueError("sparsity must be between 0 and 1 (non-inclusive of 1).")
        mask = rng.choice([0, 1], size=P.shape, p=[sparsity, 1 - sparsity])
        P = (P * mask) / np.sqrt(1 - sparsity)
    
    return P

def compute_projection_approximation_errors(A, B, c_values,seed = 1234, sparsity = 0, method="gaussian"):
    """
    For a given matrix X (to approximate X^T X), its sampling distribution p,
    and a list of sample sizes (c_values), compute the relative spectral and Frobenius errors.
    
    Returns:
      spectral_errors: list of relative spectral norm errors.
      fro_errors: list of relative Frobenius norm errors.
    """
    spectral_errors = []
    fro_errors = []
    exact = A @ B
    norm_exact_spec = np.linalg.norm(exact, 2)
    norm_exact_fro = np.linalg.norm(exact, 'fro')
    for c in c_values:
        P = generate_random_projection_matrix(A.shape[1], c, sparsity=sparsity, method=method, seed=seed)
        C, R = projection_matrix_multiplication(A, B, P)
        approx = C @ R
        err_spec = np.linalg.norm(approx - exact, 2) / norm_exact_spec
        err_fro = np.linalg.norm(approx - exact, 'fro') / norm_exact_fro
        spectral_errors.append(err_spec)
        fro_errors.append(err_fro)
    return spectral_errors, fro_errors

def run_sample_trials(matrix_generator, matrix_params, c_values, num_trials, sampling_method, seed=1234):
    """
    Generate num_trials independent matrices using matrix_generator(**matrix_params)
    and compute the approximation errors for each using the sampling probabilities given by sampling_method.
    
    Returns:
      spec_errors: an array of shape (num_trials, len(c_values)) with spectral errors.
      fro_errors: an array of shape (num_trials, len(c_values)) with Frobenius errors.
    """
    spec_errors = np.zeros((num_trials, len(c_values)))
    fro_errors = np.zeros((num_trials, len(c_values)))
    
    for t in range(num_trials):
        trial_seed = seed + t  
        A = matrix_generator(**matrix_params, seed=trial_seed)
        p = compute_sampling_probabilities(A.T, A, method=sampling_method)
        spec_err, fro_err = compute_sample_approximation_errors(A.T, A, p, c_values)
        spec_errors[t, :] = spec_err
        fro_errors[t, :] = fro_err
    return spec_errors, fro_errors

def run_projection_trials(matrix_generator, matrix_params, c_values, num_trials, projection_method,p_sparsity = 0, seed=1234):
    """
    Generate num_trials independent matrices using matrix_generator(**matrix_params)
    and compute the approximation errors for each using the sampling probabilities given by sampling_method.
    
    Returns:
      spec_errors: an array of shape (num_trials, len(c_values)) with spectral errors.
      fro_errors: an array of shape (num_trials, len(c_values)) with Frobenius errors.
    """
    spec_errors = np.zeros((num_trials, len(c_values)))
    fro_errors = np.zeros((num_trials, len(c_values)))
    
    for t in range(num_trials):
        trial_seed = seed + t
        A = matrix_generator(**matrix_params, seed=trial_seed)
        spec_err, fro_err = compute_projection_approximation_errors(A.T, A, c_values, seed=trial_seed,sparsity=p_sparsity, method=projection_method)
        spec_errors[t, :] = spec_err
        fro_errors[t, :] = fro_err
    return spec_errors, fro_errors

def run_ls_trials(A, b, x_true, r_values, sampling_method, num_trials=20, seed = 1234):
    """
    For a given matrix A (m x d) and vector b (length m), runs num_trials of the
    slow least-squares sampling algorithm for each r in r_values using the specified sampling_method.
    Returns an array of shape (num_trials, len(r_values)) with the relative errors.
    """
    m, d = A.shape
    errors = np.zeros((num_trials, len(r_values)))
    p = compute_sampling_probabilities_rows(A, method=sampling_method)
    
    for i, r in enumerate(r_values):
        for t in range(num_trials):
            trial_seed = seed + t  
            x_approx = slow_least_squares_sampling(A, b, r, p, seed=trial_seed)
            errors[t, i] = np.linalg.norm(x_approx - x_true) / np.linalg.norm(x_true)
    return errors

def run_ls_projection_trials(A, b, x_true, r_values, projection_method,sparsity = 0, num_trials=20, seed = 1234):
    """
    For a given matrix A (m x d) and vector b (length m), runs num_trials of the
    least-squares projection algorithm for each r in r_values using the specified projection_method.
    Returns an array of shape (num_trials, len(r_values)) with the relative errors.
    """
    m, d = A.shape
    errors = np.zeros((num_trials, len(r_values)))
    
    for i, r in enumerate(r_values):
        for t in range(num_trials):
            trial_seed = seed + t  
            P = generate_random_projection_matrix(r, m, sparsity=sparsity, method=projection_method, seed=trial_seed)
            x_approx = least_squares_projection(A, b, P)
            errors[t, i] = np.linalg.norm(x_approx - x_true) / np.linalg.norm(x_true)
    return errors

def compute_mean_std(errors):
    return errors.mean(axis=0), errors.std(axis=0)

def plot_error_with_variability(c_values, mean_err, std_err, ylabel, title, color):
    plt.figure(figsize=(8,6))
    plt.plot(c_values, mean_err, color=color, label='Mean error')
    plt.fill_between(c_values, mean_err - std_err, mean_err + std_err, color=color, alpha=0.3, label='±1 Std')
    plt.xlabel('Number of Samples (c)')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

