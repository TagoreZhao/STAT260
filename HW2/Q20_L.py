import ace_tools_open as tools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import time
from utils import (generate_gaussian_A, 
                   generate_random_projection_matrix, 
                   least_squares_projection,
                   )

# ----------------------------
# Parameters
# ----------------------------
d = 50                                  # fixed number of columns
n_list = list(range(2*d, 100*d+1, d))   # n from 2d to 100d, in steps of d
num_trials = 5                          # number of trials for averaging timings
oversampling_parameter = 0.5            # relative oversampling: r = oversampling_parameter * n
proj_method = 'sign'                    # using {±1} projection matrix
sparsity = 0                            # dense projection matrix (sparsity=0)
seed = 1234

# ----------------------------
# Running time and error comparisons
# ----------------------------
time_rproj = []   # average running time for random projection LS
time_qr = []      # average running time for QR-based LS
time_svd = []     # average running time for SVD-based LS

err_rproj = []    # average relative LS error for random projection LS
err_qr = []       # average relative LS error for QR-based LS
err_svd = []      # average relative LS error for SVD-based LS

for n in n_list:
    t_rproj = []
    t_qr = []
    t_svd = []
    e_rproj = []
    e_qr = []
    e_svd = []
    
    for trial in range(num_trials):
        trial_seed = seed + trial
        
        # Generate an (n x d) matrix A and a vector b (length n)
        A = generate_gaussian_A(n, d, seed=trial_seed)
        rng = np.random.default_rng(trial_seed)
        b = rng.random(n)
        
        # Compute the true LS solution using full data
        x_true = np.linalg.lstsq(A, b, rcond=None)[0]
        
        # 1. Random Projection LS method:
        # Set r = oversampling_parameter * n (as integer)
        r = int(oversampling_parameter * n)
        # Generate a random projection matrix P of size (n x r) with {±1} entries.
        # (Note: Adjust the arguments if your function expects (orig_dim, proj_dim) in a different order.)
        P = generate_random_projection_matrix(n, r, sparsity=sparsity, method=proj_method, seed=trial_seed)
        start = time.time()
        x_rproj = least_squares_projection(A, b, P)
        t_rproj.append(time.time() - start)
        e_rproj.append(np.linalg.norm(x_rproj - x_true) / np.linalg.norm(x_true))
        
        # 2. QR-based LS method:
        start = time.time()
        Q, R_mat = np.linalg.qr(A)
        x_qr = np.linalg.solve(R_mat, Q.T @ b)
        t_qr.append(time.time() - start)
        e_qr.append(np.linalg.norm(x_qr - x_true) / np.linalg.norm(x_true))
        
        # 3. SVD-based LS method:
        start = time.time()
        U, s_vals, Vt = np.linalg.svd(A, full_matrices=False)
        x_svd = Vt.T @ np.linalg.solve(np.diag(s_vals), U.T @ b)
        t_svd.append(time.time() - start)
        e_svd.append(np.linalg.norm(x_svd - x_true) / np.linalg.norm(x_true))
        
    time_rproj.append(np.mean(t_rproj))
    time_qr.append(np.mean(t_qr))
    time_svd.append(np.mean(t_svd))
    err_rproj.append(np.mean(e_rproj))
    err_qr.append(np.mean(e_qr))
    err_svd.append(np.mean(e_svd))

# ----------------------------
# Plot Running Times vs n and Save Plot
# ----------------------------
plt.figure(figsize=(10,6))
plt.plot(n_list, time_rproj, marker='o', label="Random Projection LS")
plt.plot(n_list, time_qr, marker='s', label="QR")
plt.plot(n_list, time_svd, marker='^', label="SVD")
plt.xlabel("Number of rows n")
plt.ylabel("Average Running Time (seconds)")
plt.title("Running Time Comparison vs. n (d = {})".format(d))
plt.legend()
plt.grid(True)
plt.savefig("running_time_comparison.png")
plt.close()

# ----------------------------
# Plot LS Errors vs n and Save Plot
# ----------------------------
plt.figure(figsize=(10,6))
plt.plot(n_list, err_rproj, marker='o', label="Random Projection LS")
plt.plot(n_list, err_qr, marker='s', label="QR")
plt.plot(n_list, err_svd, marker='^', label="SVD")
plt.xlabel("Number of rows n")
plt.ylabel("Average Relative LS Solution Error")
plt.title("LS Error Comparison vs. n (d = {})".format(d))
plt.legend()
plt.grid(True)
plt.savefig("ls_error_comparison.png")
plt.close()
