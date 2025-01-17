import numpy as np
import matplotlib.pyplot as plt
from AM_edge_class import X_k_edge_MBF
from EM_edge_class import R_k_edge_EM
from EM_edge_class import expectation_maximization
from AM_edge_class import alternate_maximization
from params import *
from util import *
import random

"""
Runs a single experiment for both the EM algorithm and AM algorithm.
If plot is set to true, the script also creates a plot comparing log-likelihood, theta, and norm(r).
"""

### Plot settings
####################
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
####################


### Parameters
####################
# Ground truth R= [cos(theta), sin(theta)]
theta = np.pi/4
# Starting position
X_true_0 = np.array([[1/np.sqrt(2)], [1/np.sqrt(2)]])

# Number of time steps
N = 20
# Number of outer iterations (decreases sigma2_U)
max_out_iter = 3
# Number of inner iterations (iterative algorithm) (iterative algorithm) (iterative algorithm)
max_in_iter = 50

# Observation noise values
sigma2_Z = 2e-4

# plot=True generates plots for log-likelihood, theta, and norm{r}
plot = True
# DEBUG mode prints out the Gaussian messages
DEBUG = True

# Fix random seed
np.random.seed(9)
####################


## Data curation
R_true = np.array([[np.cos(theta)], [np.sin(theta)]])
X_true = generate_ground_truth(R_true, X_true_0, N)
y_obs = generate_noisy_obs(X_true, C, sigma2_Z, N)

## EM algorithm
print("Starting EM Algorithm")
R_est_em, LL_series, _, theta_series_em, r_norm_series_em = expectation_maximization(sigma2_Z, N, y_obs, max_out_iter, max_in_iter, R_true, X_true, DEBUG)
sqe_em = squared_error(R_est_em, R_true)

## AM algorithm
print("Starting AM Algorithm")
R_est_am, X_est_am , _, theta_series_am, r_norm_series_am = alternate_maximization(sigma2_Z, N, y_obs, max_out_iter, max_in_iter, R_true, X_true, DEBUG)
sqe_am = squared_error(R_est_am, R_true)

## Final squared error
print(f"sigma2_Z = {sigma2_Z:.2e}, EM L2 error= {sqe_em:.2e}, AM L2 error= {sqe_am:.2e}")

if plot:
    # Plot LL_series
    plt.figure()
    plt.plot(range(len(LL_series)), LL_series, marker='o', linestyle='-', color='b', markersize=2, label=r"EM log-likelihood")
    plt.xlabel('Epoch')
    plt.ylabel('Log-likelihood')
    plot_epoch_borders(max_out_iter, max_in_iter)
    plt.grid(True, which="both", linestyle="--")
    plt.legend()
    plt.savefig(f"plot/single/LL_series_N{N}_max_out_iter_{max_out_iter}_max_in_iter_{max_in_iter}.pdf", bbox_inches='tight')
    # plt.show()

    # Plot theta_series
    plt.figure()
    plt.plot(range(len(theta_series_em)), theta_series_em, marker='s', linestyle='--', color='blue', label=r"EM $\hat{\theta}$", markersize=2)
    plt.plot(range(len(theta_series_am)), theta_series_am, marker='s', linestyle='--', color='orange', label=r"AM $\hat{\theta}$",  markersize=2)
    plot_epoch_borders(max_out_iter, max_in_iter)
    plt.axhline(y=theta, color='g', linestyle='--', linewidth=2, label=r'True $\theta$')  # Horizontal line
    plt.ylabel(r"$\theta$")
    plt.xlabel('Epoch')
    plt.grid(True, which="both", linestyle="--")
    plt.legend()
    plt.savefig(f"plot/single/theta_series_N{N}_max_out_iter_{max_out_iter}_max_in_iter_{max_in_iter}.pdf", bbox_inches='tight')
    # plt.show()

    # Plot theta_series
    plt.figure()
    plt.plot(range(len(r_norm_series_em)), r_norm_series_em, marker='s', linestyle='--', color='blue', label=r"EM $\lVert\hat{r}\rVert$", markersize=2)
    plt.plot(range(len(r_norm_series_am)), r_norm_series_am, marker='s', linestyle='--', color='orange', label=r"AM $\lVert\hat{r}\rVert$",  markersize=2)
    plot_epoch_borders(max_out_iter, max_in_iter)
    plt.axhline(y=1, color='g', linestyle='--', linewidth=2, label=r'True $\lVert r \rVert$')  # Horizontal line
    plt.ylabel(r"$\lVert r \rVert$")
    plt.xlabel('Epoch')
    plt.grid(True, which="both", linestyle="--")
    plt.legend()
    plt.savefig(f"plot/single/r_norm_series_{N}_max_out_iter_{max_out_iter}_max_in_iter_{max_in_iter}.pdf", bbox_inches='tight')
    # plt.show()
