import numpy as np
import matplotlib.pyplot as plt
from AM_edge_class import X_k_edge_MBF
from EM_edge_class import R_k_edge_EM
from AM_edge_class import print_matrix, convert_vect_to_rotation_matrix, generate_ground_truth, generate_noisy_obs, estimate_X, estimate_R, squared_error,print_vectors_side_by_side,print_vector,print_vectors_side_by_side_float
from EM_edge_class import expectation_maximization
from AM_edge_class import alternate_maximization
from params import *
import random

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')


### Param
####################
theta = np.pi/4
X_true_0 = np.array([[1/np.sqrt(2)], [1/np.sqrt(2)]])

## Example: adding more iterations worsens!
sigma2_Z, N = 3.59e-04, 20
############################################

max_in_iter = 5
max_out_iter = 10


plot = True
# plot = False

# Fix random seed
np.random.seed(8)

## Data curation
R_true = np.array([[np.cos(theta)], [np.sin(theta)]])
X_true = generate_ground_truth(R_true, X_true_0, N)
y_obs = generate_noisy_obs(X_true, C, sigma2_Z, N)

print("Starting EM Algorithm")
R_est_em, LL_series, _, theta_series_em, r_norm_series_em = expectation_maximization(sigma2_Z, N, y_obs, max_out_iter, max_in_iter, R_true, X_true)

print("Starting AM Algorithm")
R_est_am, X_est_am , _, theta_series_am, r_norm_series_am = alternate_maximization(sigma2_Z, N, y_obs, max_out_iter, max_in_iter, R_true, X_true)

sqe_em = squared_error(R_est_em, R_true)
sqe_am = squared_error(R_est_am, R_true)
print(f"sigma2_Z = {sigma2_Z:.2e}, EM L2 error= {sqe_em:.2e}, AM L2 error= {sqe_am:.2e}")
