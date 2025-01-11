import numpy as np
import json
import matplotlib.pyplot as plt
from AM_edge_class import X_k_edge_MBF, R_k_edge
from AM_edge_class import generate_ground_truth, generate_noisy_obs, squared_error
from EM_edge_class import expectation_maximization
from AM_edge_class import alternate_maximization
from params import *

### Param
####################
theta = np.pi/8
# Starting position
X_true_0 = np.array([[1], [0]])
# X_true_0 = np.array([[1/np.sqrt(2)], [1/np.sqrt(2)]])

N = 20
max_out_iter, max_in_iter = 50, 500
# max_out_iter, max_in_iter = 5, 50

np.random.seed(7)

plot = False
# plot = True
num_trials = 1
####################


# sigma2_Z_values = np.logspace(-5, 2, num=10)
sigma2_Z_values = [1e-4]

# Initialize storage for all SQE values
all_sqe_em = {}  # Dictionary to store lists of SQE for each sigma2_Z
all_sqe_am = {}

# Lists to store results
mean_sqe_values_em = []
mean_sqe_values_am = []

for sigma2_Z in sigma2_Z_values:
    sqe_list_em = []
    sqe_list_am = []

    for trial_index in range(num_trials):
        ## Data curation
        ####################
        R_true = np.array([[np.cos(theta)], [np.sin(theta)]])
        # Ground truth and noise observation
        X_true = generate_ground_truth(R_true, X_true_0, N)
        y_obs = generate_noisy_obs(X_true, C, sigma2_Z, N)
        ####################

        # Run the algorithm
        print("EM Algorithm")
        R_est_em, LL_series, _, _, _= expectation_maximization(sigma2_Z, N, y_obs, max_out_iter, max_in_iter, R_true, X_true)
        # Compute square error
        sqe_em = squared_error(R_est_em, R_true)
        sqe_list_em.append(sqe_em)

        print("AM Algorithm")
        R_est_am, X_est_am , _, _, _= alternate_maximization(sigma2_Z, N, y_obs, max_out_iter, max_in_iter, R_true, X_true)

        # Compute square error
        sqe_am = squared_error(R_est_am, R_true)
        sqe_list_am.append(sqe_am)

        print(f"sigma2_Z = {sigma2_Z:.2e}, Trial {trial_index}: EM L2 error= {sqe_em:.2e}, AM L2 error= {sqe_am:.2e}")

    # Store all SQE values for the current sigma2_Z
    all_sqe_em[sigma2_Z] = sqe_list_em
    all_sqe_am[sigma2_Z] = sqe_list_am

    # Compute mean for plotting
    mean_sqe_em = np.mean(sqe_list_em)
    mean_sqe_am = np.mean(sqe_list_am)

    mean_sqe_values_em.append(mean_sqe_em)
    mean_sqe_values_am.append(mean_sqe_am)
    print(f"sigma2_Z = {sigma2_Z:.2e}, EM L2 error= {mean_sqe_em:.2e}, AM L2 error= {mean_sqe_am:.2e}")

with open('sqe_data_N_{N}_max_out_iter_{max_out_iter}_max_in_iter_{max_in_iter}.json', 'w') as file:
    json.dump({'all_sqe_em': all_sqe_em, 'all_sqe_am': all_sqe_am}, file)

print("Data saved to sqe_data.json")



# Plotting section
if plot:
    plt.figure()

    # Mean line for EM
    plt.loglog(sigma2_Z_values, mean_sqe_values_em, marker='o', label='EM Mean', color='blue')
    # Mean line for AM
    plt.loglog(sigma2_Z_values, mean_sqe_values_am, marker='o', label='AM Mean', color='orange')

    # Scatter plot for all SQE values
    for sigma2_Z in sigma2_Z_values:
        plt.scatter([sigma2_Z] * len(all_sqe_em[sigma2_Z]), all_sqe_em[sigma2_Z], alpha=0.3, color='blue', label='EM Points' if sigma2_Z == sigma2_Z_values[0] else "")
        plt.scatter([sigma2_Z] * len(all_sqe_am[sigma2_Z]), all_sqe_am[sigma2_Z], alpha=0.3, color='orange', label='AM Points' if sigma2_Z == sigma2_Z_values[0] else "")


    # Axis labels and title
    plt.xlabel('sigma2_Z')
    plt.ylabel('Mean Squared L2 Error (sqe) |r - \hat{r}|^2')
    plt.title('Log-Log plot of Mean sqe vs sigma2_Z')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.savefig(f"scatter_em_am_sweep_N{N}_max_iter_50_max_pass_{max_pass}.pdf", bbox_inches='tight')
    # plt.show()



