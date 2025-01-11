import numpy as np
# import cupy as np
import matplotlib.pyplot as plt
from AM_edge_class import X_k_edge_MBF, estimate_X
from AM_edge_class import alternate_maximization, plot_x_and_r_am
from EM_edge_class import R_k_edge_EM
from EM_edge_class import expectation_maximization, plot_x_and_r_em
from util import *
from params import *
import random


############################################
## Plot settings
SMALL_SIZE = 8
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
############################################

############################################
## Parameters

# Starting position
X_true_0 = np.array([[np.cos(np.pi/3)], [np.sin(np.pi/3)]])
# X_true_0 = np.array([[1], [0]])
# X_true_0 = np.array([[0], [1]])
# X_true_0 = np.array([[1/np.sqrt(2)], [1/np.sqrt(2)]])

# Number of time steps
N = 8

# Random seed
np.random.seed(8)
############################################

############################################
## Ground truth
theta = np.pi/4
# theta = np.pi/8
R_true = np.array([[np.cos(theta)], [np.sin(theta)]])
X_true = generate_ground_truth(R_true, X_true_0, N)
plot_x_and_r_true(
    X_true=X_true,
    R_true=R_true,
    save_path= f"plot/frequency_estimation_ground_truth.png"
)
############################################


############################################
# Experiment 1: both EM and AM working
sigma2_Z = 1e-3
max_out_iter, max_in_iter = 5, 5

# Generate noisy observation
y_obs = generate_noisy_obs(X_true, C, sigma2_Z, N)

print("Starting EM Algorithm")
R_est_am, X_est_am , X_est_vis, _, _= expectation_maximization(sigma2_Z, N, y_obs, max_out_iter, max_in_iter, R_true, X_true)

for entry in X_est_vis:
    step = entry["step"]
    in_iter = entry["in_iter"]
    out_iter =entry["out_iter"]
    save_path = f"plot/exp1/em_series/plot_out_iter_{out_iter}_iter_{in_iter}_{step}.png"

    # Plot each step
    plot_x_and_r_em(
        X_vis_entry=entry["X_vis"],
        X_true=X_true,
        R_est=entry["R_est"],
        R_true=R_true,
        step = step,
        save_path=save_path
    )

print("Starting AM Algorithm")
max_out_iter, max_in_iter = 2, 5
R_est_am, X_est_am , X_est_vis, _, _= alternate_maximization(sigma2_Z, N, y_obs, max_out_iter, max_in_iter, R_true, X_true)
for entry in X_est_vis:
    step = entry["step"]
    in_iter = entry["in_iter"]
    out_iter =entry["out_iter"]
    save_path = f"plot/exp1/am_series/plot_out_iter_{out_iter}_iter_{in_iter}_{step}.png"

    # Plot each step
    plot_x_and_r_am(
        X_vis_entry=entry["X_vis"],
        X_true=X_true,
        R_est=entry["R_est"],
        R_true=R_true,
        step = step,
        save_path=save_path
    )
############################################

############################################
# Experiment 2: AM working but EM fails
sigma2_Z = 1e-1
max_out_iter, max_in_iter = 3, 50
plot = True
plot_steps = False

# Generate noisy observation
y_obs = generate_noisy_obs(X_true, C, sigma2_Z, N)

print("Starting EM Algorithm")
R_est_em, LL_series, X_est_vis, theta_series_em, r_norm_series_em = expectation_maximization(sigma2_Z, N, y_obs, max_out_iter, max_in_iter, R_true, X_true)

if plot_steps:
    for entry in X_est_vis:
        step = entry["step"]
        in_iter = entry["in_iter"]
        out_iter =entry["out_iter"]
        save_path = f"plot/exp2/em_series_fail/plot_out_iter_{out_iter}_iter_{in_iter}_{step}.png"

        plot_x_and_r_em(
            X_vis_entry=entry["X_vis"],
            X_true=X_true,
            R_est=entry["R_est"],
            R_true=R_true,
            step = step,
            save_path=save_path
        )

print("Starting AM Algorithm")
R_est_am, X_est_am , X_est_vis, theta_series_am, r_norm_series_am= alternate_maximization(sigma2_Z, N, y_obs, max_out_iter, max_in_iter, R_true, X_true)
if plot_steps:
    for entry in X_est_vis:
        step = entry["step"]
        in_iter = entry["in_iter"]
        out_iter =entry["out_iter"]
        save_path = f"plot/exp2/am_series_control/plot_out_iter_{out_iter}_iter_{in_iter}_{step}.png"

        plot_x_and_r_am(
            X_vis_entry=entry["X_vis"],
            X_true=X_true,
            R_est=entry["R_est"],
            R_true=R_true,
            step = step,
            save_path=save_path
        )

if plot:
    # Plot LL_series
    plt.figure()
    plt.plot(range(len(LL_series)), LL_series, marker='o', linestyle='-', color='b', markersize=2, label=r"EM log-likelihood")
    plt.xlabel('Epoch')
    plt.ylabel('Log-likelihood')
    plt.axvline(x=max_in_iter, color='black', linestyle='--', linewidth=1 )  # Vertical line at index 50
    plt.axvline(x=2*max_in_iter, color='black', linestyle='--', linewidth=1)# Vertical line at index 100
    plt.grid(True, which="both", linestyle="--")
    plt.legend()
    plt.savefig(f"plot/exp2/LL_series_N{N}_max_out_iter_{max_out_iter}_max_in_iter_{max_in_iter}.pdf", bbox_inches='tight')
    # plt.show()

    # Plot theta_series
    plt.figure()
    plt.plot(range(len(theta_series_em)), theta_series_em, marker='s', linestyle='--', color='blue', label=r"EM $\hat{\theta}$", markersize=2)
    plt.plot(range(len(theta_series_am)), theta_series_am, marker='s', linestyle='--', color='orange', label=r"AM $\hat{\theta}$",  markersize=2)
    plt.axvline(x=max_in_iter, color='black', linestyle='--', linewidth=1 )  # Vertical line at index 50
    plt.axvline(x=2*max_in_iter, color='black', linestyle='--', linewidth=1)# Vertical line at index 100
    plt.axhline(y=theta, color='g', linestyle='--', linewidth=2, label=r'True $\theta$')  # Horizontal line
    plt.ylabel(r"$\theta$")
    plt.xlabel('Epoch')
    plt.grid(True, which="both", linestyle="--")
    plt.legend()
    plt.savefig(f"plot/exp2/theta_series_N{N}_max_out_iter_{max_out_iter}_max_in_iter_{max_in_iter}.pdf", bbox_inches='tight')
    # plt.show()

    # Plot theta_series
    plt.figure()
    plt.plot(range(len(r_norm_series_em)), r_norm_series_em, marker='s', linestyle='--', color='blue', label=r"EM $\lVert\hat{r}\rVert$", markersize=2)
    plt.plot(range(len(r_norm_series_am)), r_norm_series_am, marker='s', linestyle='--', color='orange', label=r"AM $\lVert\hat{r}\rVert$",  markersize=2)
    plt.axvline(x=max_in_iter, color='black', linestyle='--', linewidth=1 )  # Vertical line at index 50
    plt.axvline(x=2*max_in_iter, color='black', linestyle='--', linewidth=1)# Vertical line at index 100
    plt.axhline(y=1, color='g', linestyle='--', linewidth=2, label=r'True $\lVert r \rVert$')  # Horizontal line
    plt.ylabel(r"$\lVert r \rVert$")
    plt.xlabel('Epoch')
    plt.grid(True, which="both", linestyle="--")
    plt.legend()
    plt.savefig(f"plot/exp2/r_norm_series_{N}_max_out_iter_{max_out_iter}_max_in_iter_{max_in_iter}.pdf", bbox_inches='tight')
    # plt.show()
