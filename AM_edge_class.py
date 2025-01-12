import numpy as np
# import cupy as np
from params import *
from util import *
import matplotlib.pyplot as plt

class X_k_edge_MBF:
    def __init__(self, sigma2_Z, msg_V_init, msg_W_init, DEBUG=False):
        # Forward messages
        self.msgf_m_X_k_prime = np.array([[0], [0]])
        self.msgf_m = np.array([[0], [0]])
        self.msgf_V = msg_V_init.copy()
        self.msgf_V_X_k_prime = msg_V_init.copy()

        # Auxiliary
        self.G = np.eye(2)
        self.G_dd = np.eye(2)
        self.g_dd = np.array([[0], [0]])

        # Tilde
        self.xi_tilde = np.array([[0], [0]])
        self.W_tilde_prime = msg_W_init
        self.W_tilde = msg_W_init

        # V
        self.V = msg_V_init
        # m
        self.m = np.array([[0], [0]])

        # A
        self.update_A_hat_r(R_init)

        self.sigma2_Z = sigma2_Z
        self.DEBUG = DEBUG


    def update_A_hat_r (self, r):
        self.A_hat_r = convert_vect_to_rotation_matrix(r)
        return

    def forward(self, X_k_minus_1, y_k, V_U):
        self.msgf_m_X_k_prime = self.A_hat_r @ X_k_minus_1.msgf_m                            # Eq. 3.209
        self.msgf_V_X_k_prime = symmetrize(self.A_hat_r @ X_k_minus_1.msgf_V @ self.A_hat_r.T + V_U)     # Eq. 3.210
        G = np.linalg.inv(self.sigma2_Z + C @ self.msgf_V_X_k_prime @ C.T)
        self.G = G
        self.G_dd = C.T @ G @ C
        self.g_dd = C.T @ G @ (y_k - C @ self.msgf_m_X_k_prime)

        self.msgf_m = self.msgf_m_X_k_prime + self.msgf_V_X_k_prime @ self.g_dd
        self.msgf_V = symmetrize(self.msgf_V_X_k_prime - self.msgf_V_X_k_prime @ self.G_dd @ self.msgf_V_X_k_prime)

        assert(np.linalg.det(self.msgf_V_X_k_prime) >0), f"Negative deteriminant {np.linalg.det(self.msgf_m_X_k_prime):.2e}"
        assert(np.linalg.det(self.msgf_V) > 0), f"Negative deteriminant {np.linalg.det(self.msgf_V):.2e}"

        if (self.DEBUG):
            print_matrices_side_by_side(X_k_minus_1.msgf_V, self.msgf_V_X_k_prime, "X_k_minus_1.msgf_V", "self.msgf_V_X_k_prime")
            # print_matrix(self.msgf_V_X_k_prime, "self.msgf_V_X_k_prime")
            # print_matrix(self.msgf_V, "self.msgf_V")

        return

    def backward(self, X_k_plus_1):
        # Based on MESA 3.216 - 3.220. With self = X_{k-1} in script
        F_k_plus_1 = np.eye(2) - X_k_plus_1.msgf_V_X_k_prime @ X_k_plus_1.G_dd
        xi_tilde_X_k_plus_1_prime = F_k_plus_1.T @ X_k_plus_1.xi_tilde  - X_k_plus_1.g_dd
        self.xi_tilde  = self.A_hat_r.T @ xi_tilde_X_k_plus_1_prime

        # Optional (for variance estimation) Eq. 3.219-3.220 in MESA script
        X_k_plus_1.W_tilde_prime = symmetrize(F_k_plus_1.T @ X_k_plus_1.W_tilde @ F_k_plus_1 + X_k_plus_1.G_dd)
        self.W_tilde = symmetrize(self.A_hat_r.T @ X_k_plus_1.W_tilde_prime @ self.A_hat_r)

        if (np.linalg.det(X_k_plus_1.W_tilde_prime) == 0):
            print_matrix(X_k_plus_1.W_tilde, "X_k_plus_1.W_tilde")
        if (np.linalg.det(self.W_tilde) == 0):
            print_matrix(self.W_tilde, "self.W_tilde")

        assert(np.linalg.det(X_k_plus_1.W_tilde_prime) > 0), f"X_k_plus_1.W_tilde_prime (det {np.linalg.det(X_k_plus_1.W_tilde_prime):2e})"
        assert(np.linalg.det(self.W_tilde) > 0), f"self.W_tilde (det {np.linalg.det(self.W_tilde):.2e})"

        if (self.DEBUG):
            print_vector(self.xi_tilde, "X_k.xi_tilde")
            print_matrix(X_k_plus_1.W_tilde_prime, f"X_k_plus_1.W_tilde_prime (rank {np.linalg.matrix_rank(X_k_plus_1.W_tilde_prime)})")
            print_matrix(self.W_tilde, "self.W_tilde")

        return

    def marginal(self):
        self.m = self.msgf_m - self.msgf_V @ self.xi_tilde  # Eq. 3.223
        self.V = self.msgf_V - self.msgf_V @ self.W_tilde @ self.msgf_V # Eq. 3227

        if self.DEBUG:
            print_vector(self.m, "self.m")
            print_matrix( self.V, "self.V")

        return self.m

    def get_estimate(self):
        return self.m

    def em_log_likelihood(self, y_k):
        LL_k = - (y_k - C @ self.msgf_m)**2 / (self.sigma2_Z + C@ self.msgf_V@ C.T) -np.log(2*np.pi * (self.sigma2_Z + C@ self.msgf_V@ C.T))
        return LL_k[0,0]

class R_k_edge:
    def __init__(self, msg_V_init, msg_W_init, DEBUG=False):
        # Upwards message
        self.msgb_W_norm = 0
        self.msgb_xi_norm = np.array([[0], [0]])

        # Constants
        self.A_hat_x_k_minus_1 = np.eye(2)
        self.x_k = X_init
        self.x_k_minus_1 = X_init
        self.DEBUG = DEBUG

    def update_x_hat_and_A(self, x_k, x_k_minus_1):
        self.x_k = x_k
        self.x_k_minus_1 = x_k_minus_1
        self.A_hat_x_k_minus_1 = convert_vect_to_rotation_matrix(x_k_minus_1)
        return

    def backward(self, R_k_plus_1: 'R_k_edge', V_U):
        msgb_W_X0_k = np.linalg.inv(V_U)

        self.msgb_W_norm_coeff = convert_1x1_matrix_to_scalar(self.x_k_minus_1.T @ self.x_k_minus_1)
        self.msgb_xi_norm = self.A_hat_x_k_minus_1.T @ self.x_k
        if (self.DEBUG):
            print(f"R_k.msgb_W_norm_coeff: {self.msgb_W_norm_coeff:.2f}")
            msgb_m = 1/self.msgb_W_norm_coeff * self.msgb_xi_norm
            print_vectors_side_by_side(self.msgb_xi_norm, msgb_m,  "msgb_xi_norm", "msgb_m")

        return

    def marginal_estimate(self):
        return self.m

    def update_marginal(self, est):
        self.m = est
        return

def collect_R_est (R_edges, N):
    msgb_R_xi_norm = np.array([[0], [0]])
    msgb_R_W_norm_coeff = 0
    for k in range(1, N+1):
        # Sum all msgb
        msgb_R_xi_norm = msgb_R_xi_norm + R_edges[k].msgb_xi_norm
        msgb_R_W_norm_coeff = msgb_R_W_norm_coeff + R_edges[k].msgb_W_norm_coeff
    R_est_new = 1/msgb_R_W_norm_coeff * msgb_R_xi_norm
    return R_est_new

def estimate_X(R_est, X_edges, y, V_U, N, sigma2_Z, msg_V_init, msg_W_init, DEBUG):
    for k in range(0, N+1):
        X_edges[k].__init__(sigma2_Z, msg_V_init, msg_W_init, DEBUG)

    for k in range(0, N+1): # k = 0, ..., N
        X_edges[k].update_A_hat_r(R_est)

    for k in range(1, N+1): # k = 1, ..., N
        if (DEBUG):
            print(f"Forward pass on X_{k}")
        X_edges[k].forward(X_edges[k - 1], y[k], V_U)

    # for k in range(N-1, -1, -1):
    # Computation for X[k] is done at X[k-1]
    # Backward pass X[N-1], X[N-2], ..., X[0].
    # Effectively computation for X[N], ..., X[1]
    for k in range(N-1, -1, -1): # k = N-1, N-2, ..., 0
        if (DEBUG):
            print(f"Backward pass on X_{k}")
        X_edges[k].backward(X_edges[k + 1])

    ### Part 1.5 (update X)
    X_est = []
    for k in range(0, N+1): # k = 0, ..., N
        if (DEBUG):
            print(f"Marginal pass on X_{k}")
        m_est = X_edges[k].marginal()
        X_est += [m_est]

    return X_est

def estimate_R(X_est, R_edges, V_U, N, msg_V_init, msg_W_init, DEBUG):
    for k in range(0, N+1):
        R_edges[k].__init__(msg_V_init, msg_W_init, DEBUG)

    for k in range(1, N+1):
        R_edges[k].update_x_hat_and_A(X_est[k], X_est[k-1])

    ### Part 2: fix X, estimate R
    # print(f"Step 2: Assume fixed X, estimate R.")
    # Backward R[N], ..., R[1]
    for k in range(N, 0, -1):
        if (DEBUG):
            print(f"Backward pass on R_{k}")
        R_edges[k].backward(R_edges[k], V_U)

    # Estimate R
    ##################
    R_est_new = collect_R_est(R_edges, N)
    ##################
    return R_est_new

def alternate_maximization(sigma2_Z, N, y_obs, max_out_iter, max_in_iter, R_true, X_true, DEBUG=False):
    ## Params
    msg_V_init,msg_W_init,V_U_coeff = setup_params(sigma2_Z)
    ####

    X_edges = [X_k_edge_MBF(sigma2_Z, msg_V_init, msg_W_init, DEBUG) for _ in range(N+1)]
    R_edges = [R_k_edge(msg_V_init, msg_W_init, DEBUG) for _ in range(N+1)]

    R_est = R_init
    theta_series = []
    r_norm_series = []
    X_est_vis = []

    for out_iter in range (max_out_iter):
        print(f"Outer iteration {out_iter + 1}/{max_out_iter}")
        for in_iter in range (max_in_iter):
            print(f"Iteration {in_iter + 1}/{max_in_iter}")
            # print_vector(R_est, "Current R_est")
            V_U = V_U_coeff * np.eye(2)

            ## Step 1: Estimate X while keeping R fixed
            X_est = estimate_X(R_est, X_edges, y_obs, V_U, N, sigma2_Z, msg_V_init, msg_W_init, DEBUG)

            # Print out X states
            for k in range(0, N+1):
                m_est = X_est[k]
                if (DEBUG):
                    print_vectors_side_by_side_float(m_est, X_true[k], f"X_{k}.m", f"True X_{k}")

            # Visualize X estimation
            X_vis = [{"m": x.marginal(), "V": x.V} for x in X_edges[1:]]
            X_est_vis.append({"step": "1_X_estimation", "out_iter": out_iter, "in_iter": in_iter, "X_vis": X_vis, "R_est": R_est.copy()})

            ## Step 2: Estimate R while keeping X fixed
            R_est = estimate_R(X_est, R_edges, V_U, N, msg_V_init, msg_W_init, DEBUG)

            ## Metrics
            sqe = squared_error(R_est, R_true)
            theta_hat = vector_angle(R_est)
            theta_series = theta_series + [theta_hat]
            r_norm_series = r_norm_series + [np.linalg.norm(R_est)]

            # Print out R estimate
            if (DEBUG):
                print_vectors_side_by_side(R_est, R_true, f"sqe: {sqe:.2e} >R_est", "R_true")

            # Visualize R estimation
            X_est_vis.append({"step": "2_R_estimation", "out_iter": out_iter, "in_iter": in_iter, "X_vis": X_vis, "R_est": R_est.copy()})





        V_U_coeff = (V_U_coeff * 0.5)
        print_vectors_side_by_side(R_est, R_true, f"sqe: {sqe:.2e} >R_est", "R_true")

    return R_est, X_est, X_est_vis, theta_series, r_norm_series

## Visualization
def plot_x_and_r_am(X_vis_entry, X_true, R_est, R_true, step, save_path=None):
    """
    Plot the mean and variance of X along with the estimated and true rotation vectors.

    Parameters:
        X_vis_entry (list): A list of dictionaries with keys 'm' (mean) and 'V' (variance) for each X.
        R_est (np.ndarray): Estimated rotation vector.
        R_true (np.ndarray): True rotation vector.
        save_path (str, optional): Path to save the plot. If None, the plot is displayed.
    """
    plt.figure(figsize=(8, 8))

    # Plot the unit circle
    circle = plt.Circle((0, 0), 1, color='grey', fill=False, linestyle='dotted', linewidth=1.5)
    plt.gca().add_artist(circle)

    # Plot the estimated rotation vector
    plt.quiver(0, 0, R_est[0, 0], R_est[1, 0], angles='xy', scale_units='xy', scale=1, color='blue', label=f"Estimated R ({R_est[0,0]:.2f}, {R_est[1,0]:.2f})")

    # Plot the true rotation vector
    plt.quiver(0, 0, R_true[0, 0], R_true[1, 0], angles='xy', scale_units='xy', scale=1, color='purple', label="True R")

    X_true_np = np.array(X_true)  # Assuming X_true is a list of dictionaries with 'm'
    plt.scatter(X_true_np[:, 0], X_true_np[:, 1], color='black', alpha=0.7, label="True X")

    # Plot X means and variances
    label_for_x_mean = "Estimated X"
    for x in X_vis_entry:
        mean = x['m'].flatten()
        # var = 0.2* np.sqrt(np.linalg.det(x['V']))  # control: 0.2
        var = np.sqrt(np.linalg.det(x['V']))  # working: 1
        # print(mean)
        # print(f"{var:.2e}")
        # plt.scatter(mean[0], mean[1], color='red', label="Estimated X")
        handles, labels = plt.gca().get_legend_handles_labels()
        if label_for_x_mean not in labels:
            plt.scatter(mean[0], mean[1], color='red', label=label_for_x_mean)
        else:
            plt.scatter(mean[0], mean[1], color='red')
        if (step =="1_X_estimation"):
            # We treat X as constant when estimating R.
            variance_circle = plt.Circle((mean[0], mean[1]), var, color='red', fill=False, linestyle='--', alpha=0.5)
            plt.gca().add_artist(variance_circle)

    # Configure the plot
    plt.axhline(0, color='k', linestyle='--', linewidth=0.5)
    plt.axvline(0, color='k', linestyle='--', linewidth=0.5)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend(loc='upper left')
    plt.axis('equal')
    plt.grid()
    if (step =="1_X_estimation"):
        plt.title("Step 1: Fix $R$, estimate $X$")
    else:
        plt.title("Step 2: Fix $X$, estimate $R$")
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)


    # Save or display the plot
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

    return

