import numpy as np
from AM_edge_class import X_k_edge_MBF, estimate_X, collect_R_est
from params import *
from util import *
import matplotlib.pyplot as plt

P = np.array([[0, -1], [1, 0]])
class R_k_edge_EM:
    def __init__(self, msg_V_init, msg_W_init):
        # Upwards message
        self.msgb_W =msg_W_init.copy()
        self.msgb_xi = np.array([[0], [0]])

        # Normalized upwards message
        self.msgb_W_norm_coeff = 0
        self.msgb_xi_norm = np.array([[0], [0]])

        # Constants
        # self.A = convert_vect_to_rotation_matrix(X_init)

        # Marginals
        self.m = R_init
        self.V = msg_V_init.copy()

        # self.DEBUG = True
        self.DEBUG = False

    def update_marginal(self, m):
        self.m = m
        return

    def backward(self, X_k_minus_1: 'X_k_edge_MBF', X_k: 'X_k_edge_MBF', V_U_coeff):
        A_r = convert_vect_to_rotation_matrix(self.m)

        # W matrix
        # self.msgb_W_norm_coeff = np.trace(X_k_minus_1.V + X_k_minus_1.m @ X_k_minus_1.m.T )
        self.msgb_W_norm_coeff = np.trace(X_k_minus_1.V) + convert_1x1_matrix_to_scalar(X_k_minus_1.m.T @ X_k_minus_1.m)
        self.msgb_W = 1/V_U_coeff * self.msgb_W_norm_coeff * np.eye(2)

        # Xi
        msgb_V_X_k_prime = np.linalg.inv(X_k.W_tilde_prime) - X_k.msgf_V_X_k_prime # msgb_V_X_k_prime = (W_tilde_prime^-1 - msgf_V_X_k_prime)
        # V_X_k_prime_X_k_minus_1_T = msgb_V_X_k_prime.T @ X_k.W_tilde_prime.T @ A_r @ X_k_minus_1.msgf_V.T
        V_X_k_minus_1_X_k_prime_T = X_k_minus_1.msgf_V @ A_r.T @ X_k.W_tilde_prime @ msgb_V_X_k_prime
        """
        trace_arg_1 =  V_X_k_minus_1_X_k_prime_T + X_k.m @ X_k_minus_1.m.T
        trace_arg_2 = P.T @ trace_arg_1
        Exp_X_k_minus_1_T_X_k_prime = np.trace(trace_arg_1)
        Exp_X_k_minus_1_T_P_T_X_k_prime = np.trace( trace_arg_2 )
        """
        Exp_X_k_minus_1_T_X_k_prime = np.trace(V_X_k_minus_1_X_k_prime_T) + X_k_minus_1.m.T @ X_k.m
        Exp_X_k_minus_1_T_P_T_X_k_prime = np.trace(V_X_k_minus_1_X_k_prime_T @ P) + X_k_minus_1.m.T @ P.T @ X_k.m
        # Exp_X_k_minus_1_T_X_k_prime = X_k_minus_1.m.T @ X_k.m
        # Exp_X_k_minus_1_T_P_T_X_k_prime = X_k_minus_1.m.T @ P.T @ X_k.m


        # self.msgb_xi_norm = np.array([ [ Exp_X_k_minus_1_T_X_k_prime], [Exp_X_k_minus_1_T_P_T_X_k_prime]])
        self.msgb_xi_norm = np.vstack((Exp_X_k_minus_1_T_X_k_prime, Exp_X_k_minus_1_T_P_T_X_k_prime))
        self.msgb_xi = 1/V_U_coeff * self.msgb_xi_norm

        """
        print_matrix(X_k.W_tilde_prime, "W_X_k_tilde_prime")
        print_vectors_side_by_side(msgb_m, msgb_m_prime, "msgb_m", "msgb_m_prime")
        print_vector(self.msgb_xi, "msgb_xi")
        print_vector(X_k_minus_1.m, "X_k_minus_1.m")
        print_vectors_side_by_side(X_k.m, X_k_minus_1.m, "X_k.m", "X_k_minus_1.m")
        print_matrices_side_by_side(msgb_V_X_k_prime ,X_k_minus_1.msgf_V.T, "msgb_V_X_k_prime", "X_k_minus_1.msgf_V.T")
        print_matrices_side_by_side(X_k.W_tilde_prime, X_k.msgf_V_X_k_prime, "X_k.W_tilde_prime", "X_k.msgf_V_X_k_prime")
        """
        if (self.DEBUG):
            print("------------")
            print(f"R_k.msgb_W_norm_coeff: {self.msgb_W_norm_coeff:.2f}")
            # print_vector(X_k_minus_1.m, "   X_k_minus_1.m")
            print(f"   X_k_minus_1.m.T @ X_k_minus_1.m: {convert_1x1_matrix_to_scalar(X_k_minus_1.m.T @ X_k_minus_1.m):.2f}")
            print(f"   tr(X_k_minus_1.V): {np.trace(X_k_minus_1.V):.2f} ")
            print_matrices_side_by_side(X_k_minus_1.msgf_V, X_k_minus_1.W_tilde, "   X_k_minus_1.msgf_V", "X_k_minus_1.W_tilde")
            print_matrix(X_k_minus_1.V, "   X_k_minus_1.V")
            print("------------")
            # print_matrix(X_k_minus_1.V, "    X_k_minus_1.V")
            # print_matrix(X_k_minus_1.m @ X_k_minus_1.m.T, "   m@m.T")
            msgb_m = 1/self.msgb_W_norm_coeff * self.msgb_xi_norm
            print_vectors_side_by_side(self.msgb_xi_norm, msgb_m,  "msgb_xi_norm", "msgb_m")
            print(f"   (X_k_minus_1.m.T @ X_k.m, X_k_minus_1.m.T @ P.T @ X_k.m): ({convert_1x1_matrix_to_scalar(X_k_minus_1.m.T @ X_k.m):.2e},{convert_1x1_matrix_to_scalar( X_k_minus_1.m.T @ P.T @ X_k.m):.2e})")
            # print(f"   X_k_minus_1.m.T @ P.T @ X_k.m: {convert_1x1_matrix_to_scalar( X_k_minus_1.m.T @ P.T @ X_k.m):.2e}")
            print(f"   (trace1, trace2) = ({np.trace(V_X_k_minus_1_X_k_prime_T):.2e}, {np.trace(V_X_k_minus_1_X_k_prime_T @ P):.2e})")
            print_matrices_side_by_side(V_X_k_minus_1_X_k_prime_T, V_X_k_minus_1_X_k_prime_T @ P, "V_X_k_minus_1_X_k_prime_T", "V_X_k_minus_1_X_k_prime_T @ P")
            # print(f"Exp_X_k_minus_1_T_X_k_prime: {Exp_X_k_minus_1_T_X_k_prime[0, 0]:.2f}")
            # print(f"Exp_X_k_minus_1_T_P_T_X_k_prime: {Exp_X_k_minus_1_T_P_T_X_k_prime[0,0]:.2f}")
            # print_vectors_side_by_side(X_k.m, X_k_minus_1.m, "X_k.m", "X_k_minus_1.m", 4)
            # print_matrix(X_k.m @ X_k_minus_1.m.T, "    X_k.m @ X_k_minus_1.m.T")
            # print(f"    (Exp_X_k_minus_1_T_X_k_prime, Exp_X_k_minus_1_T_P_T_X_k_prime) = ({Exp_X_k_minus_1_T_X_k_prime:.2e},  {Exp_X_k_minus_1_T_P_T_X_k_prime:.2e})")
            # print_matrix(X_k_minus_1.msgf_V, "    X_k_minus_1.msgf_V")
            # print_matrix(A_r, "    A_r")
            # print_matrix(X_k.W_tilde_prime, "    X_k.W_tilde_prime")
            # print_matrix(msgb_V_X_k_prime, "    msgb_V_X_k_prime")
            # print_matrices_side_by_side( V_X_k_prime_X_k_minus_1_T + X_k.m @ X_k_minus_1.m.T, P.T @ (V_X_k_prime_X_k_minus_1_T + X_k.m @ X_k_minus_1.m.T), "trace 1 arg", "trace 2 arg")


        return self

def expectation_maximization(sigma2_Z, N, y_obs, max_out_iter, max_in_iter, R_true, X_true):
    msg_V_init,msg_W_init,V_U_coeff = setup_params(sigma2_Z)
    ## Params
    ############
    # infty = sigma2_Z*1e6
    # eps   = sigma2_Z*1e-3
    # V_U_coeff = sigma2_Z*1e2
    # msg_V_init = infty*np.eye(2)
    # msg_W_init = eps*np.eye(2)
    ############
    # infty = sigma2_Z*1e9 # when sigma2_Z = 5e-12, eps = 5e-6
    # eps   = sigma2_Z*1e-2
    # eps   = sigma2_Z*1e-3
    # # msg_W_init = np.zeros((2,2))
    # V_U_coeff = sigma2_Z*1e3 # Need atleast +2 to have non-decreasing LL (for 1.29e-2)
    # V_U_coeff = sigma2_Z*1e2 # Need atleast +2 to have non-decreasing LL (for 1.29e-2)
    # # V_U_coeff = sigma2_Z*1e-6 # For 7.74e0-2, this gives better result but doesn't have good LL
    ####


    X_est_vis = []

    X_edges = [X_k_edge_MBF(sigma2_Z,msg_V_init, msg_W_init) for k in range(N + 1)]
    R_edges = [R_k_edge_EM(msg_V_init, msg_W_init) for k in range(N+1)]

    DEBUG = False 
    R_est = R_init
    LL_series = []
    theta_series = []
    r_norm_series = []

    for out_iter in range (max_out_iter):
        print(f"Outer iteration {out_iter + 1}/{max_out_iter}")
        for in_iter in range (max_in_iter):
            print(f"Iteration {in_iter + 1}/{max_in_iter}")
            # print_vector(R_est, "Current R_est")
            V_U = V_U_coeff * np.eye(2)

            ## Step 1: Soft-estimate X while keeping R fixed
            # (Forward-backward message passing)
            ##################
            X_est = estimate_X(R_est, X_edges, y_obs, V_U, N, sigma2_Z, msg_V_init, msg_W_init, DEBUG)
            for k in range(0, N+1):
                m_est = X_est[k]
                if (DEBUG):
                    print_vectors_side_by_side_float(m_est, X_true[k], f"X_{k}.m", f"True X_{k}")

            # Visualize X expectation
            X_vis = [{"m": x.marginal(), "V": x.V} for x in X_edges[1:]]
            X_est_vis.append({"step": "1_X_expectation", "out_iter": out_iter, "in_iter": in_iter, "X_vis": X_vis, "R_est": R_est.copy()})
            ##################

            # Calculate LL
            ##################
            LL = 0
            for k in range(1, N+1):
                y_k = y_obs[k]
                LL_k = X_edges[k].em_log_likelihood(y_k)
                LL = LL + LL_k
            # print(f"Log-likelihood: {LL:.6e}")
            LL_series = LL_series + [LL]
            ##################

            ## Step 2. Estimate R
            ##################
            # Initialize R edges with correct mean.
            # This is required because we need the correct rotation matrix to compute V_X_{k-1}_V_X_k'T
            for k in range(0, N):
                R_edges[k].__init__(msg_V_init, msg_W_init)
                R_edges[k].update_marginal(R_est)
            ##################

            # Calculate upwards message from all branches.
            ##################
            # The setup is X_{k} R_{k+1} X_{k+1}. We start k from 0.
            # so there can only be (N) upwards estimation (R_1 to R_N)
            for k in range(N, 0, -1):
                # Backward R[N], ..., R[1]
                # print("===================================")
                # print("")
                # print(f"Backward pass on R_{k}")
                R_edges[k].backward(X_edges[k-1], X_edges[k], V_U_coeff)
            ##################

            # Estimate R
            R_est = collect_R_est(R_edges, N)

            # Evaluation
            sqe = squared_error(R_est, R_true)
            theta_hat = vector_angle(R_est)
            theta_true = vector_angle(R_true)
            theta_series = theta_series + [theta_hat]
            r_norm_series = r_norm_series + [np.linalg.norm(R_est)]
            # print_vectors_side_by_side(R_est, R_true, f"sqe: {sqe:.2e} >R_est", "R_true")
            # print(f"theta_hat: {theta_hat:.3e}; theta_true: {theta_true:.3e}")

            # Visualize R estimation
            X_est_vis.append({"step": "2_R_maximization", "out_iter": out_iter, "in_iter": in_iter, "X_vis": X_vis, "R_est": R_est.copy()})
            ##################


        V_U_coeff = V_U_coeff * 0.8
        print_vectors_side_by_side(R_est, R_true, f"sqe: {sqe:.2e} >R_est", "R_true")

    return R_est, LL_series, X_est_vis, theta_series, r_norm_series

def plot_x_and_r_em(X_vis_entry, X_true, R_est, R_true, step, save_path=None):
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
        # var = 50*np.sqrt(np.linalg.det(x['V']))  # EM working: 50
        var = np.sqrt(np.linalg.det(x['V']))  # EM failing: 1
        # plt.scatter(mean[0], mean[1], color='red', label="Estimated X")
        handles, labels = plt.gca().get_legend_handles_labels()
        if label_for_x_mean not in labels:
            plt.scatter(mean[0], mean[1], color='red', label=label_for_x_mean)
        else:
            plt.scatter(mean[0], mean[1], color='red')
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
    if (step =="1_X_expectation"):
        plt.title("Step 1: Expectation")
    else:
        plt.title("Step 2: Maximization: estimate $R$")
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)


    # Save or display the plot
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

    return

