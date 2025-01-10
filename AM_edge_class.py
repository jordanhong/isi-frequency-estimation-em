import numpy as np
# import cupy as np
from params import *
from util import *
import matplotlib.pyplot as plt

class X_k_edge_MBF:
    def __init__(self, sigma2_Z, msg_V_init, msg_W_init):
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
        # self.W_tilde_prime = np.zeros((2, 2))
        # self.W_tilde = np.zeros((2, 2))
        self.W_tilde_prime = msg_W_init
        self.W_tilde = msg_W_init

        # V
        self.V = msg_V_init
        # m
        self.m = np.array([[0], [0]])

        # A
        self.update_A_hat_r(R_init)

        self.sigma2_Z = sigma2_Z

        self.DEBUG = True
        # self.DEBUG = False

    def update_A_hat_r (self, r):
        self.A_hat_r = convert_vect_to_rotation_matrix(r)
        return

    def forward(self, X_k_minus_1, y_k, V_U):
        self.msgf_m_X_k_prime = self.A_hat_r @ X_k_minus_1.msgf_m                            # Eq. 3.209
        self.msgf_V_X_k_prime = symmetrize(self.A_hat_r @ X_k_minus_1.msgf_V @ self.A_hat_r.T + V_U)     # Eq. 3.210
        # self.msgf_V_X_k_prime = self.A_hat_r @ X_k_minus_1.msgf_V @ self.A_hat_r.T + V_U     # Eq. 3.210
        G = np.linalg.inv(self.sigma2_Z + C @ self.msgf_V_X_k_prime @ C.T)
        self.G = G
        self.G_dd = C.T @ G @ C
        self.g_dd = C.T @ G @ (y_k - C @ self.msgf_m_X_k_prime)

        self.msgf_m = self.msgf_m_X_k_prime + self.msgf_V_X_k_prime @ self.g_dd
        self.msgf_V = symmetrize(self.msgf_V_X_k_prime - self.msgf_V_X_k_prime @ self.G_dd @ self.msgf_V_X_k_prime)
        # self.msgf_V = self.msgf_V_X_k_prime - self.msgf_V_X_k_prime @ self.G_dd @ self.msgf_V_X_k_prime

        assert(np.linalg.det(self.msgf_V_X_k_prime) >0)
        assert(np.linalg.det(self.msgf_V) >= 0), f"Negative deteriminant {np.linalg.det(self.msgf_V):.2e}"
        # print(f"G: {G[0,0]}")
        # print_vectors_side_by_side(self.msgf_m_X_k_prime, self.msgf_m, "m_X_k_prime", "msgf_m")
        # print_matrices_side_by_side(self.msgf_V_X_k_prime, self.msgf_V, "msgf_V_X_k_prime", "msgf_V")
        # print_matrix(X_k_minus_1.msgf_V, "X_k-1.msgf_V")
        # print_matrix(self.A_hat_r, "Ar")
        # assert_symmetric(self.A_hat_r @ X_k_minus_1.msgf_V @ self.A_hat_r.T, "first term")
        """
        assert_symmetric(X_k_minus_1.msgf_V, "X_k-1.msgf_V")
        assert_symmetric(self.msgf_V_X_k_prime, "self.msgf_V_X_k_prime")
        assert_symmetric(self.msgf_V, "self.msgf_V")
        print_matrix(self.A_hat_r, "self.A_hat_r")
        print_matrices_side_by_side(X_k_minus_1.msgf_V, self.msgf_V_X_k_prime, "X_k_minus_1.msgf_V", "self.msgf_V_X_k_prime")
        # print_matrix(self.msgf_V_X_k_prime @ self.G_dd @ self.msgf_V_X_k_prime, "Kalman V drop")
        print_matrix(self.msgf_V_X_k_prime @ self.G_dd @ self.msgf_V_X_k_prime, "second term")
        print_matrix(self.msgf_V_X_k_prime, "self.msgf_V_X_k_prime")
        print_matrix(self.msgf_V, "self.msgf_V")
        """
        # print_matrices_side_by_side(self.msgf_V_X_k_prime, self.msgf_V,"self.msgf_V_X_k_prime", "self.msgf_V")
        print_matrix(self.msgf_V, "msgf_V")
        print_vector(self.msgf_m, "msgf_m")
        # print(f"det(self.msgf_V_X_k_prime): {np.linalg.det(self.msgf_V_X_k_prime)}")
        # if (np.linalg.det(self.msgf_V_X_k_prime) <0):
        #     print("ERROR: negative determinant")
        # print(f"det(self.msgf_V): {np.linalg.det(self.msgf_V)}")
        # if (np.linalg.det(self.msgf_V) <=0):
        #     print("ERROR: negative determinant")

            
        # print_matrices_side_by_side(self.A_hat_r @ X_k_minus_1.msgf_V @ self.A_hat_r.T, "first term")
        # print_ma
        # print(f"self.G_dd(0,0): {self.G_dd[0,0]:.2e}")

        return

    def backward(self, X_k_plus_1):
        # Based on MESA 3.216 - 3.220. With self = X_{k-1} in script
        F_k_plus_1 = np.eye(2) - X_k_plus_1.msgf_V_X_k_prime @ X_k_plus_1.G_dd
        xi_tilde_X_k_plus_1_prime = F_k_plus_1.T @ X_k_plus_1.xi_tilde  - X_k_plus_1.g_dd
        self.xi_tilde  = self.A_hat_r.T @ xi_tilde_X_k_plus_1_prime

        # Optional (for variance estimation) Eq. 3.219-3.220 in MESA script
        X_k_plus_1.W_tilde_prime = symmetrize(F_k_plus_1.T @ X_k_plus_1.W_tilde @ F_k_plus_1 + X_k_plus_1.G_dd)
        # X_k_plus_1.W_tilde_prime = F_k_plus_1.T @ X_k_plus_1.W_tilde @ F_k_plus_1 + X_k_plus_1.G_dd
        self.W_tilde = symmetrize(self.A_hat_r.T @ X_k_plus_1.W_tilde_prime @ self.A_hat_r)
        # self.W_tilde = self.A_hat_r.T @ X_k_plus_1.W_tilde_prime @ self.A_hat_r

        """
        print_matrix(X_k_plus_1.G, "X_k_plus_1.G")
        print_matrices_side_by_side(X_k_plus_1.W_tilde_prime, self.W_tilde, "X_k_plus_1.W_tilde_prime", "     W_X_k_tilde")
        print_matrix(X_k_plus_1.W_tilde, "X_k_plus_1.W_tilde")
        print_matrix(F_k_plus_1, "F_k_plus_1")
        print_vector(X_k_plus_1.xi_tilde, "X_k_plus_1.xi_tilde")
        print_vector(xi_tilde_X_k_plus_1_prime, "xi_tilde_X_k_plus_1_prime")
        print_vector(self.xi_tilde, "X_k.xi_tilde")
        """
        # assert(np.linalg.det(X_k_plus_1.W_tilde_prime) >0)
        # assert(np.linalg.det(self.W_tilde) > 0), f"Negative determinant {np.linalg.det(self.W_tilde):.2e}"
        # print_matrix(X_k_plus_1.W_tilde_prime, f"X_k_plus_1.W_tilde_prime (rank {np.linalg.matrix_rank(X_k_plus_1.W_tilde_prime)}, det = {np.linalg.det(X_k_plus_1.W_tilde_prime):.2e})")
        # assert(np.linalg.matrix_rank(X_k_plus_1.W_tilde_prime) == 2), f"X_k_plus_1.W_tilde_prime (rank {np.linalg.matrix_rank(X_k_plus_1.W_tilde_prime)})"
        assert(np.linalg.det(X_k_plus_1.W_tilde_prime) != 0), f"X_k_plus_1.W_tilde_prime (rank {np.linalg.matrix_rank(X_k_plus_1.W_tilde_prime)})"
        # assert(np.linalg.matrix_rank(self.W_tilde) == 2), f"self.W_tilde (rank {np.linalg.matrix_rank(self.W_tilde)})"
        assert(np.linalg.det(self.W_tilde) != 0), f"self.W_tilde (rank {np.linalg.matrix_rank(self.W_tilde)})"
    
        # if self.DEBUG:
            # print_matrix(X_k_plus_1.W_tilde_prime, f"X_k_plus_1.W_tilde_prime (rank {np.linalg.matrix_rank(X_k_plus_1.W_tilde_prime)})")
            # print_matrix(X_k_plus_1.G_dd, "X_k_plus_1.G_dd")
            # print_vector(X_k_plus_1.g_dd, "X_k_plus_1.g_dd")
            # print_vector(xi_tilde_X_k_plus_1_prime, "xi_tilde_X_k_plus_1_prime")

        return

    def marginal(self):
        self.m = self.msgf_m - self.msgf_V @ self.xi_tilde  # Eq. 3.223
        # self.V = self.msgf_V - self.msgf_V @ self.A_hat_r.T @ self.W_tilde @ self.msgf_V # Eq. 3227
        self.V = self.msgf_V - self.msgf_V @ self.W_tilde @ self.msgf_V # Eq. 3227
        # print_matrix(self.V, "X_k.V")
        # print_matrix(self.W_tilde, "X_k.W_tilde")
        if self.DEBUG:
            # print_vectors_side_by_side(self.msgf_m, self.xi_tilde, "X_k.msgf_m", "X_k.xi_tilde")
            # print_vector(self.msgf_V @ self.xi_tilde, "sub vect")
            # print_matrices_side_by_side( self.msgf_V, self.W_tilde, "X_k.msgf_V", "self.W_tilde")
            # print_matrix( self.msgf_V @ self.W_tilde @ self.msgf_V, "self.msgf_V @ self.W_tilde @ self.msgf_V")
            print_vector(self.m, "self.m")
            print_matrix( self.V, "self.V")
        
        ## Assertion
        # assert np.all(np.diagonal(self.V) >= 0), f"Matrix diagonal contains negative values: {np.diagonal(self.V)}"
        return self.m

    def get_estimate(self):
        return self.m

    def em_log_likelihood(self, y_k):
        LL_k = - (y_k - C @ self.msgf_m)**2 / (self.sigma2_Z + C@ self.msgf_V@ C.T) -np.log(2*np.pi * (self.sigma2_Z + C@ self.msgf_V@ C.T))
        return LL_k[0,0]

class R_k_edge:
    def __init__(self, msg_V_init, msg_W_init):
        # Upwards message
        self.msgb_W =msg_W_init.copy()
        self.msgb_xi = np.array([[0], [0]])

        self.msgb_W_norm = 0
        self.msgb_xi_norm = np.array([[0], [0]])

        # Constants
        self.A_hat_x_k_minus_1 = np.eye(2)
        self.x_k = X_init # Fixed value for X_k (since X_k = \hat{x}_k)
        self.x_k_minus_1 = X_init # Fixed value for X_k (since X_k = \hat{x}_k)
        self.DEBUG = True

    def update_x_hat_and_A(self, x_k, x_k_minus_1):
        self.x_k = x_k
        self.x_k_minus_1 = x_k_minus_1
        self.A_hat_x_k_minus_1 = convert_vect_to_rotation_matrix(x_k_minus_1)
        return

    def backward(self, R_k_plus_1: 'R_k_edge', V_U):
        msgb_W_X0_k = np.linalg.inv(V_U)

        # print(f"SHAPE BEFORE: {self.msgb_W.shape}, {self.msgb_xi.shape}, {self.msgb_W_prime.shape}, {self.msgb_xi_prime.shape}")
        # Upwards message
        # print(f"{self.A_hat_x_k_minus_1.T.shape}, {msgb_W_X0_k.shape}, {self.A_hat_x_k_minus_1.shape}")
        self.msgb_W = self.A_hat_x_k_minus_1.T @ msgb_W_X0_k @ self.A_hat_x_k_minus_1
        self.msgb_xi = self.A_hat_x_k_minus_1.T @ msgb_W_X0_k @self.x_k

        self.msgb_W_norm_coeff = convert_1x1_matrix_to_scalar(self.x_k_minus_1.T @ self.x_k_minus_1)
        self.msgb_xi_norm = self.A_hat_x_k_minus_1.T @ self.x_k
        # left message (prime)
        # self.msgb_W_prime = R_k_plus_1.msgb_W_prime + R_k_plus_1.msgb_W
        # self.msgb_xi_prime = R_k_plus_1.msgb_xi_prime + R_k_plus_1.msgb_xi

        # print(f"SHAPE AFTER: {self.msgb_W.shape}, {self.msgb_xi.shape}, {self.msgb_W_prime.shape}, {self.msgb_xi_prime.shape}")
        if (self.DEBUG):
            print("------------")
            print(f"R_k.msgb_W_norm_coeff: {self.msgb_W_norm_coeff:.2f}")
            # print(f" X_k_minus_1.m: ({self.A_hat_x_k_minus_1.T[0,0]:.2e}, {self.A_hat_x_k_minus_1.T[1,0]:.2e})")
            print("------------")
            # print(f"   X_k_minus_1.m.T @ X_k.m: {convert_1x1_matrix_to_scalar(X_k_minus_1.m.T @ X_k.m):.2e}")
            # print(f"   X_k_minus_1.m.T @ P.T @ X_k.m: {convert_1x1_matrix_to_scalar( X_k_minus_1.m.T @ P.T @ X_k.m):.2e}")
            msgb_m = 1/self.msgb_W_norm_coeff * self.msgb_xi_norm
            print_vectors_side_by_side(self.msgb_xi_norm, msgb_m,  "msgb_xi_norm", "msgb_m")
            # print(f"   X_k_minus_1.m.T @ X_k.m: {self.msgb_xi_norm[0,0]:.2e}")
            # print(f"   X_k_minus_1.m.T @ P.T @ X_k.m: {self.msgb_xi_norm[1,0]:.2e}")
            print(f"   (X_k_minus_1.m.T @ X_k.m, X_k_minus_1.m.T @ P.T @ X_k.m): ({self.msgb_xi_norm[0,0]:.2e},{self.msgb_xi_norm[1,0]:.2e})")

        return self

    # def forward(self, R_k_minus_1: 'R_k_edge'):
    #     # print(f"My SHAPE: {R_k_minus_1.msgf_W_prime.shape}, {self.msgb_W.shape}")
    #     self.msgf_W_prime = R_k_minus_1.msgf_W_prime + self.msgb_W
    #     self.msgf_xi_prime = R_k_minus_1.msgf_xi_prime + self.msgb_xi

    #     # print(f"My SHAPE: {self.msgf_W_prime.shape}, {self.msgf_xi_prime.shape}")
    #     return self
    # def marginal(self):
    #     # print(f"SHAPE: {self.msgf_W_prime.shape}, {self.msgb_W_prime.shape}")
    #     V_R_prime_k = np.linalg.inv(self.msgf_W_prime + self.msgb_W_prime)
    #     m_R_prime_k = V_R_prime_k @ (self.msgf_xi_prime + self.msgb_xi_prime)
    #     # print(f"SHAPE: {V_R_prime_k.shape}, {self.msgf_xi_prime.shape}, {self.msgb_xi_prime.shape}")

    #     self.m = m_R_prime_k
    #     self.V = V_R_prime_k

    #     return m_R_prime_k, V_R_prime_k

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
        X_edges[k].__init__(sigma2_Z, msg_V_init, msg_W_init)

    for k in range(0, N+1): # k = 0, ..., N
        X_edges[k].update_A_hat_r(R_est)

    for k in range(1, N+1): # k = 1, ..., N
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

def estimate_R(X_est, R_edges, V_U, N):
    for k in range(1, N+1):
        R_edges[k].update_x_hat_and_A(X_est[k], X_est[k-1])

    ### Part 2: fix X, estimate R
    # print(f"Step 2: Assume fixed X, estimate R.")
    # print(len(R_edges))
    for k in range(N, 0, -1):
        print(f"Backward pass on R_{k}")
        # Backward R[N], ..., R[1]
        R_edges[k].backward(R_edges[k], V_U)

    ## Estimate R based on backward message on R_0
    # R_est_new = np.linalg.inv(R_edges[0].msgb_W_prime) @ R_edges[0].msgb_xi_prime
    # Estimate R 
    ##################
    R_est_new = collect_R_est(R_edges, N)
    ##################
    return R_est_new

def setup_params(sigma2_Z):
    infty = sigma2_Z*1e8 # when sigma2_Z = 5e-12, eps = 5e-6
    eps   = min(sigma2_Z*1e6, 1e-6)
    msg_V_init = infty*np.eye(2)
    msg_W_init = eps*np.eye(2)
    V_U_coeff = sigma2_Z*1e2

    return msg_V_init,msg_W_init,V_U_coeff

def alternate_maximization(sigma2_Z, N, y_obs, tol, max_pass, R_true, X_true):
    # prev_R_est = None  # To store the previous estimate of R
    ## Params
    msg_V_init,msg_W_init,V_U_coeff = setup_params(sigma2_Z)
    ####

    X_edges = [X_k_edge_MBF(sigma2_Z, msg_V_init, msg_W_init) for k in range(N+1)]
    R_edges = [R_k_edge(msg_V_init, msg_W_init) for k in range(N+1)]

    DEBUG = True
    R_est = R_init
    max_iter = 3
    theta_series = []
    r_norm_series = []
    for iter_idx in range (max_iter):
        print(f"START OF V_U iteration {iter_idx}, max_iter={max_iter}")
        for p in range (max_pass):
            print(f"=============START OF PASS {p} sigma2_Z={sigma2_Z:.2e}, N={N}, max_pass={max_pass}  ==============")
            print_vector(R_est, "Current R_est")
            V_U = V_U_coeff * np.eye(2)

            X_est = estimate_X(R_est, X_edges, y_obs, V_U, N, sigma2_Z, msg_V_init, msg_W_init, DEBUG)
            ## print out states
            for k in range(0, N+1):
                m_est = X_est[k]
                print_vectors_side_by_side_float(m_est, X_true[k], f"X_{k}.m", f"True X_{k}")

            for k in range(0, N+1):
                R_edges[k].__init__(msg_V_init, msg_W_init)

            R_est_new = estimate_R(X_est, R_edges, V_U, N)
            theta_hat = vector_angle(R_est_new)
            theta_series = theta_series + [theta_hat]
            r_norm_series = r_norm_series + [np.linalg.norm(R_est_new)]
            print_vectors_side_by_side(R_est_new, R_true, ">R_est", "R_true")
            sqe = squared_error(R_est_new, R_true)
            print(f"sqe: {sqe:.2e}")

            prev_R_est = R_est_new
            R_est = R_est_new


        V_U_coeff = (V_U_coeff * 0.5)
        # print()

    return R_est, X_est, theta_series, r_norm_series

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
        var = 0.2* np.sqrt(np.linalg.det(x['V']))  # control: 0.2
        # var = np.sqrt(np.linalg.det(x['V']))  # working: 1
        print(mean)
        print(f"{var:.2e}")
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

def alternate_maximization_log(sigma2_Z, N, y_obs, tol, max_pass, R_true, X_true):
    """
    Perform alternate maximization and return intermediate estimations, emphasizing each half algorithm.

    Parameters:
        sigma2_Z (float): Noise variance.
        N (int): Number of iterations.
        y_obs (list): Observed data.
        tol (float): Tolerance for convergence.
        max_pass (int): Maximum number of passes per iteration.
        R_true (np.ndarray): Ground truth rotation vector.
        X_true (list): Ground truth X values.

    Returns:
        R_est (np.ndarray): Final estimated rotation vector.
        X_est (list): Final estimates for X.
        X_est_vis (list): List of intermediate visualization data for X and R at each iteration.
    """
    infty = sigma2_Z * 1e9
    eps = sigma2_Z * 1e-3
    msg_V_init = infty * np.eye(2)
    msg_W_init = eps * np.eye(2)
    V_U_coeff = sigma2_Z * 1e3

    R_est = R_init
    X_edges = [X_k_edge_MBF(sigma2_Z, msg_V_init, msg_W_init) for _ in range(N + 1)]
    R_edges = [R_k_edge(msg_V_init, msg_W_init) for _ in range(N + 1)]

    X_est_vis = []

    # Add initial state before the loop
    # X_vis = [{"m": X_init , "V": msg_V_init} for x in X_edges[1:]]
    # X_est_vis.append({"step": "initial", "v_u_iter": 0, "iteration": 0, "X_vis": X_vis, "R_est": R_est.copy()})
    

    DEBUG = True 
    max_iter = 2
    for iter_idx in range(max_iter):
        print(f"V_U Iteration {iter_idx+ 1}/{max_iter}")
        # print("Hello world")
        for iteration in range(max_pass):
            print(f"Iteration {iteration + 1}/{max_pass}")

            # Update V_U dynamically
            V_U = V_U_coeff * np.eye(2)

            # Step 1: Estimate X while keeping R fixed
            X_est = estimate_X(R_est, X_edges, y_obs, V_U, N, sigma2_Z, msg_V_init, msg_W_init, DEBUG)

            # Visualize X estimation
            X_vis = [{"m": x.marginal(), "V": x.V} for x in X_edges[1:]]
            X_est_vis.append({"step": "1_X_estimation", "v_u_iter": iter_idx, "iteration": iteration, "X_vis": X_vis, "R_est": R_est.copy()})

            # Step 2: Estimate R while keeping X fixed
            R_est = estimate_R(X_est, R_edges, V_U, N)
            print_vector(R_est, f"R_est at iteration: {iteration + 1}/{max_pass}")

            # Visualize R estimation
            X_est_vis.append({"step": "2_R_estimation", "v_u_iter": iter_idx, "iteration": iteration, "X_vis": X_vis, "R_est": R_est.copy()})

        V_U_coeff = (V_U_coeff * 0.5)

    return R_est, X_est, X_est_vis
