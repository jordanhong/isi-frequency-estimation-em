import pytest
import numpy as np
from AM_edge_class import X_k_edge_MBF, generate_ground_truth, generate_noisy_obs, estimate_X, estimate_R,setup_params
from AM_edge_class import R_k_edge
from params import *
from util import *

# tolerance
TOL = 1e-3

@pytest.fixture
def setup_data(theta):
    """Setup the initial data for testing with varying theta values"""
    sigma2_Z = 5e-8    # Observation noise variance
    N = 20              # Number of time steps
    X_true_0 = np.array([[np.cos(np.pi/3)], [np.sin(np.pi/3)]])  # Starting position
    # X_true_0 = np.array([[1], [0]])  # Starting position
    R_true = np.array([[np.cos(theta)], [np.sin(theta)]])

    # Generate ground truth and noisy observations
    X_true = generate_ground_truth(R_true, X_true_0, N)
    y = generate_noisy_obs(X_true, C, sigma2_Z, N)

    DEBUG = False

    # Initialize edges
    msg_V_init,msg_W_init,V_U_coeff = setup_params(sigma2_Z)
    X_edges = [X_k_edge_MBF(sigma2_Z,msg_V_init,msg_W_init, DEBUG) for k in range(N + 1)]
    R_edges = [R_k_edge(msg_V_init,msg_W_init, DEBUG) for k in range(N+1)]
    V_U = V_U_coeff * np.eye(2)

    return {
        "X_true": X_true,
        "y": y,
        "X_edges": X_edges,
        "R_edges": R_edges,
        "V_U": V_U,
        "N": N,
        "sigma2_Z": sigma2_Z,
        "msg_V_init": msg_V_init,
        "msg_W_init": msg_W_init
    }

@pytest.mark.parametrize("theta", [np.pi/4, np.pi/2, np.pi/8])
# @pytest.mark.parametrize("theta", [np.pi/4])
def test_estimate_X(theta, setup_data):
    """Test estimate_X for varying theta values"""
    data = setup_data

    R_true = np.array([[np.cos(theta)], [np.sin(theta)]])

    DEBUG = False
    # Estimate X based on fixed R and observations
    X_est = estimate_X(R_true, data["X_edges"], data["y"], data["V_U"], data["N"], data["sigma2_Z"], data["msg_V_init"], data["msg_W_init"], DEBUG)
    ## print out states
    print(f"Testing with theta: {theta*180/np.pi} deg")
    for k in range(1, data["N"]+1):
        m_est = X_est[k]
        print_vectors_side_by_side(m_est, data['X_true'][k], f"X_{k}", f"X_{k}_true")
        # print(f"==> X_{k}: m = ({m_est[0,0]:.2f}, {m_est[1,0]:.2f}), Truth = ({data['X_true'][k][0][0]:.2f}, {data['X_true'][k][1][0]:.2f})")

    # Ensure the estimated X has the correct length
    assert len(X_est) == data["N"]+1, "Incorrect number of estimated X states"

    # Check that X_est values are within an acceptable tolerance compared to the ground truth
    for k in range(0, data["N"] + 1):
        assert np.allclose(X_est[k], data["X_true"][k], atol=TOL), f"X_est[{k}] does not match X_true[{k}] within tolerance"


@pytest.mark.parametrize("theta", [np.pi/4, np.pi/2, np.pi/8])
def test_estimate_R(theta, setup_data):
    """Test estimate_R for varying theta values"""
    data = setup_data

    DEBUG = False
    # Estimate X based on fixed R and observations
    R_est = estimate_R(data["X_true"], data["R_edges"], data["V_U"], data["N"], data["msg_V_init"], data["msg_W_init"], DEBUG)
    R_true = np.array([[np.cos(theta)], [np.sin(theta)]])

    ## print out states
    print(f"Testing with theta: {theta*180/np.pi} deg")
    print(f"==> R estimate= ({R_est[0][0]:.2e}, {R_est[1][0]:.2e}), R_true= ({R_true[0][0]:.2e}, {R_true[1][0]:.2e})")


    assert np.allclose(R_est, R_true, atol=TOL), f"R_est does not match R_true within tolerance"

# ## TODO: additional tests to add
# #   1. State noise set to 0, all states should be static
# #   2. For each of the half algorithm,
# #       - initialize at the posterior mean
# #       - Gaussian forward/backward message passing should not deviate
