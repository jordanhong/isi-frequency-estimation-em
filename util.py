import numpy as np
# import cupy as np
from params import *
import matplotlib.pyplot as plt

## Parameter setup
def setup_params(sigma2_Z):
    ### Config 1 (passes test_alternate_max)
    # infty = sigma2_Z*1e8
    # eps   = sigma2_Z*1e3
    # V_U_coeff = sigma2_Z*1e3

    ### Config 2 (doesn't pass, but EM code works without failing assertion)
    # infty = sigma2_Z*1e6
    # eps   = sigma2_Z*1e-2
    # V_U_coeff = sigma2_Z*1e2

    ### Config 3 (passes)
    # infty = sigma2_Z*1e7
    # eps   = sigma2_Z*1e1
    # V_U_coeff = sigma2_Z*1e2

    infty = min(sigma2_Z*1e7, 1e6)
    # eps   = min(sigma2_Z*1e2, 1e-6)
    eps   = min(sigma2_Z*1e-6, 1e-6)
    V_U_coeff = sigma2_Z*1e3
    # V_U_coeff = sigma2_Z*1e4

    msg_V_init = infty*np.eye(2)
    msg_W_init = eps*np.eye(2)

    return msg_V_init,msg_W_init,V_U_coeff
## Data generation and curation
def generate_ground_truth(R_true, X_true_0, N):
    X_true = [X_true_0] + [X_init for _ in range(N)]
    A_hat_r = convert_vect_to_rotation_matrix(R_true)
    for k in range(1, N + 1):
        X_true[k] = A_hat_r @ X_true[k-1]
    return X_true
def generate_noisy_obs(X_true, C, sigma2_Z, N):
    # y = [0]
    y = []
    for k in range (0, N+1):
        true_projection = C @ X_true[k]  # Project X_k onto [1, 0]
        noise = np.random.normal(0, np.sqrt(sigma2_Z))  # Add Gaussian noise
        y.append(true_projection + noise)
    return y

## Visualization
def plot_x_and_r_true(X_true, R_true, save_path=None):
    plt.figure(figsize=(8, 8))

    # Plot the unit circle
    circle = plt.Circle((0, 0), 1, color='grey', fill=False, linestyle='dotted', linewidth=1.5)
    plt.gca().add_artist(circle)

    # Plot the true rotation vector
    plt.quiver(0, 0, R_true[0, 0], R_true[1, 0], angles='xy', scale_units='xy', scale=1, color='purple', label="Rotator Vector R")

    X_true_np = np.array(X_true)  # Assuming X_true is a list of dictionaries with 'm'
    plt.scatter(X_true_np[:, 0], X_true_np[:, 1], color='black', alpha=0.7, label="Position $X_k$")

    # Configure the plot
    plt.axhline(0, color='k', linestyle='--', linewidth=0.5)
    plt.axvline(0, color='k', linestyle='--', linewidth=0.5)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend(loc='upper left')
    plt.axis('equal')
    plt.grid()
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)

    # Save or display the plot
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

    return

## Miscellany
def convert_vect_to_rotation_matrix(vect):
    r1, r2 = vect[0, 0], vect[1, 0]
    A_hat_r = np.array([[r1, -r2], [r2, r1]])
    return A_hat_r
def squared_error (R, R_true):
    assert isinstance(R, np.ndarray), "R must be a NumPy array"
    assert isinstance(R_true, np.ndarray), "R_true must be a NumPy array"
    assert R.shape == (2, 1), f"R must have shape (2, 2), but see shape {R.shape}"
    assert R_true.shape == (2, 1), f"R_true must have shape (2, 2)"
    r1, r2 = R[0,0], R[1,0]
    r_true1, r_true2 = R_true[0,0], R_true[1,0]
    return (r1 - r_true1)**2 + (r2 - r_true2)**2
def print_vector(vector, label):
    print(f"{label}: ({vector[0, 0]:.2e}, {vector[1, 0]:.2e})")
    return
# def print_vectors_side_by_side(vector1, vector2, label1="Vector 1", label2="Vector 2"):
#     print(f"{label1}:({vector1[0, 0]:.2e}, {vector1[1, 0]:.2e});  {label2}:({vector2[0, 0]:.2e}, {vector2[1, 0]:.2e})")
#     return
def print_vectors_side_by_side(vector1, vector2, label1="Vector 1", label2="Vector 2", precision=2):
    """
    Prints two vectors side by side with specified labels and precision in scientific notation.

    Args:
    vector1 (np.ndarray): First vector to display.
    vector2 (np.ndarray): Second vector to display.
    label1 (str): Label for the first vector. Default is "Vector 1".
    label2 (str): Label for the second vector. Default is "Vector 2".
    precision (int): Number of decimal places for formatting. Default is 2.
    """
    # Create a format string based on the precision parameter
    format_str = f":({{:.{precision}e}}, {{:.{precision}e}});"

    # Print the vectors using the format string
    print(f"{label1}{format_str.format(vector1[0, 0], vector1[1, 0])}  {label2}{format_str.format(vector2[0, 0], vector2[1, 0])}")
    return
def print_vectors_side_by_side_float(vector1, vector2, label1="Vector 1", label2="Vector 2"):
    print(f"{label1}:({vector1[0, 0]:.2f}, {vector1[1, 0]:.2f});  {label2}:({vector2[0, 0]:.2f}, {vector2[1, 0]:.2f})")
    return
def print_matrix(matrix, label):
    """
    Prints a 2D NumPy matrix in a clean, readable format.

    Parameters:
    matrix (numpy.ndarray): A 2D NumPy array to print.
    """
    if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
        raise ValueError("Input should be a 2D NumPy array.")

    print(label)
    # Loop through the matrix and print each row
    for row in matrix:
        print(" ".join(f"{val: .2e}" for val in row))  # Adjust the formatting as needed

    return
def print_matrices_side_by_side(matrix1, matrix2, label1="Matrix 1", label2="Matrix 2"):
    """
    Prints two 2D NumPy matrices side by side in a clean, readable format.

    Parameters:
    matrix1, matrix2 (numpy.ndarray): Two 2D NumPy arrays to print side by side.
    label1, label2 (str): Labels for the two matrices (optional).
    """
    if not isinstance(matrix1, np.ndarray) or matrix1.ndim != 2:
        raise ValueError("matrix1 should be a 2D NumPy array.")
    if not isinstance(matrix2, np.ndarray) or matrix2.ndim != 2:
        raise ValueError("matrix2 should be a 2D NumPy array.")

    # Get the number of rows in both matrices
    max_rows = max(matrix1.shape[0], matrix2.shape[0])

    print(f"{label1:<20} {label2}")
    # print("-" * 40)

    for i in range(max_rows):
        # Get the i-th row of matrix1 if it exists, otherwise fill with empty spaces
        row1 = matrix1[i] if i < matrix1.shape[0] else [""] * matrix1.shape[1]
        row1_str = " ".join(f"{val: .2e}" if isinstance(val, (int, float)) else "" for val in row1)

        # Get the i-th row of matrix2 if it exists, otherwise fill with empty spaces
        row2 = matrix2[i] if i < matrix2.shape[0] else [""] * matrix2.shape[1]
        row2_str = " ".join(f"{val: .2e}" if isinstance(val, (int, float)) else "" for val in row2)
        # row2_str = " ".join(f"{val: .2f}" if isinstance(val, (int, float)) else "" for val in row2)

        print(f"{row1_str:<20} {row2_str}")

def assert_symmetric(matrix, matrix_name="", tol=1e-3):
    if not np.allclose(matrix, matrix.T, atol=tol):
        print_matrix(matrix, matrix_name)
        raise ValueError("The matrix is not symmetric")

def symmetrize(V):
    return (V+V.T)/2

def convert_1x1_matrix_to_scalar(V):
    """
    Convert a 1x1 matrix to a scalar.

    Args:
    V (np.ndarray): A numpy array that is expected to be a 1x1 matrix.

    Returns:
    float: The scalar value from the 1x1 matrix.

    Raises:
    ValueError: If V is not a 1x1 matrix.
    """
    # Check if the input is a 1x1 matrix
    if V.shape == (1, 1):
        # Return the scalar value
        return V.item()
    else:
        # Raise an error if the matrix is not 1x1
        raise ValueError("Input is not a 1x1 matrix")

def vector_angle(vector):
    """
    Calculate the angle of a 2D vector in radians, measured counterclockwise from the positive x-axis.

    Parameters:
        vector (np.ndarray): A 2x1 NumPy matrix representing a 2D vector.

    Returns:
        float: The angle in radians.
    """
    if vector.shape != (2, 1):
        raise ValueError("Input must be a 2x1 NumPy matrix.")

    x, y = vector[0, 0], vector[1, 0]  # Extract x and y components
    return np.arctan2(y, x)
