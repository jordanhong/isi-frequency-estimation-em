import numpy as np
# import cupy as np

## Constants

## Initial values
C = np.array([[1, 0]])
# C = np.array([[1/np.sqrt(2), 1/np.sqrt(2)]])
# C = np.array([[0, 1]])
X_init = np.zeros((2, 1))
##########
# theta_init = np.pi/8
# theta_init = np.pi/8 + 0.1 * np.pi
theta_init = np.pi/2
# theta_init = np.pi/3
# theta_init = 0
##########
R_init = np.array([[np.cos(theta_init)], [np.sin(theta_init)]])
# R_init =  np.array([[8.50e-01], [1.56e-3]])


