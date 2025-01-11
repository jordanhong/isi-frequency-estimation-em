import numpy as np
import json
import matplotlib.pyplot as plt

N = 20
max_out_iter, max_in_iter = 50, 100

# Load the saved data
with open(f"sqe_data_N_{N}_max_out_iter_{max_out_iter}_max_in_iter_{max_in_iter}.json", 'r') as file:
    data = json.load(file)

# Extract the stored SQE values
all_sqe_em = data['all_sqe_em']
all_sqe_am = data['all_sqe_am']

# Ensure keys are sorted if necessary (convert keys to float)
# sigma2_Z_values = np.logspace(-5, 2, num=10)
sigma2_Z_values = sorted(map(float, all_sqe_em.keys()))

# Recalculate mean SQE values for plotting
mean_sqe_values_em = [np.mean(all_sqe_em[str(sigma2_Z)]) for sigma2_Z in sigma2_Z_values]
mean_sqe_values_am = [np.mean(all_sqe_am[str(sigma2_Z)]) for sigma2_Z in sigma2_Z_values]

# Plotting
plt.figure()

# Mean line for EM
plt.loglog(sigma2_Z_values, mean_sqe_values_em, marker='o', label='EM Mean MSE', color='blue')
# Mean line for AM
plt.loglog(sigma2_Z_values, mean_sqe_values_am, marker='o', label='AM Mean MSE', color='orange')

# Scatter plot for all SQE values
# print(sigma2_Z_values)
for sigma2_Z in sigma2_Z_values:
    sigma2_Z_str = str(sigma2_Z)  # Convert float back to string for accessing dictionary keys
    plt.scatter(
        [sigma2_Z] * len(all_sqe_em[sigma2_Z_str]),
        all_sqe_em[sigma2_Z_str],
        alpha=0.3,
        color='blue',
        label='EM Trial MSE' if sigma2_Z == sigma2_Z_values[0] else ""
    )
    plt.scatter(
        [sigma2_Z] * len(all_sqe_am[sigma2_Z_str]),
        all_sqe_am[sigma2_Z_str],
        alpha=0.3,
        color='orange',
        label='AM Trial MSE' if sigma2_Z == sigma2_Z_values[0] else ""
    )

# Plotting settings
SMALL_SIZE = 8
MEDIUM_SIZE = 14
BIGGER_SIZE = 12

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


# Axis labels and title
plt.xlabel('$\sigma^2_Z$', fontsize=MEDIUM_SIZE)
plt.ylabel('Mean Squared Error $\|r - \hat{r}\|^2$', fontsize=MEDIUM_SIZE)
plt.grid(True, which="both", ls="--")
plt.legend()
plt.savefig(f"scatter_em_am_sweep_N_{N}_max_out_iter_{max_out_iter}_max_in_iter_{max_in_iter}.pdf", bbox_inches='tight')
plt.show()
