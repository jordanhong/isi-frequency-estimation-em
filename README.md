# Frequency Estimation with New Expectation Maximization Algorithm

## Dependency
- Python 3.x
- Required Python dependencies:
  - `numpy`
  - `matplotlib`
  - `latex` (optional: for plot labels rendering, remove the latex labels if latex is not installed)
  - `pytest` (optional: for running the unit tests)


## Usage

### Initialization
Run the following to create output directories
```bash
make init
```

### Experiments
Individual experiments can be run through the following make targets.  Alternatively, you can also just run the python scripts (see Makefile).

- **`single`**: Runs a single experiment for both EM and AM algorithms with a fixed $\sigma^2$. This target also generates plots for:
  - Log-likelihood
  - Estimated r angle (theta)
  - Estimated r norm
  The output directory is `plot/single`.

- **`visual`**: Executes two small experiments to generate snapshot visualizations of EM and AM. The output directories are `plot/exp1/` (case when both EM and AM works) and `plot/exp2/` (case when AM works but EM fails).

- **`sweep`**: Performs a parameter sweep of the observation noise $\sigma^2_Z$. This stores a json file in `data/`.
  - After running `make sweep`, execute the following command to plot the experiment results:

    ```bash
    python3 plot_em_am_sweep.py
    ```
    The pdf is generated in `plot/sweep`.

- **`clean`**: Cleans up generated files and directories.

### Tests
To run unit tests for the MBF algorithm and the two half-algorithms in alternating maximization, execute:

```bash
pytest test_alternate_max.py
```

### Notes

- The underlying Python scripts are defined in the Makefile target definitions. Refer to the Makefile for further details.
- The `setup_params()` function in `util.py` plays a critical role in ensuring floating-point precision. It initializes variance matrix messages and state noise variance based on the observation noise $\sigma^2_Z$.
- If you encounter assertion failures or weird results (e.g., decreasing log-likelihood plots), consider experimenting with the scaling factors in `setup_params()`. Empirically, if log-likelihood is decreasing, increasing $\sigma^2_U$ helps.



