# Required output directories
REQ_DIRS := \
    data \
    plot/single \
    plot/sweep \
    plot/exp2/em_series_fail \
    plot/exp2/am_series_control \
    plot/exp1/am_series \
    plot/exp1/em_series

# Default
.PHONY: all
all: init single

.PHONY: init
init:
	@echo "Creating required output directories"
	@$(foreach dir,$(REQ_DIRS),\
	    mkdir -p $(dir) && echo "Created directory: $(dir)";)

# Target for a single experiment
.PHONY: single
single:
	@echo "Running a single experiment of both EM and AM."
	python3 single_experiment_em_am.py

.PHONY: visual
visual:
	@echo "Running a sets of small experiments to generate demo visualization."
	python3 visualization_em_am.py

.PHONY: sweep
sweep:
	@echo "Running a parameter sweep of observation noise"
	python3 sweep_obs_noise_em_am.py
	@echo "Plotting parameter sweep of observation noise"
	python3 plot_em_am_sweep.py

.PHONY: clean
clean:
	@echo "Cleaning up generated files and directories..."
	rm -rf plot/
	rm -rf data/
	@echo "Cleanup complete."
