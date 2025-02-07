#!/usr/bin/env python
#$ -j Y
#$ -cwd
#$ -V

import os
import sys
import yaml
import pickle
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import tensorflow as tf
print('GPU devices available:', tf.config.list_physical_devices('GPU'))

from scipy import io
import braincoder
from braincoder.utils.visualize import *
import prfpy_csenf

from os.path import join as opj
from braincoder.models import ContrastSensitivity, ContrastSensitivityWithHRF
from braincoder.hrf import SPMHRFModel, CustomHRFModel, HRFModel
from braincoder.stimuli import ContrastSensitivityStimulus
from braincoder.bprf_mcmc import BPRF
from braincoder.optimize import ParameterFitter

# --------------------------
# Argument Parsing for Output Name
# --------------------------
parser = argparse.ArgumentParser(description="Run contrast sensitivity simulation and save results.")
parser.add_argument("--output", type=str, default="output", help="Base name for output files (default: 'output')")
parser.add_argument("--config", type=str, default="config.yml", help="Path to configuration YAML file (default: 'config.yaml')")
args = parser.parse_args()

output_name = args.output
config_path = args.config

# --------------------------
# Load configuration from YAML
# --------------------------
with open(opj(os.path.dirname(__file__), config_path), 'wb') as f:
    config = yaml.safe_load(f)
bloop
# Extract configuration variables
bounds       = config['bounds']
n_vx         = config['n_vx']
hrf_tr       = config['hrf_model']['tr']
noise_level  = config['simulation']['noise']
grid_points  = config['grid_fit']['grid_points']
grid_fixed   = config['grid_fit']['fixed']
use_corr_cost= config['grid_fit']['use_correlation_cost']
mcmc_params  = config['mcmc']
init_param_values = config['initial_parameters']

# --------------------------
# Load stimulus sequences
# --------------------------
seq_path = os.path.join(os.path.dirname(prfpy_csenf.__path__[0]))
sfs_seq = np.load(opj(seq_path, 'eg_sfs_seq.npy'))
con_seq = np.load(opj(seq_path, 'eg_con_seq.npy'))
paradigm = np.vstack([sfs_seq, con_seq])

# --------------------------
# Create stimulus and model objects
# --------------------------
cs_stim = ContrastSensitivityStimulus()

# Create a DataFrame of random parameters within the specified bounds
parameters = pd.DataFrame({
    key: np.random.uniform(bounds[key][0], bounds[key][1], n_vx)
    for key in bounds
}).astype('float32')

# For plotting purposes: one color per voxel
idx_cols = sns.color_palette("hsv", n_vx)

# Create the model using the loaded HRF parameter from the YAML config
model = ContrastSensitivityWithHRF(
    parameters=parameters,
    SF_seq=sfs_seq,  
    CON_seq=con_seq,
    hrf_model=SPMHRFModel(tr=hrf_tr),
)

# --------------------------
# Simulation
# --------------------------
data = model.simulate(noise=noise_level)

# Bundle the ground truth results with the configuration settings in one pickle.
ground_truth = {
    'parameters': parameters,
    'data': data,
    'bounds': bounds,
    'config': config
}
with open(f'./{output_name}/ground_truth.pkl', 'wb') as file:
    pickle.dump(ground_truth, file)

# --------------------------
# Parameter fitting using grid search
# --------------------------
start_time = time.time()
cfitter = ParameterFitter(model, data, model.paradigm)
grid_pars = cfitter.fit_grid(
    width_r     = np.linspace(bounds['width_r'][0], bounds['width_r'][1], grid_points['width_r']),
    SFp         = np.linspace(bounds['SFp'][0], bounds['SFp'][1], grid_points['SFp']),
    CSp         = np.linspace(bounds['CSp'][0], bounds['CSp'][1], grid_points['CSp']),
    width_l     = np.linspace(bounds['width_l'][0], bounds['width_l'][1], grid_points['width_l']),
    crf_exp     = np.linspace(bounds['crf_exp'][0], bounds['crf_exp'][1], grid_points['crf_exp']),
    amplitude   = [grid_fixed['amplitude']],
    baseline    = [grid_fixed['baseline']],
    use_correlation_cost = use_corr_cost
)
ols_pars = cfitter.refine_baseline_and_amplitude(grid_pars)
refined_pars = cfitter.fit(init_pars=ols_pars)  # You can also fix parameters if desired
end_time = time.time()
elapsed_time = end_time - start_time

# Convert to hours, minutes, seconds
hours, remainder = divmod(elapsed_time, 3600)
minutes, seconds = divmod(remainder, 60)
cfit_time = f"Elapsed time - classical fitting: {int(hours)}h {int(minutes)}m {seconds:.2f}s"
print(cfit_time)
# Save the cfitter object
with open(f'./{output_name}/cfitter.pkl', 'wb') as file:
    pickle.dump({'cfitter':cfitter, 'cfit_time':cfit_time}, file)

# --------------------------
# Bayesian pRF fitting using MCMC
# --------------------------
bfitter = BPRF(model, data)
bfitter.add_bijector_from_bounds(bounds)

# Prepare initial parameters as a DataFrame. Each entry is a constant vector of length n_vx.
init_pars_dict = {
    key: np.ones(n_vx) * init_param_values[key]
    for key in init_param_values
}
init_pars = pd.DataFrame(init_pars_dict)

start_time = time.time()

bfitter.fit_mcmc(
    init_pars=init_pars,
    num_steps=mcmc_params['num_steps'],
    unrolled_leapfrog_steps=mcmc_params['unrolled_leapfrog_steps'],
    step_size=mcmc_params['step_size'],
    max_tree_depth=mcmc_params['max_tree_depth'],
    target_accept_prob=mcmc_params['target_accept_prob'],
    sampler_fn=mcmc_params['sampler_fn']
)

end_time = time.time()
elapsed_time = end_time - start_time

# Convert to hours, minutes, seconds
hours, remainder = divmod(elapsed_time, 3600)
minutes, seconds = divmod(remainder, 60)
bfit_time = f"Elapsed time - MCMC fitting: {int(hours)}h {int(minutes)}m {seconds:.2f}s "
print(bfit_time)

with open(f'./{output_name}/bfitter.pkl', 'wb') as file:
    pickle.dump({'bfitter':bfitter, 'bfit_time':bfit_time}, file)
