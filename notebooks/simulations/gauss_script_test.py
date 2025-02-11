#!/usr/bin/env python
#$ -j Y
#$ -cwd
#$ -V

import os
from os.path import join as opj
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
from scipy import io
print('GPU devices available:', tf.config.list_physical_devices('GPU'))

import braincoder
from braincoder.utils.visualize import *


from braincoder.models import ContrastSensitivity, ContrastSensitivityWithHRF
from braincoder.hrf import SPMHRFModel, CustomHRFModel, HRFModel
from braincoder.bprf_mcmc import BPRF
from braincoder.optimize import ParameterFitter

# --------------------------
# Argument Parsing for Output Name
# --------------------------
parser = argparse.ArgumentParser(description="Run simulation for gaussian model.")
parser.add_argument("--output", type=str, default="output", help="Base name for output files (default: 'output')")
parser.add_argument("--config", type=str, default="gauss_config.yml", help="Path to configuration YAML file (default: 'gauss_config.yaml')")
args = parser.parse_args()

output_name = args.output
output_name = '/data1/projects/dumoulinlab/Lab_members/Marcus/programs/braincoder_bprf/notebooks/simulations/output_gauss'
config_path = args.config

# --------------------------
# Load configuration from YAML
# --------------------------
with open(opj(os.path.dirname(__file__), config_path), 'r') as f:
    config = yaml.safe_load(f)

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

dm = io.loadmat('/data1/projects/dumoulinlab/Lab_members/Marcus/programs/braincoder_bprf/notebooks/GazeCenterFS_vd.mat')['stim']

# braincoder convention is time by x by y
paradigm = np.rollaxis(dm, 2, 0)
aspect_ratio = paradigm.shape[2] / paradigm.shape[1]

x, y = np.meshgrid(
    np.linspace(-.75, .75, 3), 
    np.linspace(-aspect_ratio *.75, aspect_ratio *.75, 3))
y_grid, x_grid = np.meshgrid(
    np.linspace(-aspect_ratio, aspect_ratio, paradigm.shape[2]), 
    np.linspace(-1., 1., paradigm.shape[1]))

grid_coordinates = np.stack((x_grid.ravel().astype(np.float32), y_grid.ravel().astype(np.float32)), 1)

y_grid, x_grid = np.meshgrid(np.linspace(-aspect_ratio, aspect_ratio, paradigm.shape[2]), 
                              np.linspace(-1., 1., paradigm.shape[1]))

grid_coordinates = np.stack((x_grid.ravel().astype(np.float32), y_grid.ravel().astype(np.float32)), 1)

# --------------------------
# Create stimulus and model objects
# --------------------------
# Create a DataFrame of random parameters within the specified bounds
parameters = pd.DataFrame({
    key: np.random.uniform(bounds[key][0], bounds[key][1], n_vx)
    for key in bounds
}).astype('float32')
from braincoder.models import GaussianPRF2DWithHRF
from braincoder.hrf import SPMHRFModel
model = GaussianPRF2DWithHRF(
    grid_coordinates, 
    paradigm=paradigm,
    parameters=parameters,
    hrf_model=SPMHRFModel(tr=hrf_tr))


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
with open(f'{output_name}/ground_truth.pkl', 'wb') as file:
    pickle.dump(ground_truth, file)

# --------------------------
# Parameter fitting using grid search
# --------------------------
start_time = time.time()
cfitter = ParameterFitter(model, data, paradigm)


grid_pars = cfitter.fit_grid(
    x=np.linspace(bounds['x'][0], bounds['x'][0], grid_points['x']),
    y=np.linspace(bounds['y'][0], bounds['y'][0], grid_points['y']),
    sd=np.linspace(bounds['sd'][0], bounds['sd'][0], grid_points['sd']), 
    baseline=grid_fixed['baseline'],
    amplitude=grid_fixed['amplitude'],
    use_correlation_cost=use_corr_cost)
ols_pars = cfitter.refine_baseline_and_amplitude(grid_pars)
refined_pars = cfitter.fit(init_pars=ols_pars)
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
bloop
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
