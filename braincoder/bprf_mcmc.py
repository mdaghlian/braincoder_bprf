# MCMC fitting for bprf
import pandas as pd
import numpy as np
import datetime
import tensorflow as tf
import os.path as op
import os
from tqdm.auto import tqdm
from .utils import format_data, format_parameters, format_paradigm, logit, get_rsq
from braincoder.stimuli import ImageStimulus
import logging
from tensorflow.math import softplus, sigmoid
from tensorflow.linalg import lstsq
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from .models import LinearModelWithBaseline

softplus_inverse = tfp.math.softplus_inverse


class BPRF(object):
    ''' Wrapper to do bayesian prf fitting
    designed by Marcus Daghlian
    '''

    def __init__(self, model, data,  **kwargs):        
        ''' __init__
        Set up the object; with important info
        '''
        self.model = model
        self.data = data.astype(np.float32)
        self.memory_limit = kwargs.pop('memory_limit', 666666666)  # 8 GB?
        self.paradigm = model.get_paradigm(model.paradigm)
        log_dir = kwargs.get('log_dir', False)
        if log_dir is None:
            log_dir = op.abspath('logs/fit')

        if log_dir is not False:
            if not op.exists(log_dir):
                os.makedirs(log_dir)
            self.summary_writer = tf.summary.create_file_writer(op.join(log_dir,
                                                                        datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
        self.kwargs = kwargs
        self.model = model        
        self.n_params = len(self.model.parameter_labels)
        self.model_labels = {l:i for i,l in enumerate(self.model.parameter_labels)}        

        # MCMC specific information
        self.model_sampler = {}                                     # A sampler for each parameter (e.g., randomly pick "x" b/w -5 and 5)
        self.model_prior = {}                                       # Prior for each parameter (e.g., normal distribution at 0 for "x")
        self.model_prior_type = {}                                  # For each parameter: can be "uniform" (just bounds), or "normal"
        self.joint_prior = []                                       # Joint priors?
        self.fixed_vals = {}                                        # We can fix some parameters. To speed things up, don't include them in the MCMC process
        self.bounds = {}
        
        # How to estimate the offset and slope for time series. "glm" or "mcmc" (is it just another MCMC parameter, or use glm )                
        # self.beta_method = kwargs.get('beta_method', 'mcmc') # glm or mcmc 
        # self.fixed_baseline = kwargs.get('fixed_baseline', None) # If using glm, fix baseline?      

    def add_prior(self, pid, prior_type, **kwargs):
        ''' 
        Adds the prior to each parameter:
        Used for 
        [1] evaluating the posterior (e.g., to enforce parameter bounds)
        [2] Initialising walker positions (if init_walker_method=='random_prior')
        > randomly sample parameters from the prior distributions

        Options:
            fixed:      will 'hide' the parameter from MCMC fitting procedure (not really a prior...)
            uniform:    uniform probability b/w the specified bounds (vmin, vmax). Otherwise infinite
            normal:     normal probability. (loc, scale)
            none:       The parameter can still vary, but it will not influence the outcome... 

        '''        
        if pid not in self.model_labels.keys(): # Is this a valid parameter to add? 
            print('error...')
            return
        self.model_prior_type[pid] = prior_type 
        if prior_type=='normal':
            # Get loc, and scale
            loc = kwargs.get('loc')    
            scale = kwargs.get('scale')    
            self.model_sampler[pid] = PriorNorm(loc, scale).sampler 
            self.model_prior[pid]   = PriorNorm(loc, scale).prior            

        elif prior_type=='uniform' :
            vmin = kwargs.get('vmin')
            vmax = kwargs.get('vmax')
            self.bounds[pid] = [vmin, vmax]
            self.model_sampler[pid] = PriorUniform(vmin, vmax).sampler
            self.model_prior[pid] = PriorUniform(vmin, vmax).prior            

        elif prior_type=='fixed':
            fixed_val = kwargs.get('fixed_val')
            self.bounds[pid] = [fixed_val, fixed_val]
            self.fixed_vals[pid] = fixed_val
            self.model_prior[pid] = PriorFixed(fixed_val).prior

        elif prior_type=='none':
            self.model_prior[pid] = PriorNone().prior            
    
    def add_priors_from_bounds(self, bounds):
        '''
        Used to setup uninformative priors: i.e., uniform between the bouds
        Can setup more informative, like a normal using the other methods        
        '''        
        for i_p, v_p in enumerate(self.model_labels.keys()):
            if v_p=='rsq':
                continue

            if bounds[v_p][0]!=bounds[v_p][1]: 
                self.add_prior(
                    pid=v_p,
                    prior_type = 'uniform',
                    vmin = bounds[v_p][0],
                    vmax = bounds[v_p][1],
                    )
            else: # If upper & lower bound are the same, make it a fixed parameter
                self.add_prior(
                    pid=v_p,
                    prior_type = 'fixed',
                    fixed_val = bounds[v_p][0],
                    )

    def fit(self, 
            num_results=1000, num_burnin_steps=500, 
            init_pars=None,
            confounds=None,
            fixed_pars=None,
            progressbar=True,
            **kwargs):

        
        
        n_voxels, n_pars = self.data.shape[1], len(self.model.parameter_labels)
        y = self.data.values

        if init_pars is None:
            init_pars = self.model.get_init_pars(
                data=y, paradigm=self.paradigm, confounds=confounds)
            print('using get_init_pars')

        init_pars = self.model._get_parameters(init_pars)
        init_pars = self.model._transform_parameters_backward(init_pars.values.astype(np.float32))

        ssq_data = tf.reduce_sum(
            (y - tf.reduce_mean(y, 0)[tf.newaxis, :])**2, 0)

        # Voxels with no variance to explain can confuse the optimizer to a large degree,
        # since the gradient landscape is completely flat.
        # Therefore, we only optimize voxels where there is variance to explain
        meaningful_ts = ssq_data > 0.0

        if fixed_pars is None:
            parameter_ix = range(n_pars)
        else:
            parameter_ix = [ix for ix, label in enumerate(self.model.parameter_labels) if label not in fixed_pars]

            print('*** Only fitting: ***')
            for ix in parameter_ix:
                print(f' * {self.model.parameter_labels[ix]}')

        parameter_ix = tf.constant(parameter_ix, dtype=tf.int32)

        n_meaningful_ts = tf.reduce_sum(tf.cast(meaningful_ts, tf.int32))
        n_trainable_pars = len(parameter_ix)

        update_feature_ix, update_parameter_ix = tf.meshgrid(tf.cast(tf.where(meaningful_ts), tf.int32), parameter_ix)
        update_ix = tf.stack((tf.reshape(update_feature_ix, tf.size(update_feature_ix)),
                              tf.reshape(update_parameter_ix, tf.size(update_parameter_ix))), 1)

        print(
            f'Number of problematic voxels (mask): {tf.reduce_sum(tf.cast(meaningful_ts == False, tf.int32))}')
        print(
            f'Number of voxels remaining (mask): {tf.reduce_sum(tf.cast(meaningful_ts == True, tf.int32))}')

        trainable_parameters = tf.Variable(initial_value=tf.gather_nd(init_pars, update_ix),
                                           shape=(n_meaningful_ts*n_trainable_pars),
                                           name='estimated_parameters', dtype=tf.float32)

        trainable_variables = [trainable_parameters]

        paradigm_ = self.model.stimulus._clean_paradigm(self.paradigm)
        priors_to_loop = list(self.model_prior.keys())

        # Define the prior
        @tf.function
        def log_prior_fn(parameters):
            # Log-prior function for the model
            print(parameters.shape)
            p_out = 0            
            if priors_to_loop==[]:
                return p_out
            for p in priors_to_loop:
                print(p)
                i_p = self.model_labels[p]
                p_out += self.model_prior[p](parameters[:,i_p])
            return p_out    
        normal_dist = tfp.distributions.Normal(loc=0.0, scale=1.0)
        @tf.function
        def log_posterior_fn(parameters, voxel_idx):
            predictions = self.model._predict(
                paradigm_[tf.newaxis, ...], parameters[tf.newaxis, ...], None)
            residuals = y[:,voxel_idx] - tf.reshape(predictions, [-1])                            
            log_likelihood = tf.reduce_sum(normal_dist.log_prob(residuals))
            log_prior = log_prior_fn(parameters)
            return log_likelihood + log_prior
        
        sampler = [] 
        for voxel_idx in range(n_voxels):
            # HMC Transition Kernel
            def target_log_prob_fn(*parameters):
                parameters = tf.stack(parameters, axis=-1)
                return log_posterior_fn(parameters, voxel_idx)
            
            kernel_type = 'hmc'
            if kernel_type=='hmc':
                # Initialize HMC sampler
                kernel = tfp.mcmc.HamiltonianMonteCarlo(
                    target_log_prob_fn=target_log_prob_fn,
                    step_size=0.1,
                    num_leapfrog_steps=3
                )

                # Adaptive step size
                kernel = tfp.mcmc.SimpleStepSizeAdaptation(
                    inner_kernel=kernel,
                    num_adaptation_steps=int(num_burnin_steps * 0.8)
                )        
            elif kernel_type == 'mh':
                # Initialize MH sampler
                kernel = tfp.mcmc.RandomWalkMetropolis(
                    target_log_prob_fn=target_log_prob_fn
                )
            elif kernel_type == 'nuts':
                # Initialize NUTS sampler
                kernel = tfp.mcmc.NoUTurnSampler(
                    target_log_prob_fn=target_log_prob_fn,
                    step_size=0.1
                )

                # Adaptive step size
                kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
                    inner_kernel=kernel,
                    num_adaptation_steps=int(num_burnin_steps * 0.8),
                    target_accept_prob=0.8
                )

            initial_state = [tf.convert_to_tensor(init_pars[voxel_idx,i], dtype=tf.float32) for i in range(n_pars)]            
            initial_state = [tf.expand_dims(i, axis=0) for i in initial_state]
            if progressbar:
                print("Running HMC...")

            
            @tf.function
            def run_chain():
                return tfp.mcmc.sample_chain(
                    num_results=num_results,
                    num_burnin_steps=num_burnin_steps,
                    current_state=initial_state,                    
                    kernel=kernel,
                    trace_fn=lambda current_state, kernel_results: kernel_results,
                    seed=1234,
                )

      
            samples, beep = run_chain()
            all_samples = tf.stack(samples, axis=-1).numpy()
            all_samples = all_samples.reshape(-1, init_pars.shape[1])

            self.estimated_parameters = pd.DataFrame(
                all_samples, columns=self.model.parameter_labels
            )
            return self.estimated_parameters 
            sampler.append(samples)
        # Convert samples to DataFrame

        # self.estimated_parameters.index.name = 'sample'

        # return self.estimated_parameters
        return sampler
    


    def ln_posterior(self, params, response):
        prior = self.ln_prior(params)
        like = self.ln_likelihood(params, response)
        return prior + like, like # save both...


    def get_predictions(self, parameters=None):

        if parameters is None:
            parameters = self.estimated_parameters

        return self.model.predict(self.paradigm, parameters, None)

    def get_residuals(self, parameters=None):

        if parameters is None:
            parameters = self.estimated_parameters

        return self.data - self.get_predictions(parameters).values

    def get_rsq(self, parameters=None):

        if parameters is None:
            parameters = self.estimated_parameters

        return get_rsq(self.data, self.get_predictions(parameters))

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        if data is None:
            self._data = None
        else:
            self._data = format_data(data)

    @property
    def paradigm(self):
        return self._paradigm

    @paradigm.setter
    def paradigm(self, paradigm):
        self._paradigm = format_paradigm(paradigm)

# *** PRIORS ***
class PriorNorm:
    def __init__(self, loc, scale):
        self.loc = loc  # mean
        self.scale = scale  # standard deviation
        self.normal_dist = tfp.distributions.Normal(loc=self.loc, scale=self.scale)

    def sampler(self, n_samples):
        # Sample from the normal distribution
        return self.normal_dist.sample(n_samples)

    def prior(self, p):
        # Return the log probability of the parameter given the normal distribution
        return self.normal_dist.log_prob(p)

class PriorUniform:
    def __init__(self, vmin, vmax):
        self.vmin = vmin
        self.vmax = vmax
        self.uniform_dist = tfp.distributions.Uniform(low=self.vmin, high=self.vmax)

    def sampler(self, n_samples):
        return self.uniform_dist.sample(n_samples)

    def prior(self, param):
        return self.uniform_dist.log_prob(param)

class PriorFixed:
    def __init__(self, fixed_val):
        self.fixed_val = fixed_val
        self.vmin = fixed_val
        self.vmax = fixed_val

    def prior(self, param):
        # Return 0 if param is equal to fixed_val, else -inf
        return tf.where(tf.equal(param, self.fixed_val), tf.zeros_like(param), -tf.math.inf)

class PriorNone:
    def __init__(self):
        self.bounds = 'None'

    def prior(self, param):
        # Return 0 for any parameter
        return tf.zeros_like(param)
