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


from braincoder.utils.mcmc import sample_hmc
softplus_inverse = tfp.math.softplus_inverse

import copy
class BPRF(object):
    ''' Wrapper to do bayesian prf fitting
    designed by Marcus Daghlian
    '''

    def __init__(self, model, data,  **kwargs):        
        ''' __init__
        Set up the object; with important info
        '''
        self.model = copy.deepcopy(model)
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
        self.n_voxels = len(self.data)
        self.model_labels = {l:i for i,l in enumerate(self.model.parameter_labels)}        

        # MCMC specific information
        self.p_prior = {}                                       # Prior for each parameter (e.g., normal distribution at 0 for "x")
        self.p_prior_fn = {} 
        self.p_bijector = {}
        self.p_bijector_list = []
        self.p_prior_type = {}                                  # For each parameter: can be "uniform" (just bounds), or "normal"
        self.p_fixed_labels = {}                                        # We can fix some parameters. To speed things up, don't include them in the MCMC process
        self.p_fitted_labels = {}
        self.bounds = {}
        self.mcmc_sampler = [None] * self.data.shape[1]             # For each voxel, we will store the MCMC samples
        self.mcmc_stats = [None] * self.data.shape[1]
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
        self.p_prior_type[pid] = prior_type 
        if prior_type=='normal':
            # Get loc, and scale
            loc = kwargs.get('loc')    
            scale = kwargs.get('scale')    
            self.p_prior[pid]   = PriorNorm(loc, scale)            

        elif prior_type=='uniform' :
            vmin = kwargs.get('vmin')
            vmax = kwargs.get('vmax')
            if vmin==vmax:
                self.add_prior(
                    pid=pid, prior_type='fixed', fixed_val=vmin,
                )
                return            
            self.bounds[pid] = [vmin, vmax]
            self.p_prior[pid] = PriorUniform(vmin, vmax)            

        elif prior_type=='latent_uniform':
            vmin = kwargs.get('vmin')
            vmax = kwargs.get('vmax')
            if vmin==vmax:
                self.add_prior(
                    pid=pid, prior_type='fixed', fixed_val=vmin,
                )
                return
            self.bounds[pid] = [vmin, vmax]
            self.p_prior[pid] = PriorLatentUniform(vmin, vmax)

        elif prior_type=='fixed':
            fixed_val = kwargs.get('fixed_val')
            self.bounds[pid] = [fixed_val, fixed_val]
            self.p_fixed_labels[pid] = tf.convert_to_tensor(fixed_val, dtype=tf.float32)
            self.p_prior[pid] = PriorFixed(fixed_val)

        elif prior_type=='none':
            self.p_prior[pid] = PriorNone()            
        self.p_prior_fn[pid] = self.p_prior[pid].prior
        self.p_bijector[pid] = self.p_prior[pid].bijector
    
    def add_priors_from_bounds(self, bounds, prior_type='uniform'):
        '''
        Used to setup uninformative priors: i.e., uniform between the bouds
        Can setup more informative, like a normal using the other methods        
        '''        
        for i_p, v_p in enumerate(self.model_labels.keys()):
            if v_p=='rsq':
                continue
 
            self.add_prior(
                pid=v_p,
                prior_type = prior_type,
                vmin = bounds[v_p][0],
                vmax = bounds[v_p][1],
                )
            
    def prep_for_fitting(self):
        ''' Get everything ready for fitting...
        '''
        # Set the bijectors (to project to a useful fitting space)
        self.p_bijector_list = []
        for p in self.model_labels:
            self.p_bijector_list.append(
                self.p_bijector[p]
            )
        self.p_bijector_list_BACK = []
        for p in self.model_labels:
            self.p_bijector_list_BACK.append(
                self.p_bijector[p].inverse
            )        
        # Only loop through those parameters with a prior
        self.priors_to_loop = [
            p for p,t in self.p_prior_type.items() if t not in ('fixed', 'none')
        ]
        # Create an index for those parameters we are fitting... 
        self.p_fitted_labels = {}
        i_p = 0
        for k in self.model_labels:
            if k in self.p_fixed_labels:
                continue
            self.p_fitted_labels[k] = i_p
            i_p += 1    
        self.n_params2fit = len(self.p_fitted_labels)
    
    @tf.function
    def _bprf_transform_parameters_forward(self, parameters):
        # Loop through parameters & bijectors (forward)
        return tf.concat([
            self.p_bijector_list[i](parameters[:, i][:, tf.newaxis]) for i in range(self.n_params)
            ], axis=1)

    @tf.function
    def _bprf_transform_parameters_backward(self, parameters):
        # Loop through parameters & bijectors... (but apply inverse)
        return tf.concat([
            self.p_bijector_list[i].inverse(parameters[:, i][:, tf.newaxis]) for i in range(self.n_params)
            ], axis=1)


    def fit(self, 
            num_results=1000, 
            num_burnin_steps=500, 
            init_pars=None,
            confounds=None, # ...
            # progressbar=True,
            idx = None, 
            **kwargs):

        # Which voxels to fit?    
        if idx is None: # all of them
            idx = range(self.n_voxels)
        elif isinstance(idx, int):
            idx = [idx]            
        
        y = self.data.values
        # Initial parameters 
        if init_pars is None:
            init_pars = self.model.get_init_pars(
                data=y, paradigm=self.paradigm, confounds=confounds)
            print('using get_init_pars')
        init_pars = self.model._get_parameters(init_pars)
        # Use the bprf bijectors (not the model ones...)
        init_pars = self._bprf_transform_parameters_forward(init_pars.values.astype(np.float32))
        # Clean the paradigm 
        paradigm_ = self.model.stimulus._clean_paradigm(self.paradigm)        
        
        # Define the prior in 'tf'
        @tf.function
        def log_prior_fn(parameters):
            # Log-prior function for the model
            p_out = 0            
            if self.priors_to_loop==[]:
                return p_out
            for p in self.priors_to_loop:
                i_p = self.p_fitted_labels[p]
                p_out += self.p_prior_fn[p](parameters[:,i_p])
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
        
        # Loop through the voxels to fit!
        for voxel_idx in idx:
            initial_state = [tf.convert_to_tensor(init_pars[voxel_idx,i], dtype=tf.float32) for i in range(self.n_params)]                          
            initial_state = [tf.expand_dims(i, axis=0) for i in initial_state]
            # Define the target log probability function
            def target_log_prob_fn(*parameters):
                parameters = tf.stack(parameters, axis=-1)
                return log_posterior_fn(parameters, voxel_idx)
            
            # BASED ON GILLES OG mcmc (but not for decoding)                                            
            # Quick test

            bloop = target_log_prob_fn(*initial_state)
            print(bloop)
            # return

            samples, stats = sample_hmc(
                init_state = initial_state, 
                target_log_prob_fn=target_log_prob_fn, 
                unconstraining_bijectors=self.p_bijector_list, 
                num_steps=num_results, 
                # OTHER STUFF TO OPTIMIZE
                step_size=1, 
                # burnin=n_burnin,
                # target_accept_prob=target_accept_prob, 
                # unrolled_leapfrog_steps=unrolled_leapfrog_steps,
                # max_tree_depth=max_tree_depth
                )                
                                
            # stuff to save...
            all_samples = tf.stack(samples, axis=-1).numpy()
            all_samples = all_samples.reshape(-1, init_pars.shape[1])
            # Apply bijectors                        
            estimated_p_dict = {}
            for i,p in enumerate(self.model_labels):
                if self.p_prior_type[p] == 'latent_uniform':
                    estimated_p_dict[p] = all_samples[:,i]
                    # estimated_p_dict[f'L_{p}'] = all_samples[:,i]
                    # estimated_p_dict[f'L_{p}'] = self.p_bijector[p].inverse(all_samples[:,i])
                else:
                    estimated_p_dict[p] = all_samples[:,i]
            estimated_parameters = pd.DataFrame(estimated_p_dict)
            self.mcmc_sampler[voxel_idx] = estimated_parameters            
            self.mcmc_stats[voxel_idx] = stats
        return     

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

    def get_rsq_for_idx(self, idx, parameters=None):
        this_data = self.data.iloc[:, idx]
        # Repeat n times 
        n_estimates = len(parameters)
        this_data = np.tile(this_data, (n_estimates, 1)).T
        this_pred = self.get_predictions(parameters)
        this_rsq = get_rsq(this_data, this_pred)[:-1]
        return this_rsq



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
class PriorBase():
    bijector = tfp.bijectors.Identity()

class PriorNorm(PriorBase):
    def __init__(self, loc, scale):
        self.loc = loc  # mean
        self.scale = scale  # standard deviation
        self.distribution = tfp.distributions.Normal(loc=self.loc, scale=self.scale)

    def prior(self, p):
        # Return the log probability of the parameter given the normal distribution
        return self.distribution.log_prob(p)

class PriorUniform(PriorBase):
    def __init__(self, vmin, vmax):
        self.vmin = vmin
        self.vmax = vmax
        self.distribution = tfp.distributions.Uniform(low=self.vmin, high=self.vmax)

    def prior(self, param):
        return self.distribution.log_prob(param)


class PriorLatentUniform(PriorBase):  
    '''Special case
    Where the parameter is transformed by the NCDF
    So the latent parameter is sampled from a normal distribution (mean 0, std 1)
    And the actual parameter is transformed by the NCDF - so it is uniform
    between the bounds
    '''  
    def __init__(self, vmin, vmax):
        self.vmin = vmin
        self.vmax = vmax
        self.distribution = tfp.distributions.Normal(loc=0, scale=1)
        # Set the bijector!
        self.bijector = tfp.bijectors.Chain([
                tfp.bijectors.Shift(shift=self.vmin),  # Scale to [vmin, vmax]
                tfp.bijectors.Scale(scale= self.vmax - self.vmin),  # Scale to [vmin, vmax]
                tfp.bijectors.NormalCDF(),  # Transform to uniform
                ])

    def prior(self, param):
        return self.distribution.log_prob(param)


class PriorFixed(PriorBase):

    def __init__(self, fixed_val):
        print('NOT WORKING YET')

        self.fixed_val = fixed_val
        self.tf_fixed_val = tf.convert_to_tensor(fixed_val, dtype=tf.float32)
        self.vmin = fixed_val
        self.vmax = fixed_val
        # Set the bijector to just return the param
        self.bijector = tfp.bijectors.Identity()
        # self.bijector = tfp.bijectors.Chain([
        #     tfp.bijectors.Chain([
        #         tfp.bijectors.Shift(shift=self.tf_fixed_val),
        #         tfp.bijectors.Scale(scale=0.0),  
        #     ])
        # ])
    # def prior(self, param):
    #     # Return 0 if param is equal to fixed_val, else -inf
    #     return tf.where(tf.equal(param, self.fixed_val), tf.zeros_like(param), -tf.math.inf)

    def prior(self,param):
        return tf.zeros_like(param) 
    

class PriorNone(PriorBase):
    def __init__(self):
        self.bounds = 'None'

    def prior(self, param):
        # Return 0 for any parameter
        return tf.zeros_like(param)
