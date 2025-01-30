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
        Set up and run "bayesian" pRF analysis. This class contains:
        * model + stimulus/paradigm (from .models, used to generate predictions)
        * prior definitions (per parameter)        

        TODO: 
        * add option to fix parameters
        * more prior options
        * work out how this jacobian thingy works with the priors + bijectors...


        '''    
        self.model = copy.deepcopy(model)
        self.data = data.astype(np.float32)
        self.kwargs = kwargs
        self.paradigm = model.get_paradigm(model.paradigm)
        self.n_params = len(self.model.parameter_labels)
        self.n_voxels = self.data.shape[-1]
        self.model_labels = {l:i for i,l in enumerate(self.model.parameter_labels)} # useful to have it as a dict per entry        
        
        # MCMC specific information
        # Prior for each parameter (e.g., normal distribution at 0 for "x")      
        # -> default no prior
        self.p_prior = {p:PriorNone() for p in self.model_labels}                                        
        self.p_prior_type = {p:'none' for p in self.model_labels}
        # -> default no bijector 
        self.p_bijector = {p:tfp.bijectors.Identity() for p in self.model_labels} # What to apply to the 
        self.p_bijector_bw = {p:tfp.bijectors.Identity() for p in self.model_labels} # What to apply to the 

        # Per voxel (row in data) - save the output of the MCMC sampler 
        self.mcmc_sampler = [None] * self.data.shape[1]
        self.mcmc_stats = [None] * self.data.shape[1]
        
    def add_prior(self, pid, prior_type, **kwargs):
        ''' 
        Adds the prior to each parameter:

        Options:
            uniform:    uniform probability b/w the specified bounds (vmin, vmax). Otherwise infinite
            normal:     normal probability. (loc, scale)
            none:       The parameter can still vary, but it will not influence the outcome... 

        '''        
        if pid not in self.model_labels.keys(): # Is this a valid parameter to add? 
            print(f'error... not {pid} not in model labels')
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
            self.p_prior[pid] = PriorUniform(vmin, vmax)            

        elif prior_type=='none':
            self.p_prior[pid] = PriorNone()  
        else:
            raise ValueError(f"Prior type '{prior_type}' is not implemented")
    
    def add_priors_from_bounds(self, bounds):
        '''
        Used to setup uninformative priors: i.e., uniform between the bouds
        Can setup more informative, like a normal using the other methods        
        '''        
        for p in bounds.keys():
            self.add_prior(
                pid=p,
                prior_type = 'uniform',
                vmin = bounds[p][0],
                vmax = bounds[p][1],
                )
            
    def add_bijector(self, pid, bijector_type, **kwargs):
        ''' add transformations to parameters so that they are fit smoothly        
        
        identity        - do nothing
        softplus        - don't let anything be negative

        '''
        if bijector_type == 'identity':
            self.p_bijector[pid] = tfp.bijector.Identity()        
        elif bijector_type == 'softplus':
            # Don't let anything be negative
            self.p_bijector[pid] = tfp.bijectors.Softplus()
        elif bijector_type == 'sigmoid':
            self.p_bijector[pid] = tfp.bijectors.Sigmoid(
                low=kwargs.get('low'), high=kwargs.get('high')
            )

    def add_bijector_from_bounds(self, bounds):
        for p in bounds.keys():
            self.add_bijector(
                pid=p,
                bijector_type = 'sigmoid',
                vmin = bounds[p][0],
                vmax = bounds[p][1],
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
        # Only loop through those parameters with a prior
        self.priors_to_loop = [
            p for p,t in self.p_prior_type.items()# if t not in ('fixed', 'none')
        ]
    
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
            idx = None, 
            init_pars=None,
            num_results=1000, 
            **kwargs):
        
        step_size = kwargs.pop('step_size', 1) # rest of the kwargs go to "hmc_sample"        

        # Which voxels to fit?    
        if idx is None: # all of them
            idx = range(self.n_voxels)
        elif isinstance(idx, int):
            idx = [idx]            
        
        y = self.data.values
        init_pars = self.model._get_parameters(init_pars)
        # Use the bprf bijectors (not the model ones...)
        init_pars = init_pars.values.astype(np.float32) # ???        
        # # Maybe should be applying some transform here? 
        # init_pars = self._bprf_transform_parameters_forward(init_pars.values.astype(np.float32)) # ???        

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
                p_out += self.p_prior[p].prior(parameters[:,self.model_labels[p]])
            return p_out       

        # Calculating the likelihood, based on the assumption that 
        # the residuals are all normally distributed...
        # simple - but I think it works; and has been applied in this context before (see Invernizzi et al)
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
            # -> make sure we are in the correct dtype 
            initial_state = [tf.convert_to_tensor(init_pars[voxel_idx,i], dtype=tf.float32) for i in range(self.n_params)]                          
            # -> also make sure that the shape of the tensor is one that the "model" class likes 
            initial_state = [tf.expand_dims(i, axis=0) for i in initial_state]
            # Define the target log probability function (for this voxel)
            
            def target_log_prob_fn(*parameters):
                parameters = tf.stack(parameters, axis=-1)
                return log_posterior_fn(parameters, voxel_idx)
            
            print('Lets run some checks with everything...')
            # Check the gradient with respect to each parameter
            with tf.GradientTape() as tape:
                tape.watch(initial_state)
                log_prob = target_log_prob_fn(*initial_state)
            gradients = tape.gradient(log_prob, initial_state)
            print('Using tape.gradient to check gradients w/respect to each parameter')
            for i, grad in enumerate(gradients):
                print(f'Gradient for parameter {i}: {grad.numpy()}')
            # CALLING GILLES' "sample_hmc" from .utils.mcmc
            # quick test - does it work?
            initial_ll = target_log_prob_fn(*initial_state)
            print(f'idx={idx}; initial_ll={initial_ll}')
            print('Calling "sample_hmc" from braincoder.utils.mcmc')

            samples, stats = sample_hmc(
                init_state = initial_state, 
                target_log_prob_fn=target_log_prob_fn, 
                unconstraining_bijectors=self.p_bijector_list, 
                num_steps=num_results, 
                # OTHER STUFF TO OPTIMIZE
                step_size=step_size, 
                **kwargs
                )                
                                
            # stuff to save...
            all_samples = tf.stack(samples, axis=-1).numpy()
            all_samples = all_samples.reshape(-1, init_pars.shape[1])
            # Apply bijectors                        
            estimated_p_dict = {}
            for i,p in enumerate(self.model_labels):
                estimated_p_dict[p] = all_samples[:,i]
            estimated_parameters = pd.DataFrame(estimated_p_dict)
            self.mcmc_sampler[voxel_idx] = estimated_parameters            
            self.mcmc_stats[voxel_idx] = stats

    def fit_all(self, 
            idx = None, 
            init_pars=None,
            num_results=1000, 
            **kwargs):
        '''
        Experimental - can we fit everything at once?
        Does that even make sense?
        '''
        
        step_size = kwargs.pop('step_size', 1) # rest of the kwargs go to "hmc_sample"                
        y = self.data.values
        init_pars = self.model._get_parameters(init_pars)
        # Use the bprf bijectors (not the model ones...)
        init_pars = init_pars.values.astype(np.float32) # ???        
        # # Maybe should be applying some transform here? 
        # init_pars = self._bprf_transform_parameters_forward(init_pars.values.astype(np.float32)) # ???        

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
                p_out += self.p_prior[p].prior(parameters[:,self.model_labels[p]])
            return p_out       

        # Calculating the likelihood, based on the assumption that 
        # the residuals are all normally distributed...
        # simple - but I think it works; and has been applied in this context before (see Invernizzi et al)
        normal_dist = tfp.distributions.Normal(loc=0.0, scale=1.0)
        # @tf.function
        def log_posterior_fn(parameters):
            predictions = self.model._predict(
                paradigm_[tf.newaxis, ...], parameters[tf.newaxis, ...], None)
            
            residuals = y - predictions[0]
            log_likelihood = tf.reduce_sum(normal_dist.log_prob(residuals))
            log_prior = tf.reduce_sum(log_prior_fn(parameters))
            return log_likelihood + log_prior
        
        # -> make sure we are in the correct dtype 
        initial_state = [tf.convert_to_tensor(init_pars[:,i], dtype=tf.float32) for i in range(self.n_params)]                          
        # Define the target log probability function (for this voxel)
        
        def target_log_prob_fn(*parameters):
            parameters = tf.stack(parameters, axis=-1)
            return log_posterior_fn(parameters)
        
        print('Lets run some checks with everything...')
        # Check the gradient with respect to each parameter
        log_prob = target_log_prob_fn(*initial_state)
        # with tf.GradientTape() as tape:
        #     tape.watch(initial_state)
        #     log_prob = target_log_prob_fn(*initial_state)
        # gradients = tape.gradient(log_prob, initial_state)
        # print('Using tape.gradient to check gradients w/respect to each parameter')
        # for i, grad in enumerate(gradients):
        #     print(f'Gradient for parameter {i}: {grad.numpy()}')

        # # CALLING GILLES' "sample_hmc" from .utils.mcmc
        # # quick test - does it work?
        # initial_ll = target_log_prob_fn(*initial_state)
        # print(f'idx={idx}; initial_ll={initial_ll}')
        # print('Calling "sample_hmc" from braincoder.utils.mcmc')

        samples, stats = sample_hmc(
            init_state = initial_state, 
            target_log_prob_fn=target_log_prob_fn, 
            unconstraining_bijectors=self.p_bijector_list, 
            num_steps=num_results, 
            # OTHER STUFF TO OPTIMIZE
            step_size=step_size, 
            **kwargs
            )                
                            
        # stuff to save...        
        all_samples = tf.stack(samples, axis=-1).numpy()
        # nsteps, n_voxels, n_params
        
        for ivx in range(self.n_voxels):
            estimated_p_dict = {}
            for i,p in enumerate(self.model_labels):
                estimated_p_dict[p] = all_samples[:,ivx,i]
            self.mcmc_sampler[ivx] = pd.DataFrame(estimated_p_dict)
        self.mcmc_stats = stats            

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
    prior_type = 'base'
    def prior(self, param):
        return self.distribution.log_prob(param)
    
class PriorNorm(PriorBase):
    def __init__(self, loc, scale):
        self.prior_type = 'norm'
        self.loc = loc  # mean
        self.scale = scale  # standard deviation
        self.distribution = tfp.distributions.Normal(loc=self.loc, scale=self.scale)

class PriorUniform(PriorBase):
    def __init__(self, vmin, vmax):
        self.prior_type = 'uniform'
        self.vmin = vmin
        self.vmax = vmax
        self.distribution = tfp.distributions.Uniform(low=self.vmin, high=self.vmax)

class PriorNone(PriorBase):
    def __init__(self):
        self.prior_type = 'none'
    def prior(self, param):
        return tf.zeros_like(param)
# *** BIJECTORS ***
