import pandas as pd
import numpy as np
from .utils import format_data, format_paradigm, get_rsq, calculate_log_prob_t, calculate_log_prob_gauss_loc0, calculate_log_prob_gauss, format_parameters
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb

from .bprf_mcmc import *

import copy
class BPRF_hier(BPRF):
    ''' Wrapper to do bayesian prf fitting + hiearchical 
    designed by Marcus Daghlian
    '''

    def __init__(self, model, data,  **kwargs):        
        ''' __init__
        Set up and run "bayesian" pRF analysis. This class contains:
        * model + stimulus/paradigm (from .models, used to generate predictions)
        * prior definitions (per parameter)        


        '''    
        super().__init__(model, data, **kwargs)
        # Do we want to fit our priors? 
        # -> hierarchical priors
        self.h_prior_to_apply = {} # How is the hierarchical prior applied to the parameters? i.e., take loc and scale...
        # The rest follows as similar form as the non-hiearchical priors, but with "h_" in front
        self.h_labels = {} # What are the labels for the hierarchical parameters? & their idx
        self.h_bijector = {} # What bijector to apply to the hierarchical parameters
        self.h_prior_type = {} # Meta prior
        self.h_prior = {}
        self.h_mcmc_sampler = {}

                
    def h_add_param(self, pid, h_prior_to_apply='normal', **kwargs):
        ''' Add the parameters for the hierarchical priors
        i.e., fit the prior across all vertices

        Say we look at parameter 'x' 
        we could make the prior for 'x' be N(x_loc,x_scale); 
        Here we need to add the extra parameters x_loc, x_scale
        -> these parameters in turn can have their own priors...
        '''
        # [1] make sure the 'p_prior' is none, because we are doing hierarchical priors now! (i.e., the direct prior is not used)
        self.p_prior[pid] = PriorNone
        self.p_prior_type[pid] = 'none'

        # [2]
        self.h_prior_to_apply[pid] = h_prior_to_apply # This tells us we need to apply the learnt prior to our parameter
        # Only doing hierarchical priors for normal distributions for now...
        if h_prior_to_apply=='normal':
            # [2] add the h_priors to the model_labels
            self.h_labels[f'{pid}_loc'] = len(self.h_labels) 
            self.h_labels[f'{pid}_scale'] = len(self.h_labels)
            self.h_add_bijector(pid=f'{pid}_loc', bijector_type=tfb.Identity())
            self.h_add_bijector(pid=f'{pid}_scale', bijector_type=tfb.Exp())

            # [3] add the priors for the new parameters
            # ... none for now...
            self.h_add_prior(pid=f'{pid}_loc', prior_type='none', **kwargs)
            self.h_add_prior(pid=f'{pid}_scale', prior_type='none', **kwargs)

    
    def h_add_bijector(self, pid, bijector_type, **kwargs):
        ''' add transformations to parameters so that they are fit smoothly        
        
        identity        - do nothing
        softplus        - don't let anything be negative

        '''
        if bijector_type == 'identity':
            self.h_bijector[pid] = tfb.Identity()        
        elif bijector_type == 'softplus':
            # Don't let anything be negative
            self.h_bijector[pid] = tfb.Softplus()
        elif bijector_type == 'sigmoid':
            self.h_bijector[pid] = tfb.Sigmoid(
                low=kwargs.get('low'), high=kwargs.get('high')
            )
        else:
            self.h_bijector[pid] = bijector_type

    def h_add_prior(self, pid, prior_type, **kwargs):
        ''' 
        Adds the prior to each parameter:

        Options:
            uniform:    uniform probability b/w the specified bounds (low, high). Otherwise infinite
            normal:     normal probability. (loc, scale)
            none:       The parameter can still vary, but it will not influence the outcome... 

        '''        
        if pid not in self.h_labels.keys(): # Is this a valid parameter to add? 
            print(f'error... not {pid} not in h labels')
            return
        self.h_prior_type[pid] = prior_type 
        if prior_type=='normal':
            # Get loc, and scale
            loc = kwargs.get('loc')    
            scale = kwargs.get('scale')    
            self.h_prior[pid]   = PriorNorm(loc, scale)            
        elif prior_type=='uniform' :
            low = kwargs.get('low')
            high = kwargs.get('high')        
            self.h_prior[pid] = PriorUniform(low, high)            
        elif prior_type=='none':
            self.h_prior[pid] = PriorNone()  
        elif prior_type=='fixed':
            fixed_val = kwargs.get('fixed_val')
            self.h_prior[pid] = PriorFixed(fixed_val)
        else:
            self.h_prior[pid] = PriorGeneral(prior_type=prior_type, distribution=kwargs.get('distribution'))
    

    def h_prep_for_fitting(self, **kwargs):
        ''' Get everything ready for fitting...
        '''        
        # Ok lets map everything so we can fix some parameters
        # Are there any parameters to fix? 
        if (len(self.h_fixed_pars) != 0):
            # Build the indices and update values lists.
            indices_list = []
            updates_list = []

            # Create a tensor for row indices: (number of vx being fit)
            rows = tf.range(1) # 1 row - this is hierarchical!
            for param_name,fix_value in self.h_fixed_pars.items():
                # Where to put the values
                col_idx = self.h_labels[param_name]                                
                # Create a tensor of column indices (same column for every row)
                cols = tf.fill(tf.shape(rows), col_idx)
                
                # Stack rows and cols to create indices of shape [n_vx_to_fit, 2]
                param_indices = tf.stack([rows, cols], axis=1)
                indices_list.append(param_indices)
                
                # Create the update values: a vector of length n_vx_to_fit with the fixed value.
                param_updates = tf.fill(tf.shape(rows), fix_value)
                updates_list.append(param_updates)
            # Concatenate all the indices and updates from each parameter fix.
            self.h_fix_update_index = tf.concat(indices_list, axis=0)  # shape: [num_updates, 2]
            self.h_fix_update_value = tf.concat(updates_list, axis=0)    # shape: [num_updates]
            
            # Define the update function
            self.h_fix_update_fn = FixUdateFn(self.h_fix_update_index, self.h_fix_update_value).update_fn             

            # Also ensure that priors & bijectors are correct
            for p in self.h_fixed_pars.keys():
                self.h_prior_type[p] = 'none'
                self.h_bijector[p] = tfb.Identity()
            
        else:
            self.h_fix_update_fn = FixUdateFn().update_fn             
        
        # Set the bijectors (to project to a useful fitting space)
        self.h_bijector_list = []
        for p in self.h_labels.keys():            
            self.h_bijector_list.append(
                self.h_bijector[p]
            )      
        # Only loop through those parameters with a prior
        self.h_priors_to_loop = [
            p for p,t in self.h_prior_type.items() if t not in ('fixed', 'none')
        ]


    def fit_mcmc_hier(self, 
            init_pars=None,
            h_init_pars=None,
            num_steps=100, 
            idx = None,
            fixed_pars={},
            **kwargs):
        '''
        Experimental - can we fit everything at once?
        Does that even make sense?
        '''
        if idx is None: # all of them?
            idx = np.arange(self.n_voxels).tolist()
        elif isinstance(idx, int):
            idx = [idx]
        vx_bool = np.zeros(self.n_voxels, dtype=bool)
        vx_bool[idx] = True
        self.n_vx_to_fit = len(idx)
        self.fixed_pars = fixed_pars
        if not isinstance(self.fixed_pars, pd.DataFrame):
            self.fixed_pars = pd.DataFrame.from_dict(self.fixed_pars, orient='index').T.astype('float32')        
        self.h_fixed_pars = kwargs.pop('h_fixed_pars', {})        
        if not isinstance(self.h_fixed_pars, pd.DataFrame):
            self.h_fixed_pars = pd.DataFrame.from_dict(self.h_fixed_pars, orient='index').T.astype('float32')

        self.prep_for_fitting(**kwargs)
        self.n_params = len(self.model_labels)
        self.model_labels_inv = {v:k for k,v in self.model_labels.items()}
        self.h_prep_for_fitting(**kwargs)
        self.h_n_params = len(self.h_labels)
        self.h_labels_inv = {v:k for k,v in self.h_labels.items()}

        step_size = kwargs.pop('step_size', 0.0001) # rest of the kwargs go to "hmc_sample"                
        paradigm = kwargs.pop('paradigm', self.paradigm)
        
        y = self.data.values
        init_pars = format_parameters(init_pars)
        init_pars = init_pars.values.astype(np.float32) 
        h_init_pars = format_parameters(h_init_pars)
        h_init_pars = h_init_pars.values.astype(np.float32)

        # Clean the paradigm 
        paradigm_ = self.model.stimulus._clean_paradigm(paradigm)        
        
        # Define the prior in 'tf'
        @tf.function
        def log_prior_fn(parameters, h_parameters):
            # Log-prior function for the model
            p_out = 0.0            
            for p in self.priors_to_loop:
                p_out += tf.reduce_sum(self.p_prior[p].prior(parameters[:,self.model_labels[p]]))
            # Also apply the hierarchical priors
            for h in self.h_prior_to_apply.keys():
                if self.h_prior_to_apply[h]=='normal':
                    p_for_prior = parameters[:,self.model_labels[h]]
                    loc_for_prior = h_parameters[:,self.h_labels[h+'_loc']]
                    scale_for_prior = h_parameters[:,self.h_labels[h+'_scale']]
                    p_out = tf.reduce_sum(calculate_log_prob_gauss(
                        data=p_for_prior, loc=loc_for_prior, scale=scale_for_prior
                    ))
                    
            for h in self.h_priors_to_loop:
                p_out += tf.reduce_sum(self.h_prior[h].prior(h_parameters[:,self.h_labels[h]]))
            return p_out       

        # Calculating the likelihood
        # First lets make the function which returns the likelihood of the residuals
        residual_ln_likelihood_fn = self._create_residual_ln_likelihood_fn()
        @tf.function
        def log_posterior_fn(parameters, h_parameters):
            parameters = self.fix_update_fn(parameters)            
            h_parameters = self.h_fix_update_fn(h_parameters)
            par4pred = parameters[:,:self.n_model_params] # chop out any hyper / noise parameters
            predictions = self.model._predict(
                paradigm_[tf.newaxis, ...], par4pred[tf.newaxis, ...], None)     # Only include those parameters that are fed to the model
            residuals = y[:, vx_bool] - predictions[0]                        
            
            # -> rescale based on std...
            log_likelihood = residual_ln_likelihood_fn(parameters, residuals)
            log_prior = log_prior_fn(parameters, h_parameters)            
            # Return vector of length idx (optimize each chain separately)
            return tf.reduce_sum(log_likelihood + log_prior)
        
        # -> make sure we are in the correct dtype 
        p_initial_state = [tf.convert_to_tensor(init_pars[vx_bool,i], dtype=tf.float32) for i in range(self.n_params)]                          
        h_initial_state = [tf.convert_to_tensor(h_init_pars[:,i], dtype=tf.float32) for i in range(self.h_n_params)]        

        def target_log_prob_fn(*all_parameters):
            parameters = tf.stack(all_parameters[:self.n_params], axis=-1)
            h_parameters = tf.stack(all_parameters[self.n_params:], axis=-1)
            return log_posterior_fn(parameters, h_parameters)
        all_initial_state = [*p_initial_state, *h_initial_state]

        print('Lets run some checks with everything...')
        # quick test - does it work?
        initial_ll = target_log_prob_fn(*all_initial_state)
        print(f'initial_ll={initial_ll}')                    
        # Check the gradient with respect to each parameter
        with tf.GradientTape() as tape:
            tape.watch(all_initial_state)
            log_prob = target_log_prob_fn(*all_initial_state)
        gradients = tape.gradient(log_prob, all_initial_state)
        print('Using tape.gradient to check gradients w/respect to each parameter')
        for i, grad in enumerate(gradients[self.n_params:]):
            print(f' Hierarchical {self.h_labels_inv[i]}: {grad.numpy()}')            
        for i, grad in enumerate(gradients[:self.n_params]):
            print(f' Gradient for {self.model_labels_inv[i]}: {grad.numpy()}')

        # CALLING GILLES' "sample_hmc" from .utils.mcmc
        print(f"Starting NUTS sampling...")
        samples, stats = bprf_sample_NUTS(
            init_state = all_initial_state, 
            target_log_prob_fn=target_log_prob_fn, 
            unconstraining_bijectors=[*self.p_bijector_list, *self.h_bijector_list], 
            num_steps=num_steps, 
            # OTHER STUFF TO OPTIMIZE
            step_size=step_size, 
            **kwargs
            )                
        # stuff to save...        
        all_samples = tf.stack(samples[:self.n_params], axis=-1).numpy()
        # nsteps, n_voxels, n_params
        
        for ivx_loc,ivx_fit in enumerate(idx):
            estimated_p_dict = {}
            for i,p in enumerate(self.model_labels):
                estimated_p_dict[p] = all_samples[:,ivx_loc,i]
            for p,v in self.fixed_pars.items():
                estimated_p_dict[p] = estimated_p_dict[p]*0 + v.values
            self.mcmc_sampler[ivx_fit] = pd.DataFrame(estimated_p_dict)
        
        h_samples = tf.stack(samples[self.n_params:], axis=-1).numpy().squeeze()
        for h in self.h_labels:
            self.h_mcmc_sampler[h] = h_samples[:,self.h_labels[h]]
        self.h_mcmc_sampler = pd.DataFrame(self.h_mcmc_sampler)
        self.mcmc_stats = stats            

    def _create_residual_ln_likelihood_fn(self):
        # Calculating the likelihood
        if self.noise_method == 'fit_tdist':
            @tf.function
            def residual_ln_likelihood_fn(parameters, residuals):                    
                resid_ln_likelihood = calculate_log_prob_t(
                    data=residuals, scale=parameters[:,self.model_labels['noise_scale']], dof=parameters[:,self.model_labels['noise_dof']]
                )
                resid_ln_likelihood = tf.reduce_sum(resid_ln_likelihood, axis=0)       
                # Return vector of length idx (optimize each chain separately)
                return resid_ln_likelihood
        elif self.noise_method == 'fit_norm':
            # Assume residuals are normally distributed (loc=0.0)
            # Add the scale as an extra parameters to be fit             
            @tf.function
            def residual_ln_likelihood_fn(parameters, residuals):                    
                # -> rescale based on std...
                resid_ln_likelihood = calculate_log_prob_gauss_loc0(
                    data=residuals, scale=parameters[:,self.model_labels['noise_scale']],
                )
                resid_ln_likelihood = tf.reduce_sum(resid_ln_likelihood, axis=0)       
                return resid_ln_likelihood              
        
        elif self.noise_method == 'none': 
            # Do not fit the noise - assume it is normally distributed
            # -> calculate scale based on the standard deviation of the residuals 
            @tf.function
            def residual_ln_likelihood_fn(parameters, residuals):                    
                # [1] Use N(0, std)         
                residuals_std  = tf.math.reduce_std(residuals, axis=0)
                # -> rescale based on std...
                resid_ln_likelihood = calculate_log_prob_gauss_loc0(
                    data=residuals, scale=residuals_std,
                )
                resid_ln_likelihood = tf.reduce_sum(resid_ln_likelihood, axis=0)       
                return resid_ln_likelihood     
        
        return residual_ln_likelihood_fn