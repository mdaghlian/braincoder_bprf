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
        self.h_gp_function = {} # Using gaussian process?
        self.h_prior = {}
        self.h_mcmc_sampler = {}
        self.h_mcmc_summary = None
        self.h_mcmc_mean = None
        # MAP
        self.h_MAP_parameters = None

                
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
        elif h_prior_to_apply=='gp_dists':
            # Gaussian process based on geodesic distance
            # -> Gaussian process generates a covariance matrix
            self.h_labels[f'{pid}_gp_lengthscale'] = len(self.h_labels) 
            self.h_labels[f'{pid}_gp_variance'] = len(self.h_labels)
            self.h_gp_function[pid] = GPdists(
                dists=kwargs.pop('dists'),
                **kwargs
            )

            self.h_add_bijector(pid=f'{pid}_gp_lengthscale', bijector_type=tfb.Softplus())
            self.h_add_bijector(pid=f'{pid}_gp_variance', bijector_type=tfb.Softplus())

            # self.h_add_bijector(pid=f'{pid}_gp_lengthscale', bijector_type=tfb.Exp())
            # self.h_add_bijector(pid=f'{pid}_gp_variance', bijector_type=tfb.Exp())

            # [3] add the priors for the new parameters - make them very broad...
            self.h_add_prior(
                pid=f'{pid}_gp_lengthscale', 
                prior_type='HalfNormal', 
                distribution=tfd.HalfNormal(scale=100)
                )
            self.h_add_prior(
                pid=f'{pid}_gp_variance', 
                prior_type='HalfNormal', 
                distribution=tfd.HalfNormal(scale=100)
                )
        
        elif h_prior_to_apply=='gp_dists_full':
            # Gaussian process based on geodesic distance
            # include mean + nugget
            # -> Gaussian process generates a covariance matrix
            self.h_labels[f'{pid}_gp_lengthscale'] = len(self.h_labels) 
            self.h_labels[f'{pid}_gp_variance'] = len(self.h_labels)
            self.h_labels[f'{pid}_gp_mean'] = len(self.h_labels)
            self.h_labels[f'{pid}_gp_nugget'] = len(self.h_labels)
            self.h_gp_function[pid] = GPdists(
                dists=kwargs.pop('dists'),
                **kwargs
            )
            self.h_add_bijector(pid=f'{pid}_gp_lengthscale', bijector_type=tfb.Softplus())
            self.h_add_bijector(pid=f'{pid}_gp_variance', bijector_type=tfb.Softplus())
            self.h_add_bijector(pid=f'{pid}_gp_mean', bijector_type=tfb.Identity())
            self.h_add_bijector(pid=f'{pid}_gp_nugget', bijector_type=tfb.Softplus())

            # [3] add the priors for the new parameters
            self.h_add_prior(
                pid=f'{pid}_gp_lengthscale', 
                prior_type='HalfNormal', 
                distribution=tfd.HalfNormal(scale=20)
                )
            self.h_add_prior(
                pid=f'{pid}_gp_variance', 
                prior_type='HalfNormal', 
                distribution=tfd.HalfNormal(scale=10)
                )
            self.h_add_prior(
                pid=f'{pid}_gp_nugget', 
                prior_type='HalfNormal', 
                distribution=tfd.HalfNormal(scale=5)
                )
            self.h_add_prior(
                pid=f'{pid}_gp_mean', 
                prior_type='none', 
                )                
        elif h_prior_to_apply=='gp_dists_m':
            # Gaussian process based on geodesic distance
            # include... everything
            # Need to have premade the GPDistsM object 
            self.h_gp_function[pid] = kwargs.pop('gp_obj')
            current_n_labels = len(self.h_labels)
            new_h_labels = []
            for k in self.h_gp_function[pid].pids_inv.keys():
                # Add the keys + index
                self.h_labels[f'{pid}_{k}'] = self.h_gp_function[pid].pids_inv[k]+current_n_labels
                new_h_labels.append(f'{pid}_{k}')
            # Bijectors
            for k in new_h_labels:
                if any([substr in k for substr in ['_l', '_v', '_nugget']]):
                    self.h_add_bijector(pid=k, bijector_type=tfb.Softplus())        
                else:
                    self.h_add_bijector(pid=k, bijector_type=tfb.Identity())        

            for k in new_h_labels:
                if any([substr in k for substr in ['_l', '_v', '_nugget']]):
                    self.h_add_prior(
                        pid=k, 
                        prior_type='HalfNormal', 
                        distribution=tfd.HalfNormal(scale=20)
                        )
                else:
                    self.h_add_prior(pid=k, prior_type='none')

    
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
            # Remove any entry of h_fixed pars which is not in h_labels
            k_check = list(self.h_fixed_pars.keys())
            for k in k_check:
                if k not in self.h_labels.keys():
                    self.h_fixed_pars.pop(k)
            if not isinstance(self.h_fixed_pars, pd.DataFrame):
                self.h_fixed_pars = pd.DataFrame.from_dict(self.h_fixed_pars, orient='index').T.astype('float32')        

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
        self.h_labels_inv = {v:k for k,v in self.h_labels.items()}

    @tf.function
    def _h_bprf_transform_parameters_forward(self, h_parameters):
        # Loop through parameters & bijectors (forward)
        h_out = [
            self.h_bijector_list[i].forward(h_parameters[:,i]) for i in range(self.h_n_params)
            ]
        return tf.stack(h_out, axis=-1)

    @tf.function
    def _h_bprf_transform_parameters_backward(self, h_parameters):
        # Loop through parameters & bijectors (backward)
        h_out = [
            self.h_bijector_list[i].inverse(h_parameters[:,i]) for i in range(self.h_n_params)
            ]
        return tf.stack(h_out, axis=-1)
    
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
        self.n_inducers = kwargs.pop('n_inducers', None)
        if idx is None: # all of them?
            idx = np.arange(self.n_voxels).tolist()
        elif isinstance(idx, int):
            idx = [idx]
        vx_bool = np.zeros(self.n_voxels, dtype=bool)
        vx_bool[idx] = True
        self.n_vx_to_fit = len(idx)
        self.fixed_pars = fixed_pars
        self.h_fixed_pars = kwargs.pop('h_fixed_pars', {})        
        self.prep_for_fitting(**kwargs)
        self.n_params = len(self.model_labels)
        self.model_labels_inv = {v:k for k,v in self.model_labels.items()}
        self.h_prep_for_fitting(**kwargs)
        self.h_n_params = len(self.h_labels)
        self.h_labels_inv = {v:k for k,v in self.h_labels.items()}

        step_size = kwargs.pop('step_size', 0.0001) # rest of the kwargs go to "hmc_sample"                
        self.total_n_params = self.n_params + self.h_n_params
        step_size = [tf.constant(step_size, np.float32) for _ in range(self.total_n_params)]
        paradigm = kwargs.pop('paradigm', self.paradigm)
        
        y = self.data.values
        init_pars = self.sort_parameters(init_pars)
        init_pars = format_parameters(init_pars)
        init_pars = init_pars.values.astype(np.float32) 
        h_init_pars = self.sort_h_parameters(h_init_pars)
        h_init_pars = format_parameters(h_init_pars)
        h_init_pars = h_init_pars.values.astype(np.float32)

        # Clean the paradigm 
        paradigm_ = self.model.stimulus._clean_paradigm(paradigm)        
        
        # Define the prior in 'tf'
        log_prior_fn = self._create_log_prior_fn()    

        # Calculating the likelihood
        # First lets make the function which returns the likelihood of the residuals
        residual_ln_likelihood_fn = self._create_residual_ln_likelihood_fn()
        
        @tf.function
        def log_posterior_fn(parameters, h_parameters):
            # Not needed - it does this under the hood with HMC
            # parameters = self._bprf_transform_parameters_forward(parameters)
            # h_parameters = self._h_bprf_transform_parameters_forward(h_parameters)
            parameters = self.fix_update_fn(parameters)            
            h_parameters = self.h_fix_update_fn(h_parameters)
            par4pred = parameters[:,:self.n_model_params] # chop out any hyper / noise parameters
            predictions = self.model._predict(
                paradigm_[tf.newaxis, ...], par4pred[tf.newaxis, ...], None)     # Only include those parameters that are fed to the model
            residuals = y[:, vx_bool] - predictions[0]                        
            log_likelihood = residual_ln_likelihood_fn(parameters, residuals)            
            log_prior = log_prior_fn(parameters, h_parameters)            
            return tf.reduce_sum(log_prior + log_likelihood)        
        
        # -> make sure we are in the correct dtype 
        p_initial_state = [tf.convert_to_tensor(init_pars[vx_bool,i], dtype=tf.float32, name=n) for i,n in enumerate(self.model_labels.keys())]                          
        h_initial_state = [tf.convert_to_tensor(h_init_pars[:,i], dtype=tf.float32, name=n) for i,n in enumerate(self.h_labels)]        

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
            try: 
                print(f' Hierarchical {self.h_labels_inv[i]}: {grad.numpy()}')            
            except:
                print(f'No gradient for {self.h_labels_inv[i]}')            
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
                estimated_p_dict[p] = estimated_p_dict[p]*0 + v[ivx_fit]
            self.mcmc_sampler[ivx_fit] = pd.DataFrame(estimated_p_dict)
        
        h_samples = tf.stack(samples[self.n_params:], axis=-1).numpy().squeeze()
        print(h_samples.shape)
        for h in self.h_labels:
            print(self.h_labels[h])
            self.h_mcmc_sampler[h] = h_samples[:,self.h_labels[h]]
        self.h_mcmc_sampler = pd.DataFrame(self.h_mcmc_sampler)
        self.mcmc_stats = stats            

    def get_mcmc_summary(self, burnin=100, pc_range=25):
        burnin = 100
        pc_range = 25
        bpars = {}
        bpars_m = {}
        for p in list(self.model_labels.keys()): 
            m = []
            q1 = []
            q2 = []
            uc = []
            for idx in range(self.n_voxels):
                this_p = self.mcmc_sampler[idx][p][burnin:].to_numpy()
                m.append(np.percentile(this_p,50))
                tq1 = np.percentile(this_p, pc_range)
                tq2 = np.percentile(this_p, 100-pc_range)
                tuc = tq2 - tq1
                
                q1.append(tq1)
                q2.append(tq2)
                uc.append(tuc)
            bpars_m[p] = np.array(m)
            bpars[f'm_{p}'] = np.array(m)
            bpars[f'q1_{p}'] = np.array(q1)
            bpars[f'q2_{p}'] = np.array(q2)
            bpars[f'uc_{p}'] = np.array(uc)
            
        self.mcmc_summary = pd.DataFrame(bpars)
        self.mcmc_mean = pd.DataFrame(bpars_m)

        hpars = {}
        hpars_m = {}
        for p in list(self.h_labels.keys()): 

            this_p = self.h_mcmc_sampler[p][burnin:].to_numpy()
            m = np.percentile(this_p,50)
            tq1 = np.percentile(this_p, pc_range)
            tq2 = np.percentile(this_p, 100-pc_range)
            tuc = tq2 - tq1
            
            hpars_m[p] = np.array(m)
            hpars[f'm_{p}'] = np.array(m)
            hpars[f'q1_{p}'] = np.array(q1)
            hpars[f'q2_{p}'] = np.array(q2)
            hpars[f'uc_{p}'] = np.array(uc)
            
        self.h_mcmc_summary = pd.DataFrame(hpars)
        self.h_mcmc_mean = pd.DataFrame.from_dict(hpars_m, orient='index').T.astype('float32')

    def fit_MAP_hier(self,
                init_pars=None,
                h_init_pars=None,
                num_steps=100, 
                idx = None,
                fixed_pars={},
                **kwargs):
        '''
        Maximum a posteriori (MAP) optimization for hierarchical model.
        Finds the parameter values that maximize the posterior distribution.
        '''
        optimizer_type = kwargs.get('optimizer', 'adam' )
        self.n_inducers = kwargs.pop('n_inducers', None)
        adam_kwargs = kwargs.pop('adam_kwargs', {})
        if idx is None: # all of them?
            idx = np.arange(self.n_voxels).tolist()
        elif isinstance(idx, int):
            idx = [idx]
        vx_bool = np.zeros(self.n_voxels, dtype=bool)
        vx_bool[idx] = True
        self.n_vx_to_fit = len(idx)
        self.fixed_pars = fixed_pars
        self.h_fixed_pars = kwargs.pop('h_fixed_pars', {})                
        self.prep_for_fitting(**kwargs)
        self.n_params = len(self.model_labels)
        self.h_prep_for_fitting(**kwargs)
        self.h_n_params = len(self.h_labels)

        self.total_n_params = self.n_params + self.h_n_params
        paradigm = kwargs.pop('paradigm', self.paradigm)
        
        y = self.data.values
        init_pars = self.sort_parameters(init_pars)
        init_pars = format_parameters(init_pars)
        init_pars = init_pars.values.astype(np.float32) 
        h_init_pars = self.sort_h_parameters(h_init_pars)
        h_init_pars = format_parameters(h_init_pars)
        h_init_pars = h_init_pars.values.astype(np.float32)

        # Clean the paradigm 
        paradigm_ = self.model.stimulus._clean_paradigm(paradigm)  

        # Define the prior in 'tf'
        log_prior_fn = self._create_log_prior_fn()
        
        # Calculating the likelihood
        # -> based on our 'noise_method'
        residual_ln_likelihood_fn = self._create_residual_ln_likelihood_fn()

        # Now create the log_posterior_fn
        @tf.function
        def log_posterior_fn(parameters, h_parameters):
            parameters = self._bprf_transform_parameters_forward(parameters)
            parameters = self.fix_update_fn(parameters)            
            h_parameters = self._h_bprf_transform_parameters_forward(h_parameters)
            h_parameters = self.h_fix_update_fn(h_parameters)
            par4pred = parameters[:,:self.n_model_params] # chop out any hyper / noise parameters
            predictions = self.model._predict(
                paradigm_[tf.newaxis, ...], par4pred[tf.newaxis, ...], None)     # Only include those parameters that are fed to the model
            residuals = y[:, vx_bool] - predictions[0]                        
            log_likelihood = residual_ln_likelihood_fn(parameters, residuals)            
            log_prior = log_prior_fn(parameters, h_parameters)            
            return tf.reduce_sum(log_prior + log_likelihood)

        # -> make sure we are in the correct dtype
        # Convert initial parameters to tensors
        p_state_tensors = [
            tf.convert_to_tensor(init_pars[vx_bool, i], dtype=tf.float32, name=name) 
            for i, name in enumerate(self.model_labels.keys())
        ]

        h_state_tensors = [
            tf.convert_to_tensor(h_init_pars[:, i], dtype=tf.float32, name=name) 
            for i, name in enumerate(self.h_labels)
        ]

        # Transform initial states using bijectors
        p_unconstrained = [bijector.inverse(state) for bijector, state in zip(self.p_bijector_list, p_state_tensors)]
        h_unconstrained = [bijector.inverse(state) for bijector, state in zip(self.h_bijector_list, h_state_tensors)]

        print("Starting MAP optimization...")

        # Initialize optimization variables
        p_opt_vars = [tf.Variable(state, name=self.model_labels_inv[i]) for i,state in enumerate(p_unconstrained)]
        h_opt_vars = [tf.Variable(state, name=self.h_labels_inv[i]) for i,state in enumerate(h_unconstrained)]        

        all_opt_vars = [*p_opt_vars, *h_opt_vars]

        @tf.function
        def neg_log_posterior_fn():
            return -log_posterior_fn(
                tf.stack(all_opt_vars[:self.n_params], axis=-1), 
                tf.stack(all_opt_vars[self.n_params:], axis=-1)
                )
        
        # **** quick test **** 
        initial_ll = neg_log_posterior_fn()
        print(f'initial neg ll={initial_ll}')                    
        # Check the gradient with respect to each parameter
        with tf.GradientTape() as tape:
            loss = neg_log_posterior_fn()
        gradients = tape.gradient(loss, all_opt_vars)
        print('Using tape.gradient to check gradients w/respect to each parameter')
        for i, grad in enumerate(gradients[self.n_params:]):
            try: 
                print(f' Hierarchical {self.h_labels_inv[i]}: {grad.numpy()}')            
            except:
                print(f'No gradient for {self.h_labels_inv[i]}')            
        for i, grad in enumerate(gradients[:self.n_params]):
            print(f' Gradient for {self.model_labels_inv[i]}: {grad.numpy()}')
        # **** **** **** **** 

        optimizer = tf.optimizers.Adam(**adam_kwargs)
        # Optimization loop with tqdm progress bar
        progress_bar = tqdm(tf.range(num_steps), desc="MAP Optimization")
        for step in progress_bar:
            with tf.GradientTape() as tape:
                loss = neg_log_posterior_fn()
            gradients = tape.gradient(loss, all_opt_vars)
            optimizer.apply_gradients(zip(gradients, all_opt_vars))
            progress_bar.set_description(f"MAP Optimization, Loss: {loss.numpy():.4f}")        

        # Extract optimized parameters
        # -> transform parameters forward after fitting
        p_opt_vars = all_opt_vars[:self.n_params]
        p_opt_vars = [self.p_bijector_list[i](ov).numpy() for i,ov in enumerate(p_opt_vars)]

        h_opt_vars = all_opt_vars[self.n_params:]
        h_opt_vars = [self.h_bijector_list[i](ov).numpy() for i,ov in enumerate(h_opt_vars)]
        
        df_list = []
        # Save optimized parameters
        for ivx_loc,ivx_fit in enumerate(idx):
            estimated_p_dict = {}
            for i,p in enumerate(self.model_labels):
                estimated_p_dict[p] = p_opt_vars[i][ivx_loc]
            for p,v in self.fixed_pars.items():
                estimated_p_dict[p] = estimated_p_dict[p]*0 + v[ivx_fit]
            
            df = pd.DataFrame(estimated_p_dict, index=[ivx_fit]) # use map_sampler instead of mcmc_sampler
            df_list.append(df)
        self.MAP_parameters = pd.concat(df_list).reindex(idx)
        
        hdf = {}
        for i,h in enumerate(self.h_labels):
            hdf[h] = h_opt_vars[self.h_labels[h]]
        self.h_MAP_parameters = pd.DataFrame(hdf)
        print('MAP optimization finished.')    
    
    def _create_log_prior_fn(self):
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
                    p_out += tf.reduce_sum(calculate_log_prob_gauss(
                        data=p_for_prior, loc=loc_for_prior, scale=scale_for_prior
                    ))
                    
                elif self.h_prior_to_apply[h]=='gp_dists':
                    param_values = parameters[:, self.model_labels[h]] # Values of parameter 'h' for vertices being fit
                    gp_lengthscale = h_parameters[:, self.h_labels[f'{h}_gp_lengthscale']] # Current value of GP lengthscale hyperparameter
                    gp_variance = h_parameters[:, self.h_labels[f'{h}_gp_variance']] # Current value of GP variance hyperparameter
                    p_out += self.h_gp_function[h].return_log_prob(
                        gp_lengthscale=gp_lengthscale, 
                        gp_variance=gp_variance, 
                        parameter=param_values, 
                        n_inducers=self.n_inducers,
                    )
                elif self.h_prior_to_apply[h]=='gp_dists_full':
                    param_values = parameters[:, self.model_labels[h]] # Values of parameter 'h' for vertices being fit
                    gp_lengthscale  = h_parameters[:, self.h_labels[f'{h}_gp_lengthscale']] # Current value of GP lengthscale hyperparameter
                    gp_variance     = h_parameters[:, self.h_labels[f'{h}_gp_variance']] # Current value of GP variance hyperparameter
                    gp_mean         = h_parameters[:, self.h_labels[f'{h}_gp_mean']] # Current value of GP variance hyperparameter
                    gp_nugget       = h_parameters[:, self.h_labels[f'{h}_gp_nugget']] # Current value of GP variance hyperparameter
                    p_out += self.h_gp_function[h].return_log_prob(
                        parameter=param_values, gp_lengthscale=gp_lengthscale, gp_variance=gp_variance, 
                        gp_mean=gp_mean, gp_nugget=gp_nugget,
                        n_inducers=self.n_inducers,
                    )
                elif self.h_prior_to_apply[h]=='gp_dists_m':
                    param_values = parameters[:, self.model_labels[h]] # Values of parameter 'h' for vertices being fit
                    gpkwargs = {}
                    for k in self.h_labels:
                        if h in k:
                            gpkwargs[k.split(f'{h}_')[-1]] = h_parameters[:, self.h_labels[k]]
                    p_out += self.h_gp_function[h].return_log_prob(
                        parameter=param_values, n_inducers=self.n_inducers,
                        **gpkwargs,
                    )

                    
            for h in self.h_priors_to_loop:
                p_out += tf.reduce_sum(self.h_prior[h].prior(h_parameters[:,self.h_labels[h]]))
            return p_out     
        return log_prior_fn
    
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
        elif self.noise_method == 'fit_normal':
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
        
        else: # self.noise_method == 'none': 
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
    
    def sort_h_parameters(self, h_parameters):
        # Make sure pd.Dataframe parameters are in order
        h_parameters = reorder_dataframe_columns(pd_to_fix=h_parameters, dict_index=self.h_labels)
        return h_parameters

    def fit_mcmc_hier_GP_only(self,
                pid,            # Which parameter is this GP associated with 
                pid_pars,       # The values of these parameters (e.g., obtained with a basic grid fit)
                h_init_pars,    
                num_steps=100, 
                idx = None,
                **kwargs):
        '''
        Maximum a posteriori (MAP) optimization for hierarchical model. Fitting the GP only
        The idea is to:
        [1] Do some basic traditional parameter fitting on the data (e.g., pRF, nCSF)
        [2] You expect that some of these parameters vary spatially
        - so you want to apply a GP prior
        - But you don't know how to set the hyperparameters (lengthscale, variance, etc)
        - So you fit these on the output of [1] 
        - There is no interaction with the data
        [3] You fit your hyperparameters **here** and then fix them for subsequent fitting
        - now using your "informed" prior

        A sort of "empirical bayesian approach

        This is an alternative to fitting everything at the same time. Which is also an option in 
        '''
        n_inducers = kwargs.pop('n_inducers', None)
        if idx is None: # all of them?
            idx = np.arange(self.n_voxels).tolist()
        elif isinstance(idx, int):
            idx = [idx]
        vx_bool = np.zeros(self.n_voxels, dtype=bool)
        vx_bool[idx] = True
        self.n_vx_to_fit = len(idx)
        self.h_fixed_pars = kwargs.pop('h_fixed_pars', {})                
        self.n_params = len(self.model_labels)
        self.h_prep_for_fitting(**kwargs)
        self.h_n_params = len(self.h_labels)
        step_size = kwargs.pop('step_size', 0.0001) # rest of the kwargs go to "hmc_sample"                
        step_size = [tf.constant(step_size, np.float32) for _ in range(self.h_n_params)]        
        
        pid_pars = pid_pars.values.astype(np.float32) # These stay the same...
        pid_pars = tf.convert_to_tensor(pid_pars[vx_bool], dtype=tf.float32, name=pid) 
        h_init_pars = self.sort_h_parameters(h_init_pars)
        h_init_pars = format_parameters(h_init_pars)
        h_init_pars = h_init_pars.values.astype(np.float32)

        # Define the log prior function
        @tf.function
        def log_prior_fn(h_parameters):
            p_out = 0.0
            for h in self.h_priors_to_loop:
                p_out += tf.reduce_sum(self.h_prior[h].prior(h_parameters[:,self.h_labels[h]]))            
            return p_out 

        # Now create the log_posterior_fn
        @tf.function
        def log_posterior_fn(h_parameters):
            h_parameters = self._h_bprf_transform_parameters_forward(h_parameters)
            h_parameters = self.h_fix_update_fn(h_parameters)
            log_prior = log_prior_fn(h_parameters)            
            # Apply gp to parameters
            if self.h_prior_to_apply[pid]=='gp_dists':
                gp_lengthscale = h_parameters[:, self.h_labels[f'{pid}_gp_lengthscale']] # Current value of GP lengthscale hyperparameter
                gp_variance = h_parameters[:, self.h_labels[f'{pid}_gp_variance']] # Current value of GP variance hyperparameter
                gp_likelihood = self.h_gp_function[pid].return_log_prob(
                    parameter=pid_pars, gp_lengthscale=gp_lengthscale, gp_variance=gp_variance, 
                    n_inducers=n_inducers,
                )
            elif self.h_prior_to_apply[pid]=='gp_dists_full':
                gp_lengthscale  = h_parameters[:, self.h_labels[f'{pid}_gp_lengthscale']] # Current value of GP lengthscale hyperparameter
                gp_variance     = h_parameters[:, self.h_labels[f'{pid}_gp_variance']] # Current value of GP variance hyperparameter
                gp_mean         = h_parameters[:, self.h_labels[f'{pid}_gp_mean']] # Current value of GP variance hyperparameter
                gp_nugget       = h_parameters[:, self.h_labels[f'{pid}_gp_nugget']] # Current value of GP variance hyperparameter
                gp_likelihood = self.h_gp_function[pid].return_log_prob(
                    parameter=pid_pars, gp_lengthscale=gp_lengthscale, gp_variance=gp_variance, 
                    gp_mean=gp_mean, gp_nugget=gp_nugget, n_inducers=n_inducers,
                )
            elif self.h_prior_to_apply[pid]=='gp_dists_m':
                gpkwargs = {}
                for k in self.h_labels:
                    if pid in k:
                        gpkwargs[k.split(f'{pid}_')[-1]] = h_parameters[:, self.h_labels[k]]
                gp_likelihood = self.h_gp_function[pid].return_log_prob(
                    parameter=pid_pars, n_inducers=self.n_inducers,
                    **gpkwargs,
                )                
            else:
                raise AssertionError
            return tf.reduce_sum(log_prior + gp_likelihood)

        # -> make sure we are in the correct dtype
        h_initial_state = [
            tf.convert_to_tensor(h_init_pars[:,i], dtype=tf.float32, name=n) for i,n in enumerate(self.h_labels)]        
        
        def target_log_prob_fn(*h_parameters):
            return log_posterior_fn(tf.stack(h_parameters, axis=-1))

        # **** quick test **** 

        print('Lets run some checks with everything...')
        # quick test - does it work?
        initial_ll = target_log_prob_fn(*h_initial_state)
        print(f'initial_ll={initial_ll}')                    
        # Check the gradient with respect to each parameter
        with tf.GradientTape() as tape:
            tape.watch(h_initial_state)
            log_prob = target_log_prob_fn(*h_initial_state)
        gradients = tape.gradient(log_prob, h_initial_state)
        print('Using tape.gradient to check gradients w/respect to each parameter')
        for i, grad in enumerate(gradients):
            try: 
                print(f' Hierarchical {self.h_labels_inv[i]}: {grad.numpy()}')            
            except:
                print(f'No gradient for {self.h_labels_inv[i]}')            
    
        # **** **** **** **** 

        # CALLING GILLES' "sample_hmc" from .utils.mcmc
        print(f"Starting NUTS sampling...")
        samples, stats = bprf_sample_NUTS(
            init_state = h_initial_state, 
            target_log_prob_fn=target_log_prob_fn, 
            unconstraining_bijectors=self.h_bijector_list, 
            num_steps=num_steps, 
            # OTHER STUFF TO OPTIMIZE
            step_size=step_size, 
            **kwargs
            )                
        # stuff to save...        
        
        h_samples = tf.stack(samples, axis=-1).numpy().squeeze()
        for h in self.h_labels:
            self.h_mcmc_sampler[h] = h_samples[:,self.h_labels[h]]
        self.h_mcmc_sampler = pd.DataFrame(self.h_mcmc_sampler)
        self.mcmc_stats = stats                

    def fit_MAP_hier_GP_only(self,
                pid,            # Which parameter is this GP associated with 
                pid_pars,       # The values of these parameters (e.g., obtained with a basic grid fit)
                h_init_pars,    
                num_steps=100, 
                idx = None,
                **kwargs):
        '''
        Maximum a posteriori (MAP) optimization for hierarchical model. Fitting the GP only
        The idea is to:
        [1] Do some basic traditional parameter fitting on the data (e.g., pRF, nCSF)
        [2] You expect that some of these parameters vary spatially
        - so you want to apply a GP prior
        - But you don't know how to set the hyperparameters (lengthscale, variance, etc)
        - So you fit these on the output of [1] 
        - There is no interaction with the data
        [3] You fit your hyperparameters **here** and then fix them for subsequent fitting
        - now using your "informed" prior

        A sort of "empirical bayesian approach

        This is an alternative to fitting everything at the same time. Which is also an option in 
        '''
        optimizer_type = kwargs.get('optimizer', 'adam' )
        adam_kwargs = kwargs.pop('adam_kwargs', {})
        self.n_inducers = kwargs.pop('n_inducers', None)
        if idx is None: # all of them?
            idx = np.arange(self.n_voxels).tolist()
        elif isinstance(idx, int):
            idx = [idx]
        vx_bool = np.zeros(self.n_voxels, dtype=bool)
        vx_bool[idx] = True
        self.n_vx_to_fit = len(idx)
        self.h_fixed_pars = kwargs.pop('h_fixed_pars', {})                
        self.n_params = len(self.model_labels)
        self.h_prep_for_fitting(**kwargs)
        self.h_n_params = len(self.h_labels)
        
        pid_pars = pid_pars.values.astype(np.float32) # These stay the same...
        pid_pars = tf.convert_to_tensor(pid_pars[vx_bool], dtype=tf.float32, name=pid) 
        h_init_pars = self.sort_h_parameters(h_init_pars)
        h_init_pars = format_parameters(h_init_pars)
        h_init_pars = h_init_pars.values.astype(np.float32)

        # Define the log prior function
        @tf.function
        def log_prior_fn(h_parameters):
            p_out = 0.0
            for h in self.h_priors_to_loop:
                p_out += tf.reduce_sum(self.h_prior[h].prior(h_parameters[:,self.h_labels[h]]))            
            return p_out 
        print(self.h_prior_to_apply[pid])

        # Now create the log_posterior_fn
        @tf.function
        def log_posterior_fn(h_parameters):
            h_parameters = self._h_bprf_transform_parameters_forward(h_parameters)
            h_parameters = self.h_fix_update_fn(h_parameters)
            log_prior = log_prior_fn(h_parameters)            
            # Apply gp to parameters
            if self.h_prior_to_apply[pid]=='gp_dists':
                gp_lengthscale = h_parameters[:, self.h_labels[f'{pid}_gp_lengthscale']] # Current value of GP lengthscale hyperparameter
                gp_variance = h_parameters[:, self.h_labels[f'{pid}_gp_variance']] # Current value of GP variance hyperparameter
                gp_likelihood = self.h_gp_function[pid].return_log_prob(
                    parameter=pid_pars, gp_lengthscale=gp_lengthscale, gp_variance=gp_variance, 
                    n_inducers=self.n_inducers,
                )
            elif self.h_prior_to_apply[pid]=='gp_dists_full':
                gp_lengthscale  = h_parameters[:, self.h_labels[f'{pid}_gp_lengthscale']] # Current value of GP lengthscale hyperparameter
                gp_variance     = h_parameters[:, self.h_labels[f'{pid}_gp_variance']] # Current value of GP variance hyperparameter
                gp_mean         = h_parameters[:, self.h_labels[f'{pid}_gp_mean']] # Current value of GP variance hyperparameter
                gp_nugget       = h_parameters[:, self.h_labels[f'{pid}_gp_nugget']] # Current value of GP variance hyperparameter
                gp_likelihood = self.h_gp_function[pid].return_log_prob(
                    parameter=pid_pars, gp_lengthscale=gp_lengthscale, gp_variance=gp_variance, 
                    gp_mean=gp_mean, gp_nugget=gp_nugget, n_inducers=self.n_inducers,
                )
            elif self.h_prior_to_apply[pid]=='gp_dists_m':
                gpkwargs = {}
                for k in self.h_labels:
                    if pid in k:
                        gpkwargs[k.split(f'{pid}_')[-1]] = h_parameters[:, self.h_labels[k]]
                gp_likelihood = self.h_gp_function[pid].return_log_prob(
                    parameter=pid_pars, n_inducers=self.n_inducers,
                    **gpkwargs,
                )                     
            else:
                raise AssertionError
            return tf.reduce_sum(log_prior + gp_likelihood)

        # -> make sure we are in the correct dtype
        h_state_tensors = [
            tf.convert_to_tensor(h_init_pars[:, i], dtype=tf.float32, name=name) 
            for i, name in enumerate(self.h_labels)
        ]

        # Transform initial states using bijectors
        h_unconstrained = [bijector.inverse(state) for bijector, state in zip(self.h_bijector_list, h_state_tensors)]
        print("Starting MAP optimization...")
        # Initialize optimization variables
        h_opt_vars = [tf.Variable(state, name=self.h_labels_inv[i]) for i,state in enumerate(h_unconstrained)]        

        @tf.function
        def neg_log_posterior_fn():
            return -log_posterior_fn(tf.stack(h_opt_vars, axis=-1))
        
        # **** quick test **** 
        initial_ll = neg_log_posterior_fn()
        print(f'initial neg ll={initial_ll}')                    
        # Check the gradient with respect to each parameter
        with tf.GradientTape() as tape:
            loss = neg_log_posterior_fn()
        gradients = tape.gradient(loss, h_opt_vars)
        print('Using tape.gradient to check gradients w/respect to each parameter')
        for i, grad in enumerate(gradients):
            try: 
                print(f' Hierarchical {self.h_labels_inv[i]}: {grad.numpy()}')            
            except:
                print(f'No gradient for {self.h_labels_inv[i]}')            
    
        # **** **** **** **** 
        # Define early stopping parameters
        patience = 10  # Number of steps with no improvement before stopping
        min_delta = 1e-9  # Minimum change in loss to be considered an improvement
        best_loss = float("inf")
        patience_counter = 0

        optimizer = tf.optimizers.Adam(**adam_kwargs)
        progress_bar = tqdm(tf.range(num_steps), desc="MAP Optimization")

        for step in progress_bar:
            with tf.GradientTape() as tape:
                loss = neg_log_posterior_fn()
            gradients = tape.gradient(loss, h_opt_vars)
            optimizer.apply_gradients(zip(gradients, h_opt_vars))

            # # Early stopping check
            # if loss.numpy() < best_loss - min_delta:
            #     best_loss = loss.numpy()
            #     patience_counter = 0  # Reset patience if improvement is found
            # else:
            #     patience_counter += 1  # Increment patience counter

            # if patience_counter >= patience:
            #     print(f"Early stopping at step {step}, Loss: {best_loss:.4f}")
            #     break

            progress_bar.set_description(f"MAP Optimization, Loss: {loss.numpy():.4f}")


        # Extract optimized parameters
        # -> transform parameters forward after fitting
        h_opt_vars = [self.h_bijector_list[i](ov).numpy() for i,ov in enumerate(h_opt_vars)]
        
        hdf = {}
        for i,h in enumerate(self.h_labels):
            hdf[h] = h_opt_vars[self.h_labels[h]]
        self.h_MAP_parameters = pd.DataFrame(hdf)
        print('MAP optimization finished.')    

    def fit_grid_hier_GP_only(self,
                pid,            # Which parameter is this GP associated with 
                pid_pars,       # The values of these parameters (e.g., obtained with a basic grid fit)
                grids={},
                idx = None,
                **kwargs):
        '''
        Maximum a posteriori (MAP) optimization for hierarchical model using a grid search over
        specified hyperparameters (e.g., GP lengthscale, variance, etc.). Instead of continuously
        optimizing these hyperparameters with gradient descent, we instead evaluate the objective
        over a grid of candidate values and select the combination that minimizes the negative log
        posterior. The remaining parts of the hierarchical model parameters are left at their
        initial (or previously computed) values.
        '''
        # Setup voxel indices
        if idx is None:  # Use all voxels
            idx = np.arange(self.n_voxels).tolist()
        elif isinstance(idx, int):
            idx = [idx]
        vx_bool = np.zeros(self.n_voxels, dtype=bool)
        vx_bool[idx] = True
        self.n_vx_to_fit = len(idx)
        self.h_fixed_pars = kwargs.pop('h_fixed_pars', {})
        self.n_params = len(self.model_labels)
        self.h_prep_for_fitting(**kwargs)
        self.h_n_params = len(self.h_labels)

        # Prepare parameter tensors
        pid_pars = pid_pars.values.astype(np.float32)
        pid_pars = tf.convert_to_tensor(pid_pars[vx_bool], dtype=tf.float32, name=pid)
        
        
        g_list = [grids[i] for i in grids.keys()]
        # Meshgrid 
        mg_list = np.meshgrid(*g_list)
        g_list = [i.flatten() for i in mg_list]
        g_list = np.vstack(g_list).T

        # Define the log prior function (same as in MAP fit)
        @tf.function
        def log_prior_fn(h_parameters):
            p_out = 0.0
            for h in self.h_priors_to_loop:
                p_out += tf.reduce_sum(self.h_prior[h].prior(h_parameters[:, self.h_labels[h]]))
            return p_out

        # Define the log posterior function (including GP likelihood)
        @tf.function
        def log_posterior_fn(h_parameters):
            # h_parameters = self._h_bprf_transform_parameters_forward(h_parameters)
            h_parameters = self.h_fix_update_fn(h_parameters)
            log_prior = log_prior_fn(h_parameters)
            # Apply GP to parameters based on the type specified
            if self.h_prior_to_apply[pid] == 'gp_dists':
                gp_lengthscale = h_parameters[:, self.h_labels[f'{pid}_gp_lengthscale']]
                gp_variance = h_parameters[:, self.h_labels[f'{pid}_gp_variance']]
                gp_likelihood = self.h_gp_function[pid].return_log_prob(
                    parameter=pid_pars, gp_lengthscale=gp_lengthscale, gp_variance=gp_variance, 
                    n_inducers=kwargs.pop('n_inducers', None),
                )
            elif self.h_prior_to_apply[pid] == 'gp_dists_full':
                gp_lengthscale  = h_parameters[:, self.h_labels[f'{pid}_gp_lengthscale']]
                gp_variance     = h_parameters[:, self.h_labels[f'{pid}_gp_variance']]
                gp_mean         = h_parameters[:, self.h_labels[f'{pid}_gp_mean']]
                gp_nugget       = h_parameters[:, self.h_labels[f'{pid}_gp_nugget']]
                gp_likelihood = self.h_gp_function[pid].return_log_prob(
                    parameter=pid_pars, gp_lengthscale=gp_lengthscale, gp_variance=gp_variance, 
                    gp_mean=gp_mean, gp_nugget=gp_nugget, n_inducers=kwargs.pop('n_inducers', None),
                )
            else:
                raise AssertionError("Unrecognized GP prior type")
            return tf.reduce_sum(log_prior + gp_likelihood)

        best_loss = np.inf
        best_candidate = None
        loss_list = []
        # Grid search: iterate over all combinations of candidate values
        for i,g in enumerate(g_list):# Convert the current grid point to a TensorFlow tensor
            candidate = tf.constant(g, dtype=tf.float32)
            candidate = tf.expand_dims(candidate, axis=0) # Add a batch dimension
            loss = -log_posterior_fn(candidate).numpy()
            loss_list.append(loss)
            # Update the best candidate if the current loss is lower
            if loss < best_loss:
                best_loss = loss
                best_candidate = candidate.numpy()

        if best_candidate is None:
            raise ValueError("Grid search did not yield a valid candidate.")
        # Organize parameters into a DataFrame for storage
        hdf = {}
        for h, idx_h in self.h_labels.items():
            hdf[h] = best_candidate[:,idx_h]
        self.h_MAP_parameters = pd.DataFrame(hdf)
        return g_list, loss_list