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
                kernel=kwargs.pop('kernel', 'RBF'),
                **kwargs
            )
            # self.h_add_bijector(pid=f'{pid}_gp_lengthscale', bijector_type=tfb.Identity())
            # self.h_add_bijector(pid=f'{pid}_gp_variance', bijector_type=tfb.Identity())

            self.h_add_bijector(pid=f'{pid}_gp_lengthscale', bijector_type=tfb.Softplus())
            self.h_add_bijector(pid=f'{pid}_gp_variance', bijector_type=tfb.Softplus())

            # [3] add the priors for the new parameters
            # ... none for now...
            self.h_add_prior(
                pid=f'{pid}_gp_lengthscale', 
                prior_type='none', 
                **kwargs)
            self.h_add_prior(pid=f'{pid}_gp_variance', prior_type='none', **kwargs)
        
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
                kernel=kwargs.pop('kernel', 'RBF'),
                **kwargs
            )
            self.h_add_bijector(pid=f'{pid}_gp_lengthscale', bijector_type=tfb.Softplus())
            self.h_add_bijector(pid=f'{pid}_gp_variance', bijector_type=tfb.Softplus())
            self.h_add_bijector(pid=f'{pid}_gp_mean', bijector_type=tfb.Identity())
            self.h_add_bijector(pid=f'{pid}_gp_nugget', bijector_type=tfb.Softplus())

            # [3] add the priors for the new parameters
            # ... none for now...
            for h_added in ['lengthscale', 'variance', 'mean', 'nugget']:
                self.h_add_prior(
                    pid=f'{pid}_gp_{h_added}', 
                    prior_type='none', 
                    **kwargs)

    
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


    @tf.function
    def _h_bprf_transform_parameters_forward(self, h_parameters):
        # Loop through parameters & bijectors (forward)
        h_out = [
            self.h_bijector_list[i](h_parameters[:,i]) for i in range(self.h_n_params)
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
            parameters = self.fix_update_fn(parameters)            

            nan_mask = tf.math.is_nan(parameters)            
            if tf.reduce_any(nan_mask):
                nan_indices = tf.where(nan_mask)
                tf.print("2:NaN values found in parameters at indices:", nan_indices)

            h_parameters = self.h_fix_update_fn(h_parameters)
            par4pred = parameters[:,:self.n_model_params] # chop out any hyper / noise parameters
            predictions = self.model._predict(
                paradigm_[tf.newaxis, ...], par4pred[tf.newaxis, ...], None)     # Only include those parameters that are fed to the model
            residuals = y[:, vx_bool] - predictions[0]                        
            tf.debugging.assert_all_finite(predictions, "NaN or Inf found in predictions!")                         
            tf.debugging.assert_all_finite(residuals, "NaN or Inf found in predictions!")                                     
            # -> rescale based on std...
            log_likelihood = residual_ln_likelihood_fn(parameters, residuals)
            log_prior = log_prior_fn(parameters, h_parameters)            
            return tf.reduce_sum(log_likelihood + log_prior)
        
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
        adam_kwargs = kwargs.pop('adam_kwargs', {})
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
            
            # -> rescale based on std...
            log_likelihood = residual_ln_likelihood_fn(parameters, residuals)
            log_prior = log_prior_fn(parameters, h_parameters)            
            # Return vector of length idx (optimize each chain separately)
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
        
        optimizer = tf.optimizers.Adam(**adam_kwargs)
        # Optimization loop with tqdm progress bar
        for step in tqdm(tf.range(num_steps), desc="MAP Optimization"):
            with tf.GradientTape() as tape:
                loss = neg_log_posterior_fn()
            print(f"Step {step.numpy()} Loss = {loss.numpy()}")
            gradients = tape.gradient(loss, all_opt_vars)
            # gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=1.0)
            optimizer.apply_gradients(zip(gradients, all_opt_vars))         
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
                estimated_p_dict[p] = estimated_p_dict[p]*0 + v.values
            
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
                        gp_lengthscale=gp_lengthscale, gp_variance=gp_variance, parameter=param_values
                    )
                elif self.h_prior_to_apply[h]=='gp_dists_full':
                    param_values = parameters[:, self.model_labels[h]] # Values of parameter 'h' for vertices being fit
                    gp_lengthscale  = h_parameters[:, self.h_labels[f'{h}_gp_lengthscale']] # Current value of GP lengthscale hyperparameter
                    gp_variance     = h_parameters[:, self.h_labels[f'{h}_gp_variance']] # Current value of GP variance hyperparameter
                    gp_mean         = h_parameters[:, self.h_labels[f'{h}_gp_mean']] # Current value of GP variance hyperparameter
                    gp_nugget       = h_parameters[:, self.h_labels[f'{h}_gp_mean']] # Current value of GP variance hyperparameter
                    p_out += self.h_gp_function[h].return_log_prob(
                        parameter=param_values, gp_lengthscale=gp_lengthscale, gp_variance=gp_variance, 
                        gp_mean=gp_mean, gp_nugget=gp_nugget
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
    
class GPdists():
    def __init__(self, dists, kernel='rbf', **kwargs):
        self.psd_control = kwargs.get('psd_control', 'euclidean')
        self.dists_dtype = kwargs.get('dists_dtype', tf.float64)        
        # Create the matrix + cholesky once... (faster...)
        self.fixed_params = kwargs.get('fixed_params', False)        
        self.f_gp_variance = kwargs.get('gp_variance', None)
        self.f_gp_lengthscale = kwargs.get('gp_lengthscale', None)
        self.f_gp_mean = kwargs.get('gp_mean', 0.0)
        self.f_gp_nugget = kwargs.get('gp_nugget', 0.0)

        # dmatrix = n x n matrix of distances (i.e., cortical distance)
        self.dists_raw = tf.convert_to_tensor(dists, dtype=self.dists_dtype, name='dists')
        self.dists_raw = (self.dists_raw + tf.transpose(self.dists_raw)) / 2.0

        if self.psd_control == 'euclidean':
            # Create a euclidean embedding to ensure that the matrix
            # produced will be positive semidefinite
            X = mds_embedding(tf.cast(self.dists_raw, dtype=self.dists_dtype))
            self.dists = compute_euclidean_distance_matrix(X)
        else:
            self.dists = self.dists_raw            
        self.kernel = kernel
        self.nvx = self.dists.shape[0]

        if self.fixed_params:
            cov_matrix = self.return_sigma(
                gp_lengthscale=self.f_gp_lengthscale,
                gp_variance=self.f_gp_variance,
                gp_nugget=self.f_gp_nugget,
            )
            chol = tf.linalg.cholesky(tf.cast(cov_matrix, dtype=self.dists_dtype))                
            self.gp_prior_dist = tfd.MultivariateNormalTriL(
                loc=tf.zeros(self.nvx, dtype=tf.float32) + self.f_gp_mean,
                scale_tril=tf.cast(chol, dtype=tf.float32), 
                allow_nan_stats=False,
                )            
        else:
            self.gp_prior_dist = []
    @tf.function
    def return_sigma(self, gp_lengthscale, gp_variance, gp_nugget=0.0):
        ''' Return the correlation matrix         
        '''                
        if self.kernel=='RBF':
            # RBF
            # K(x,x') = exp ( - abs(x-x')^2 / (2*sigma^2) )
            # self.dists = abs(x-x')^2 
            gp_variance = tf.cast(gp_variance, dtype=self.dists_dtype)
            gp_lengthscale = tf.cast(gp_lengthscale, dtype=self.dists_dtype)
            cov_matrix = gp_variance * tf.exp(
                - tf.square(self.dists) / (2.0*tf.square(gp_lengthscale))
            )
            # cov_matrix = (cov_matrix + tf.transpose(cov_matrix)) / 2.0  # Keep symmetry enforcement for now
            cov_matrix += tf.eye(self.nvx, dtype=self.dists_dtype) * (1e-6 + gp_nugget)

        return cov_matrix    

    @tf.function
    def return_log_prob(self, parameter, gp_lengthscale, gp_variance, gp_mean=0.0, gp_nugget=0.0):    
        if self.fixed_params:
            log_prob = self.gp_prior_dist.log_prob(parameter) # Log-prior contribution for this parameter    
        else:
            cov_matrix = self.return_sigma(gp_lengthscale=gp_lengthscale, gp_variance=gp_variance, gp_nugget=gp_nugget)
            chol = tf.linalg.cholesky(tf.cast(cov_matrix, dtype=self.dists_dtype))                
            gp_prior_dist = tfd.MultivariateNormalTriL(
                loc=tf.zeros(self.nvx, dtype=tf.float32) + gp_mean,
                scale_tril=tf.cast(chol, dtype=tf.float32), 
                allow_nan_stats=False,
                )   
            log_prob = gp_prior_dist.log_prob(parameter) # Log-prior contribution for this parameter
        return log_prob

def mds_embedding(distance_matrix, embedding_dim=10, eps=1e-3):
    """
    Converts a geodesic distance matrix into a Euclidean embedding using classical MDS.
    
    Args:
        distance_matrix: A [n x n] tensor of geodesic distances.
        embedding_dim: Optional integer specifying the number of dimensions for the embedding.
            If None, it uses the number of positive eigenvalues.
        eps: A threshold to consider eigenvalues as positive.
        
    Returns:
        A [n x d] tensor of embedded coordinates.
    """
    # Compute squared distances
    D2 = tf.square(distance_matrix)
    
    # Number of points
    n = tf.shape(distance_matrix)[0]
    n_float = tf.cast(n, distance_matrix.dtype)
    
    # Create centering matrix: J = I - 1/n * 11^T
    I = tf.eye(n, dtype=distance_matrix.dtype)
    ones = tf.ones((n, n), dtype=distance_matrix.dtype)
    J = I - ones / n_float
    
    # Compute the Gram matrix: K = -0.5 * J D2 J
    K = -0.5 * tf.matmul(J, tf.matmul(D2, J))
    
    # Compute eigen decomposition (eigenvalues are in ascending order)
    eigenvalues, eigenvectors = tf.linalg.eigh(K)
    
    # Determine embedding dimension if not provided by counting positive eigenvalues
    if embedding_dim is None:
        positive_mask = eigenvalues > eps
        embedding_dim = tf.reduce_sum(tf.cast(positive_mask, tf.int32))
        print(embedding_dim)
        # bloop
    
    # Select the largest 'embedding_dim' eigenvalues and corresponding eigenvectors
    eigenvalues = eigenvalues[-embedding_dim:]
    eigenvectors = eigenvectors[:, -embedding_dim:]
    
    # Form the embedding X = eigenvectors * sqrt(eigenvalues)
    # Ensure non-negative eigenvalues before taking sqrt
    eigenvalues = tf.maximum(eigenvalues, 0)
    X = eigenvectors * tf.sqrt(eigenvalues)
    return X

def compute_euclidean_distance_matrix(X, eps=1e-6):
    """
    Computes the pairwise Euclidean distance matrix from the embedding X.
    
    Args:
        X: A [n x d] tensor of embedded coordinates.
        eps: A small number for numerical stability.
        
    Returns:
        A [n x n] tensor of Euclidean distances.
    """
    # Expand dims for broadcasting
    X_expanded1 = tf.expand_dims(X, axis=1)  # shape: (n, 1, d)
    X_expanded2 = tf.expand_dims(X, axis=0)  # shape: (1, n, d)
    diff = X_expanded1 - X_expanded2
    # Compute the Euclidean distances
    D_euc = tf.sqrt(tf.reduce_sum(tf.square(diff), axis=-1) + eps)
    return D_euc