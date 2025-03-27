import pandas as pd
import numpy as np
from .utils import format_data, format_paradigm, get_rsq, calculate_log_prob_t, calculate_log_prob_gauss_loc0, format_parameters
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb
from tqdm import tqdm
import math

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


        '''    
        self.model = copy.deepcopy(model)
        self.data = data.astype(np.float32)
        self.kwargs = kwargs
        self.paradigm = model.get_paradigm(model.paradigm)
        self.n_voxels = self.data.shape[-1]
        self.model_labels = {l:i for i,l in enumerate(self.model.parameter_labels)} # useful to have it as a dict per entry        
        self.n_model_params = len(self.model_labels) # length of model parameters only... 

        # We can also fit noise -> 
        self.noise_method = kwargs.get('noise_method', 'fit_tdist')  # 'fit_normal' 'none'
        print(self.noise_method)
        assert self.noise_method in ('fit_tdist', 'fit_normal', 'none') 
        if self.noise_method=='fit_tdist':
            # Fit the t-distribution including dof, scale 
            self.model_labels['noise_dof'] = len(self.model_labels) 
            self.model_labels['noise_scale'] = len(self.model_labels) 
        elif self.noise_method=='fit_normal':
            self.model_labels['noise_scale'] = len(self.model_labels)  
        self.n_params = len(self.model_labels)

        # MCMC specific information
        self.fixed_pars = {}
        # Prior for each parameter (e.g., normal distribution at 0 for "x")      
        # -> default no prior
        self.p_prior = {p:PriorNone() for p in self.model_labels}                                        
        self.p_prior_type = {p:'none' for p in self.model_labels}
        # -> default no bijector 
        self.p_bijector = {p:tfb.Identity() for p in self.model_labels} # What to apply to the 
        # If fitting noise - apply these priors and bijectors by default
        if self.noise_method=='fit_tdist':
            self.add_bijector(pid='noise_dof', bijector_type=tfb.Softplus())
            self.add_bijector(pid='noise_scale', bijector_type=tfb.Exp())
            self.add_prior(pid='noise_dof', prior_type='Exponential', distribution=tfd.Exponential(rate=0.8))
            self.add_prior(pid='noise_scale', prior_type='HalfNormal', distribution=tfd.HalfNormal(scale=1.0))
        elif self.noise_method=='fit_normal':
            self.add_bijector(pid='noise_scale', bijector_type=tfb.Exp())
            self.add_prior(pid='noise_scale', prior_type='HalfNormal', distribution=tfd.HalfNormal(scale=1.0))                        
        # Per voxel (row in data) - save the output of the MCMC sampler 
        self.mcmc_sampler = [None] * self.data.shape[1]
        self.mcmc_stats = [None] * self.data.shape[1]
        self.mcmc_summary = None
        self.mcmc_mean = None
        # MAP 
        self.MAP_parameters = [None] * self.data.shape[1]
        
    def add_prior(self, pid, prior_type, **kwargs):
        ''' 
        Adds the prior to each parameter:

        Options:
            uniform:    uniform probability b/w the specified bounds (low, high). Otherwise infinite
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
            low = kwargs.get('low')
            high = kwargs.get('high')        
            self.p_prior[pid] = PriorUniform(low, high)            
        elif prior_type=='none':
            self.p_prior[pid] = PriorNone()  
        elif prior_type=='fixed':
            fixed_val = kwargs.get('fixed_val')
            self.p_prior[pid] = PriorFixed(fixed_val)
        elif prior_type == 'gp_dists':
            dists = kwargs.pop('dists')
            fixed_params = kwargs.pop('fixed_params', True)
            self.p_prior[pid] = GPdists(dists, fixed_params=fixed_params, **kwargs)
        else:
            self.p_prior[pid] = PriorGeneral(prior_type=prior_type, distribution=kwargs.get('distribution'))
    
    def add_priors_from_bounds(self, bounds):
        '''
        Used to setup uninformative priors: i.e., uniform between the bouds
        Can setup more informative, like a normal using the other methods        
        '''        
        for p in bounds.keys():
            if bounds[p][0]!=bounds[p][1]:
                self.add_prior(
                    pid=p,
                    prior_type = 'uniform',
                    low = bounds[p][0],
                    high = bounds[p][1],
                    )
            else:
                self.add_prior(
                    pid=p,
                    prior_type = 'fixed',
                    fixed_val = bounds[p][0],
                    )    
            
    
    def sample_from_priors(self, n):
        samples = []
        for p in self.model_labels:
            samples.append(self.p_prior[p].sampler(n))
        return tf.stack(samples, axis=-1)
    
    def add_bijector(self, pid, bijector_type, **kwargs):
        ''' add transformations to parameters so that they are fit smoothly        
        
        identity        - do nothing
        softplus        - don't let anything be negative

        '''
        if bijector_type == 'identity':
            self.p_bijector[pid] = tfb.Identity()        
        elif bijector_type == 'softplus':
            # Don't let anything be negative
            self.p_bijector[pid] = tfb.Softplus()
        elif bijector_type == 'sigmoid':
            self.p_bijector[pid] = tfb.Sigmoid(
                low=kwargs.get('low'), high=kwargs.get('high')
            )
        else:
            self.p_bijector[pid] = bijector_type

    def add_bijector_from_bounds(self, bounds):
        for p in bounds.keys():
            self.add_bijector(
                pid=p,
                bijector_type = 'sigmoid',
                low = bounds[p][0],
                high = bounds[p][1],
                )
        
    def prep_for_fitting(self, **kwargs):
        ''' Get everything ready for fitting...
        '''        
        # Ok lets map everything so we can fix some parameters
        # Are there any parameters to fix? 
        if (len(self.fixed_pars) != 0):
            k_check = list(self.fixed_pars.keys())
            for k in k_check:
                if k not in self.model_labels.keys():
                    self.fixed_pars.pop(k)                        
            if not isinstance(self.fixed_pars, pd.DataFrame):
                self.fixed_pars = pd.DataFrame.from_dict(self.fixed_pars, orient='index').T.astype('float32')        
            indices_list = []
            updates_list = []
            if len(self.fixed_pars) == 1:
                self.fixed_pars = pd.concat([self.fixed_pars] * self.n_vx_to_fit, ignore_index=True)

            # Create a tensor for row indices: (number of vx being fit)
            rows = tf.range(self.n_vx_to_fit)
            for param_name,fix_value in self.fixed_pars.items():
                # Where to put the values
                col_idx = self.model_labels[param_name]                                
                # Create a tensor of column indices (same column for every row)
                cols = tf.fill(tf.shape(rows), col_idx)
                
                # Stack rows and cols to create indices of shape [n_vx_to_fit, 2]
                param_indices = tf.stack([rows, cols], axis=1)
                indices_list.append(param_indices)
                
                # Create the update values: a vector of length n_vx_to_fit with the fixed value.
                updates_list.append(tf.convert_to_tensor(fix_value, dtype=tf.float32))
            # Concatenate all the indices and updates from each parameter fix.
            self.fix_update_index = tf.concat(indices_list, axis=0)  # shape: [num_updates, 2]
            self.fix_update_value = tf.concat(updates_list, axis=0)    # shape: [num_updates]            
            # Define the update function
            self.fix_update_fn = FixUdateFn(self.fix_update_index, self.fix_update_value).update_fn             

            # Also ensure that priors & bijectors are correct
            for p in self.fixed_pars.keys():
                self.p_prior_type[p] = 'none'
                self.p_bijector[p] = tfb.Identity()
            
        else:
            self.fix_update_fn = FixUdateFn().update_fn             
        
        # Set the bijectors (to project to a useful fitting space)
        self.p_bijector_list = []
        for p in self.model_labels:            
            self.p_bijector_list.append(
                self.p_bijector[p]
            )      
        # Only loop through those parameters with a prior
        self.priors_to_loop = [
            p for p,t in self.p_prior_type.items() if t not in ('fixed', 'none')
        ]
        self.model_labels_inv = {v:k for k,v in self.model_labels.items()}

    @tf.function
    def _bprf_transform_parameters_forward(self, parameters):
        # Loop through parameters & bijectors (forward)
        p_out = [
            self.p_bijector_list[i].forward(parameters[:,i]) for i in range(self.n_params)
            ]
        return tf.stack(p_out, axis=-1)
    @tf.function
    def _bprf_transform_parameters_backward(self, parameters):
        # Loop through parameters & bijectors... (but apply inverse)
        p_out = [
            self.p_bijector_list[i].inverse(parameters[:,i]) for i in range(self.n_params)
            ]
        return tf.stack(p_out, axis=-1)

    def fit_mcmc(self, 
            init_pars=None,
            num_steps=100, 
            idx = None,
            fixed_pars={},
            **kwargs):
        '''
        Find the distribution of parameters 
        '''
        if idx is None: # all of them?
            idx = np.arange(self.n_voxels).tolist()
        elif isinstance(idx, int):
            idx = [idx]
        vx_bool = np.zeros(self.n_voxels, dtype=bool)
        vx_bool[idx] = True
        self.n_vx_to_fit = len(idx)
        self.fixed_pars = fixed_pars
        self.prep_for_fitting(**kwargs)
        self.n_params = len(self.model_labels)
        step_size = kwargs.pop('step_size', 0.0001) # rest of the kwargs go to "hmc_sample"                
        step_size = [tf.constant(step_size, np.float32) for _ in range(self.n_params)]
        paradigm = kwargs.pop('paradigm', self.paradigm)
        
        y = self.data.values
        init_pars = self.sort_parameters(init_pars)
        # init_pars = format_parameters(init_pars)
        init_pars = init_pars.values.astype(np.float32) 

        # Clean the paradigm 
        paradigm_ = self.model.stimulus._clean_paradigm(paradigm)        
        
        # Define the prior in 'tf'
        log_prior_fn = self._create_log_prior_fn()
        
        # Calculating the likelihood
        # -> based on our 'noise_method'
        residual_ln_likelihood_fn = self._create_residual_ln_likelihood_fn()

        # Now create the log_posterior_fn
        # normal_dist = tfd.Normal(loc=0.0, scale=1.0)
        @tf.function
        def log_posterior_fn(parameters):
            parameters = self.fix_update_fn(parameters)
            par4pred = parameters[:,:self.n_model_params] # chop out any hyper / noise parameters
            predictions = self.model._predict(
                paradigm_[tf.newaxis, ...], par4pred[tf.newaxis, ...], None)     # Only include those parameters that are fed to the model
            residuals = y[:, vx_bool] - predictions[0]                                    
            log_likelihood = residual_ln_likelihood_fn(parameters, residuals)
            log_prior = log_prior_fn(parameters) 
            # Return vector of length idx (optimize each chain separately)
            return log_prior + log_likelihood
        
        # -> make sure we are in the correct dtype 
        initial_state = [tf.convert_to_tensor(init_pars[vx_bool,i], dtype=tf.float32) for i in range(self.n_params)]                          
        # Define the target log probability function (for this voxel)
        def target_log_prob_fn(*parameters):
            parameters = tf.stack(parameters, axis=-1)
            return log_posterior_fn(parameters)
        
        print('Lets run some checks with everything...')
        # Check the gradient with respect to each parameter
        log_prob = target_log_prob_fn(*initial_state)
        print(f'log prob {log_prob}')
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
        print(f'initial_ll={initial_ll}')            
        print(f"Starting NUTS sampling...")
        samples, stats = bprf_sample_NUTS(
            init_state = initial_state, 
            target_log_prob_fn=target_log_prob_fn, 
            unconstraining_bijectors=self.p_bijector_list, 
            num_steps=num_steps, 
            # OTHER STUFF TO OPTIMIZE
            step_size=step_size, 
            **kwargs
            )                
        print('Finished NUTS sampling...')
        # stuff to save...        
        all_samples = tf.stack(samples, axis=-1).numpy()
        # nsteps, n_voxels, n_params
        
        for ivx_loc,ivx_fit in enumerate(idx):
            estimated_p_dict = {}
            for i,p in enumerate(self.model_labels):
                estimated_p_dict[p] = all_samples[:,ivx_loc,i]
            for p,v in self.fixed_pars.items():
                estimated_p_dict[p] = estimated_p_dict[p]*0 + v[ivx_fit]
            self.mcmc_sampler[ivx_fit] = pd.DataFrame(estimated_p_dict)
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



    def fit_MAP(self,
                    init_pars=None,
                    num_steps=100,
                    idx = None,
                    fixed_pars={},
                    **kwargs):
        '''
        Maximum a posteriori (MAP) optimization 
        '''
        if idx is None: # all of them?
            idx = np.arange(self.n_voxels).tolist()
        elif isinstance(idx, int):
            idx = [idx]
        vx_bool = np.zeros(self.n_voxels, dtype=bool)
        vx_bool[idx] = True
        self.n_vx_to_fit = len(idx)
        self.fixed_pars = fixed_pars
        self.prep_for_fitting(**kwargs)
        self.n_params = len(self.model_labels)
        paradigm = kwargs.pop('paradigm', self.paradigm)
        adam_kwargs = kwargs.pop('adam_kwargs', {})
        y = self.data.values
        init_pars = self.sort_parameters(init_pars)
        # init_pars = format_parameters(init_pars)
        init_pars = init_pars.values.astype(np.float32) 

        # Clean the paradigm 
        paradigm_ = self.model.stimulus._clean_paradigm(paradigm)  

        # Define the prior in 'tf'
        log_prior_fn = self._create_log_prior_fn()
        
        # Calculating the likelihood
        # -> based on our 'noise_method'
        residual_ln_likelihood_fn = self._create_residual_ln_likelihood_fn()

        # Now create the log_posterior_fn
        @tf.function
        def log_posterior_fn(parameters):                        
            parameters = self._bprf_transform_parameters_forward(parameters)
            parameters = self.fix_update_fn(parameters)            

            # *** NAN DEBUGGING ***            
            # nan_mask = tf.math.is_nan(parameters)            
            # if tf.reduce_any(nan_mask):
            #     nan_indices = tf.where(nan_mask)
            #     tf.print("NaN values found in parameters at indices:", nan_indices)

            par4pred = parameters[:,:self.n_model_params] # chop out any hyper / noise parameters
            predictions = self.model._predict(
                paradigm_[tf.newaxis, ...], par4pred[tf.newaxis, ...], None)     # Only include those parameters that are fed to the model
            residuals = y[:, vx_bool] - predictions[0]                                    
            
            # *** NAN DEBUGGING ***
            # nan_mask = tf.math.is_nan(predictions)   
            # if tf.reduce_any(nan_mask):
            #     nan_indices = tf.where(nan_mask)
            #     tf.print("NaN values found in parameters at indices:", nan_indices[0])            
            # tf.debugging.assert_all_finite(predictions, f"NaN or Inf found in predictions!")                         
            # tf.debugging.assert_all_finite(residuals, f"NaN or Inf found in residuals!")                         

            log_likelihood = residual_ln_likelihood_fn(parameters, residuals)
            log_prior = log_prior_fn(parameters) 
            # Return vector of length idx (optimize each chain separately)
            return log_prior + log_likelihood

        # -> make sure we are in the correct dtype 
        initial_state = [tf.convert_to_tensor(init_pars[vx_bool,i], dtype=tf.float32) for i in range(self.n_params)]                          
        # Define the target log probability function (for this voxel)
        
        print('Starting MAP optimization...')

        # Optimization setup
        # -> transform parameters backwards before fit
        opt_vars = [
            tf.Variable(self.p_bijector_list[i].inverse(init_state)) for i,init_state in enumerate(initial_state)]

        @tf.function
        def neg_log_posterior_fn():
            return -log_posterior_fn(tf.stack(opt_vars, axis=-1))
        
        # **** quick test **** 
        initial_ll = neg_log_posterior_fn()
        print(f'initial neg ll={initial_ll}')                    
        # Check the gradient with respect to each parameter
        with tf.GradientTape() as tape:
            loss = neg_log_posterior_fn()
        gradients = tape.gradient(loss, opt_vars)
        print('Using tape.gradient to check gradients w/respect to each parameter')
        for i, grad in enumerate(gradients):
            print(f' Gradient for {self.model_labels_inv[i]}: {grad.numpy()}')
        # **** **** **** **** 
        optimizer = tf.optimizers.Adam(**adam_kwargs)
        # Optimization loop with tqdm progress bar
        progress_bar = tqdm(tf.range(num_steps), desc="MAP Optimization")
        for step in progress_bar:
            with tf.GradientTape() as tape:
                loss = neg_log_posterior_fn()
            gradients = tape.gradient(loss, opt_vars)
            optimizer.apply_gradients(zip(gradients, opt_vars))
            progress_bar.set_description(f"MAP Optimization, Mean loss: {loss.numpy().mean():.4f}")        
          
        # Extract optimized parameters
        # -> transform parameters forward after fitting
        opt_vars = [self.p_bijector_list[i](ov) for i,ov in enumerate(opt_vars)]
        optimized_samples = [var.numpy() for var in opt_vars]
        df_list = []
        # Save optimized parameters
        for ivx_loc,ivx_fit in enumerate(idx):
            estimated_p_dict = {}
            for i,p in enumerate(self.model_labels):
                estimated_p_dict[p] = optimized_samples[i][ivx_loc]
            for p,v in self.fixed_pars.items():
                estimated_p_dict[p] = estimated_p_dict[p]*0 + v[ivx_fit]
            
            df = pd.DataFrame(estimated_p_dict, index=[ivx_fit]) # use map_sampler instead of mcmc_sampler
            df_list.append(df)
        self.MAP_parameters = pd.concat(df_list).reindex(idx)
        print('MAP optimization finished.')    

    def _create_log_prior_fn(self):
        @tf.function
        def log_prior_fn(parameters):
            # Log-prior function for the model
            p_out = tf.zeros(parameters.shape[0])  
            if self.priors_to_loop==[]:
                return p_out
            for p in self.priors_to_loop:                
                p_out += self.p_prior[p].prior(parameters[:,self.model_labels[p]])
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
    def sort_parameters(self, parameters):
        # Make sure pd.Dataframe parameters are in order
        parameters = reorder_dataframe_columns(pd_to_fix=parameters, dict_index=self.model_labels)
        return parameters
        
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


# *** FIX UPDATE FN **
class FixUdateFn():
    '''Replace lambda to make it pickleable'''
    def __init__(self, fix_update_index=None, fix_update_value=None):
        self.fix_update_index = fix_update_index
        self.fix_update_value = fix_update_value

    def update_fn(self, parameters):
        if self.fix_update_index is None:
            return parameters
        else:
            return tf.tensor_scatter_nd_update(parameters, self.fix_update_index, self.fix_update_value)             

# *** PRIORS ***
class PriorBase():
    prior_type = 'base'
    def prior(self, param):
        return self.distribution.log_prob(param)
    def sampler(self, n):
        # Sample n instances
        return self.distribution.sample(n)
    
class PriorNorm(PriorBase):
    def __init__(self, loc, scale):
        self.prior_type = 'norm'
        self.loc = loc  # mean
        self.scale = scale  # standard deviation
        self.distribution = tfp.distributions.Normal(loc=self.loc, scale=self.scale)

class PriorUniform(PriorBase):
    def __init__(self, low, high):
        self.prior_type = 'uniform'
        self.low = low
        self.high = high
        self.distribution = tfp.distributions.Uniform(low=self.low, high=self.high)

class PriorNone(PriorBase):
    def __init__(self):
        self.prior_type = 'none'
    def prior(self, param):
        return tf.zeros_like(param)        

class PriorFixed(PriorBase):
    def __init__(self, fixed_val):
        self.prior_type = 'fixed'
        self.fixed_val = tf.constant(fixed_val, dtype=tf.float32)
    def prior(self, param):
        return tf.zeros_like(param)
    def sampler(self, n):
        return tf.fill([n], self.fixed_val)

class PriorGeneral(PriorBase):
    def __init__(self, prior_type, distribution):
        self.prior_type = prior_type
        self.distribution = distribution


# *** SAMPLER ***
from timeit import default_timer as timer
import tensorflow as tf
import tensorflow_probability as tfp

# Copied from .utils.mcmc
@tf.function
def bprf_sample_NUTS(
        init_state,
        step_size,
        target_log_prob_fn,
        unconstraining_bijectors,
        target_accept_prob=0.85,
        unrolled_leapfrog_steps=1,
        max_tree_depth=10,
        num_steps=50,
        burnin=50):


    def trace_fn(_, pkr):
        return {
            'log_prob': pkr.inner_results.inner_results.target_log_prob,
            'diverging': pkr.inner_results.inner_results.has_divergence,
            'is_accepted': pkr.inner_results.inner_results.is_accepted,
            'accept_ratio': tf.exp(pkr.inner_results.inner_results.log_accept_ratio),
            'leapfrogs_taken': pkr.inner_results.inner_results.leapfrogs_taken,
            'step_size': pkr.inner_results.inner_results.step_size}

    hmc = tfp.mcmc.NoUTurnSampler(
        target_log_prob_fn,
        unrolled_leapfrog_steps=unrolled_leapfrog_steps,
        max_tree_depth=max_tree_depth,
        step_size=step_size)

    hmc = tfp.mcmc.TransformedTransitionKernel(
        inner_kernel=hmc,
        bijector=unconstraining_bijectors)

    adaptive_sampler = tfp.mcmc.DualAveragingStepSizeAdaptation(
        inner_kernel=hmc,
        num_adaptation_steps=int(0.8 * burnin),
        target_accept_prob=target_accept_prob,
        # NUTS inside of a TTK requires custom getter/setter functions.
        step_size_setter_fn=lambda pkr, new_step_size: pkr._replace(
            inner_results=pkr.inner_results._replace(
                step_size=new_step_size)
        ),
        step_size_getter_fn=lambda pkr: pkr.inner_results.step_size,
        log_accept_prob_getter_fn=lambda pkr: pkr.inner_results.log_accept_ratio,
    )
    
    start = timer()
    # Sampling from the chain.
    samples, stats = tfp.mcmc.sample_chain(
        num_results=burnin + num_steps,
        current_state=init_state,
        kernel=adaptive_sampler,
        trace_fn=trace_fn)

    duration = timer() - start
    stats['elapsed_time'] = duration

    return samples, stats

import operator

def reorder_dataframe_columns(pd_to_fix, dict_index):
    """
    Reorders the columns of the given DataFrame based on dict_index.
    
    Parameters:
    - pd_to_fix: pandas DataFrame whose columns will be reordered.
    - dict_index: Dictionary mapping column names to desired index positions.
    
    Returns:
    - DataFrame with columns reordered according to dict_index.
    """
    # Sort the dict_index items by their value using operator.itemgetter
    sorted_columns = [col for col, _ in sorted(dict_index.items(), key=operator.itemgetter(1))]
    # Return DataFrame with columns ordered based on sorted_columns
    return pd_to_fix[sorted_columns]



# *************

class GPdists():
    def __init__(self, dists, **kwargs):
        """
        Initialize the GPdists class.
        Objective: take geodesic distances (dists) per vertex on the cortex
        Use this to construct a covariance matrix 
        which we can use as a prior for smoothness during encoding model fitting

        Args:
            dists (array-like): The input distance matrix.
            **kwargs: Optional parameters for controlling behavior, such as:
                - psd_control: Method for ensuring positive semidefiniteness.
                - dists_dtype: Data type for tensor conversion.
                - fixed_params: Fixing (some) of the GP parameters? Precomputes what is possible for efficiency
                    'unfixed'       Everything can change
                    'fixed_vl'      variance, lengthscale are fixed, others can change
                    'fixed_all'     Everything is fixed
                - gp_variance, gp_lengthscale, gp_mean, gp_nugget: GP hyperparameters.
                - kernel: Choice of covariance function (default: 'RBF').
        """
        self.log_prob_method = kwargs.get('log_prob_method', 'tf')  # 'tf' or 'precision'
        self.full_norm = kwargs.get('full_norm', False) # Use full normalization in precision method
        self.kernel = kwargs.get('kernel', 'RBF')
        self.fixed_params = kwargs.get('fixed_params', 'unfixed') # fixed_vl, fixed_all

        # Fixed GP hyperparameters (when fixed_params is True)
        self.f_gp_variance = kwargs.get('gp_variance', None)
        self.f_gp_lengthscale = kwargs.get('gp_lengthscale', None)
        self.f_gp_mean = kwargs.get('gp_mean', 0.0)
        self.f_gp_nugget = kwargs.get('gp_nugget', 0.0)

        # Setup distance matrix and positive semidefinite control
        self.psd_control   = kwargs.get('psd_control', 'euclidean')  # 'euclidean' or 'none'
        self.embedding_dim = kwargs.get('embedding_dim', 10)
        self.dists_dtype   = kwargs.get('dists_dtype', tf.float64)
        self.dists_raw = tf.convert_to_tensor(dists, dtype=self.dists_dtype)
        self.dists_raw = (self.dists_raw + tf.transpose(self.dists_raw)) / 2.0

        if self.psd_control == 'euclidean':
            print('Embedding in Euclidean space...')
            X = mds_embedding(self.dists_raw, self.embedding_dim)
            self.dists = compute_euclidean_distance_matrix(X)
        else:
            self.dists = self.dists_raw

        self.n_vx = self.dists.shape[0]

        # Precompute covariance related matrices if parameters are fixed
        if self.fixed_params == 'unfixed':
            # Nothing is fixed...
            self.cov_matrix = None
            self.prec_matrix = None
            self.chol = None
            self.gp_prior_dist = None
        elif self.fixed_params == 'fixed_all':
            print('Precomputing covariance matrix...')
            self.cov_matrix = self.return_sigma(
                gp_lengthscale=self.f_gp_lengthscale,
                gp_variance=self.f_gp_variance,
                gp_nugget=self.f_gp_nugget,
            )
            # Compute precision matrix once
            self.prec_matrix = tf.cast(tf.linalg.inv(self.cov_matrix), dtype=tf.float32)
            print('Precomputing Cholesky decomposition...')
            self.chol = tf.linalg.cholesky(tf.cast(self.cov_matrix, dtype=self.dists_dtype))
            # Create the fixed prior distribution
            self.gp_prior_dist = tfd.MultivariateNormalTriL(
                loc=tf.cast(tf.fill([self.n_vx], self.f_gp_mean), dtype=tf.float32),
                scale_tril=tf.cast(self.chol, dtype=tf.float32),
                allow_nan_stats=False,
            )
        elif self.fixed_params == 'fixed_vl':
            raise NotImplementedError("Fixed parameters with 'fixed_vl' option is not implemented yet.")
        else:
            raise ValueError(f"Invalid fixed_params option: {self.fixed_params}. Choose from 'unfixed', 'fixed_vl', or 'fixed_all'.")
        self.set_log_prob()

    @tf.function
    def return_sigma(self, gp_lengthscale, gp_variance, gp_nugget=0.0):
        """
        Computes the covariance matrix using the chosen kernel.

        Args:
            gp_lengthscale (float): Lengthscale parameter.
            gp_variance (float): Variance parameter.
            gp_nugget (float): Nugget (noise) term.

        Returns:
            tf.Tensor: Covariance matrix.
        """
        gp_nugget = tf.cast(gp_nugget, dtype=self.dists_dtype)
        gp_variance = tf.cast(gp_variance, dtype=self.dists_dtype)
        gp_lengthscale = tf.cast(gp_lengthscale, dtype=self.dists_dtype)

        if self.kernel == 'RBF':
            cov_matrix = tf.square(gp_variance) * tf.exp(
                -tf.square(self.dists) / (2.0 * tf.square(gp_lengthscale))
            )
        elif self.kernel == 'matern52':
            sqrt5 = tf.cast(tf.sqrt(5.0), dtype=self.dists_dtype)
            frac1 = (sqrt5 * self.dists) / gp_lengthscale
            frac2 = (5.0 * tf.square(self.dists)) / (3.0 * tf.square(gp_lengthscale))
            cov_matrix = tf.square(gp_variance) * (1 + frac1 + frac2) * tf.exp(-frac1)
        elif self.kernel == 'laplace':
            cov_matrix = tf.square(gp_variance) * tf.exp(-self.dists / gp_lengthscale)
        else:
            raise ValueError("Unsupported kernel: {}".format(self.kernel))
        # Add nugget term for numerical stability
        return cov_matrix + tf.eye(self.n_vx, dtype=self.dists_dtype) * (1e-6 + gp_nugget)
    
    @tf.function
    def _compute_precision_and_conditional_log_prob(self, parameter, prec_matrix, gp_mean, full_norm):
        if full_norm:
            # Full normalization version:
            diff = parameter - gp_mean
            log_det_prec = tf.linalg.logdet(prec_matrix)
            quad_form = tf.tensordot(diff, tf.linalg.matvec(prec_matrix, diff), axes=1)
            n = tf.cast(tf.shape(parameter)[0], diff.dtype)
            two_pi = tf.constant(2 * math.pi, dtype=diff.dtype)
            return 0.5 * log_det_prec - 0.5 * quad_form - 0.5 * n * tf.math.log(two_pi)
        else:
            # Original element-wise (conditional) formulation:
            diff = parameter - gp_mean
            Qp = tf.linalg.matvec(prec_matrix, diff)
            diag_Q = tf.linalg.diag_part(prec_matrix)
            log_probs = 0.5 * tf.math.log(diag_Q) - 0.5 * tf.math.log(2 * math.pi) - 0.5 * (Qp ** 2) / diag_Q
            return tf.reduce_sum(log_probs)

    def set_log_prob(self):
        """
        Set the log probability method based on whether parameters are fixed and the chosen method.
        """
        if self.fixed_params == 'fixed_all':
            # When hyperparameters are fixed, use the precomputed distribution or precision
            if self.log_prob_method == 'tf':
                self.return_log_prob = self._return_log_prob_fixed_tf
            else:
                self.return_log_prob = self._return_log_prob_fixed_prec
        elif self.fixed_params == 'fixed_vl':
            raise NotImplementedError("Fixed parameters with 'fixed_vl' option is not implemented yet.")
        elif self.fixed_params == 'unfixed':
            if self.log_prob_method == 'tf':
                self.return_log_prob = self._return_log_prob_unfixed_tf
            else:
                self.return_log_prob = self._return_log_prob_unfixed_prec
        else:
            raise ValueError(f"Invalid fixed_params option: {self.fixed_params}. Choose from 'unfixed', 'fixed_vl', or 'fixed_all'.")


    @tf.function
    def _return_log_prob_fixed_tf(self, parameter, gp_lengthscale, gp_variance, gp_mean=0.0, gp_nugget=0.0):
        """
        Fixed parameters using TensorFlow distribution.
        The extra hyperparameter inputs are included to maintain the gradient graph,
        even though they are not used.
        """
        # Note: The extra terms are added via tf.stop_gradient to keep gradients flowing
        extra = tf.stop_gradient(gp_lengthscale + gp_variance + gp_mean + gp_nugget) * 0.0
        return self.gp_prior_dist.log_prob(parameter) + extra

    @tf.function
    def _return_log_prob_unfixed_tf(self, parameter, gp_lengthscale, gp_variance, gp_mean=0.0, gp_nugget=0.0):
        """
        Unfixed parameters using TensorFlow distribution.
        Recompute covariance and Cholesky decomposition on the fly.
        """
        cov_matrix = self.return_sigma(gp_lengthscale, gp_variance, gp_nugget)
        chol = tf.linalg.cholesky(tf.cast(cov_matrix, dtype=self.dists_dtype))
        
        gp_prior_dist = tfd.MultivariateNormalTriL(
            loc=tf.fill([self.n_vx], tf.squeeze(gp_mean)),
            scale_tril=tf.cast(chol, dtype=tf.float32),
            allow_nan_stats=False,
        )
        return gp_prior_dist.log_prob(parameter)

    @tf.function
    def _return_log_prob_fixed_prec(self, parameter, gp_lengthscale, gp_variance, gp_mean=0.0, gp_nugget=0.0):
        """
        Fixed parameters using precomputed precision matrix.
        """
        extra = tf.stop_gradient(gp_lengthscale + gp_variance + gp_mean + gp_nugget) * 0.0
        return self._compute_precision_and_conditional_log_prob(parameter, self.prec_matrix, gp_mean, self.full_norm) + extra

    @tf.function
    def _return_log_prob_unfixed_prec(self, parameter, gp_lengthscale, gp_variance, gp_mean=0.0, gp_nugget=0.0):
        """
        Unfixed parameters using precision matrix. Recompute the covariance and precision matrices on the fly.
        """
        cov_matrix = self.return_sigma(gp_lengthscale, gp_variance, gp_nugget)
        prec_matrix = tf.cast(tf.linalg.inv(cov_matrix), dtype=tf.float32)
        return self._compute_precision_and_conditional_log_prob(parameter, prec_matrix, gp_mean, self.full_norm)


# ****************************
# **************************** 
@tf.function
def mds_embedding(distance_matrix, embedding_dim=10, eps=1e-3):
    """
    Converts a geodesic distance matrix into a Euclidean embedding using classical MDS.
    
    This transformation is necessary because I want to use geodesic distances to generate 
    covariance matrices, which must be positive definite. Directly using geodesic distances 
    may not yield a positive definite covariance matrix. By applying classical multidimensional 
    scaling (MDS), we recover a Gram matrix (an inner product matrix) from the squared distances 
    that is positive semi-definite. Then, by selecting only the positive eigenvalues and 
    corresponding eigenvectors, we obtain an embedding whose reconstructed covariance matrix 
    (X X^T) is guaranteed to be positive semi-definite, thus suitable for use as a covariance matrix.
    
    Args:
        distance_matrix: A [n x n] tensor of geodesic distances.
        embedding_dim: Optional integer specifying the number of dimensions for the embedding.
            If None, it uses the number of positive eigenvalues.
        eps: A threshold to consider eigenvalues as positive.
        
    Returns:
        A [n x d] tensor of embedded coordinates.
    """
    # 1. Compute squared distances:
    #    Classical MDS begins with the squared distance matrix D^2, where each element is (d_ij)^2.
    D2 = tf.square(distance_matrix)
    
    # 2. Determine the number of points:
    #    'n' is the number of data points. We also cast it to the same type as distance_matrix.
    n = tf.shape(distance_matrix)[0]
    n_float = tf.cast(n, distance_matrix.dtype)
    
    # 3. Create the centering matrix J:
    #    J = I - (1/n) * 11^T, where I is the identity matrix and 1 is a vector of ones.
    #    This matrix centers the data by subtracting the mean from each coordinate.
    I = tf.eye(n, dtype=distance_matrix.dtype)
    ones = tf.ones((n, n), dtype=distance_matrix.dtype)
    J = I - ones / n_float
    
    # 4. Compute the Gram matrix (inner product matrix):
    #    K = -0.5 * J * D2 * J. This operation is known as double centering, which recovers the inner products
    #    from the squared distances. The Gram matrix can be written as K = X X^T, where X are the embedded coordinates.
    K = -0.5 * tf.matmul(J, tf.matmul(D2, J))
    
    # 5. Compute the eigen decomposition of K:
    #    Since K is symmetric, we can decompose it into its eigenvalues and eigenvectors.
    #    tf.linalg.eigh returns eigenvalues in ascending order.
    eigenvalues, eigenvectors = tf.linalg.eigh(K)
    
    # 6. Determine the embedding dimension if not provided:
    #    If embedding_dim is None, count how many eigenvalues are greater than the threshold eps.
    #    Only positive eigenvalues indicate meaningful dimensions; near-zero or negative values may be due to numerical errors.
    if embedding_dim is None:
        positive_mask = eigenvalues > eps
        embedding_dim = tf.reduce_sum(tf.cast(positive_mask, tf.int32))
    
    # 7. Select the largest 'embedding_dim' eigenvalues and corresponding eigenvectors:
    #    Since the eigenvalues are in ascending order, we slice from the end to get the largest ones.
    eigenvalues = eigenvalues[-embedding_dim:]
    eigenvectors = eigenvectors[:, -embedding_dim:]
    
    # 8. Form the final embedding:
    #    Multiply the eigenvectors by the square root of the eigenvalues.
    #    This is derived from the factorization K = UΛU^T, so setting X = U * sqrt(Λ) gives X X^T = K.
    #    Use tf.maximum to ensure numerical stability by avoiding the square root of negative numbers.
    eigenvalues = tf.maximum(eigenvalues, 0)
    X = eigenvectors * tf.sqrt(eigenvalues)
    return X


@tf.function
def compute_euclidean_distance_matrix(X, eps=1e-6):
    """
    Computes the pairwise Euclidean distance matrix from the embedding X.
    
    Args:
        X: A [n x d] tensor of embedded coordinates.
        eps: A small number for numerical stability.
        
    Returns:
        A [n x n] tensor of Euclidean distances.
    """
    # 1. Expand dimensions of X for broadcasting:
    #    X_expanded1 will have shape (n, 1, d) and X_expanded2 will have shape (1, n, d).
    #    This setup allows us to compute the difference between every pair of points.
    X_expanded1 = tf.expand_dims(X, axis=1)
    X_expanded2 = tf.expand_dims(X, axis=0)
    
    # 2. Compute pairwise differences:
    #    For each pair of points i and j, this computes (X_i - X_j).
    diff = X_expanded1 - X_expanded2
    
    # 3. Compute the Euclidean distance matrix:
    #    For each pair (i, j), calculate the square root of the sum of squared differences across dimensions.
    #    Adding eps inside the square root ensures numerical stability (avoiding sqrt(0) issues).
    D_euc = tf.sqrt(tf.reduce_sum(tf.square(diff), axis=-1) + eps)
    return D_euc
