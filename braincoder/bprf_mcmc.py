import pandas as pd
import numpy as np
from .utils import format_data, format_paradigm, get_rsq, calculate_log_prob_t, calculate_log_prob_gauss_loc0, format_parameters
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb


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
            # Build the indices and update values lists.
            indices_list = []
            updates_list = []

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
                param_updates = tf.fill(tf.shape(rows), fix_value)
                updates_list.append(param_updates)
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

    def fit_mcmc(self, 
            init_pars=None,
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
        self.prep_for_fitting(**kwargs)
        self.n_params = len(self.model_labels)
        step_size = kwargs.pop('step_size', 0.0001) # rest of the kwargs go to "hmc_sample"                
        step_size = [tf.constant(step_size, np.float32) for _ in range(self.n_params)]
        paradigm = kwargs.pop('paradigm', self.paradigm)
        
        y = self.data.values
        init_pars = format_parameters(init_pars)
        init_pars = init_pars.values.astype(np.float32) 

        # Clean the paradigm 
        paradigm_ = self.model.stimulus._clean_paradigm(paradigm)        
        
        # Define the prior in 'tf'
        @tf.function
        def log_prior_fn(parameters):
            # Log-prior function for the model
            p_out = tf.zeros(parameters.shape[0])  
            if self.priors_to_loop==[]:
                return p_out
            for p in self.priors_to_loop:
                p_out += self.p_prior[p].prior(parameters[:,self.model_labels[p]])
            return p_out       

        # Calculating the likelihood
        # First lets make the function which returns the likelihood of the residuals
        # -> based on our 'noise_method'
        residual_ln_likelihood_fn = self._create_residual_ln_likelihood_fn()
        # normal_dist = tfd.Normal(loc=0.0, scale=1.0)
        @tf.function
        def log_posterior_fn(parameters):
            parameters = self.fix_update_fn(parameters)
            par4pred = parameters[:,:self.n_model_params] # chop out any hyper / noise parameters
            predictions = self.model._predict(
                paradigm_[tf.newaxis, ...], par4pred[tf.newaxis, ...], None)     # Only include those parameters that are fed to the model
            residuals = y[:, vx_bool] - predictions[0]                        
            
            # -> rescale based on std...
            # residuals_std  = tf.math.reduce_std(residuals, axis=0)
            # log_likelihood = normal_dist.log_prob(residuals/residuals_std) - tf.math.log(residuals_std)
            # log_likelihood = tf.reduce_sum(log_likelihood, axis=0)            # -> rescale based on std...
            log_likelihood = residual_ln_likelihood_fn(parameters, residuals)
            log_prior = log_prior_fn(parameters) 
            print(f'log prior={log_prior}')           
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
                estimated_p_dict[p] = estimated_p_dict[p]*0 + v.values
            self.mcmc_sampler[ivx_fit] = pd.DataFrame(estimated_p_dict)
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