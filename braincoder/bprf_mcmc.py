import pandas as pd
import numpy as np
from .utils import format_data, format_paradigm, get_rsq, calculate_log_prob_t, calculate_log_prob_gauss_loc0, format_parameters
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb
from tqdm import tqdm
import math
from timeit import default_timer as timer
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
        self.include_jacobian = kwargs.get('include_jacobian', True)  # Include the jacobian in the model?
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
        
        # Used to select a sub set of vx to sample...
        # 2-element seed for stateless RNG; will be incremented each call
        self._seed = tf.Variable([0, 0], dtype=tf.int32, trainable=False)        
        self.dists         = kwargs.get('dists', None)
        if self.dists is not None:
            self.dists = tf.convert_to_tensor(self.dists, dtype=tf.float32)
        self._dists_idx = None

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
            fixed_params = kwargs.pop('fixed_params', 'fixed_all')
            self.p_prior[pid] = GPdists(dists, fixed_params=fixed_params, **kwargs)
        elif prior_type == 'gp_dists_m':
            self.p_prior[pid] = kwargs.pop('gp_obj')
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
        if self.dists is not None:  
            # If we are using "close" for inducer selection...      
            self._dists_idx = tf.gather(tf.gather(self.dists, self.idx_to_fit, axis=0), self.idx_to_fit, axis=1)

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
        print(f'Priors to loop through: {self.priors_to_loop}')
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
        self.idx_to_fit = idx
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
        self.idx_to_fit = idx
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
        log_jac_fn = self._create_log_jac_fn()
        # Calculating the likelihood
        # -> based on our 'noise_method'
        residual_ln_likelihood_fn = self._create_residual_ln_likelihood_fn()

        # Now create the log_posterior_fn
        @tf.function
        def log_posterior_fn(parameters_unc):                        
            # [1] Mask fixed parameters
            f_parameters_unc = self.fix_update_fn(parameters_unc)            

            # [2] Jacobian
            log_jac = log_jac_fn(f_parameters_unc) # before transformation 
            
            # [3] bijectors
            f_parameters = self._bprf_transform_parameters_forward(f_parameters_unc)

            # [4] Prior
            log_prior = log_prior_fn(f_parameters) 

            # [5] Likelihood
            par4pred = f_parameters[:,:self.n_model_params] # chop out any hyper / noise parameters
            predictions = self.model._predict(
                paradigm_[tf.newaxis, ...], par4pred[tf.newaxis, ...], None)     # Only include those parameters that are fed to the model
            residuals = y[:, vx_bool] - predictions[0]                                                
            log_likelihood = residual_ln_likelihood_fn(f_parameters, residuals)
            # Return vector of length idx (optimize each chain separately)
            return log_prior + log_likelihood + log_jac

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

    @tf.function
    def _return_inducing_idx(self, n_inducers):
        if self.n_inducers is None:
            # Return all indices if no inducing points are specified
            return tf.range(self.n_vx_to_fit, dtype=tf.int32)
        # increment the seed each call - necessary because otherwise when graph is drawn it doesn't change...
        new_seed = self._seed.assign_add([1, 1])
        # FOR DEBUGGING -> PRINT IT
        # 2) print it for debugging
        # tf.print("debug — new_seed:", new_seed)        
        if self.inducer_selection == 'random':
            # Randomly select indices for inducing points
            inducing_indices = tf.random.experimental.stateless_shuffle(
                tf.range(self.n_vx_to_fit),
                seed=new_seed)[:n_inducers]
            inducing_indices = tf.sort(inducing_indices)  # Keep them sorted for easier indexing

        elif self.inducer_selection == 'close':
            # centre_idx = np.random.randint(0, self.n_vx)
            centre_idx = tf.random.stateless_uniform(
                shape=[],
                minval=0,
                maxval=self.n_vx_to_fit,
                dtype=tf.int32,
                seed=new_seed
            )
            # Get the indices of the closest n_inducers points
            _, inducing_indices = tf.math.top_k(-self._dists_idx[centre_idx, :], k=n_inducers)
            inducing_indices = tf.sort(inducing_indices)  # Keep sorted

        else:
            raise ValueError(f"Unknown inducer selection method: {self.inducer_selection}")
        
        return inducing_indices
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

    def _create_log_jac_fn(self):
        if self.include_jacobian:
            @tf.function
            def log_jac_fn(parameters):
                # Log-prior function for the model
                p_out = tf.zeros(parameters.shape[0])  
                for p in self.priors_to_loop:
                    p_out += self.p_bijector[p].forward_log_det_jacobian(
                        parameters[:,self.model_labels[p]], 
                        event_ndims=0
                    )
                return p_out           
        else:
            @tf.function
            def log_jac_fn(parameters):
                return tf.zeros(parameters.shape[0])
        return log_jac_fn        
    
    def _create_residual_ln_likelihood_fn(self):
        # Calculating the likelihood
        if self.noise_method == 'fit_tdist':
            @tf.function
            def residual_ln_likelihood_fn(parameters, residuals):                    
                # Return vector of length idx (optimize each chain separately)
                # Usef tfd.StudentT for t-distribution
                resid_dist = tfd.StudentT(
                    df=parameters[:,self.model_labels['noise_dof']],
                    loc=0.0, 
                    scale=parameters[:,self.model_labels['noise_scale']]
                )
                resid_ln_likelihood = resid_dist.log_prob(residuals)
                resid_ln_likelihood = tf.reduce_sum(resid_ln_likelihood, axis=0)
                return resid_ln_likelihood
        elif self.noise_method == 'fit_normal':
            # Assume residuals are normally distributed (loc=0.0)
            # Add the scale as an extra parameters to be fit             
            @tf.function
            def residual_ln_likelihood_fn(parameters, residuals):                    
                # [1] Use N(0, std) from tensorflow
                resid_dist = tfd.Normal(
                    loc=0.0, 
                    scale=parameters[:,self.model_labels['noise_scale']]
                )
                resid_ln_likelihood = resid_dist.log_prob(residuals)
                resid_ln_likelihood = tf.reduce_sum(resid_ln_likelihood, axis=0)
                return resid_ln_likelihood              
        
        elif self.noise_method == 'none': 
            # Do not fit the noise - assume it is normally distributed
            # -> calculate scale based on the standard deviation of the residuals 
            @tf.function
            def residual_ln_likelihood_fn(parameters, residuals):                    
                # [1] Use N(0, std)         
                residuals_std  = tf.math.reduce_std(residuals, axis=0)
                resid_dist = tfd.Normal(
                    loc=0.0, 
                    scale=residuals_std
                )
                resid_ln_likelihood = resid_dist.log_prob(residuals)
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
class FixUdateFn:
    """Mask out ‘fixed’ entries in `parameters` using the same init args."""
    def __init__(self, fix_update_index=None, fix_update_value=None):
        # fix_update_index: an [N,1] or [N,D] int64 tensor of indices into the V-vector
        # fix_update_value: a length-N float tensor of the values you want to hold fixed
        if fix_update_index is not None:            
            self.fix_update_index = tf.convert_to_tensor(fix_update_index, dtype=tf.int32)
            self.fix_update_value = tf.convert_to_tensor(fix_update_value, dtype=tf.float32)
        else:
            self.fix_update_index = None
            self.fix_update_value = None

    def update_fn(self, parameters):
        # parameters: a [V] float tensor of your raw_x
        if self.fix_update_index is None:
            return parameters

        # 1) build a full-length mask of 1s and zero out the fixed positions
        mask = tf.ones_like(parameters)
        zeros_for_mask = tf.zeros(tf.shape(self.fix_update_value), dtype=mask.dtype)
        mask = tf.tensor_scatter_nd_update(mask,
                                           self.fix_update_index,
                                           zeros_for_mask)
        # 2) build a full-length "fixed values" vector: 0s everywhere except
        #    your desired fix_update_value at the fixed positions
        fixed_vals = tf.zeros_like(parameters)
        fixed_vals = tf.tensor_scatter_nd_update(
            fixed_vals,
            self.fix_update_index,
            tf.cast(self.fix_update_value, parameters.dtype)
        )
        # 3) fuse it all: keep parameters where mask==1, else use fixed_vals
        return mask * parameters + (1.0 - mask) * fixed_vals
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
        self.eps = kwargs.get('eps', 1e-6) # for numerical stability
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

        self.n_vx = tf.Variable(self.dists.shape[0], dtype=tf.int32, name="n_vx")

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
        else:
            raise ValueError(f"Invalid fixed_params option: {self.fixed_params}. Choose from 'unfixed', 'fixed_vl', or 'fixed_all'.")
        self.set_log_prob()
    def update_n_vx(self, new_value):
        self.n_vx.assign(new_value)

    @tf.function
    def prior(self, param):
        # Compute the conditional log-probability of the parameter under the GP prior
        diff = param - self.f_gp_mean
        raw_score = tf.linalg.matvec(self.prec_matrix, diff)
        prec_ii = tf.linalg.diag_part(self.prec_matrix)
        logZ = 0.5 * (tf.math.log(prec_ii) - tf.math.log(2 * math.pi))
        quadratic = -0.5 * tf.square(raw_score) / prec_ii
        log_probs = logZ + quadratic
        return log_probs
        
        
    @tf.function
    def return_sigma(self, gp_lengthscale, gp_variance, gp_nugget=0.0, dists=None):
        """
        Computes the covariance matrix using the chosen kernel.

        Args:
            gp_lengthscale (float): Lengthscale parameter.
            gp_variance (float): Variance parameter.
            gp_nugget (float): Nugget (noise) term.

        Returns:
            tf.Tensor: Covariance matrix.
        """
        if dists is None: 
            dists = self.dists
        gp_nugget = tf.cast(gp_nugget, dtype=self.dists_dtype)
        gp_variance = tf.cast(gp_variance, dtype=self.dists_dtype)
        gp_lengthscale = tf.cast(gp_lengthscale, dtype=self.dists_dtype)

        if self.kernel == 'RBF':
            cov_matrix = tf.square(gp_variance) * tf.exp(
                -tf.square(dists) / (2.0 * tf.square(gp_lengthscale))
            )
        elif self.kernel == 'matern52':
            sqrt5 = tf.cast(tf.sqrt(5.0), dtype=self.dists_dtype)
            frac1 = (sqrt5 * dists) / gp_lengthscale
            frac2 = (5.0 * tf.square(dists)) / (3.0 * tf.square(gp_lengthscale))
            cov_matrix = tf.square(gp_variance) * (1 + frac1 + frac2) * tf.exp(-frac1)
        elif self.kernel == 'laplace':
            cov_matrix = tf.square(gp_variance) * tf.exp(-dists / gp_lengthscale)
        else:
            raise ValueError("Unsupported kernel: {}".format(self.kernel))
        # Add nugget term for numerical stability
        return cov_matrix + tf.eye(cov_matrix.shape[0], dtype=self.dists_dtype) * tf.cast(self.eps + gp_nugget, dtype=self.dists_dtype)


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
    def _return_log_prob_fixed_tf(self, parameter, gp_lengthscale, gp_variance, gp_mean=0.0, gp_nugget=0.0, n_inducers=None, inducing_indices=None):
        """
        Fixed parameters using TensorFlow distribution.
        The extra hyperparameter inputs are included to maintain the gradient graph,
        even though they are not used.
        """
        # Note: The extra terms are added via tf.stop_gradient to keep gradients flowing
        # extra = tf.stop_gradient(gp_lengthscale + gp_variance + gp_mean + gp_nugget) * 0.0
        # bloop
        extra = (gp_lengthscale + gp_variance + gp_mean + gp_nugget) * 0.0
        return self.gp_prior_dist.log_prob(parameter) + extra

    @tf.function
    def _return_log_prob_unfixed_tf(self, parameter, gp_lengthscale, gp_variance, gp_mean=0.0, gp_nugget=0.0, n_inducers=None, inducing_indices=None):
        """
        Unfixed parameters using TensorFlow distribution.
        Recompute covariance and Cholesky decomposition on the fly.
        Optionally uses random selection of n_inducers for sparse GP approximation.
        """
        if n_inducers is None or n_inducers >= self.n_vx:
            # Full GP
            cov_matrix = self.return_sigma(gp_lengthscale, gp_variance, gp_nugget)
            chol = tf.linalg.cholesky(tf.cast(cov_matrix, dtype=self.dists_dtype))

            gp_prior_dist = tfd.MultivariateNormalTriL(
                loc=tf.fill([self.n_vx], tf.squeeze(gp_mean)),
                scale_tril=tf.cast(chol, dtype=tf.float32),
                allow_nan_stats=False,
            )
            return gp_prior_dist.log_prob(parameter)
        else:
            # Sparse GP approximation using inducing points
            if n_inducers <= 0:
                raise ValueError("n_inducers must be a positive integer.")

            inducing_indices, inducing_dists = self._return_inducing_idx_and_dists(n_inducers=n_inducers, inducing_indices=inducing_indices)
            # Get the parameter values at the inducing points
            inducing_parameter = parameter

            # Calculate the covariance matrix for the inducing points
            inducing_cov_matrix = self.return_sigma(gp_lengthscale, gp_variance, gp_nugget, dists=inducing_dists)
            inducing_chol = tf.linalg.cholesky(tf.cast(inducing_cov_matrix, dtype=self.dists_dtype))

            inducing_gp_prior_dist = tfd.MultivariateNormalTriL(
                loc=tf.fill([n_inducers], tf.squeeze(gp_mean)),
                scale_tril=tf.cast(inducing_chol, dtype=tf.float32),
                allow_nan_stats=False,
            )
            return inducing_gp_prior_dist.log_prob(inducing_parameter)
    @tf.function
    def _return_inducing_idx_and_dists(self, n_inducers, inducing_indices=None):
        if n_inducers is None or n_inducers >= self.n_vx:
            inducing_indices = tf.range(self.n_vx)


        # Gather distances for the selected inducing points
        inducing_dists = tf.gather(tf.gather(self.dists, inducing_indices, axis=0), inducing_indices, axis=1)
        
        return inducing_indices, inducing_dists
# ****************************
class GPdistsM():
    def __init__(self, n_vx, **kwargs):
        """
        More complex GPdists class -> 
        
        Objective: contruct a multivariate normal distribution to obtain the 
        log probability of a list of values

        N(m,K+nugget)
        -> we construct m, the mean function
        -> we construct K, the covariance function
        -> with a nugget, for stability        

        Args:
            dists (dict of dists) : The input distance matrix.
            **kwargs: Optional parameters for controlling behavior, such as:
                - psd_control: Method for ensuring positive semidefiniteness.
                - dists_dtype: Data type for tensor conversion.
                - kernel: Choice of covariance function (default: 'RBF').

        ****************** TODO ********************
        - clean up code

        Possible ideas:
        - mean function with univariate case? 
        - use gpflow for more flexible, and robust GPs...
        - nugget fit with LBO
        - 1 x mean function? - put it all in together
        - better combination / chaining abilities? How combine kernels?
        - make it modular? 
        """
        self.n_vx = tf.Variable(n_vx, dtype=tf.int32, name="n_vx")

        # Setup distance matrix and positive semidefinite control
        self.psd_control   = kwargs.get('psd_control', 'euclidean')  # 'euclidean' or 'none'
        self.eps           = kwargs.get('eps', 1e-6)
        self.embedding_dim = kwargs.get('embedding_dim', 10)
        self.dists_dtype   = kwargs.get('dists_dtype', tf.float64)

        self.stat_kernel_list = []
        self.lin_kernel_list = []
        self.spec_kernel_list = []
        self.mfunc_list = []
        self.mfunc_bijector = tfb.Identity()
        self.Xs = {}
        self.dXs = {}
        self.eig_vals = []
        self.eig_vects = []
        self.kernel_type = {}
        self.pids = {}
        # Index of parameters to be passed...
        self.pids[0] = 'gpk_nugget' # Global nugget term
        self.pids[1] = 'gpk_var'    # Global variance term
        self.pids[2] = 'mfunc_mean' # Global mean term 
        self.pids_inv = {}
        self._update_pids_inv()
        self.return_log_prob = self._return_log_prob_unfixed # by default, return log prob unfixed...
        self.gp_prior_dist = []

    def _update_pids_inv(self):
        self.pids_inv = {}
        self.pids_inv = {v:k for k,v in self.pids.items()}
    
    def update_n_vx(self, new_value):
        self.n_vx.assign(new_value)
        
    @tf.function
    def prior(self, param):
        # Compute the conditional log-probability of the parameter under the GP prior
        diff = param - self.m_vect
        raw_score = tf.linalg.matvec(self.prec_matrix, diff)
        prec_ii = tf.linalg.diag_part(self.prec_matrix)
        logZ = 0.5 * (tf.math.log(prec_ii) - tf.math.log(2 * math.pi))
        quadratic = -0.5 * tf.square(raw_score) / prec_ii
        log_probs = logZ + quadratic
        return log_probs
        
        
    # **************** MEAN FUNCTIONS ***************
    def add_xid_linear_mfunc(self, xid, **kwargs):
        ''' add a kernel
        '''
        Xs = kwargs.get('Xs', None)
        Xs = tf.convert_to_tensor(Xs, dtype=self.dists_dtype)
        if len(Xs.shape) == 1:
            Xs = tf.expand_dims(Xs, axis=-1)
        self.Xs[xid] = Xs
        self.mfunc_list.append(xid)        
        Ds = self.Xs[xid].shape[1]
        for i in range(Ds):
            self.pids[len(self.pids)] = f'mfunc{xid}_slope{i}'

        # Update the inverse dictionary
        self._update_pids_inv()
    
    @tf.function
    def _return_mfunc(self, inducing_indices=None, **kwargs):
        '''Return the mean function
        '''
        if inducing_indices is None:
            inducing_indices = tf.range(self.n_vx)        
        n_ind = tf.shape(inducing_indices)[0]
        # Start of with zero then add global mean
        m_out = tf.zeros(n_ind, dtype=self.dists_dtype) + tf.cast(kwargs['mfunc_mean'], self.dists_dtype) # global mean...
        # then add any regressors...
        for m in self.mfunc_list:
            slopes = tf.stack([kwargs[f'mfunc{m}_slope{i}'] for i in range(self.Xs[m].shape[1])], axis=0)  # [D]
            Xs = tf.gather(self.Xs[m], inducing_indices, axis=0)  # [N, D]
            m_out += tf.reduce_sum(tf.cast(slopes, dtype=self.dists_dtype) * tf.transpose(Xs), axis=0) 
        return self.mfunc_bijector(tf.cast(m_out, dtype=tf.float32))
    
    def add_mfunc_bijector(self, bijector_type, **kwargs):
        ''' add transformations to parameters so that they are fit smoothly        
        
        identity        - do nothing
        softplus        - don't let anything be negative

        '''
        dtype = kwargs.get('dtype', self.dists_dtype)
        if bijector_type == 'identity':
            self.mfunc_bijector = tfb.Identity()        
        elif bijector_type == 'softplus':
            # Don't let anything be negative
            self.mfunc_bijector = tfb.Softplus()
        elif bijector_type == 'sigmoid':
            self.mfunc_bijector = tfb.Sigmoid(
                low=kwargs.get('low'), high=kwargs.get('high'),
            )
        else:
            self.mfunc_bijector = bijector_type

    # *************************************************
    # *************************************************
    # *************************************************
    
    # *** KERNELS ***
    # -> add Stationary kernels 
    def add_xid_stationary_kernel(self, xid, **kwargs):
        ''' add a kernel 
        '''
        Xs = kwargs.get('Xs', None)
        dXs = kwargs.get('dXs', None)
        psd_control = kwargs.get('psd_control', self.psd_control)
        embedding_dim = kwargs.get('embedding_dim', self.embedding_dim)
        self.kernel_type[xid] = kwargs.get('kernel_type', 'RBF')                
        self.stat_kernel_list.append(xid)

        if dXs is None:
            # Get distances from 
            dXs = compute_euclidean_distance_matrix(Xs[...,np.newaxis])        
        if psd_control == 'euclidean':
            print('Embedding in Euclidean space...')
            dXs = mds_embedding(dXs, embedding_dim)
            dXs = compute_euclidean_distance_matrix(dXs)
        self.dXs[xid] = tf.convert_to_tensor(dXs, dtype=self.dists_dtype)
        self.dXs[xid] = (self.dXs[xid] + tf.transpose(self.dXs[xid])) / 2.0        
        # Add a lengthscale & a variance
        self.pids[len(self.pids)] = f'gpk{xid}_l'
        self.pids[len(self.pids)] = f'gpk{xid}_v'

        # Update the inverse dictionary
        self._update_pids_inv()        
    
    # -> add Linear kernels
    def add_xid_linear_kernel(self, xid, **kwargs):
        ''' add a kernel
        '''
        Xs = kwargs.get('Xs', None)
        self.kernel_type[xid] = 'linear'
        self.lin_kernel_list.append(xid)
        
        self.Xs[xid] = tf.expand_dims(tf.convert_to_tensor(Xs, dtype=self.dists_dtype), axis=1)
        # Add a lengthscale & a variance
        self.pids[len(self.pids)] = f'gpk{xid}_slope'
        self.pids[len(self.pids)] = f'gpk{xid}_const'

        # Update the inverse dictionary
        self._update_pids_inv()        
    
    # -> add Spectral kernels (based on LBO)
    def add_xid_spectral_kernel(self, xid, **kwargs):
        ''' add a spectral kernel
        I know - this is not all spectral strictly speaking
        I will clean it up later
        '''
        self.kernel_type[xid] = kwargs.get('kernel_type', 'spec_exp')                
        # store eigenvalues and eigenvectors
        self.eig_vals = tf.convert_to_tensor(kwargs['eig_vals'], dtype=self.dists_dtype)
        self.eig_vects = tf.convert_to_tensor(kwargs['eig_vects'], dtype=self.dists_dtype)

        # if spectral, expect precomputed eigenpairs
        if self.kernel_type[xid] == 'spec_exp':
            self.pids[len(self.pids)] = f'gpk{xid}_spec_l'
            self.pids[len(self.pids)] = f'gpk{xid}_spec_v'

        elif self.kernel_type[xid] == 'spec_heat':
            self.pids[len(self.pids)] = f'gpk{xid}_spec_t'

        elif self.kernel_type[xid] == 'spec_ratquad':
            self.pids[len(self.pids)] = f'gpk{xid}_spec_alpha'
            self.pids[len(self.pids)] = f'gpk{xid}_spec_beta'

        elif 'spec_LBOwarp' in self.kernel_type[xid]:
            # Warp distance and/or lengthscale & variance using LBO eigenvectors
            #       spec_LBOwarp_dx    → dx only
            #       spec_LBOwarp_dxl   → dx + l
            #       spec_LBOwarp_dxv   → dx + v
            #       spec_LBOwarp_dxlv  → dx + l + v            
            use_l = 'l' in self.kernel_type[xid]
            use_v = 'v' in self.kernel_type[xid]   
            
            # Distance for RBF type kernel comes from warped LBO
            self.pids[len(self.pids)] = f'gpk{xid}_spec_dxm0'# 
            for i in range(1, self.eig_vals.shape[0]+1):
                self.pids[len(self.pids)] = f'gpk{xid}_spec_dxm{i}' 
            
            # Lengthscale 
            if use_l:
                self.pids[len(self.pids)] = f'gpk{xid}_spec_lm0' 
                for i in range(1, self.eig_vals.shape[0]+1):
                    self.pids[len(self.pids)] = f'gpk{xid}_spec_lm{i}'             
            else:
                self.pids[len(self.pids)] = f'gpk{xid}_spec_l'
            # Variance
            if use_v:
                self.pids[len(self.pids)] = f'gpk{xid}_spec_vm0' 
                for i in range(1, self.eig_vals.shape[0]+1):
                    self.pids[len(self.pids)] = f'gpk{xid}_spec_vm{i}'             
            else:
                self.pids[len(self.pids)] = f'gpk{xid}_spec_v'
        
        elif self.kernel_type[xid] == 'spec_LBOGibbs':
            self.pids[len(self.pids)] = f'gpk{xid}_spec_v'
            self.pids[len(self.pids)] = f'gpk{xid}_spec_lm0'
            for i in range(1, self.eig_vals.shape[0]+1):
                self.pids[len(self.pids)] = f'gpk{xid}_spec_lm{i}'
            
            Xs = kwargs.get('Xs', None)
            dXs = kwargs.get('dXs', None)
            psd_control = kwargs.get('psd_control', self.psd_control)
            embedding_dim = kwargs.get('embedding_dim', self.embedding_dim)
            if dXs is None:
                # Get distances from 
                dXs = compute_euclidean_distance_matrix(Xs[...,np.newaxis])        
            if psd_control == 'euclidean':
                print('Embedding in Euclidean space...')
                dXs = mds_embedding(dXs, embedding_dim)
                dXs = compute_euclidean_distance_matrix(dXs)
            self.dXs[xid] = tf.convert_to_tensor(dXs, dtype=self.dists_dtype)
            self.dXs[xid] = (self.dXs[xid] + tf.transpose(self.dXs[xid])) / 2.0        
                                
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type[xid]}.")
        self.spec_kernel_list.append(xid)
        self._update_pids_inv()

        

    @tf.function
    def _return_sigma(self, inducing_indices=None, **kwargs):
        ''' Putting all the kernels together - > return the covariance matrix
        '''
        if inducing_indices is None:
            inducing_indices = tf.range(self.n_vx)
        # Start covariance matrix from zero...
        s_out = tf.zeros((len(inducing_indices),len(inducing_indices)), dtype=self.dists_dtype)
        
        # Add in any linear kernels
        for s in self.lin_kernel_list:
            s_out += self._return_sigma_xid_linear(
                gpk_slope=kwargs[f'gpk{s}_slope'],
                gpk_const=kwargs[f'gpk{s}_const'],
                Xs=tf.gather(self.Xs[s], inducing_indices, axis=0)
            )
        
        # Add in any stationary kernels (e.g., RBF)
        for s in self.stat_kernel_list:
            s_out += self._return_sigma_xid_stationary(
                gpk_l=kwargs[f'gpk{s}_l'],
                gpk_v=kwargs[f'gpk{s}_v'],
                dXs=tf.gather(tf.gather(self.dXs[s], inducing_indices, axis=0), inducing_indices, axis=1),
                kernel_type=self.kernel_type[s]
            )

        for s in self.spec_kernel_list:
            spec_kwargs = {i.replace(s,''):k for i,k in kwargs.items() if 'spec' in i}
            if 'Gibbs' in self.kernel_type[s]:
                # Gibbs kernel
                spec_kwargs['dXs'] = tf.gather(tf.gather(self.dXs[s], inducing_indices, axis=0), inducing_indices, axis=1)
            s_out += self._return_sigma_xid_spectral(
                eig_vals = self.eig_vals,
                eig_vects = tf.gather(self.eig_vects, inducing_indices, axis=0),
                kernel_type=self.kernel_type[s],
                **spec_kwargs,
            )

        
        s_out *= tf.cast(kwargs['gpk_var'], dtype=self.dists_dtype)
        # Add the nugget term
        s_out += tf.eye(s_out.shape[0], dtype=self.dists_dtype) * tf.cast(self.eps + kwargs[f'gpk_nugget'], dtype=self.dists_dtype)
        return s_out        
        
    @tf.function
    def _return_sigma_xid_stationary(self, gpk_l, gpk_v, dXs, kernel_type):
        """
        Computes the covariance matrix using the chosen kernel.

        Args:
            gp_l (float): Lengthscale parameter.
            gp_v (float): Variance parameter.

        Returns:
            tf.Tensor: Covariance matrix.
        """
        gpk_v = tf.cast(gpk_v, dtype=self.dists_dtype)
        gpk_l = tf.cast(gpk_l, dtype=self.dists_dtype)

        if kernel_type == 'RBF':
            cov_matrix = tf.square(gpk_v) * tf.exp(
                -tf.square(dXs) / (2.0 * tf.square(gpk_l))
            )
        elif kernel_type == 'matern52':
            sqrt5 = tf.cast(tf.sqrt(5.0), dtype=self.dists_dtype)
            frac1 = (sqrt5 * dXs) / gpk_l
            frac2 = (5.0 * tf.square(dXs)) / (3.0 * tf.square(gpk_l))
            cov_matrix = tf.square(gpk_v) * (1 + frac1 + frac2) * tf.exp(-frac1)
        elif kernel_type == 'laplace':
            cov_matrix = tf.square(gpk_l) * tf.exp(-dXs / gpk_l)
        else:
            raise ValueError("Unsupported kernel: {}".format(kernel_type))
        # Add nugget term for numerical stability
        return cov_matrix 
    
    @tf.function
    def _return_sigma_xid_linear(self, gpk_slope, gpk_const, Xs):
        '''linear kernel
        '''
        gpk_slope = tf.cast(gpk_slope, dtype=self.dists_dtype)
        gpk_const = tf.cast(gpk_const, dtype=self.dists_dtype)        
        cov_matrix = gpk_slope**2 * (Xs-gpk_const) * (tf.transpose(Xs) - gpk_const)
        return cov_matrix

    @tf.function
    def _return_sigma_xid_spectral(self, eig_vals, eig_vects, kernel_type, **kwargs):
        """
        Compute covariance matrix from spectral kernel:
        K = eig_vects @ diag(var * f(eig_vals)) @ eig_vects^T

        Args:
            eig_vals: Tensor of eigenvalues [M]
            eig_vects: Tensor of eigenvectors [N, M]
            kernel_type: str, e.g., 'spec_exp', 'spec_heat', etc.
            **kwargs: Hyperparameters depending on kernel_type

        Returns:
            Covariance matrix [N, N]
        """
        eig_vals = tf.cast(eig_vals, dtype=self.dists_dtype)
        
        if kernel_type == 'spec_exp':            
            gpk_l = tf.cast(kwargs['gpk_spec_l'], dtype=self.dists_dtype)
            gpk_v = tf.cast(kwargs['gpk_spec_v'], dtype=self.dists_dtype)
            f_lambda = tf.exp(-tf.sqrt(eig_vals) / gpk_l)
            # Scale filter
            filt = gpk_v * f_lambda  # shape [M]
            # Form covariance matrix
            cov_matrix = tf.matmul(eig_vects * filt[tf.newaxis, :], eig_vects, transpose_b=True)
        
        elif kernel_type == 'spec_heat':
            t = tf.cast(kwargs['gpk_spec_t'], dtype=self.dists_dtype)
            filt = tf.exp(-t * eig_vals)
            # Form covariance matrix
            cov_matrix = tf.matmul(eig_vects * filt[tf.newaxis, :], eig_vects, transpose_b=True)

        elif kernel_type == 'spec_ratquad':
            alpha = tf.cast(kwargs['gpk_spec_alpha'], dtype=self.dists_dtype)
            beta = tf.cast(kwargs['gpk_spec_beta'], dtype=self.dists_dtype)
            filt = tf.math.pow(1.0 + eig_vals / alpha, -beta)
            # Form covariance matrix
            cov_matrix = tf.matmul(eig_vects * filt[tf.newaxis, :], eig_vects, transpose_b=True)        

        elif 'spec_LBOwarp' in kernel_type:             
            # Warp distance and/or lengthscale & variance using LBO eigenvectors
            #       spec_LBOwarp_dx    → dx only
            #       spec_LBOwarp_dxl   → dx + l
            #       spec_LBOwarp_dxv   → dx + v
            #       spec_LBOwarp_dxlv  → dx + l + v
            use_l = 'l' in kernel_type
            use_v = 'v' in kernel_type            
            # [1] Weighted sum of eigenvectors (LBOwarp)
            dxLBO_w0 = tf.cast(kwargs['gpk_spec_dxm0'], dtype=self.dists_dtype)
            dxLBO_wX = tf.stack(
                [kwargs[f"gpk_spec_dxm{i}"] for i in range(1, len(eig_vals)+1)],
                axis=0
            )
            dxLBO_wX = tf.cast(dxLBO_wX, dtype=self.dists_dtype)
            # Warped distances
            warped_X = dxLBO_w0 + tf.matmul(eig_vects, dxLBO_wX) # [N, 1]
            warped_X = tf.squeeze(warped_X, axis=-1) # [N,]            
            dXs = compute_euclidean_distance_matrix(warped_X[...,tf.newaxis])        

            # [2] Lengthscale 
            if use_l:
                lLBO_w0 = tf.cast(kwargs['gpk_spec_lm0'], dtype=self.dists_dtype)
                lLBO_wX = tf.stack(
                    [kwargs[f"gpk_spec_lm{i}"] for i in range(1, len(eig_vals)+1)],
                    axis=0
                )
                lLBO_wX = tf.cast(lLBO_wX, dtype=self.dists_dtype)
                # Warped lengthscale
                warped_l = lLBO_w0 + tf.matmul(eig_vects, lLBO_wX) # [N, 1]
                warped_l = tf.squeeze(warped_l, axis=-1)
                # -> force positive
                gpk_l = tf.exp(warped_l) + tf.cast(self.eps, dtype=self.dists_dtype)                
                # If lengthscale is warped, we need to compute the norm_ for RBF kernel
                two_l_lT = 2 * tf.matmul(gpk_l[:,None], gpk_l[None,:])
                lsq_lTsq = tf.square(gpk_l[:,None]) + tf.square(gpk_l[None,:])
                norm_ = tf.sqrt(two_l_lT / lsq_lTsq)                
            else:
                gpk_l = tf.cast(kwargs['gpk_spec_l'], dtype=self.dists_dtype)

            # [3] Variance
            if use_v:
                # we model log-variance offsets per eigenvector, same pattern:
                vLBO_w0 = tf.cast(kwargs['gpk_spec_vm0'], dtype=self.dists_dtype)
                vLBO_wX = tf.stack(
                    [kwargs[f"gpk_spec_vm{i}"] for i in range(1, len(eig_vals)+1)],
                    axis=0
                )
                vLBO_wX = tf.cast(vLBO_wX, dtype=self.dists_dtype)

                warped_v = vLBO_w0 + tf.matmul(eig_vects, vLBO_wX)   # [N,1]
                warped_v = tf.squeeze(warped_v, axis=-1)            # [N,]
                # make positive
                gpk_v = tf.exp(warped_v) + tf.cast(self.eps, dtype=self.dists_dtype)
                gpk_v_square = (gpk_v[:,None] * gpk_v[None,:])
            else:
                gpk_v = tf.cast(kwargs['gpk_spec_v'], dtype=self.dists_dtype)                
                gpk_v_square = tf.square(gpk_v) 

            # Kernel 
            if use_l:
                cov_matrix = gpk_v_square * norm_ * tf.exp(-tf.square(dXs) / lsq_lTsq)    
            else:
                cov_matrix = gpk_v_square * tf.exp(
                    -tf.square(dXs) / (2.0 * tf.square(gpk_l))
                )

        elif kernel_type == 'spec_LBOGibbs':
            # * i know not really gibss...
            # https://gpss.cc/gpss21/slides/Heinonen2021.pdf equation 27)
            # Like the standard RBF kernel, but we let the lengthscale vary
            # -> defined by a weighted sum of eigenvectors
            
            # Load from kwargs
            gpk_v = tf.cast(kwargs['gpk_spec_v'], dtype=self.dists_dtype)
            dXs = kwargs['dXs']
            lLBO_w0 = tf.cast(kwargs['gpk_spec_lm0'], dtype=self.dists_dtype)
            lLBO_wX = tf.stack(
                [kwargs[f"gpk_spec_lm{i}"] for i in range(1, len(eig_vals)+1)],
                axis=0
            )
            lLBO_wX = tf.cast(lLBO_wX, dtype=self.dists_dtype)
            # Warped lengthscale
            warped_l = lLBO_w0 + tf.matmul(eig_vects, lLBO_wX) # [N, 1]
            warped_l = tf.squeeze(warped_l, axis=-1)
            # -> force positive
            gpk_l = tf.exp(warped_l) + tf.cast(self.eps, dtype=self.dists_dtype)                
            # If lengthscale is warped, we need to compute the norm_ for RBF kernel
            two_l_lT = 2 * tf.matmul(gpk_l[:,None], gpk_l[None,:])
            lsq_lTsq = tf.square(gpk_l[:,None]) + tf.square(gpk_l[None,:])
            norm_ = tf.sqrt(two_l_lT / lsq_lTsq)                

            # RBF
            # cov_matrix = tf.square(gpk_v) * norm_ * tf.exp(-tf.square(dXs) / lsq_lTsq)
            cov_matrix = (gpk_l[:,None] * gpk_l[None,:]) * tf.exp(-tf.square(dXs) / (2.0 * tf.square(gpk_v)))

            # LAPLACE
            # cov_matrix     = tf.square(gpk_v) * norm_ * tf.exp(-dXs / lsq_lTsq)

            # MATRERN
            # sqrt5 = tf.cast(tf.sqrt(5.0), dtype=self.dists_dtype)
            # frac1 = (sqrt5 * dXs) / gpk_l
            # frac2 = (5.0 * tf.square(dXs)) / (3.0 * tf.square(lsq_lTsq))
            # cov_matrix = tf.square(gpk_v) * norm_ * (1 + frac1 + frac2) * tf.exp(-frac1)


            # debug: eigenvalues
            # eigs = tf.linalg.eigvalsh(K)
            # tf.print("min(eigs):", eigs[0], "max(eigs):", eigs[-1])

            # optionally assert no big negatives
            # tf.debugging.check_numerics(eigs, message="NaN or Inf in eigenvalues!")
            # tf.debugging.assert_equal(eigs[0] >= -1e-8, True,
            #     message="Covariance is not PSD: negative eigenvalue detected."
            # )
        else:
            raise ValueError(f"Unknown spectral kernel type: {kernel_type}")


        return cov_matrix    
    
    def set_log_prob_fixed(self,**kwargs):
        # Create a one off covariance matrix -> then use it to get probability each time...
        # Get cov matrix
        self.cov_matrix = self._return_sigma(**kwargs)
        self.chol = tf.linalg.cholesky(tf.cast(self.cov_matrix, dtype=self.dists_dtype))
        # Get mean vector
        self.m_vect = self._return_mfunc(**kwargs)
        self.prec_matrix = tf.cast(tf.linalg.inv(self.cov_matrix), dtype=tf.float32)

        self.gp_prior_dist = tfd.MultivariateNormalTriL(
            loc=tf.squeeze(tf.cast(self.m_vect, dtype=tf.float32)), 
            scale_tril=tf.cast(self.chol, dtype=tf.float32),
            allow_nan_stats=False,
        )
        self.return_log_prob = self._return_log_prob_fixed

    @tf.function
    def _return_log_prob_unfixed(self, parameter, n_inducers=None, **kwargs):
        """
        Unfixed parameters using TensorFlow distribution.
        Recompute covariance and Cholesky decomposition on the fly.
        Optionally uses random selection of n_inducers for sparse GP approximation.
        """
        inducing_indices = kwargs.pop('inducing_indices', None)
        inducing_indices = self._return_inducing_idx(n_inducers=n_inducers, inducing_indices=inducing_indices)
        inducing_parameter = parameter
        # Get cov matrix
        cov_matrix = self._return_sigma(
            inducing_indices=inducing_indices,
            **kwargs,            
            )
        chol = tf.linalg.cholesky(tf.cast(cov_matrix, dtype=self.dists_dtype))
        # Get mean vector
        m_vect = self._return_mfunc(
            inducing_indices=inducing_indices,
            **kwargs
            )
        gp_prior_dist = tfd.MultivariateNormalTriL(
            loc=tf.squeeze(tf.cast(m_vect, dtype=tf.float32)), #tf.fill([self.n_vx], tf.squeeze(m_vect)),
            scale_tril=tf.cast(chol, dtype=tf.float32),
            allow_nan_stats=False,
        )
        return gp_prior_dist.log_prob(inducing_parameter)

    @tf.function
    def _return_log_prob_fixed(self, parameter, **kwargs):
        """
        Unfixed parameters using TensorFlow distribution.
        Recompute covariance and Cholesky decomposition on the fly.
        Optionally uses random selection of n_inducers for sparse GP approximation.
        """
        return self.gp_prior_dist.log_prob(parameter)

    @tf.function
    def _return_inducing_idx(self, n_inducers, inducing_indices=None):        
        if n_inducers is None or n_inducers >= self.n_vx:
            # Everything...
            return tf.range(self.n_vx)

        return inducing_indices 


class GPSpec(GPdistsM):
    
    def __init__(self, n_vx, eig_vects, eig_vals, **kwargs):
        """
        A ** TRUE ** spectral GP 
               
        Objective: LBO to construct a spectral GP 

        """
        super().__init__(n_vx, **kwargs)
        # Setup distance matrix and positive semidefinite control
        self.spec_method         = kwargs.get('spec_method', 'kappa') # Dense
        self.eps           = kwargs.get('eps', 1e-6)
        self.dists_dtype   = tf.float32 #kwargs.get('dists_dtype', tf.float64)
        self.eig_vals = tf.convert_to_tensor(eig_vals, dtype=self.dists_dtype)
        self.eig_vects = tf.convert_to_tensor(eig_vects, dtype=self.dists_dtype)
        self.mass_matrix = kwargs.get('mass_matrix', None)
        if self.mass_matrix is None:
            self.mass_matrix = tf.ones([self.n_vx], dtype=self.dists_dtype)
        else:
            self.mass_matrix = tf.constant(self.mass_matrix, dtype=self.dists_dtype)   # [V]
        
        self.n_lbo = eig_vects.shape[-1]

        self.kappa_kernel_list = []
        self.return_log_prob = self._return_log_prob_unfixed # by default, return log prob unfixed...
        self.gp_prior_dist = []

    def _update_pids_inv(self):
        self.pids_inv = {}
        self.pids_inv = {v:k for k,v in self.pids.items()}

    @tf.function
    def prior(self, param):
        # Compute the conditional log-probability of the parameter under the GP prior
        return NotImplementedError

    # *************************************************
    # *************************************************
    # *************************************************
    
    # *** KERNELS ***
    # -> add Spectral kernels (based on LBO)
    def add_xid_kappa_kernel(self, xid, **kwargs):
        ''' CHANGED - may alter original as well
        '''
        self.kernel_type[xid] = kwargs.get('kernel_type', 'spec_exp')                
        # if spectral, expect precomputed eigenpairs
        if self.kernel_type[xid] == 'spec_exp':
            self.pids[len(self.pids)] = f'gpk{xid}_kappa_l'
            self.pids[len(self.pids)] = f'gpk{xid}_kappa_v'

        elif self.kernel_type[xid] == 'spec_heat':
            self.pids[len(self.pids)] = f'gpk{xid}_kappa_t'

        elif self.kernel_type[xid] == 'spec_ratquad':
            self.pids[len(self.pids)] = f'gpk{xid}_kappa_alpha'
            self.pids[len(self.pids)] = f'gpk{xid}_kappa_beta'

        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type[xid]}.")
        self.kappa_kernel_list.append(xid)
        self._update_pids_inv()

    def add_spec_warp(self):
        ''' Add a warping function '''
        for i in range(1, self.eig_vals.shape[0]+1):
            self.pids[len(self.pids)] = f'gpk_warp_kappa_m{i}'                     
        self._update_pids_inv()
    
    @tf.function
    def _return_sigma_component(self, inducing_indices=None, **kwargs):
        ''' Return the full Vx x Vx covariance matrix components
        '''
        if inducing_indices is None:
            inducing_indices = tf.range(self.n_vx)
        # Start covariance matrix from zero...
        s_out = tf.zeros((len(inducing_indices),len(inducing_indices)), dtype=self.dists_dtype)

        # Add in any linear kernels
        for s in self.lin_kernel_list:
            s_out += self._return_sigma_xid_linear(
                gpk_slope=kwargs[f'gpk{s}_slope'],
                gpk_const=kwargs[f'gpk{s}_const'],
                Xs=tf.gather(self.Xs[s], inducing_indices, axis=0)
            )
        
        # Add in any stationary kernels (e.g., RBF)
        for s in self.stat_kernel_list:
            s_out += self._return_sigma_xid_stationary(
                gpk_l=kwargs[f'gpk{s}_l'],
                gpk_v=kwargs[f'gpk{s}_v'],
                dXs=tf.gather(tf.gather(self.dXs[s], inducing_indices, axis=0), inducing_indices, axis=1),
                kernel_type=self.kernel_type[s]
            )

        for s in self.spec_kernel_list:
            eig_vects = tf.gather(self.eig_vects, inducing_indices, axis=0)
            spec_kwargs = {i.replace(s,''):k for i,k in kwargs.items() if 'spec' in i}
            s_out += self._return_sigma_xid_spectral(
                eig_vals = self.eig_vals,
                eig_vects = eig_vects,
                kernel_type=self.kernel_type[s],
                **spec_kwargs,
            )
        return s_out
    
    # *** KAPPA 
    @tf.function
    def _return_kappa_diag_component(self, inducing_indices=None, **kwargs):
        ''' Return the kappa diagonal component
        '''
        if inducing_indices is None:
            inducing_indices = tf.range(self.n_vx)
        # Start covariance matrix from zero...
        k_out = np.zeros(self.n_lbo, dtype=self.dists_dtype)
        for s in self.kappa_kernel_list:
            spec_kwargs = {i.replace(s,''):k for i,k in kwargs.items() if 'kappa' in i}
            k_out += self._return_kappa_xid_spectral(
                eig_vals = self.eig_vals,
                kernel_type=self.kernel_type[s],
                **spec_kwargs,
            )
        
        return k_out        
    
    @tf.function     
    def _sigma_to_kappa_diag(self,sigma):
        K_proj = tf.matmul(self.eig_vects, tf.matmul(sigma, self.eig_vects, transpose_a=True))
        # # 2) Form the summed covariance in subspace
        # K_sum = K_proj + tf.linalg.diag(kappa)  # [n,n]
        # For now cheat and just get diagonal
        return tf.linalg.diag_part(K_proj)
        
    @tf.function
    def _return_kappa_xid_spectral(
        self,
        eig_vals,        # Tensor shape [M]
                            kernel_type,     # 'spec_exp', 'spec_heat', or 'spec_ratquad'
                            **kwargs) -> tf.Tensor:  # returns Tensor [M]
        """
        Compute spectral variances kappa_i = var * f(lambda_i) for each eigenvalue.

        Args:
            eig_vals:    Tensor of shape [M], the Laplacian eigenvalues.
            kernel_type: 'spec_exp', 'spec_heat', or 'spec_ratquad'.
            **kwargs:    Hyperparameters:
                - spec_exp:     gpk_spec_l, gpk_spec_v
                - spec_heat:    gpk_spec_t
                - spec_ratquad: gpk_spec_alpha, gpk_spec_beta

        Returns:
            kappa: Tensor [M], the diagonal of the spectral covariance.
        """
        # cast to model dtype
        eig_vals = tf.cast(eig_vals, dtype=self.dists_dtype)

        if kernel_type == 'spec_exp':
            # exponential kernel: kappa_i = sigma2 * exp(-sqrt(lambda_i)/l)
            l = tf.cast(kwargs['gpk_kappa_l'], dtype=self.dists_dtype)
            v = tf.cast(kwargs['gpk_kappa_v'], dtype=self.dists_dtype)
            filt = tf.exp(-tf.sqrt(eig_vals) / l)
            kappa = v * v * filt

        elif kernel_type == 'spec_heat':
            # heat kernel: kappa_i = exp(-t * lambda_i)
            t = tf.cast(kwargs['gpk_kappa_t'], dtype=self.dists_dtype)
            kappa = tf.exp(-t * eig_vals)

        elif kernel_type == 'spec_ratquad':
            # rational quadratic: kappa_i = (1 + lambda_i/alpha)^(-beta)
            alpha = tf.cast(kwargs['gpk_kappa_alpha'], dtype=self.dists_dtype)
            beta  = tf.cast(kwargs['gpk_kappa_beta'], dtype=self.dists_dtype)
            kappa = tf.pow(1.0 + eig_vals / alpha, -beta)

        else:
            raise ValueError("Unknown spectral kernel type: {}".format(kernel_type))

        return kappa


    @tf.function
    def _return_log_prob_unfixed(self, parameter, n_inducers=None, **kwargs):
        """
        Unfixed parameters using TensorFlow distribution.
        Recompute covariance and Cholesky decomposition on the fly.
        Optionally uses random selection of n_inducers for sparse GP approximation.
        """
        inducing_indices = kwargs.pop('inducing_indices', None)
        inducing_indices = self._return_inducing_idx(n_inducers=n_inducers, inducing_indices=inducing_indices)
        eig_vects = tf.gather(self.eig_vects, inducing_indices, axis=0)
        mass_matrix = tf.gather(self.mass_matrix, inducing_indices, axis=0)

        # [1] Get mean function
        m_vect = self._return_mfunc(
            inducing_indices=inducing_indices,
            **kwargs
            )
        nvx = tf.shape(m_vect)[0]
        # -> demean 
        dm_parameter = parameter - m_vect      
        # Project y into the spectral domain: z = Phi^T y
        # -> not including mass_matrix
        # proj_dm_parameter = tf.linalg.matvec(eig_vects, dm_parameter, transpose_a=True)  # [M]    
        # Including mass matrix
        w = tf.sqrt(mass_matrix)
        yw = dm_parameter * w
        phiw = eig_vects * tf.reshape(w, [-1, 1])
        proj_dm_parameter = tf.linalg.matvec(phiw, yw, transpose_a=True)
        # Warping - if appropriate 
        warping_vect = tf.stack(
            [kwargs.get(f"gpk_warp_spec_m{i}", 1.0) for i in range(1, self.n_lbo+1)],
        )
        warping_vect = tf.squeeze(warping_vect)
        warped_proj = proj_dm_parameter #* warping_vect        
        # Compute residual norm^2: r = y - Phi z
        y_recon = tf.linalg.matvec(eig_vects, proj_dm_parameter)         # [N]
        r2 = tf.reduce_sum((dm_parameter - y_recon) ** 2)                # scalar        

        # Get kappa
        kappa = self._return_kappa(
            inducing_indices=inducing_indices,
            **kwargs,            
            ) 
        # kappa = kappa * (warping_vect**2)

        gp_prior_dist = tfd.MultivariateNormalDiag(loc=tf.zeros(self.n_lbo), scale_diag=tf.sqrt(kappa))
        log_prob = gp_prior_dist.log_prob(warped_proj)

        # Residual treated as iid noise: ||r||^2 / (2 σ2_obs) + const
        gp_nugget = tf.cast(self.eps + kwargs[f'gpk_nugget'], dtype=self.dists_dtype)
        lr = -0.5 * (r2 / gp_nugget + (tf.cast(nvx, self.dists_dtype) - tf.cast(self.n_lbo, self.dists_dtype)) * tf.math.log(2 * tf.constant(np.pi, dtype=self.dists_dtype) * gp_nugget))
        return log_prob + lr
    
    def set_log_prob_fixed(self, **kwargs):
        return None

    @tf.function
    def _return_log_prob_fixed(self, parameter, **kwargs):
        # -> demean 
        dm_parameter = parameter - self.m_vect      
        # Project y into the spectral domain: z = Phi^T y
        # -> not including mass_matrix
        # proj_dm_parameter = tf.linalg.matvec(eig_vects, dm_parameter, transpose_a=True)  # [M]    
        # Including mass matrix
        yw = dm_parameter * self.w
        phiw = self.eig_vects * tf.reshape(self.w, [-1, 1])
        proj_dm_parameter = tf.linalg.matvec(phiw, yw, transpose_a=True)
        warped_proj = proj_dm_parameter * self.warping_vect        
        # Compute residual norm^2: r = y - Phi z
        y_recon = tf.linalg.matvec(self.eig_vects, proj_dm_parameter)         # [N]
        r2 = tf.reduce_sum((dm_parameter - y_recon) ** 2)                # scalar        
        log_prob = self.gp_prior_dist.log_prob(warped_proj)

        # Residual treated as iid noise: ||r||^2 / (2 σ2_obs) + const
        
        lr = -0.5 * \
            (r2 / self.gp_nugget + (tf.cast(self.n_vxvx, self.dists_dtype) \
            - tf.cast(self.n_lbo, self.dists_dtype)) * tf.math.log(2 * tf.constant(np.pi, dtype=self.dists_dtype) * self.gp_nugget))
        return log_prob + lr

# **************************** 
# @tf.function
def sigma2kappa(sigma, eig_vects):
    """
    Project a dense covariance matrix into the spectral domain.

    Args:
        sigma: tf.Tensor of shape (N, N), the covariance matrix in real space.
        eig_vects: tf.Tensor of shape (N, L), the first L LBO eigenvectors.

    Returns:
        kappa: tf.Tensor of shape (L,), diagonal variances in spectral domain.
    """
    # Ensure symmetry for numerical stability
    # sigma = (sigma + tf.transpose(sigma)) * 0.5

    # Project covariance onto eigenbasis: phi^T * sigma * phi
    # For efficiency compute sigma * phi first
    sigma_phi = tf.matmul(sigma, eig_vects)  # shape (N, L)
    # Then compute diagonal entries: sum over N: phi_i * (sigma * phi)_i
    kappa = tf.reduce_sum(eig_vects * sigma_phi, axis=0)

    return kappa


# @tf.function
def kappa2sigma(kappa, eig_vects):
    """
    Reconstruct the dense covariance matrix from spectral variances.

    Args:
        kappa: tf.Tensor of shape (L,), spectral variances.
        eig_vects: tf.Tensor of shape (N, L), the first L LBO eigenvectors.

    Returns:
        sigma: tf.Tensor of shape (N, N), reconstructed covariance matrix.
    """

    # Form diagonal spectral matrix
    Lambda = tf.linalg.diag(kappa)

    # Reconstruct covariance: phi * Lambda * phi^T
    sigma_recon = tf.matmul(eig_vects, tf.matmul(Lambda, eig_vects, transpose_b=True))

    # Ensure symmetry
    sigma_recon = (sigma_recon + tf.transpose(sigma_recon)) * 0.5

    return sigma_recon

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


# *** SAMPLER ***


# Copied from .utils.mcmc
# @tf.function(autograph=False, jit_compile=False)
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
        burnin=50,
        parallel_iterations=1,
        ):

    # bloop
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
    
    # start = tf.timestamp()
    # Sampling from the chain.
    samples, stats = tfp.mcmc.sample_chain(
        num_results=burnin + num_steps,
        current_state=init_state,
        kernel=adaptive_sampler,
        trace_fn=trace_fn,
        parallel_iterations=parallel_iterations,
        )

    # duration = tf.timestamp() - start
    # stats['elapsed_time'] = duration

    return samples, stats
# Summarize MCMC

def get_mcmc_summary(sampler, burnin=100, pc_range=25):
    keys = sampler[0].keys()
    n_voxels = len(sampler)
    bpars = {}
    bpars_m = {}
    for p in list(keys): 
        m = []
        q1 = []
        q2 = []
        uc = []
        std = []
        for idx in range(n_voxels):
            this_p = sampler[idx][p][burnin:].to_numpy()
            m.append(np.percentile(this_p,50))
            tq1 = np.percentile(this_p, pc_range)
            tq2 = np.percentile(this_p, 100-pc_range)
            tuc = tq2 - tq1
            
            q1.append(tq1)
            q2.append(tq2)
            uc.append(tuc)
            std.append(np.std(this_p))
        bpars_m[p] = np.array(m)
        bpars[f'm_{p}'] = np.array(m)
        bpars[f'q1_{p}'] = np.array(q1)
        bpars[f'q2_{p}'] = np.array(q2)
        bpars[f'uc_{p}'] = np.array(uc)
        bpars[f'std_{p}'] = np.array(std)
    return pd.DataFrame(bpars) #, pd.DataFrame(bpars_m)
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
