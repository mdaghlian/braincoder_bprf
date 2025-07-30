import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb
import math
import copy

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
        cov_matrix = self._kernel_output(gp_lengthscale, gp_variance, dists)
        # Add nugget term for numerical stability / noise 
        return cov_matrix + tf.eye(cov_matrix.shape[0], dtype=self.dists_dtype) * tf.cast(self.eps + gp_nugget, dtype=self.dists_dtype)
    
    @tf.function
    def _kernel_output(self, gp_lengthscale, gp_variance, dists):
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
        return cov_matrix
    
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
        # if n_inducers is None or n_inducers >= self.n_vx:
        #     # Full GP
        #     cov_matrix = self.return_sigma(gp_lengthscale, gp_variance, gp_nugget)
        #     chol = tf.linalg.cholesky(tf.cast(cov_matrix, dtype=self.dists_dtype))

        #     gp_prior_dist = tfd.MultivariateNormalTriL(
        #         loc=tf.fill([self.n_vx], tf.squeeze(gp_mean)),
        #         scale_tril=tf.cast(chol, dtype=tf.float32),
        #         allow_nan_stats=False,
        #     )
        #     return gp_prior_dist.log_prob(parameter)
        # else:
        # bleee
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
    def _return_inducing_idx_and_dists(self, n_inducers, inducing_indices):
        # if n_inducers is None or n_inducers >= self.n_vx:
        #     inducing_indices = tf.range(self.n_vx)
        # Gather distances for the selected inducing points
        inducing_dists = tf.gather(tf.gather(self.dists, inducing_indices, axis=0), inducing_indices, axis=1)
        return inducing_indices, inducing_dists
    

    # ****************** PREDICTION ********************
    def _return_sigma_for_pred(self, **kwargs):
        '''
        '''
        gp_lengthscale = kwargs.pop('gp_lengthscale')
        gp_variance = kwargs.pop('gp_variance')
        gp_nugget = kwargs.pop('gp_nugget', 0.0)
        dists = kwargs.pop('dists', None)
        if dists is None:
            dists = self.dists
        # If dists is symetric - use standard method
        if dists.shape[0] == dists.shape[1] and tf.reduce_all(tf.equal(dists, tf.transpose(dists))):
            return self.return_sigma(gp_lengthscale, gp_variance, gp_nugget, dists=dists) 
        else:
            # If dists is not symmetric - need the kernel output (avoid the 'eye' )
            return self._kernel_output(gp_lengthscale, gp_variance, dists)

    def _predict(self, **kwargs):
        par = kwargs.pop('par', None)
        assert par is not None, "Parameter must be provided for prediction."
        Xs = kwargs.pop('Xs', None)
        dists = kwargs.pop('dists', None)
        if (dists is None) & (Xs is not None):
            dists = pairwise_euclidean_distance(Xs, Xs)
        if dists is None:
            dists = self.dists # [N X N]
        dists_new = kwargs.pop('dists_new', None) # [M X M]
        dists_old_new = kwargs.pop('dists_old_new', None) # [N X M]
        if dists_new is None and dists_old_new is None:
            assert Xs is not None, "Xs must be provided if dists_new or dists_old_new are not specified."
            Xs_new = kwargs.pop('Xs_new', None)
            assert Xs_new is not None, "Xs_new must be provided if dists_new or dists_old_new are not specified."
            dists_new = pairwise_euclidean_distance(Xs_new,Xs_new)
            dists_old_new = pairwise_euclidean_distance(Xs, Xs_new)
            print(dists_old_new.shape)
        
        K = self.return_sigma(dists= dists,**kwargs).numpy()
        K_s = self._return_sigma_for_pred(dists= dists_old_new, **kwargs).numpy()
        K_ss = self._return_sigma_for_pred(dists=dists_new, **kwargs).numpy()
        
        # 2. Compute inverse of K (for stability you could use cholesky + solve)
        L = np.linalg.cholesky(K)           # K = L Lᵀ        
        # Solve for alpha = K^{-1} z via two triangular solves:
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, par))

        # 3. Predictive mean: K_sᵀ α
        z_mean = K_s.T @ alpha                 # shape (m,)

        # 4. Predictive variance:
        #    v = solve(L, K_s)   so that vᵀ v = K_sᵀ K^{-1} K_s
        v = np.linalg.solve(L, K_s)
        cov_new = K_ss - v.T @ v               # shape (m, m)
        # numerical errors can make cov_new slightly non-PSD → clip
        z_var = np.clip(np.diag(cov_new), 0, None)
        z_std = np.sqrt(z_var)

        return z_mean, z_std


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
        s_out += tf.eye(len(inducing_indices), dtype=self.dists_dtype) * tf.cast(self.eps + kwargs[f'gpk_nugget'], dtype=self.dists_dtype)
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

class GPSpecMM(GPdistsM):
    
    def __init__(self, n_vx, eig_vects, eig_vals, **kwargs):
        """        
        Objective: LBO to construct a spectral GP 
        -> WITH MASS MATRIX
        """
        super().__init__(n_vx, **kwargs)
        # Setup distance matrix and positive semidefinite control
        self.spec_method         = kwargs.get('spec_method', 'diag') # dense
        if self.spec_method == 'diag':
            self._return_kappa = self._return_kappa_diag     
            self._return_kappa_dist = self._return_kappa_dist_diag
        if self.spec_method == 'dense':
            self._return_kappa = self._return_kappa_dense     
            self._return_kappa_dist = self._return_kappa_dist_dense               
        self.eps           = kwargs.get('eps', 1e-6)
        self.dists_dtype   = tf.float32 #kwargs.get('dists_dtype', tf.float64)
        self.eig_vals = tf.convert_to_tensor(eig_vals, dtype=self.dists_dtype)
        self.eig_vects = tf.convert_to_tensor(eig_vects, dtype=self.dists_dtype)
        self.mass_matrix = kwargs.get('mass_matrix', None)
        if self.mass_matrix is None:
            self.mass_matrix = tf.ones([self.n_vx], dtype=self.dists_dtype)
        else:
            self.mass_matrix = tf.constant(self.mass_matrix, dtype=self.dists_dtype)   # [V]
        self.mass_matrix_sqrt = tf.sqrt(self.mass_matrix)  # useful for computations
        
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
    def _return_kappa_diag(self, inducing_indices, **kwargs):
        ''' To return the (L,) tensor (for diagonal)'''
        if inducing_indices is None:
            inducing_indices = tf.range(self.n_vx)
        eig_vects = tf.gather(self.eig_vects, inducing_indices, axis=0)
        mass_matrix_sqrt = tf.gather(self.mass_matrix_sqrt, inducing_indices, axis=0)

        k_out = tf.zeros(self.n_lbo, dtype=self.dists_dtype)            
        s_out = self._return_sigma_component(inducing_indices=inducing_indices, **kwargs)
        phiw = eig_vects * mass_matrix_sqrt[:, None]    # shape [N, L]
        # And weight sigma*phi as well:
        sigma_phi = tf.matmul(s_out * mass_matrix_sqrt[:, None], phiw)  # [N, L]
        # Then the diagonal variances:
        k_out += tf.reduce_sum(phiw * sigma_phi, axis=0)  # [L]        
        
        for s in self.kappa_kernel_list:
            spec_kwargs = {i.replace(s,''):k for i,k in kwargs.items() if 'kappa' in i}
            k_out += self._return_kappa_xid_spectral(
                eig_vals = self.eig_vals,
                kernel_type=self.kernel_type[s],
                **spec_kwargs,
            )
                                
        k_out *= tf.cast(kwargs['gpk_var'], dtype=self.dists_dtype)    
        return k_out

    @tf.function
    def _return_kappa_dense(self, inducing_indices, **kwargs):
        ''' To return the (L,) tensor (for diagonal)'''
        if inducing_indices is None:
            inducing_indices = tf.range(self.n_vx)
        eig_vects = tf.gather(self.eig_vects, inducing_indices, axis=0)
        mass_matrix = tf.gather(self.mass_matrix, inducing_indices, axis=0)
        k_out = tf.zeros((self.n_lbo,self.n_lbo), dtype=self.dists_dtype)
        
        s_out = self._return_sigma_component(inducing_indices=inducing_indices, **kwargs)
        k_out += sigma2kappa_dense(s_out, eig_vects, mass_matrix)        

        for s in self.kappa_kernel_list:
            spec_kwargs = {i.replace(s,''):k for i,k in kwargs.items() if 'kappa' in i}
            k_diag = self._return_kappa_xid_spectral(
                eig_vals = self.eig_vals,
                kernel_type=self.kernel_type[s],
                **spec_kwargs,
            )
            k_out += tf.linalg.diag(k_diag)
                                
        k_out *= tf.cast(kwargs['gpk_var'], dtype=self.dists_dtype)            
        return k_out + tf.eye(self.n_lbo, dtype=self.dists_dtype) * tf.cast(self.eps+kwargs['gpk_nugget'], dtype=self.dists_dtype)

    @tf.function
    def _return_kappa_dist_diag(self, kappa):
        return tfd.MultivariateNormalDiag(loc=tf.zeros(self.n_lbo), scale_diag=tf.sqrt(kappa))
    @tf.function
    def _return_kappa_dist_dense(self, kappa):
        chol = tf.linalg.cholesky(tf.cast(kappa, dtype=tf.float64))
        gp_prior_dist = tfd.MultivariateNormalTriL(
            loc=tf.zeros(self.n_lbo), 
            scale_tril=tf.cast(chol, dtype=tf.float32),
            allow_nan_stats=False,
        )        
        return gp_prior_dist

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
        mass_matrix_sqrt = tf.gather(self.mass_matrix_sqrt, inducing_indices, axis=0)[...,None]

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
        yw = dm_parameter *tf.squeeze(mass_matrix_sqrt,-1)
        phiw = eig_vects * mass_matrix_sqrt
        proj_dm_parameter = tf.tensordot(phiw, yw, axes=[[0],[0]])  # [M]

        # Compute residual norm^2: r = y - Phi z
        y_recon = tf.tensordot(phiw, proj_dm_parameter, axes=[[1],[0]])  # [N]
        r2 = tf.reduce_sum((dm_parameter - y_recon) ** 2)                # scalar        

        # Get kappa
        kappa = self._return_kappa(
            inducing_indices=inducing_indices,
            **kwargs,            
            ) 
        gp_prior_dist = self._return_kappa_dist(kappa)
        log_prob = gp_prior_dist.log_prob(proj_dm_parameter)

        # Residual treated as iid noise: ||r||^2 / (2 σ2_obs) + const
        gp_nugget = tf.cast(self.eps + kwargs[f'gpk_nugget'], dtype=self.dists_dtype)
        lr = -0.5 * (r2 / gp_nugget + (tf.cast(nvx, self.dists_dtype) - tf.cast(self.n_lbo, self.dists_dtype)) * tf.math.log(2 * tf.constant(np.pi, dtype=self.dists_dtype) * gp_nugget))
        return log_prob + lr
    
    def set_log_prob_fixed(self, **kwargs):
        return None

# **************************** 
@tf.function
def sigma2kappa_diag(sigma, eig_vects, mass_matrix_sqrt):
    """
    Project a dense covariance matrix into the spectral domain,
    accounting for the mass matrix weights.

    Args:
        sigma: tf.Tensor [N, N], the covariance matrix Σ.
        eig_vects: tf.Tensor [N, L], the first L LBO eigenvectors Φ.
        mass_matrix_sqrt: tf.Tensor [N], the sqrt of the mass matrix diag(M)^(1/2).

    Returns:
        kappa: tf.Tensor [L], the diagonal variances in spectral domain.
    """
    # weight eigenvectors:
    phiw = eig_vects * tf.expand_dims(mass_matrix_sqrt, -1)  # [N, L]

    # apply sigma, then weight the result:
    sigma_phiw = tf.matmul(sigma, phiw)                      # [N, L]
    weighted = sigma_phiw * tf.expand_dims(mass_matrix_sqrt, -1)  # [N, L]

    # extract diagonal entries:
    kappa = tf.reduce_sum(phiw * weighted, axis=0)           # [L]

    return kappa

@tf.function
def sigma2kappa_dense(sigma, eig_vects, mass_matrix):
    """
    Project a dense covariance matrix into the spectral domain,
    accounting for the mass‐matrix weights.

    Args:
        sigma: tf.Tensor of shape (N, N), the covariance matrix Σ.
        eig_vects: tf.Tensor of shape (N, L), the first L LBO eigenvectors Φ.
        mass_matrix: tf.Tensor of shape (N,), the diagonal of M.

    Returns:
        kappa: tf.Tensor of shape (L, L), the full spectral covariance.
    """
    # weight the eigenvectors: M * Φ
    mp = eig_vects * tf.expand_dims(mass_matrix, -1)        # [N, L]

    # apply Σ to the weighted eigenvectors: Σ (M Φ)
    sigma_mp = tf.matmul(sigma, mp)                         # [N, L]

    # form Φ^T M Σ M Φ = (M Φ)^T (Σ (M Φ))
    kappa = tf.matmul(mp, sigma_mp, transpose_a=True)       # [L, L]

    return kappa

# @tf.function
@tf.function
def kappa_diag2sigma(kappa, eig_vects):
    """
    Reconstruct the dense covariance matrix from spectral variances,
    accounting for the mass matrix.
    
    Args:
        kappa: tf.Tensor of shape (L,), spectral variances.
        eig_vects: tf.Tensor of shape (N, L), the first L LBO eigenvectors Φ.
    
    Returns:
        sigma_recon: tf.Tensor of shape (N, N), reconstructed Σ.
    """
    kappa_matrix = tf.linalg.diag(kappa)                          # [n_modes, n_modes]
    intermediate = tf.linalg.matmul(eig_vects, kappa_matrix)            # [nvx, n_modes]
    vertex_cov   = tf.linalg.matmul(intermediate, eig_vects, transpose_b=True)  # [nvx, nvx]
    return vertex_cov



@tf.function
def kappa_dense2sigma(kappa, eig_vects):
    """
    Reconstruct the dense covariance matrix from full spectral covariance
    
    Args:
        kappa: tf.Tensor of shape (L, L), full spectral covariance.
        eig_vects: tf.Tensor of shape (N, L), the first L LBO eigenvectors Φ.
    
    Returns:
        sigma_recon: tf.Tensor of shape (N, N), reconstructed Σ.
    """
    intermediate = tf.linalg.matmul(eig_vects, kappa)         # [nvx, n_modes]
    vertex_cov   = tf.linalg.matmul(intermediate, eig_vects, transpose_b=True)  # [nvx, nvx]
    return vertex_cov



# *****
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

def pairwise_euclidean_distance(X1,X2):
    """
    Computes the pairwise Euclidean distance matrix 
    
    Args:
        X1: A [n x d] tensor of embedded coordinates.
        X2: A [m x d] tensor of embedded coordinates.
        eps: A small number for numerical stability.
        
    Returns:
        A [n x m] tensor of Euclidean distances.
    """
    # 1. Expand dimensions of X for broadcasting:
    X1_expanded = X1[:,None, :]
    X2_expanded = X2[None, :, :]
    # 2. Compute pairwise differences:
    diff = X1_expanded - X2_expanded
    # 3. Compute the Euclidean distance matrix:
    D_euc = np.sqrt(np.sum(diff**2, axis=-1))
    return D_euc 