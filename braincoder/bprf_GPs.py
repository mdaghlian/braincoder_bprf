import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb
import math
import copy


# ******************* return log probability ****************
# - distance based gp prior 
class GP():
    def __init__(self, n_vx, **kwargs):
        """
        Gaussian process        

        N(m,K+nugget)
        -> we construct m, the mean function
        -> we construct K, the covariance function
        -> with a nugget, for stability        

        Args:
            **kwargs: Optional parameters for controlling behavior, such as:
                - psd_control: Method for ensuring positive semidefiniteness.
                - gp_dtype: Data type for tensor conversion.
                - kernel: Choice of covariance function (default: 'RBF').
        """
        self.n_vx = tf.Variable(n_vx, dtype=tf.int32, name="n_vx")

        # Setup distance matrix and positive semidefinite control
        self.psd_control   = kwargs.get('psd_control', 'euclidean')  # 'euclidean' or 'none'
        self.eps           = kwargs.get('eps', 1e-6)
        self.embedding_dim = kwargs.get('embedding_dim', 10)
        self.gp_dtype   = kwargs.get('gp_dtype', tf.float64)

        self.stat_kernel_list = []
        self.lin_kernel_list = []
        self.warp_kernel_list = []
        self.mfunc_list = []
        self.mfunc_bijector = tfb.Identity()
        self.Xs = {}
        self.dXs = {}
        self.n_inducers = None  # if None, use all points
        self.inducer_idx = None  # if None, use all points
        self.nystrom = False
        self.kernel_type = {}
        self.pids = {}
        # Index of parameters to be passed...
        self.pids[0] = 'gpk_nugget' # Global nugget term
        self.pids[1] = 'mfunc_mean' # Global mean term 
        self.pids_inv = {}
        self._update_pids_inv()
        self.return_log_prob = self._return_log_prob_unfixed # by default, return log prob unfixed...
        self.gp_prior_dist = None

    def _update_pids_inv(self):
        self.pids_inv = {}
        self.pids_inv = {v:k for k,v in self.pids.items()}
    
    def update_n_vx(self, new_value):
        self.n_vx.assign(new_value)
                        
    # **************** MEAN FUNCTIONS ***************
    def add_xid_linear_mfunc(self, xid, **kwargs):
        ''' add linear mean function
        '''
        Xs = kwargs.get('Xs', None)
        Xs = tf.convert_to_tensor(Xs, dtype=self.gp_dtype)
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
    def _return_mfunc(self, **kwargs):
        '''Return the mean function
        '''
        # Start of with zero then add global mean
        m_out = tf.zeros(self.n_vx, dtype=self.gp_dtype) + tf.cast(kwargs['mfunc_mean'], self.gp_dtype) # global mean...
        # then add any regressors...
        for m in self.mfunc_list:
            slopes = tf.stack([kwargs[f'mfunc{m}_slope{i}'] for i in range(self.Xs[m].shape[1])], axis=0)  # [D]
            m_out += tf.reduce_sum(tf.cast(slopes, dtype=self.gp_dtype) * tf.transpose(self.Xs[m]), axis=0) 
        return self.mfunc_bijector(tf.cast(m_out, dtype=tf.float32))
    
    def add_mfunc_bijector(self, bijector_type, **kwargs):
        ''' add transformations to parameters so that they are fit smoothly        
        
        identity        - do nothing
        softplus        - don't let anything be negative

        '''
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
        self.dXs[xid] = tf.convert_to_tensor(dXs, dtype=self.gp_dtype)
        self.dXs[xid] = (self.dXs[xid] + tf.transpose(self.dXs[xid])) / 2.0        
        # Add a lengthscale & a variance
        self.pids[len(self.pids)] = f'gpk{xid}_l'
        self.pids[len(self.pids)] = f'gpk{xid}_v'

        # Update the inverse dictionary
        self._update_pids_inv()        
    
    # # -> add Linear kernels
    # def add_xid_linear_kernel(self, xid, **kwargs):
    #     ''' add a kernel
    #     '''
    #     Xs = kwargs.get('Xs', None)
    #     self.kernel_type[xid] = 'linear'
    #     self.lin_kernel_list.append(xid)
        
    #     self.Xs[xid] = tf.expand_dims(tf.convert_to_tensor(Xs, dtype=self.gp_dtype), axis=1)
    #     # Add a lengthscale & a variance
    #     self.pids[len(self.pids)] = f'gpk{xid}_slope'
    #     self.pids[len(self.pids)] = f'gpk{xid}_const'

    #     # Update the inverse dictionary
    #     self._update_pids_inv()    
    
    def add_xid_warp_kernel(self, xid, Xs, **kwargs):
        self.Xs[xid] = tf.convert_to_tensor(Xs, dtype=self.gp_dtype)
        self.pids[len(self.pids)] = f'gpk{xid}_v'
        # Distance for RBF type kernel comes from warped LBO
        for i in range(self.Xs[xid].shape[1]):
            self.pids[len(self.pids)] = f'gpk{xid}_w{i}' 
        self.warp_kernel_list.append(xid)
        self._update_pids_inv()
                        

    def add_nystrom_approximation(self, n_inducers, inducer_idx=None):
        ''' Use nystrom approximation to speed up the GP
        '''
        self.n_inducers = n_inducers
        self.inducer_idx = inducer_idx
        self.nystrom = True
        if self.inducer_idx is not None:
            self.inducer_idx = tf.convert_to_tensor(self.inducer_idx, dtype=tf.int32)
        else:
            self.inducer_idx = tf.random.shuffle(tf.range(self.n_vx))[:self.n_inducers]
        self.return_log_prob = self._return_log_prob_nystrom


    @tf.function
    def _return_sigma_full(self, **kwargs):
        ''' Putting all the kernels together - > return the full covariance matrix
        '''
        # Start covariance matrix from zero...
        s_out = tf.zeros((self.n_vx,self.n_vx), dtype=self.gp_dtype)        
        
        # Add in any stationary kernels (e.g., RBF)
        for s in self.stat_kernel_list:
            s_out += self._return_sigma_xid_stationary(
                gpk_l=kwargs[f'gpk{s}_l'],
                gpk_v=kwargs[f'gpk{s}_v'],
                dXs=self.dXs[s],
                kernel_type=self.kernel_type[s]
            )
        for s in self.warp_kernel_list:
            s_kernel_type = s.split('_')[-1]
            dXs = self._return_warp_dXs(
                s, **kwargs,
            )
            s_out += self._return_sigma_xid_stationary(
                gpk_l=1.0,
                gpk_v=kwargs[f'gpk{s}_v'],
                dXs=dXs,
                kernel_type=s_kernel_type
            )   
        # # Add in any linear kernels
        # for s in self.lin_kernel_list:
        #     s_out += self._return_sigma_xid_linear(
        #         gpk_slope=kwargs[f'gpk{s}_slope'],
        #         gpk_const=kwargs[f'gpk{s}_const'],
        #         Xs=self.Xs[s],
        #     )
        # Add the nugget term
        s_out += tf.linalg.diag(tf.ones(self.n_vx, dtype=self.gp_dtype)) * tf.cast(self.eps + kwargs[f'gpk_nugget'], dtype=self.gp_dtype)
        return s_out        
    
    @tf.function
    def _return_warp_dXs(self, xid, **kwargs):
        # [1] Weighted sum of eigenvectors (LBOwarp)
        wX = tf.stack(
            [kwargs[f"gpk{xid}_w{i}"] for i in range(self.Xs[xid].shape[1])],
            axis=0
        )
        wX = tf.cast(wX, dtype=self.gp_dtype)
        # Warped distances
        warp_X = tf.matmul(self.Xs[xid], wX) # [N, 1]
        warp_X = tf.squeeze(warp_X, axis=-1) # [N,]            
        warp_dXs = compute_euclidean_distance_matrix(warp_X[...,tf.newaxis])                
        return warp_dXs 
    
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
        gpk_v = tf.cast(gpk_v, dtype=self.gp_dtype)
        gpk_l = tf.cast(gpk_l, dtype=self.gp_dtype)

        if kernel_type == 'RBF':
            cov_matrix = tf.square(gpk_v) * tf.exp(
                -tf.square(dXs) / (2.0 * tf.square(gpk_l))
            )
        elif kernel_type == 'matern52':
            sqrt5 = tf.cast(tf.sqrt(5.0), dtype=self.gp_dtype)
            frac1 = (sqrt5 * dXs) / gpk_l
            frac2 = (5.0 * tf.square(dXs)) / (3.0 * tf.square(gpk_l))
            cov_matrix = tf.square(gpk_v) * (1 + frac1 + frac2) * tf.exp(-frac1)
        elif kernel_type == 'laplace':
            cov_matrix = tf.square(gpk_v) * tf.exp(-dXs / gpk_l)
        else:
            raise ValueError("Unsupported kernel: {}".format(kernel_type))
        # Add nugget term for numerical stability
        return cov_matrix 
    
    # @tf.function
    # def _return_sigma_xid_linear(self, gpk_slope, gpk_const, Xs):
    #     '''linear kernel
    #     '''
    #     gpk_slope = tf.cast(gpk_slope, dtype=self.gp_dtype)
    #     gpk_const = tf.cast(gpk_const, dtype=self.gp_dtype)        
    #     cov_matrix = gpk_slope**2 * (Xs-gpk_const) * (tf.transpose(Xs) - gpk_const)
    #     return cov_matrix

    def set_log_prob_fixed(self,**kwargs):
        # Create a one off covariance matrix -> then use it to get probability each time...
        # Get cov matrix
        self.cov_matrix = self._return_sigma_full(**kwargs)
        self.chol = tf.linalg.cholesky(tf.cast(self.cov_matrix, dtype=self.gp_dtype))
        # Get mean vector
        self.m_vect = self._return_mfunc(**kwargs)

        self.gp_prior_dist = tfd.MultivariateNormalTriL(
            loc=tf.squeeze(tf.cast(self.m_vect, dtype=tf.float32)), 
            scale_tril=tf.cast(self.chol, dtype=tf.float32),
            allow_nan_stats=False,
        )
        self.return_log_prob = self._return_log_prob_fixed
        
    @tf.function
    def _return_log_prob_nystrom(self, parameter, **kwargs):
        ''' Return the log probability using nystrom approximation
        '''
        gpk_nugget = tf.cast(kwargs['gpk_nugget'], dtype=self.gp_dtype)        
        m_vect = self._return_mfunc(**kwargs)        
        y = tf.cast(parameter - m_vect, self.gp_dtype) # remove mean function from parameter

        K_full = self._return_sigma_full(**kwargs)        
        # For full calculation we include the nugget term
        # - so have to remove it here
        K_full = K_full - tf.linalg.diag(tf.ones(self.n_vx, dtype=self.gp_dtype)) * tf.cast(self.eps + kwargs[f'gpk_nugget'], dtype=self.gp_dtype)
        
        K_uu=tf.gather(tf.gather(K_full, self.inducer_idx, axis=0), self.inducer_idx, axis=1)
        # Add small jitter to for PD-ness
        K_uu += tf.cast(self.eps, self.gp_dtype) * tf.eye(self.n_inducers, dtype=self.gp_dtype)

        K_vu=tf.gather(K_full, self.inducer_idx, axis=1)

        # Build S = K_uu + (1/nugget) K_vu^T K_vu  (m x m)
        K_vutK_vu = tf.matmul(tf.transpose(K_vu), K_vu)   # (m,m)
        S = K_uu + (1.0 / gpk_nugget) * K_vutK_vu

        # --- Calculate the quadratic term: y^T * inv(K) * y ---
        # According to the Woodbury identity
        # First term: (1/tau^2) * y^T * y
        quad_term_1 = (1.0 / gpk_nugget) * tf.reduce_sum(tf.square(y))

        # Second term: -(1/tau^4) * y^T * K_vu * inv(S) * K_vu^T * y
        chol_S = tf.linalg.cholesky(S)
        K_vu_T_y = tf.matmul(tf.transpose(K_vu), tf.expand_dims(y, -1))
        
        # Solve S * x = K_vu^T * y -> x = inv(S) * K_vu^T * y
        x_vec = tf.linalg.cholesky_solve(chol_S, K_vu_T_y)
        
        # Calculate K_vu * x_vec
        K_vu_x = tf.matmul(K_vu, x_vec)
        
        quad_term_2 = tf.squeeze(tf.matmul(tf.transpose(
            tf.expand_dims(y, -1)), (1.0 / (gpk_nugget*gpk_nugget)) * K_vu_x))    
        quadratic_term = quad_term_1 - quad_term_2    
        # --- Calculate the log-determinant term: log(|K|) ---
        # log(|K|) = n * log(tau^2) + log(|S|) - log(|K_uu|)
        chol_K_uu = tf.linalg.cholesky(K_uu)    
        logdet_K_uu = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol_K_uu)))
        logdet_S = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol_S)))    
        n_float = tf.cast(self.n_vx, self.gp_dtype)
        logdet_K = n_float * tf.math.log(gpk_nugget) + logdet_S - logdet_K_uu
        
        # --- Combine terms to get the final log-probability ---
        log2pi = tf.math.log(2.0 * tf.constant(math.pi, dtype=self.gp_dtype))    
        log_prob = -0.5 * quadratic_term - 0.5 * logdet_K - 0.5 * n_float * log2pi    
        return tf.cast(tf.reshape(log_prob, []), tf.float32)    

    @tf.function
    def _return_log_prob_unfixed(self, parameter, **kwargs):
        """
        Unfixed parameters using TensorFlow distribution.
        Recompute covariance and Cholesky decomposition on the fly.
        Optionally uses random selection of n_inducers for sparse GP approximation.
        """
        # Get cov matrix
        cov_matrix = self._return_sigma_full(**kwargs)
        chol = tf.linalg.cholesky(tf.cast(cov_matrix, dtype=self.gp_dtype))
        # Get mean vector
        m_vect = self._return_mfunc(**kwargs)
        gp_prior_dist = tfd.MultivariateNormalTriL(
            loc=tf.squeeze(tf.cast(m_vect, dtype=tf.float32)), 
            scale_tril=tf.cast(chol, dtype=tf.float32),
            allow_nan_stats=False,
        )
        return gp_prior_dist.log_prob(parameter)

    @tf.function
    def _return_log_prob_fixed(self, parameter, **kwargs):
        """
        Unfixed parameters using TensorFlow distribution.
        Recompute covariance and Cholesky decomposition on the fly.
        Optionally uses random selection of n_inducers for sparse GP approximation.
        """
        # stop the gradients
        k_list = 0
        for k in self.pids.keys():
            k_list += kwargs[self.pids[k]]*0.0
        return self.gp_prior_dist.log_prob(parameter)+k_list

    def _predict(self, **kwargs):
        ''' GP prediction
        given new Xs (M x D), old Xs (N x D), parameters (N,), & hyperparameters for GP
        return mean (M,), std (M,) for new Xs

        Required kwargs:
            - parameter : (N,) training observations

        Optional kwargs (one of these must allow determination of M):
            - Xs_new : (M, D) new input locations
            - d_new_train : (M, N)  submatrix of distances between new & train (used for ALL stationary kernels
                            unless per-kernel d_new_train_{xid} is provided)
            - d_new_new   : (M, M)  distances between new & new (used for ALL stationary kernels unless per-kernel provided)
            - d_new_train_{xid} : per-kernel (M,N) distances for stationary kernel xid
            - d_new_new_{xid}   : per-kernel (M,M) distances for stationary kernel xid
            - Xs_train : (N, D) training inputs (used to compute distances if not provided)
            - mfunc_new : (M,) mean-function values at new points (if not provided we use the global mean or replicate training mean)

        Returns:
            mean_pred: (M,) tf.float32
            std_pred:  (M,) tf.float32
        '''
        # --- basic tensors ---
        if 'parameter' not in kwargs:
            raise ValueError("Missing 'parameter' in kwargs (observations at training points).")
        y = tf.convert_to_tensor(kwargs['parameter'], dtype=self.gp_dtype)
        y = tf.reshape(y, [-1])
        N = tf.shape(y)[0]

        # helper: pairwise euclidean distances A (P,D) vs B (Q,D) -> (P,Q)
        def pairwise_dists(A, B):
            A = tf.cast(A, dtype=self.gp_dtype)
            B = tf.cast(B, dtype=self.gp_dtype)
            # (P,1,D) - (1,Q,D)
            diff = tf.expand_dims(A, 1) - tf.expand_dims(B, 0)
            d2 = tf.reduce_sum(tf.square(diff), axis=-1)
            return tf.sqrt(tf.maximum(d2, tf.cast(0.0, self.gp_dtype)))

        # Determine M from provided inputs
        M = None
        if 'd_new_train' in kwargs:
            dnt = tf.convert_to_tensor(kwargs['d_new_train'], dtype=self.gp_dtype)
            M = tf.shape(dnt)[0]
        elif len(self.warp_kernel_list) > 0 and any([f"d_new_train_{s}" in kwargs for s in self.warp_kernel_list]):
            # get M from any provided per-warp matrix
            for s in self.warp_kernel_list:
                key = f'd_new_train_{s}'
                if key in kwargs:
                    dnt = tf.convert_to_tensor(kwargs[key], dtype=self.gp_dtype)
                    M = tf.shape(dnt)[0]
                    break
        elif 'Xs_new' in kwargs:
            Xs_new = tf.convert_to_tensor(kwargs['Xs_new'], dtype=self.gp_dtype)
            if len(Xs_new.shape) == 1:
                Xs_new = tf.expand_dims(Xs_new, -1)
            M = tf.shape(Xs_new)[0]
        else:
            raise ValueError("Cannot determine M (number of prediction points). Provide 'Xs_new' or 'd_new_train'.")

        # Cast Xs_new / Xs_train if present
        Xs_new = None
        if 'Xs_new' in kwargs:
            Xs_new = tf.convert_to_tensor(kwargs['Xs_new'], dtype=self.gp_dtype)
            if len(Xs_new.shape) == 1:
                Xs_new = tf.expand_dims(Xs_new, -1)
        Xs_train = None
        if 'Xs_train' in kwargs:
            Xs_train = tf.convert_to_tensor(kwargs['Xs_train'], dtype=self.gp_dtype)
            if len(Xs_train.shape) == 1:
                Xs_train = tf.expand_dims(Xs_train, -1)
            if tf.shape(Xs_train)[0] != N:
                raise ValueError("Provided Xs_train must have same first-dimension length as 'parameter'.")

        # --- build train covariance and mean (uses existing class functions) ---
        K_train = self._return_sigma_full(**kwargs)  # (N,N), includes nugget
        if tf.shape(K_train)[0] != N:
            # sanity check
            N_cov = tf.shape(K_train)[0]
            if N_cov != N:
                raise ValueError("Length of 'parameter' does not match GP covariance size.")
        chol_K = tf.linalg.cholesky(K_train)
        m_train = self._return_mfunc(**kwargs)
        m_train = tf.reshape(tf.cast(m_train, dtype=self.gp_dtype), [-1])
        y_minus_m = tf.reshape(tf.cast(y, dtype=self.gp_dtype) - m_train, [-1, 1])
        alpha = tf.linalg.cholesky_solve(chol_K, y_minus_m)  # (N,1)

        # --- build cross-covariance K_nN (M,N) and K_nn diag (M) or full (M,M) ---
        K_nN = tf.zeros((M, N), dtype=self.gp_dtype)
        K_nn = tf.zeros((M, M), dtype=self.gp_dtype)

        # stationary kernels
        for s in self.stat_kernel_list:
            kernel_type = self.kernel_type.get(s, 'RBF')
            l = tf.cast(kwargs[f'gpk{s}_l'], dtype=self.gp_dtype)
            v = tf.cast(kwargs[f'gpk{s}_v'], dtype=self.gp_dtype)

            # priority for distances:
            # 1) per-kernel provided: d_new_train_{s}, d_new_new_{s}
            # 2) global provided: d_new_train, d_new_new
            # 3) compute from Xs_train & Xs_new (requires Xs_train & Xs_new)
            key_nt = f'd_new_train_{s}'
            key_nn = f'd_new_new_{s}'
            if key_nt in kwargs:
                d_new_train = tf.convert_to_tensor(kwargs[key_nt], dtype=self.gp_dtype)
            elif 'd_new_train' in kwargs:
                d_new_train = tf.convert_to_tensor(kwargs['d_new_train'], dtype=self.gp_dtype)
            else:
                if Xs_train is None or Xs_new is None:
                    raise ValueError(f"Need distances or Xs for stationary kernel '{s}'. Provide '{key_nt}' or 'd_new_train' or both Xs_train & Xs_new.")
                d_new_train = pairwise_dists(Xs_new, Xs_train)  # (M,N)

            if key_nn in kwargs:
                d_new_new = tf.convert_to_tensor(kwargs[key_nn], dtype=self.gp_dtype)
            elif 'd_new_new' in kwargs:
                d_new_new = tf.convert_to_tensor(kwargs['d_new_new'], dtype=self.gp_dtype)
            else:
                # if we have d_new_train we can compute d_new_new from Xs_new or from distances
                if Xs_new is not None:
                    d_new_new = pairwise_dists(Xs_new, Xs_new)
                else:
                    # approximate d_new_new as zeros on diagonal and large elsewhere (fallback)
                    d_new_new = tf.zeros((M, M), dtype=self.gp_dtype)

            # apply kernel formula
            if kernel_type == 'RBF':
                K_nN += (v**2) * tf.exp(-tf.square(d_new_train) / (2.0 * l**2))
                K_nn += (v**2) * tf.exp(-tf.square(d_new_new) / (2.0 * l**2))
            elif kernel_type == 'matern52':
                sqrt5 = tf.cast(tf.sqrt(5.0), dtype=self.gp_dtype)
                f1 = (sqrt5 * d_new_train) / l
                f2 = (5.0 * tf.square(d_new_train)) / (3.0 * tf.square(l))
                K_nN += (v**2) * (1.0 + f1 + f2) * tf.exp(-f1)

                f1nn = (sqrt5 * d_new_new) / l
                f2nn = (5.0 * tf.square(d_new_new)) / (3.0 * tf.square(l))
                K_nn += (v**2) * (1.0 + f1nn + f2nn) * tf.exp(-f1nn)
            elif kernel_type == 'laplace':
                K_nN += (v**2) * tf.exp(-d_new_train / l)
                K_nn += (v**2) * tf.exp(-d_new_new / l)
            else:
                raise ValueError(f"Unsupported stationary kernel: {kernel_type}")

        # warp kernels (simpler: allow precomputed d_new_train_{s} else compute via stored self.Xs and weights)
        for s in self.warp_kernel_list:
            s_kernel_type = s.split('_')[-1]
            key_nt = f'd_new_train_{s}'
            key_nn = f'd_new_new_{s}'
            if key_nt in kwargs:
                d_new_train = tf.convert_to_tensor(kwargs[key_nt], dtype=self.gp_dtype)
            else:
                # compute warp projections
                w_list = [kwargs[f"gpk{s}_w{i}"] for i in range(self.Xs[s].shape[1])]
                w = tf.stack(w_list, axis=0)
                w = tf.cast(w, dtype=self.gp_dtype)
                X_train_s = tf.cast(self.Xs[s], dtype=self.gp_dtype)  # (N, Dw)
                warp_train = tf.reshape(tf.matmul(X_train_s, w), [N])  # (N,)
                if Xs_new is None:
                    raise ValueError(f"Need 'Xs_new' to compute warp distances for kernel '{s}' or provide '{key_nt}'.")
                warp_new = tf.reshape(tf.matmul(Xs_new, w), [tf.shape(Xs_new)[0]])
                d_new_train = pairwise_dists(tf.expand_dims(warp_new, -1), tf.expand_dims(warp_train, -1))  # (M,N)

            if key_nn in kwargs:
                d_new_new = tf.convert_to_tensor(kwargs[key_nn], dtype=self.gp_dtype)
            else:
                if Xs_new is not None:
                    # compute warp_new if not already
                    if 'warp_new' not in locals():
                        w_list = [kwargs[f"gpk{s}_w{i}"] for i in range(self.Xs[s].shape[1])]
                        w = tf.stack(w_list, axis=0)
                        w = tf.cast(w, dtype=self.gp_dtype)
                        warp_new = tf.reshape(tf.matmul(Xs_new, w), [tf.shape(Xs_new)[0]])
                    d_new_new = pairwise_dists(tf.expand_dims(warp_new, -1), tf.expand_dims(warp_new, -1))
                else:
                    d_new_new = tf.zeros((M, M), dtype=self.gp_dtype)

            v = tf.cast(kwargs[f'gpk{s}_v'], dtype=self.gp_dtype)
            l = tf.cast(1.0, dtype=self.gp_dtype)  # as used elsewhere for warp kernels
            if s_kernel_type == 'RBF':
                K_nN += (v**2) * tf.exp(-tf.square(d_new_train) / (2.0 * l**2))
                K_nn += (v**2) * tf.exp(-tf.square(d_new_new) / (2.0 * l**2))
            elif s_kernel_type == 'matern52':
                sqrt5 = tf.cast(tf.sqrt(5.0), dtype=self.gp_dtype)
                f1 = (sqrt5 * d_new_train) / l
                f2 = (5.0 * tf.square(d_new_train)) / (3.0 * tf.square(l))
                K_nN += (v**2) * (1.0 + f1 + f2) * tf.exp(-f1)

                f1nn = (sqrt5 * d_new_new) / l
                f2nn = (5.0 * tf.square(d_new_new)) / (3.0 * tf.square(l))
                K_nn += (v**2) * (1.0 + f1nn + f2nn) * tf.exp(-f1nn)
            elif s_kernel_type == 'laplace':
                K_nN += (v**2) * tf.exp(-d_new_train / l)
                K_nn += (v**2) * tf.exp(-d_new_new / l)
            else:
                raise ValueError(f"Unsupported warp kernel: {s_kernel_type}")

        # small jitter for numerical stability on K_nn diagonal
        K_nn += tf.cast(self.eps, dtype=self.gp_dtype) * tf.eye(M, dtype=self.gp_dtype)

        # --- mean at new points ---
        if 'mfunc_new' in kwargs:
            m_new = tf.reshape(tf.cast(kwargs['mfunc_new'], dtype=self.gp_dtype), [-1, 1])
        else:
            # fallback: use global mean if present, otherwise repeat mean of training mean
            if 'mfunc_mean' in kwargs:
                global_mean = tf.cast(kwargs['mfunc_mean'], dtype=self.gp_dtype)
                m_new = tf.ones([M, 1], dtype=self.gp_dtype) * global_mean
            else:
                # replicate mean of m_train
                mean_train = tf.reduce_mean(m_train)
                m_new = tf.ones([M, 1], dtype=self.gp_dtype) * mean_train

        # --- predictive mean ---
        pred_mean = m_new + tf.matmul(K_nN, alpha)  # (M,1)

        # --- predictive variance (diagonal) ---
        K_nN_T = tf.transpose(K_nN)  # (N,M)
        solved = tf.linalg.cholesky_solve(chol_K, K_nN_T)  # (N,M)
        cov_cond = K_nn - tf.matmul(K_nN, solved)  # (M,M)
        var_pred = tf.linalg.diag_part(cov_cond)
        var_pred = tf.maximum(var_pred, tf.cast(0.0, dtype=self.gp_dtype))
        std_pred = tf.sqrt(var_pred)

        pred_mean = tf.cast(tf.reshape(pred_mean, [M]), dtype=tf.float32)
        std_pred = tf.cast(tf.reshape(std_pred, [M]), dtype=tf.float32)
        return pred_mean, std_pred



# ******* SUPPORTING FUNCTIONS *********
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

@tf.function
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