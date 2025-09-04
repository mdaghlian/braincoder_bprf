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
        parameter_dm = tf.cast(parameter - m_vect, self.gp_dtype) # remove mean function from parameter

        K_full = self._return_sigma_full(**kwargs) # we do the nugget later, so have to remove it here...        
        # might change this later...
        K_full -= tf.linalg.diag(tf.ones(self.n_vx, dtype=self.gp_dtype)) * tf.cast(self.eps + kwargs[f'gpk_nugget'], dtype=self.gp_dtype)
        A=tf.gather(tf.gather(K_full, self.inducer_idx, axis=0), self.inducer_idx, axis=1)
        B=tf.gather(K_full, self.inducer_idx, axis=1)
        # Add small jitter to A for PD-ness
        A += tf.cast(self.eps, self.gp_dtype) * tf.eye(self.n_inducers, dtype=self.gp_dtype)

        # Build S = A + (1/nugget) B^T B  (m x m)
        BtB = tf.matmul(tf.transpose(B), B)   # (m,m)
        S = A + (1.0 / gpk_nugget) * BtB

        # Cholesky S
        Ls = tf.linalg.cholesky(S) # (m,m)
        # Solve S x = B^T y
        Bt_y = tf.matmul(tf.transpose(B), tf.expand_dims(parameter_dm, -1))  # (m,1)
        x = tf.linalg.cholesky_solve(Ls, Bt_y)                       # (m,1)

        # Compute quadratic term via Woodbury K^{-1} y = (1/sigma2) y - (1/sigma2^2) B x
        Bx = tf.matmul(B, x)                                        # (n,1)
        v = (1.0 / gpk_nugget) * tf.expand_dims(parameter_dm, -1) - (1.0 / (gpk_nugget * gpk_nugget)) * Bx
        quad = tf.squeeze(tf.matmul(tf.transpose(tf.expand_dims(parameter_dm, -1)), v))  # scalar
        # Log-determinant via determinant lemma:
        # log|K| = n log sigma2 - log|A| + log|S|
        La = tf.linalg.cholesky(A)
        logdetA = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(La)))
        logdetS = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(Ls)))
        n_float = tf.cast(self.n_vx, self.gp_dtype)
        logdetK = n_float * tf.math.log(gpk_nugget) - logdetA + logdetS        
        log2pi = tf.math.log(2.0 * tf.constant(math.pi, dtype=self.gp_dtype))
        logp = -0.5 * quad - 0.5 * logdetK - 0.5 * n_float * log2pi
        return tf.cast(tf.reshape(logp, []), tf.float32)

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
        return self.gp_prior_dist.log_prob(parameter)


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