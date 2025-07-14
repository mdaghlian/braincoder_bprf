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

from braincoder.bprf_mcmc import GPdistsM

class GPSpecMM(GPdistsM):
    
    def __init__(self, n_vx, eig_vects, eig_vals, **kwargs):
        """
        A ** TRUE ** spectral GP 
               
        Objective: LBO to construct a spectral GP 

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
def kappa_diag2sigma(kappa, eig_vects, mass_matrix):
    """
    Reconstruct the dense covariance matrix from spectral variances,
    accounting for the mass matrix.
    
    Args:
        kappa: tf.Tensor of shape (L,), spectral variances.
        eig_vects: tf.Tensor of shape (N, L), the first L LBO eigenvectors Φ.
        mass_matrix: tf.Tensor of shape (N,), the diagonal entries of M.
    
    Returns:
        sigma_recon: tf.Tensor of shape (N, N), reconstructed Σ.
    """
    # compute sqrt and inv-sqrt of M
    m_sqrt = tf.sqrt(mass_matrix)                             # [N]
    m_inv_sqrt = tf.math.reciprocal(m_sqrt)                   # [N]

    # weight eigenvectors by sqrt(M)
    phiw = eig_vects * tf.expand_dims(m_sqrt, -1)             # [N, L]

    # build mid: (M^{1/2}Φ) diag(kappa) (M^{1/2}Φ)^T
    Lambda = tf.linalg.diag(kappa)                            # [L, L]
    mid = tf.matmul(phiw, tf.matmul(Lambda, phiw, transpose_b=True))  # [N, N]

    # un-weight by M^{-1/2} on both sides
    sigma_recon = (m_inv_sqrt[:,None] * mid) * m_inv_sqrt[None,:]      # [N, N]

    # ensure symmetry
    return 0.5 * (sigma_recon + tf.transpose(sigma_recon))


@tf.function
def kappa_dense2sigma(kappa, eig_vects, mass_matrix):
    """
    Reconstruct the dense covariance matrix from full spectral covariance,
    accounting for the mass matrix.
    
    Args:
        kappa: tf.Tensor of shape (L, L), full spectral covariance.
        eig_vects: tf.Tensor of shape (N, L), the first L LBO eigenvectors Φ.
        mass_matrix: tf.Tensor of shape (N,), the diagonal entries of M.
    
    Returns:
        sigma_recon: tf.Tensor of shape (N, N), reconstructed Σ.
    """
    # compute sqrt and inv-sqrt of M
    m_sqrt = tf.sqrt(mass_matrix)                             # [N]
    m_inv_sqrt = tf.math.reciprocal(m_sqrt)                   # [N]

    # weight eigenvectors by sqrt(M)
    phiw = eig_vects * tf.expand_dims(m_sqrt, -1)             # [N, L]

    # mid = (M^{1/2}Φ) K (M^{1/2}Φ)^T
    mid = tf.matmul(phiw, tf.matmul(kappa, phiw, transpose_a=True))  # [N, N]

    # un-weight by M^{-1/2} on both sides
    sigma_recon = (m_inv_sqrt[:,None] * mid) * m_inv_sqrt[None,:]     # [N, N]

    # ensure symmetry
    return 0.5 * (sigma_recon + tf.transpose(sigma_recon))

