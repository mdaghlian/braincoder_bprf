import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb
import math
import copy


# ******************* return log probability ****************
# - distance based gp prior
#
# GP objects are treated as immutable once built: all vertex-specific data (distance matrices,
# coordinates, eigenvectors, inducing points) are captured as constants the first time a
# tf.function method is traced. To work on a different set of vertices, call gp.subset(idx),
# which returns a *new* GP (with its own traces) rather than editing this one in place.

_LOG2PI = math.log(2.0 * math.pi)


def _pairwise_dist_np(A, B=None):
    """Exact Euclidean distances between the rows of A [P, D] and B [Q, D] (B = A if None)"""
    A = np.asarray(A, dtype='float64')
    if A.ndim == 1:
        A = A[:, None]
    B = A if B is None else np.asarray(B, dtype='float64').reshape(len(B), -1)
    d2 = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T
    return np.sqrt(np.maximum(d2, 0.0))


class GP():
    def __init__(self, n_vx, **kwargs):
        """
        Gaussian process prior over vertices

        N(m, K + nugget*I)
        -> m: the mean function (global mean + optional linear regressors, optional bijector)
        -> K: sum of covariance functions (stationary kernels on distances, warp kernels)
        -> nugget (a variance) + a small relative jitter on the diagonal, for stability

        Args:
            n_vx: number of vertices
            **kwargs: Optional parameters, such as:
                - psd_control: 'none' (default) or 'euclidean'. 'euclidean' replaces each distance
                  matrix with distances in a classical-MDS embedding (embedding_dim), which makes
                  e.g. geodesic distances give a positive definite kernel.
                - embedding_dim: dimensions for the MDS embedding (default 10)
                - gp_dtype: dtype for the covariance algebra (default tf.float64)
                - eps: relative jitter added to the diagonal: eps * (1 + total kernel variance)
        """
        self.n_vx = int(n_vx)
        self.psd_control   = kwargs.get('psd_control', 'none')  # 'euclidean' or 'none'
        self.eps           = kwargs.get('eps', 1e-6)
        self.embedding_dim = kwargs.get('embedding_dim', 10)
        self.gp_dtype      = kwargs.get('gp_dtype', tf.float64)

        self.stat_kernel_list = []
        self.warp_kernel_list = []
        self.mfunc_list = []
        self.mfunc_bijector = tfb.Identity()
        self.Xs = {}            # coordinates per kernel xid (if given) and per mean-function xid
        self.dXs = {}           # [N, N] distances per stationary kernel
        self.median_dX = {}     # typical distance per stationary kernel (for default lengthscale priors)
        self.kernel_type = {}
        self.n_inducers = None
        self.inducer_idx = None
        self.nystrom = False
        self.is_fixed = False
        self._fixed_kwargs = None
        self.pids = {}
        # Index of parameters to be passed...
        self.pids[0] = 'gpk_nugget' # Global nugget term (a variance)
        self.pids[1] = 'mfunc_mean' # Global mean term
        self.pids_inv = {}
        self._update_pids_inv()
        self._log_prob_mode = 'unfixed'
        self.gp_prior_dist = None

    def _update_pids_inv(self):
        self.pids_inv = {v: k for k, v in self.pids.items()}

    def _add_pid(self, name):
        self.pids[len(self.pids)] = name
        self._update_pids_inv()

    def update_n_vx(self, new_value):
        raise NotImplementedError(
            'GPs are immutable (tf.function traces capture their data). Use gp.subset(idx) instead.')

    def _sc(self, x):
        """Hyperparameter (scalar or shape-(1,) tensor) -> scalar in gp_dtype"""
        return tf.reshape(tf.cast(x, self.gp_dtype), [])

    def _hyper_kwargs(self, kwargs):
        """Only the hyperparameters this GP knows about"""
        return {k: kwargs[k] for k in self.pids_inv}

    # **************** HYPERPARAMETER INFO (used by BPRF_hier) ***************
    def positive_pids(self):
        """Hyperparameters that must be > 0 (lengthscales, variances, nugget, spectral parameters)"""
        pos = {'gpk_nugget'}
        for s in self.stat_kernel_list:
            pos |= {f'gpk{s}_l', f'gpk{s}_v'}
        for s in self.warp_kernel_list:
            pos.add(f'gpk{s}_v')
        return pos

    def default_hyperpriors(self):
        """Default prior per hyperparameter (None = no prior).
        Lengthscales: LogNormal around half the median pairwise distance (so in the units of the distances,
        and kept away from 0, where the GP collapses into the nugget). Other positive ones: HalfNormal(20)."""
        priors = {k: None for k in self.pids_inv}
        for k in self.positive_pids():
            priors[k] = tfd.HalfNormal(scale=20.0)
        for s in self.stat_kernel_list:
            priors[f'gpk{s}_l'] = tfd.LogNormal(loc=np.float32(np.log(self.median_dX[s] / 2)), scale=1.5)
        return priors

    # **************** MEAN FUNCTIONS ***************
    def add_xid_linear_mfunc(self, xid, **kwargs):
        ''' add linear mean function: slopes on the columns of Xs [N] or [N, D]
        '''
        Xs = np.asarray(kwargs.get('Xs'), dtype='float64')
        if Xs.ndim == 1:
            Xs = Xs[:, None]
        self.Xs[xid] = tf.convert_to_tensor(Xs, dtype=self.gp_dtype)
        self.mfunc_list.append(xid)
        for i in range(Xs.shape[1]):
            self._add_pid(f'mfunc{xid}_slope{i}')

    def _mean_from(self, Xs_by_xid, n, **kwargs):
        """Mean function at n points with regressors Xs_by_xid {xid: [n, D]} (gp_dtype)"""
        m_out = self._sc(kwargs['mfunc_mean']) * tf.ones(n, dtype=self.gp_dtype)
        for m in self.mfunc_list:
            X = Xs_by_xid[m]
            slopes = tf.stack([self._sc(kwargs[f'mfunc{m}_slope{i}']) for i in range(X.shape[1])], axis=0)
            m_out = m_out + tf.linalg.matvec(X, slopes)
        return self.mfunc_bijector(m_out)

    @tf.function
    def _return_mfunc(self, **kwargs):
        '''Return the mean function at the training vertices [N] (gp_dtype)
        '''
        return self._mean_from({m: self.Xs[m] for m in self.mfunc_list}, self.n_vx, **kwargs)

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
    # *** KERNELS ***
    # -> add Stationary kernels
    def add_xid_stationary_kernel(self, xid, **kwargs):
        ''' add a stationary kernel (RBF, matern52, laplace), from coordinates Xs [N] / [N, D]
        or a precomputed distance matrix dXs [N, N]. Adds hyperparameters gpk{xid}_l, gpk{xid}_v
        '''
        Xs = kwargs.get('Xs', None)
        dXs = kwargs.get('dXs', None)
        psd_control = kwargs.get('psd_control', self.psd_control)
        embedding_dim = kwargs.get('embedding_dim', self.embedding_dim)
        self.kernel_type[xid] = kwargs.get('kernel_type', 'RBF')
        if self.kernel_type[xid] not in ('RBF', 'matern52', 'laplace'):
            raise ValueError("Unsupported kernel: {}".format(self.kernel_type[xid]))
        self.stat_kernel_list.append(xid)

        if dXs is None:
            Xs = np.asarray(Xs, dtype='float64')
            if Xs.ndim == 1:
                Xs = Xs[:, None]
            self.Xs[xid] = tf.convert_to_tensor(Xs, dtype=self.gp_dtype)
            dXs = _pairwise_dist_np(Xs)
        dXs = np.array(dXs, dtype='float64')
        if psd_control == 'euclidean':
            print('Embedding in Euclidean space...')
            emb = mds_embedding(tf.constant(dXs), embedding_dim).numpy()
            dXs = _pairwise_dist_np(emb)
            self.Xs.pop(xid, None)            # distances no longer match the raw coordinates
        dXs = (dXs + dXs.T) / 2.0
        np.fill_diagonal(dXs, 0.0)
        self.dXs[xid] = tf.convert_to_tensor(dXs, dtype=self.gp_dtype)
        off = dXs[np.triu_indices_from(dXs, 1)]
        self.median_dX[xid] = float(np.median(off[off > 0])) if np.any(off > 0) else 1.0
        # Add a lengthscale & a variance
        self._add_pid(f'gpk{xid}_l')
        self._add_pid(f'gpk{xid}_v')

    def add_xid_warp_kernel(self, xid, Xs, **kwargs):
        ''' warp kernel: distances along a learnt projection Xs @ w (lengthscale absorbed in w).
        xid must end in the kernel type, e.g. 'lbo_RBF'. Adds gpk{xid}_v and gpk{xid}_w{i}
        '''
        Xs = np.asarray(Xs, dtype='float64')
        if Xs.ndim == 1:
            Xs = Xs[:, None]
        self.Xs[xid] = tf.convert_to_tensor(Xs, dtype=self.gp_dtype)
        self._add_pid(f'gpk{xid}_v')
        for i in range(Xs.shape[1]):
            self._add_pid(f'gpk{xid}_w{i}')
        self.warp_kernel_list.append(xid)

    def _kernel(self, d, l, v, kernel_type):
        """Stationary kernel on distances d with lengthscale l and std v"""
        if kernel_type == 'RBF':
            return tf.square(v) * tf.exp(-tf.square(d) / (2.0 * tf.square(l)))
        elif kernel_type == 'matern52':
            sqrt5 = tf.constant(np.sqrt(5.0), dtype=self.gp_dtype)
            frac1 = (sqrt5 * d) / l
            frac2 = (5.0 * tf.square(d)) / (3.0 * tf.square(l))
            return tf.square(v) * (1 + frac1 + frac2) * tf.exp(-frac1)
        elif kernel_type == 'laplace':
            return tf.square(v) * tf.exp(-d / l)
        raise ValueError("Unsupported kernel: {}".format(kernel_type))

    def _warp_coord(self, xid, X, **kwargs):
        """Projection X @ w for warp kernel xid -> [P]"""
        w = tf.stack([self._sc(kwargs[f'gpk{xid}_w{i}']) for i in range(self.Xs[xid].shape[1])], axis=0)
        return tf.linalg.matvec(tf.cast(X, self.gp_dtype), w)

    def _warp_dist(self, a, b):
        # small eps inside the sqrt keeps the gradient finite at d = 0
        return tf.sqrt(tf.square(a[:, None] - b[None, :]) + tf.constant(1e-12, self.gp_dtype))

    def _kernel_sum(self, rows, cols, **kwargs):
        """Sum of all kernels between vertex index sets rows and cols (None = all vertices)"""
        def take(D):
            if rows is not None:
                D = tf.gather(D, rows, axis=0)
            if cols is not None:
                D = tf.gather(D, cols, axis=1)
            return D
        K = 0.0
        for s in self.stat_kernel_list:
            K = K + self._kernel(take(self.dXs[s]), self._sc(kwargs[f'gpk{s}_l']),
                                 self._sc(kwargs[f'gpk{s}_v']), self.kernel_type[s])
        for s in self.warp_kernel_list:
            a = self._warp_coord(s, self.Xs[s], **kwargs)
            ar = a if rows is None else tf.gather(a, rows)
            ac = a if cols is None else tf.gather(a, cols)
            K = K + self._kernel(self._warp_dist(ar, ac), tf.constant(1.0, self.gp_dtype),
                                 self._sc(kwargs[f'gpk{s}_v']), s.split('_')[-1])
        return K

    def _jitter(self, **kwargs):
        """Relative jitter: eps * (1 + total kernel variance)"""
        var = tf.constant(0.0, self.gp_dtype)
        for s in self.stat_kernel_list + self.warp_kernel_list:
            var += tf.square(self._sc(kwargs[f'gpk{s}_v']))
        return self.eps * (1.0 + var)

    @tf.function
    def _return_sigma_full(self, **kwargs):
        ''' Putting all the kernels together - > return the full covariance matrix [N, N] (incl. nugget + jitter)
        '''
        K = self._kernel_sum(None, None, **kwargs)
        diag = self._sc(kwargs['gpk_nugget']) + self._jitter(**kwargs)
        return K + diag * tf.eye(self.n_vx, dtype=self.gp_dtype)

    # *************************************************
    def add_nystrom_approximation(self, n_inducers, inducer_idx=None, seed=None):
        ''' Use a Nystrom approximation (low rank through n_inducers inducing vertices + nugget)
        Only the [N, m] and [m, m] kernel blocks are computed per step, never the full [N, N].
        '''
        self.n_inducers = int(n_inducers)
        if inducer_idx is None:
            inducer_idx = np.random.default_rng(seed).choice(self.n_vx, self.n_inducers, replace=False)
        self.inducer_idx = np.sort(np.asarray(inducer_idx, dtype='int64'))
        self.nystrom = True
        self._log_prob_mode = 'nystrom'

    def set_log_prob_univariate(self, **kwargs):
        ''' Independent prior per vertex: N(m, nugget) (nugget is a variance, as everywhere else)'''
        self._log_prob_mode = 'univariate'

    def set_log_prob_fixed(self, **kwargs):
        ''' Fix the hyperparameters: build (and factorise) the covariance once, reuse it every call.
        BPRF_hier detects this (gp.is_fixed) and holds the matching h-parameters fixed.
        '''
        self._fixed_kwargs = {k: np.float64(np.asarray(kwargs[k]).ravel()[0]) for k in self.pids_inv}
        self.cov_matrix = self._return_sigma_full(**self._fixed_kwargs)
        self.chol = tf.linalg.cholesky(self.cov_matrix)
        self.m_vect = self._return_mfunc(**self._fixed_kwargs)
        self.gp_prior_dist = tfd.MultivariateNormalTriL(
            loc=self.m_vect, scale_tril=self.chol, allow_nan_stats=False)
        self.is_fixed = True
        self._log_prob_mode = 'fixed'

    def return_log_prob(self, parameter, **kwargs):
        ''' Scalar (float32) log prior of parameter [N] given the hyperparameters in kwargs '''
        mode = self._log_prob_mode
        if mode == 'fixed':
            return self._return_log_prob_fixed(parameter)
        if mode == 'nystrom':
            return self._return_log_prob_nystrom(parameter, **self._hyper_kwargs(kwargs))
        if mode == 'univariate':
            return self._return_log_prob_univariate(parameter, **self._hyper_kwargs(kwargs))
        return self._return_log_prob_unfixed(parameter, **self._hyper_kwargs(kwargs))

    @tf.function
    def _return_log_prob_unfixed(self, parameter, **kwargs):
        """ Exact log prob; covariance + Cholesky recomputed every call (all in gp_dtype) """
        chol = tf.linalg.cholesky(self._return_sigma_full(**kwargs))
        y = tf.cast(tf.reshape(parameter, [-1]), self.gp_dtype) - self._return_mfunc(**kwargs)
        a = tf.linalg.triangular_solve(chol, y[:, None], lower=True)
        n = tf.cast(self.n_vx, self.gp_dtype)
        log_prob = -0.5 * (tf.reduce_sum(tf.square(a))
                           + 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)))
                           + n * _LOG2PI)
        return tf.cast(log_prob, tf.float32)

    @tf.function
    def _return_log_prob_fixed(self, parameter, **kwargs):
        """ Log prob with the covariance built in set_log_prob_fixed (any hyperparameters passed are ignored) """
        y = tf.cast(tf.reshape(parameter, [-1]), self.gp_dtype)
        return tf.cast(self.gp_prior_dist.log_prob(y), tf.float32)

    @tf.function
    def _return_log_prob_univariate(self, parameter, **kwargs):
        y = tf.cast(tf.reshape(parameter, [-1]), self.gp_dtype) - self._return_mfunc(**kwargs)
        var = self._sc(kwargs['gpk_nugget']) + self._jitter(**kwargs)
        n = tf.cast(self.n_vx, self.gp_dtype)
        log_prob = -0.5 * (tf.reduce_sum(tf.square(y)) / var + n * tf.math.log(var) + n * _LOG2PI)
        return tf.cast(log_prob, tf.float32)

    @tf.function
    def _return_log_prob_nystrom(self, parameter, **kwargs):
        ''' Log probability under K ~= K_vu K_uu^-1 K_uv + s2*I  (Woodbury; O(N m^2) per call)
        '''
        u = tf.constant(self.inducer_idx)
        jit = self._jitter(**kwargs)
        s2 = self._sc(kwargs['gpk_nugget']) + jit
        K_uu = self._kernel_sum(u, u, **kwargs) + jit * tf.eye(self.n_inducers, dtype=self.gp_dtype)
        K_vu = self._kernel_sum(None, u, **kwargs)                                # [N, m]
        y = tf.cast(tf.reshape(parameter, [-1]), self.gp_dtype) - self._return_mfunc(**kwargs)

        # S = K_uu + (1/s2) K_vu^T K_vu  (m x m)
        S = K_uu + tf.matmul(K_vu, K_vu, transpose_a=True) / s2
        chol_S = tf.linalg.cholesky(S)
        chol_K_uu = tf.linalg.cholesky(K_uu)
        # quadratic term: y^T K^-1 y = |y|^2/s2 - (1/s2^2) (K_vu^T y)^T S^-1 (K_vu^T y)
        b = tf.linalg.triangular_solve(chol_S, tf.matmul(K_vu, y[:, None], transpose_a=True), lower=True)
        quad = tf.reduce_sum(tf.square(y)) / s2 - tf.reduce_sum(tf.square(b)) / tf.square(s2)
        # log|K| = n log s2 + log|S| - log|K_uu|
        n = tf.cast(self.n_vx, self.gp_dtype)
        logdet = (n * tf.math.log(s2)
                  + 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol_S)))
                  - 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol_K_uu))))
        return tf.cast(-0.5 * (quad + logdet + n * _LOG2PI), tf.float32)

    # *************************************************
    def subset(self, idx, seed=None):
        ''' New GP restricted to vertices idx (same hyperparameter names / mode).
        Nystrom inducers inside idx are kept (remapped); missing ones are replaced by random subset vertices.
        '''
        idx = np.asarray(idx, dtype='int64')
        new = copy.copy(self)
        new.n_vx = len(idx)
        new.Xs = {k: tf.gather(v, idx, axis=0) for k, v in self.Xs.items()}
        new.dXs = {k: tf.gather(tf.gather(v, idx, axis=0), idx, axis=1) for k, v in self.dXs.items()}
        for attr in ['stat_kernel_list', 'warp_kernel_list', 'mfunc_list']:
            setattr(new, attr, list(getattr(self, attr)))
        new.kernel_type, new.median_dX = dict(self.kernel_type), dict(self.median_dX)
        new.pids, new.pids_inv = dict(self.pids), dict(self.pids_inv)
        new.is_fixed, new.gp_prior_dist = False, None
        new._log_prob_mode = 'univariate' if self._log_prob_mode == 'univariate' else 'unfixed'
        if self.nystrom:
            pos = {v: i for i, v in enumerate(idx)}
            keep = [pos[v] for v in self.inducer_idx if v in pos]
            n_ind = min(self.n_inducers, new.n_vx)
            if len(keep) < n_ind:
                rest = np.setdiff1d(np.arange(new.n_vx), keep)
                keep += list(np.random.default_rng(seed).choice(rest, n_ind - len(keep), replace=False))
            new.add_nystrom_approximation(n_ind, inducer_idx=np.array(keep[:n_ind]))
        if self.is_fixed:
            new.set_log_prob_fixed(**self._fixed_kwargs)
        return new

    # *************************************************
    def _predict(self, **kwargs):
        ''' GP prediction
        given new points, training values `parameter` (N,) & hyperparameters,
        return mean (M,), std (M,) at the new points

        Required kwargs:
            - parameter : (N,) training observations
            - the hyperparameters (gpk_nugget, mfunc_mean, ...)
        Distances / coordinates (one of these must allow determination of M):
            - Xs_new : (M, D) new input locations (stationary kernels built from Xs, and warp kernels)
            - d_new_train : (M, N) distances new <-> train (all stationary kernels), d_new_new : (M, M)
            - d_new_train_{xid}, d_new_new_{xid} : per-kernel versions
            - Xs_train : (N, D) training inputs (default: the Xs stored for that kernel)
        Mean function:
            - mfunc_new : (M,) mean at the new points, or
            - mfunc_Xs_new : {xid: (M, D)} regressors at the new points (required if there are linear mean terms)
        '''
        if 'parameter' not in kwargs:
            raise ValueError("Missing 'parameter' in kwargs (observations at training points).")
        hyp = self._hyper_kwargs(kwargs)
        y = tf.reshape(tf.cast(kwargs['parameter'], self.gp_dtype), [-1])
        if int(y.shape[0]) != self.n_vx:
            raise ValueError("Length of 'parameter' does not match GP size.")

        def as2d(X):
            X = tf.convert_to_tensor(np.asarray(X, dtype='float64'), dtype=self.gp_dtype)
            return X[:, None] if len(X.shape) == 1 else X

        Xs_new = as2d(kwargs['Xs_new']) if 'Xs_new' in kwargs else None
        Xs_train = as2d(kwargs['Xs_train']) if 'Xs_train' in kwargs else None
        pdist = lambda A, B: tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(A[:, None, :] - B[None, :, :]), -1),
                                                tf.constant(0.0, self.gp_dtype)))

        K_nN, K_nn = 0.0, 0.0
        for s in self.stat_kernel_list:
            l, v = self._sc(hyp[f'gpk{s}_l']), self._sc(hyp[f'gpk{s}_v'])
            Xt = Xs_train if Xs_train is not None else self.Xs.get(s)
            if f'd_new_train_{s}' in kwargs or 'd_new_train' in kwargs:
                d_nt = tf.cast(kwargs.get(f'd_new_train_{s}', kwargs.get('d_new_train')), self.gp_dtype)
            elif Xs_new is not None and Xt is not None:
                d_nt = pdist(Xs_new, Xt)
            else:
                raise ValueError(f"Need distances or coordinates for stationary kernel '{s}'.")
            if f'd_new_new_{s}' in kwargs or 'd_new_new' in kwargs:
                d_nn = tf.cast(kwargs.get(f'd_new_new_{s}', kwargs.get('d_new_new')), self.gp_dtype)
            elif Xs_new is not None:
                d_nn = pdist(Xs_new, Xs_new)
            else:   # only the diagonal (d = 0) is used for the predictive std
                d_nn = tf.zeros((d_nt.shape[0], d_nt.shape[0]), dtype=self.gp_dtype)
            K_nN += self._kernel(d_nt, l, v, self.kernel_type[s])
            K_nn += self._kernel(d_nn, l, v, self.kernel_type[s])
        for s in self.warp_kernel_list:
            if Xs_new is None:
                raise ValueError(f"Need 'Xs_new' for warp kernel '{s}'.")
            v, one = self._sc(hyp[f'gpk{s}_v']), tf.constant(1.0, self.gp_dtype)
            a_new = self._warp_coord(s, Xs_new, **hyp)
            a_train = self._warp_coord(s, self.Xs[s], **hyp)
            K_nN += self._kernel(tf.abs(a_new[:, None] - a_train[None, :]), one, v, s.split('_')[-1])
            K_nn += self._kernel(tf.abs(a_new[:, None] - a_new[None, :]), one, v, s.split('_')[-1])
        M = int(K_nN.shape[0])
        K_nn += self._jitter(**hyp) * tf.eye(M, dtype=self.gp_dtype)

        # --- mean at the new points ---
        if 'mfunc_new' in kwargs:
            m_new = tf.reshape(tf.cast(kwargs['mfunc_new'], self.gp_dtype), [-1])
        else:
            if self.mfunc_list and 'mfunc_Xs_new' not in kwargs:
                raise ValueError("This GP has linear mean terms: pass 'mfunc_Xs_new' ({xid: X}) or 'mfunc_new'.")
            Xm = {m: as2d(kwargs['mfunc_Xs_new'][m]) for m in self.mfunc_list}
            m_new = self._mean_from(Xm, M, **hyp)

        # --- condition on the training values ---
        chol_K = tf.linalg.cholesky(self._return_sigma_full(**hyp))
        alpha = tf.linalg.cholesky_solve(chol_K, (y - self._return_mfunc(**hyp))[:, None])
        pred_mean = m_new + tf.reshape(tf.matmul(K_nN, alpha), [-1])
        solved = tf.linalg.cholesky_solve(chol_K, tf.transpose(K_nN))
        var_pred = tf.maximum(tf.linalg.diag_part(K_nn - tf.matmul(K_nN, solved)),
                              tf.constant(0.0, self.gp_dtype))
        return tf.cast(pred_mean, tf.float32), tf.cast(tf.sqrt(var_pred), tf.float32)


class GPSpec(GP):
    '''
    Spectral GP, using (Laplace-Beltrami) eigenpairs:
        y_w = sqrt(M) (y - m) ~ N(0, Phi_w diag(kappa) Phi_w^T + nugget I),   Phi_w = sqrt(M) Phi
    kappa = gpk_var * sum_k f_k(eigenvalues). Exact for any set of rows of Phi (via Woodbury),
    so it also works after gp.subset(idx).
    '''
    def __init__(self, n_vx, eig_vects, eig_vals, **kwargs):
        super().__init__(n_vx, **kwargs)
        self.eig_vals = tf.convert_to_tensor(eig_vals, dtype=self.gp_dtype)
        self.eig_vects = tf.convert_to_tensor(eig_vects, dtype=self.gp_dtype)
        mass_matrix = kwargs.get('mass_matrix', None)
        self.mass_matrix = (tf.ones([self.n_vx], dtype=self.gp_dtype) if mass_matrix is None
                            else tf.constant(mass_matrix, dtype=self.gp_dtype))   # [V]
        self.n_lbo = int(self.eig_vects.shape[-1])
        self._precompute()
        self.spec_kernel_list = []
        self._add_pid('gpk_var')

    def _precompute(self):
        self.mass_matrix_sqrt = tf.sqrt(self.mass_matrix)
        self.phiw = self.eig_vects * self.mass_matrix_sqrt[..., tf.newaxis]     # [N, M]
        self.gram = tf.matmul(self.phiw, self.phiw, transpose_a=True)            # [M, M] (= I on the full mesh)
        # orthonormal basis (e.g. LBO eigenvectors on exactly these vertices): the M x M system is diagonal,
        # so the log prob costs O(N M) with no Cholesky
        self.orthonormal = bool(np.max(np.abs(self.gram.numpy() - np.eye(self.n_lbo))) < 1e-6)

    def default_hyperpriors(self):
        """As GP, but no default prior on gpk_var: it scales the spectral density (e.g. v^2 * 2 pi l^2 for a
        2-D heat kernel), so a fixed-scale prior on it biases the lengthscale"""
        priors = super().default_hyperpriors()
        priors['gpk_var'] = None
        return priors

    def positive_pids(self):
        pos = super().positive_pids() | {'gpk_var'}
        for s in self.spec_kernel_list:
            pos |= {f'gpk{s}_kappa_{n}' for n in self._spec_names(self.kernel_type[s])}
        return pos

    @staticmethod
    def _spec_names(kernel_type):
        return {'spec_exp': ['l'], 'spec_heat': ['t'], 'spec_ratquad': ['alpha', 'beta']}[kernel_type]

    # *** KERNELS ***
    # -> add Spectral kernels (based on LBO)
    def add_xid_kappa_kernel(self, xid, **kwargs):
        ''' spectral kernel: spec_exp (gpk{xid}_kappa_l), spec_heat (gpk{xid}_kappa_t),
        spec_ratquad (gpk{xid}_kappa_alpha, gpk{xid}_kappa_beta). Overall scale: gpk_var.
        '''
        self.kernel_type[xid] = kwargs.get('kernel_type', 'spec_exp')
        if self.kernel_type[xid] not in ('spec_exp', 'spec_heat', 'spec_ratquad'):
            raise ValueError(f"Unknown kernel type: {self.kernel_type[xid]}.")
        for n in self._spec_names(self.kernel_type[xid]):
            self._add_pid(f'gpk{xid}_kappa_{n}')
        self.spec_kernel_list.append(xid)

    @tf.function
    def _return_kappa(self, **kwargs):
        ''' To return the (M,) spectral variances'''
        k_out = tf.zeros(self.n_lbo, dtype=self.gp_dtype)
        for s in self.spec_kernel_list:
            p = lambda n: self._sc(kwargs[f'gpk{s}_kappa_{n}'])
            t = self.kernel_type[s]
            if t == 'spec_exp':
                k_out += tf.exp(-p('l') * self.eig_vals)
            elif t == 'spec_heat':
                k_out += tf.exp(-p('t') * self.eig_vals)
            else:   # spec_ratquad
                k_out += (1.0 + self.eig_vals / (2.0 * p('alpha') * p('beta'))) ** (-p('alpha'))
        return k_out * self._sc(kwargs['gpk_var'])

    def set_log_prob_fixed(self, **kwargs):
        # the spectral log prob is already cheap: just freeze the hyperparameters
        self._fixed_kwargs = {k: np.float64(np.asarray(kwargs[k]).ravel()[0]) for k in self.pids_inv}
        self.is_fixed = True
        self._log_prob_mode = 'fixed'

    def return_log_prob(self, parameter, **kwargs):
        if self._log_prob_mode == 'fixed':
            kwargs = self._fixed_kwargs
        return self._return_log_prob_unfixed(parameter, **self._hyper_kwargs(kwargs))

    @tf.function
    def _return_log_prob_unfixed(self, parameter, **kwargs):
        """
        Log probability using the spectral representation (Woodbury on the M x M system).
        """
        s2 = self._sc(kwargs['gpk_nugget']) + self.eps
        y = tf.cast(tf.reshape(parameter, [-1]), self.gp_dtype) - self._return_mfunc(**kwargs)
        yw = y * self.mass_matrix_sqrt
        z = tf.linalg.matvec(self.phiw, yw, transpose_a=True)                 # [M]
        kappa = self._return_kappa(**kwargs)                                  # [M]
        n = tf.cast(self.n_vx, self.gp_dtype)
        if self.orthonormal:      # A = I + diag(kappa) / s2
            a = 1.0 + kappa / s2
            quad = (tf.reduce_sum(tf.square(yw)) - tf.reduce_sum(kappa * tf.square(z) / a) / s2) / s2
            logdet = n * tf.math.log(s2) + tf.reduce_sum(tf.math.log(a))
            return tf.cast(-0.5 * (quad + logdet + n * _LOG2PI), tf.float32)
        ks = tf.sqrt(kappa)
        A = tf.eye(self.n_lbo, dtype=self.gp_dtype) + ks[:, None] * self.gram * ks[None, :] / s2
        chol_A = tf.linalg.cholesky(A)
        u = tf.linalg.triangular_solve(chol_A, (ks * z)[:, None], lower=True)
        quad = (tf.reduce_sum(tf.square(yw)) - tf.reduce_sum(tf.square(u)) / s2) / s2
        logdet = n * tf.math.log(s2) + 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol_A)))
        return tf.cast(-0.5 * (quad + logdet + n * _LOG2PI), tf.float32)

    def subset(self, idx, seed=None):
        idx = np.asarray(idx, dtype='int64')
        new = super().subset(idx, seed=seed)
        new.spec_kernel_list = list(self.spec_kernel_list)
        new.eig_vects = tf.gather(self.eig_vects, idx, axis=0)
        new.mass_matrix = tf.gather(self.mass_matrix, idx)
        new._precompute()
        if self.is_fixed:
            new.set_log_prob_fixed(**self._fixed_kwargs)
        return new

    def _predict(self, **kwargs):
        raise NotImplementedError('Prediction is not implemented for GPSpec')


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
    # 1. Compute squared distances
    D2 = tf.square(distance_matrix)
    # 2. Number of points
    n = tf.shape(distance_matrix)[0]
    n_float = tf.cast(n, distance_matrix.dtype)
    # 3. Centering matrix J = I - (1/n) * 11^T
    I = tf.eye(n, dtype=distance_matrix.dtype)
    ones = tf.ones((n, n), dtype=distance_matrix.dtype)
    J = I - ones / n_float
    # 4. Gram matrix K = -0.5 * J * D2 * J (double centering)
    K = -0.5 * tf.matmul(J, tf.matmul(D2, J))
    # 5. Eigen decomposition (ascending eigenvalues)
    eigenvalues, eigenvectors = tf.linalg.eigh(K)
    # 6. Embedding dimension: number of eigenvalues > eps, if not given
    if embedding_dim is None:
        positive_mask = eigenvalues > eps
        embedding_dim = tf.reduce_sum(tf.cast(positive_mask, tf.int32))
    # 7. Largest 'embedding_dim' eigenpairs
    eigenvalues = eigenvalues[-embedding_dim:]
    eigenvectors = eigenvectors[:, -embedding_dim:]
    # 8. X = U * sqrt(Lambda), so X X^T = K
    eigenvalues = tf.maximum(eigenvalues, 0)
    X = eigenvectors * tf.sqrt(eigenvalues)
    return X

@tf.function
def compute_euclidean_distance_matrix(X, eps=1e-6):
    """
    Computes the pairwise Euclidean distance matrix from the embedding X [n x d].
    eps inside the sqrt keeps gradients finite (so the diagonal is ~sqrt(eps), not 0).
    """
    X_expanded1 = tf.expand_dims(X, axis=1)
    X_expanded2 = tf.expand_dims(X, axis=0)
    diff = X_expanded1 - X_expanded2
    D_euc = tf.sqrt(tf.reduce_sum(tf.square(diff), axis=-1) + eps)
    return D_euc

def pairwise_euclidean_distance(X1, X2):
    """
    Computes the pairwise Euclidean distance matrix between X1 [n x d] and X2 [m x d] -> [n x m]
    """
    return _pairwise_dist_np(X1, X2)
