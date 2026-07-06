"""Tests for GP (Gaussian Process) class, MDS/geodesic embedding, and related utilities."""
import numpy as np
import pytest
import tensorflow as tf


N_VX = 12       # small enough for Cholesky to be fast
EMBED_DIM = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_distance_matrix(n, rng=None):
    """Symmetric zero-diagonal distance matrix from random 2D coordinates."""
    if rng is None:
        rng = np.random.default_rng(0)
    coords = rng.uniform(0, 10, size=(n, 2)).astype(np.float32)
    diff = coords[:, None, :] - coords[None, :, :]
    D = np.sqrt((diff ** 2).sum(axis=-1)).astype(np.float32)
    return tf.constant(D), coords


def _random_coordinates(n, rng=None):
    if rng is None:
        rng = np.random.default_rng(1)
    return rng.uniform(0, 5, size=(n,)).astype(np.float32)


def _make_gp_with_kernel(n_vx=N_VX):
    """Construct a GP with float32 dtype and one RBF stationary kernel."""
    from braincoder.bprf_GPs import GP
    coords = _random_coordinates(n_vx)
    gp = GP(n_vx=n_vx, gp_dtype=tf.float32)
    gp.add_xid_stationary_kernel('k0', Xs=coords, kernel_type='RBF')
    return gp, coords


def _gp_kwargs_for_log_prob(gp):
    """Return the minimal kwargs dict needed to call return_log_prob."""
    kwargs = {name: tf.constant(1.0, dtype=gp.gp_dtype)
              for name in gp.pids_inv.keys()}
    return kwargs


def _gp_params_tensor(gp, n_vx=N_VX):
    """A random parameter vector in the GP's dtype."""
    rng = np.random.default_rng(5)
    vals = rng.normal(0, 1, size=(n_vx,))
    return tf.constant(vals, dtype=gp.gp_dtype)


# ---------------------------------------------------------------------------
# mds_embedding tests
# ---------------------------------------------------------------------------

class TestMDSEmbedding:

    def test_output_shape(self):
        from braincoder.bprf_GPs import mds_embedding
        D, _ = _random_distance_matrix(N_VX)
        X = mds_embedding(D, embedding_dim=EMBED_DIM)
        assert X.shape == (N_VX, EMBED_DIM)

    def test_reconstruction_is_close(self):
        """Embedded Gram matrix should approximate original Gram matrix."""
        from braincoder.bprf_GPs import mds_embedding
        D, _ = _random_distance_matrix(N_VX)
        X = mds_embedding(D, embedding_dim=N_VX)   # full rank
        K_reconstructed = tf.matmul(X, tf.transpose(X)).numpy()
        # Double-centered squared distance is the Gram matrix
        D2 = (D.numpy() ** 2).astype(np.float64)
        n = N_VX
        J = np.eye(n) - np.ones((n, n)) / n
        K_true = -0.5 * J @ D2 @ J
        # Only compare positive-eigenvalue part
        evals = np.linalg.eigvalsh(K_true)
        # Frobenius norm of difference should be small relative to signal
        err = np.linalg.norm(K_reconstructed - K_true) / np.linalg.norm(K_true)
        assert err < 0.1

    def test_output_is_real_valued(self):
        from braincoder.bprf_GPs import mds_embedding
        D, _ = _random_distance_matrix(N_VX)
        X = mds_embedding(D, embedding_dim=EMBED_DIM)
        assert not np.any(np.isnan(X.numpy()))
        assert not np.any(np.isinf(X.numpy()))

    def test_auto_embedding_dim(self):
        """With embedding_dim=None it should still return a 2D tensor."""
        from braincoder.bprf_GPs import mds_embedding
        D, _ = _random_distance_matrix(N_VX)
        X = mds_embedding(D, embedding_dim=None)
        assert len(X.shape) == 2
        assert X.shape[0] == N_VX

    def test_zero_distance_zero_embedding(self):
        """Identical points → zero distance → embedding should be near zero."""
        from braincoder.bprf_GPs import mds_embedding
        D = tf.zeros((N_VX, N_VX), dtype=tf.float32)
        X = mds_embedding(D, embedding_dim=EMBED_DIM)
        np.testing.assert_allclose(X.numpy(), 0.0, atol=1e-5)


# ---------------------------------------------------------------------------
# compute_euclidean_distance_matrix tests
# ---------------------------------------------------------------------------

class TestComputeEuclideanDistanceMatrix:
    # compute_euclidean_distance_matrix expects 2D input [n, d]

    def test_output_shape(self):
        from braincoder.bprf_GPs import compute_euclidean_distance_matrix
        X = tf.constant(_random_coordinates(N_VX)[:, np.newaxis])  # (N_VX, 1)
        D = compute_euclidean_distance_matrix(X)
        assert D.shape == (N_VX, N_VX)

    def test_symmetry(self):
        from braincoder.bprf_GPs import compute_euclidean_distance_matrix
        X = tf.constant(_random_coordinates(N_VX)[:, np.newaxis])
        D = compute_euclidean_distance_matrix(X).numpy()
        np.testing.assert_allclose(D, D.T, atol=1e-5)

    def test_diagonal_is_small(self):
        """Diagonal is ~sqrt(eps) due to numerical stabilisation in the function."""
        from braincoder.bprf_GPs import compute_euclidean_distance_matrix
        X = tf.constant(_random_coordinates(N_VX)[:, np.newaxis])
        D = compute_euclidean_distance_matrix(X).numpy()
        np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-2)

    def test_non_negative(self):
        from braincoder.bprf_GPs import compute_euclidean_distance_matrix
        X = tf.constant(_random_coordinates(N_VX)[:, np.newaxis])
        D = compute_euclidean_distance_matrix(X).numpy()
        assert np.all(D >= 0)

    def test_triangle_inequality(self):
        from braincoder.bprf_GPs import compute_euclidean_distance_matrix
        rng = np.random.default_rng(7)
        X = tf.constant(rng.uniform(0, 5, size=(5, 2)).astype(np.float32))
        D = compute_euclidean_distance_matrix(X).numpy()
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    assert D[i, j] <= D[i, k] + D[k, j] + 1e-5


# ---------------------------------------------------------------------------
# GP class tests
# ---------------------------------------------------------------------------

class TestGPInit:

    def test_default_pids(self):
        from braincoder.bprf_GPs import GP
        gp = GP(n_vx=N_VX, gp_dtype=tf.float32)
        assert 'gpk_nugget' in gp.pids_inv
        assert 'mfunc_mean' in gp.pids_inv

    def test_n_vx_stored(self):
        from braincoder.bprf_GPs import GP
        gp = GP(n_vx=N_VX, gp_dtype=tf.float32)
        assert gp.n_vx.numpy() == N_VX

    def test_empty_kernel_lists(self):
        from braincoder.bprf_GPs import GP
        gp = GP(n_vx=N_VX, gp_dtype=tf.float32)
        assert gp.stat_kernel_list == []
        assert gp.warp_kernel_list == []
        assert gp.mfunc_list == []


class TestGPAddStatKernel:

    def test_adds_lengthscale_and_variance_pids(self):
        from braincoder.bprf_GPs import GP
        coords = _random_coordinates(N_VX)
        gp = GP(n_vx=N_VX, gp_dtype=tf.float32)
        n_before = len(gp.pids)
        gp.add_xid_stationary_kernel('k0', Xs=coords, kernel_type='RBF')
        assert len(gp.pids) == n_before + 2
        # pid names use xid directly: f'gpk{xid}_l'
        assert 'gpkk0_l' in gp.pids_inv
        assert 'gpkk0_v' in gp.pids_inv

    def test_kernel_type_stored(self):
        from braincoder.bprf_GPs import GP
        coords = _random_coordinates(N_VX)
        gp = GP(n_vx=N_VX, gp_dtype=tf.float32)
        gp.add_xid_stationary_kernel('k0', Xs=coords, kernel_type='matern52')
        assert gp.kernel_type['k0'] == 'matern52'

    def test_distance_matrix_is_symmetric(self):
        from braincoder.bprf_GPs import GP
        coords = _random_coordinates(N_VX)
        gp = GP(n_vx=N_VX, gp_dtype=tf.float32)
        gp.add_xid_stationary_kernel('k0', Xs=coords, kernel_type='RBF')
        D = gp.dXs['k0'].numpy()
        np.testing.assert_allclose(D, D.T, atol=1e-5)

    def test_multiple_kernels(self):
        from braincoder.bprf_GPs import GP
        coords = _random_coordinates(N_VX)
        gp = GP(n_vx=N_VX, gp_dtype=tf.float32)
        gp.add_xid_stationary_kernel('k0', Xs=coords, kernel_type='RBF')
        gp.add_xid_stationary_kernel('k1', Xs=coords, kernel_type='laplace')
        assert len(gp.stat_kernel_list) == 2


class TestGPAddLinearMfunc:

    def test_adds_slope_pids(self):
        from braincoder.bprf_GPs import GP
        coords = _random_coordinates(N_VX)
        gp = GP(n_vx=N_VX, gp_dtype=tf.float32)
        n_before = len(gp.pids)
        gp.add_xid_linear_mfunc('x0', Xs=coords)
        # 1D coords → 1 slope parameter; pid name is f'mfunc{xid}_slope{i}'
        assert len(gp.pids) == n_before + 1
        assert 'mfuncx0_slope0' in gp.pids_inv

    def test_multi_dim_coords_add_multiple_slopes(self):
        from braincoder.bprf_GPs import GP
        rng = np.random.default_rng(3)
        coords_2d = rng.uniform(0, 5, size=(N_VX, 2)).astype(np.float32)
        gp = GP(n_vx=N_VX, gp_dtype=tf.float32)
        n_before = len(gp.pids)
        gp.add_xid_linear_mfunc('x0', Xs=coords_2d)
        assert len(gp.pids) == n_before + 2
        assert 'mfuncx0_slope0' in gp.pids_inv
        assert 'mfuncx0_slope1' in gp.pids_inv


class TestGPCovarianceMatrix:

    def test_rbf_covariance_is_psd(self):
        from braincoder.bprf_GPs import GP
        coords = _random_coordinates(N_VX)
        gp = GP(n_vx=N_VX, gp_dtype=tf.float32)
        gp.add_xid_stationary_kernel('k0', Xs=coords, kernel_type='RBF')
        kwargs = _gp_kwargs_for_log_prob(gp)
        sigma = gp._return_sigma_full(**kwargs).numpy()
        eigvals = np.linalg.eigvalsh(sigma)
        assert np.all(eigvals > -1e-5)

    def test_matern52_covariance_is_psd(self):
        from braincoder.bprf_GPs import GP
        coords = _random_coordinates(N_VX)
        gp = GP(n_vx=N_VX, gp_dtype=tf.float32)
        gp.add_xid_stationary_kernel('k0', Xs=coords, kernel_type='matern52')
        kwargs = _gp_kwargs_for_log_prob(gp)
        sigma = gp._return_sigma_full(**kwargs).numpy()
        eigvals = np.linalg.eigvalsh(sigma)
        assert np.all(eigvals > -1e-5)

    def test_laplace_covariance_is_psd(self):
        from braincoder.bprf_GPs import GP
        coords = _random_coordinates(N_VX)
        gp = GP(n_vx=N_VX, gp_dtype=tf.float32)
        gp.add_xid_stationary_kernel('k0', Xs=coords, kernel_type='laplace')
        kwargs = _gp_kwargs_for_log_prob(gp)
        sigma = gp._return_sigma_full(**kwargs).numpy()
        eigvals = np.linalg.eigvalsh(sigma)
        assert np.all(eigvals > -1e-5)

    def test_covariance_is_symmetric(self):
        from braincoder.bprf_GPs import GP
        coords = _random_coordinates(N_VX)
        gp = GP(n_vx=N_VX, gp_dtype=tf.float32)
        gp.add_xid_stationary_kernel('k0', Xs=coords, kernel_type='RBF')
        kwargs = _gp_kwargs_for_log_prob(gp)
        sigma = gp._return_sigma_full(**kwargs).numpy()
        np.testing.assert_allclose(sigma, sigma.T, atol=1e-5)


class TestGPLogProb:

    def test_return_log_prob_unfixed_is_scalar_per_voxel(self):
        gp, _ = _make_gp_with_kernel()
        params = _gp_params_tensor(gp)
        kwargs = _gp_kwargs_for_log_prob(gp)
        lp = gp._return_log_prob_unfixed(params, **kwargs)
        assert np.isscalar(lp.numpy()) or lp.numpy().ndim == 0

    def test_log_prob_is_finite(self):
        gp, _ = _make_gp_with_kernel()
        params = _gp_params_tensor(gp)
        kwargs = _gp_kwargs_for_log_prob(gp)
        lp = gp._return_log_prob_unfixed(params, **kwargs).numpy()
        assert np.isfinite(lp)

    def test_set_log_prob_fixed_changes_mode(self):
        gp, _ = _make_gp_with_kernel()
        kwargs = _gp_kwargs_for_log_prob(gp)
        gp.set_log_prob_fixed(**kwargs)
        assert gp.return_log_prob == gp._return_log_prob_fixed

    def test_set_log_prob_univariate_changes_mode(self):
        gp, _ = _make_gp_with_kernel()
        gp.set_log_prob_univariate()
        assert gp.return_log_prob == gp._return_log_prob_univariate

    def test_nystrom_approximation_changes_mode(self):
        from braincoder.bprf_GPs import GP
        gp = GP(n_vx=N_VX, gp_dtype=tf.float32)
        coords = _random_coordinates(N_VX)
        gp.add_xid_stationary_kernel('k0', Xs=coords, kernel_type='RBF')
        gp.add_nystrom_approximation(n_inducers=4)
        assert gp.return_log_prob == gp._return_log_prob_nystrom
        assert gp.n_inducers == 4


class TestGPUpdateNVx:

    def test_update_n_vx(self):
        from braincoder.bprf_GPs import GP
        gp = GP(n_vx=10)
        gp.update_n_vx(20)
        assert gp.n_vx.numpy() == 20


class TestGPMfuncBijector:

    def test_softplus_mfunc_bijector(self):
        from braincoder.bprf_GPs import GP
        from tensorflow_probability import bijectors as tfb
        gp = GP(n_vx=N_VX, gp_dtype=tf.float32)
        gp.add_mfunc_bijector('softplus')
        assert isinstance(gp.mfunc_bijector, tfb.Softplus)

    def test_sigmoid_mfunc_bijector(self):
        from braincoder.bprf_GPs import GP
        from tensorflow_probability import bijectors as tfb
        gp = GP(n_vx=N_VX, gp_dtype=tf.float32)
        gp.add_mfunc_bijector('sigmoid', low=0.0, high=1.0)
        assert isinstance(gp.mfunc_bijector, tfb.Sigmoid)
