"""Regression tests for the GP / BPRF_hier fixes (one or more per bug). Checked against exact
dense-covariance log densities from scipy wherever possible."""
import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from scipy.stats import multivariate_normal, norm

from braincoder.bprf_GPs import GP, GPSpec
from braincoder.bprf_hierarchical import BPRF_hier
from braincoder.models import CRFWithHRF
from braincoder.hrf import SPMHRFModel

N, T, M_EIG = 40, 60, 8
RNG = np.random.default_rng(0)
XY = RNG.uniform(0, 5, (N, 2))
DX = np.sqrt(((XY[:, None] - XY[None]) ** 2).sum(-1))
HP = dict(gpk_nugget=0.3, mfunc_mean=0.5, gpkk_l=1.5, gpkk_v=1.2)
Y = RNG.normal(size=N)


def rbf(d, l, v):
    return v ** 2 * np.exp(-d ** 2 / (2 * l ** 2))


def exact_lp(y, mean, K):
    return multivariate_normal(mean=np.broadcast_to(mean, len(y)), cov=K).logpdf(y)


def make_gp(n=N, dX=DX):
    gp = GP(n_vx=n)
    gp.add_xid_stationary_kernel('k', dXs=dX)
    return gp


def make_model():
    sf = np.repeat(np.where(np.arange(6) % 2, 3.0, 0.3), 10)
    con = np.repeat(RNG.choice([.1, 1.], 6), 10)
    return CRFWithHRF(SF_seq=sf, CON_seq=con, hrf_model=SPMHRFModel(tr=1.), crf_type='sqrt',
                      flexible_hrf_parameters=True)


def init_pars(m, n=N):
    p = pd.DataFrame(1.0, index=range(n), columns=m.parameter_labels + ['noise_scale']).astype('float32')
    p['hrf_delay'] = 6.0
    return p


def spec_gp(n=N, kernel='spec_heat', xid='s'):
    evecs = np.linalg.qr(np.random.default_rng(1).normal(size=(n, M_EIG)))[0]
    evals = np.linspace(0, 4, M_EIG)
    g = GPSpec(n, evecs, evals)
    g.add_xid_kappa_kernel(xid, kernel_type=kernel)
    return g, evecs, evals


# ----------------------------------------------------------------------------- GP log densities
class TestExactDensities:

    def test_unfixed_matches_scipy(self):
        gp = make_gp()
        K = rbf(DX, 1.5, 1.2) + (0.3 + gp._jitter(**HP).numpy()) * np.eye(N)
        assert np.isclose(gp.return_log_prob(tf.constant(Y, tf.float32), **HP).numpy(),
                          exact_lp(Y, 0.5, K), rtol=1e-4)

    def test_fixed_matches_unfixed(self):
        gp = make_gp()
        lp_unfixed = gp.return_log_prob(tf.constant(Y, tf.float32), **HP).numpy()
        gp.set_log_prob_fixed(**HP)
        assert np.isclose(gp.return_log_prob(tf.constant(Y, tf.float32)).numpy(), lp_unfixed, rtol=1e-5)

    def test_univariate_nugget_is_a_variance(self):
        gp = make_gp()
        gp.set_log_prob_univariate()
        var = 0.3 + gp._jitter(**HP).numpy()
        expected = norm(0.5, np.sqrt(var)).logpdf(Y).sum()
        assert np.isclose(gp.return_log_prob(tf.constant(Y, tf.float32), **HP).numpy(), expected, rtol=1e-4)

    def test_nystrom_with_all_inducers_is_exact(self):
        gp = make_gp()
        exact = gp.return_log_prob(tf.constant(Y, tf.float32), **HP).numpy()
        gp.add_nystrom_approximation(N, inducer_idx=np.arange(N))
        assert np.isclose(gp.return_log_prob(tf.constant(Y, tf.float32), **HP).numpy(), exact, rtol=1e-3)

    def test_two_d_coordinates(self):
        gp = GP(n_vx=N)
        gp.add_xid_stationary_kernel('k', Xs=XY)
        np.testing.assert_allclose(gp.dXs['k'].numpy(), DX, atol=1e-8)

    def test_default_psd_control_keeps_distances(self):
        assert np.allclose(make_gp().dXs['k'].numpy(), DX)


# ----------------------------------------------------------------------------- subsets (stale traces)
class TestSubset:

    def test_subset_after_tracing_uses_new_distances(self):
        gp = make_gp()
        gp.return_log_prob(tf.constant(Y, tf.float32), **HP)            # trace on all vertices
        for sub in [np.arange(20), np.arange(20, 40)]:                  # two same-size subsets
            g = gp.subset(sub)
            K = rbf(DX[np.ix_(sub, sub)], 1.5, 1.2) + (0.3 + g._jitter(**HP).numpy()) * np.eye(20)
            assert np.isclose(g.return_log_prob(tf.constant(Y[sub], tf.float32), **HP).numpy(),
                              exact_lp(Y[sub], 0.5, K), rtol=1e-4)
        # the original is untouched
        g20 = gp.subset(np.arange(20))      # keep a reference: tf.function methods hold their object weakly
        np.testing.assert_allclose(gp._return_sigma_full(**HP).numpy()[:20, :20],
                                   g20._return_sigma_full(**HP).numpy())

    def test_subset_keeps_fixed_mode(self):
        gp = make_gp()
        gp.set_log_prob_fixed(**HP)
        g = gp.subset(np.arange(10, 30))
        assert g.is_fixed
        K = rbf(DX[10:30, 10:30], 1.5, 1.2) + (0.3 + g._jitter(**HP).numpy()) * np.eye(20)
        assert np.isclose(g.return_log_prob(tf.constant(Y[10:30], tf.float32)).numpy(),
                          exact_lp(Y[10:30], 0.5, K), rtol=1e-4)

    def test_nystrom_inducers_remapped(self):
        gp = make_gp()
        gp.add_nystrom_approximation(10, inducer_idx=np.arange(30, 40))   # none inside the subset below
        g = gp.subset(np.arange(25))
        assert g.n_inducers == 10 and g.inducer_idx.max() < 25
        assert np.isfinite(g.return_log_prob(tf.constant(Y[:25], tf.float32), **HP).numpy())
        g2 = gp.subset(np.arange(28, 40))                                  # inducers 30..39 -> 2..11
        np.testing.assert_array_equal(g2.inducer_idx, np.arange(2, 12))

    def test_mean_function_subset(self):
        gp = make_gp()
        gp.add_xid_linear_mfunc('m', Xs=XY[:, 0])
        hp = {**HP, 'mfuncm_slope0': 0.7}
        g = gp.subset(np.arange(15))
        np.testing.assert_allclose(g._return_mfunc(**hp).numpy(), 0.5 + 0.7 * XY[:15, 0])


# ----------------------------------------------------------------------------- GPSpec
class TestGPSpec:
    HPS = dict(gpk_nugget=0.2, mfunc_mean=0.0, gpk_var=2.0, gpks_kappa_t=0.5)

    def _exact(self, evecs, evals, rows):
        kappa = 2.0 * np.exp(-0.5 * evals)
        Phi = evecs[rows]
        return Phi @ np.diag(kappa) @ Phi.T + (0.2 + 1e-6) * np.eye(len(rows))

    def test_full_matches_dense(self):
        g, evecs, evals = spec_gp()
        lp = g.return_log_prob(tf.constant(Y, tf.float32), **self.HPS)
        assert lp.dtype == tf.float32
        assert np.isclose(lp.numpy(), exact_lp(Y, 0.0, self._exact(evecs, evals, np.arange(N))), rtol=1e-4)

    def test_subset_matches_dense(self):
        g, evecs, evals = spec_gp()
        rows = np.arange(5, 30)
        lp = g.subset(rows).return_log_prob(tf.constant(Y[rows], tf.float32), **self.HPS).numpy()
        assert np.isclose(lp, exact_lp(Y[rows], 0.0, self._exact(evecs, evals, rows)), rtol=1e-4)

    def test_kernel_name_k_and_two_kernels(self):
        g, evecs, evals = spec_gp(xid='k')                          # 'k' used to be stripped from 'gpk...'
        g.add_xid_kappa_kernel('e', kernel_type='spec_exp')
        hp = dict(gpk_nugget=0.2, mfunc_mean=0.0, gpk_var=2.0, gpkk_kappa_t=0.5, gpke_kappa_l=1.0)
        kappa = 2.0 * (np.exp(-0.5 * evals) + np.exp(-1.0 * evals))
        np.testing.assert_allclose(g._return_kappa(**hp).numpy(), kappa, rtol=1e-6)

    def test_in_bprf_hier(self):
        m = make_model()
        g, _, _ = spec_gp()
        bp = BPRF_hier(m, RNG.normal(size=(T, N)).astype('float32'), noise_method='fit_normal')
        bp.h_add_param('baseline', 'gp', gp=g)
        h0 = pd.DataFrame([{k: 0.5 for k in bp.h_labels}])
        bp.fit_MAP_hier(init_pars=init_pars(m), h_init_pars=h0, num_steps=3)
        bp.fit_MAP_hier_GP_only(pid='baseline', pid_pars=init_pars(m), h_init_pars=h0, num_steps=2, idx=np.arange(25))
        assert np.all(np.isfinite(bp.h_MAP_parameters.values))


# ----------------------------------------------------------------------------- prediction
class TestPredict:

    def test_linear_mean_used_at_new_points(self):
        gp = GP(n_vx=N)
        gp.add_xid_stationary_kernel('k', Xs=XY)
        gp.add_xid_linear_mfunc('m', Xs=XY[:, 0])
        hp = {**HP, 'mfuncm_slope0': 2.0}
        far = np.array([[100.0, 100.0]])                    # no covariance with the data -> prior mean
        mean, std = gp._predict(parameter=Y, Xs_new=far, mfunc_Xs_new={'m': far[:, 0]}, **hp)
        assert np.isclose(mean.numpy()[0], 0.5 + 2.0 * 100.0, rtol=1e-5)
        assert np.isclose(std.numpy()[0], 1.2, rtol=1e-3)
        with pytest.raises(ValueError):
            gp._predict(parameter=Y, Xs_new=far, **hp)


# ----------------------------------------------------------------------------- BPRF_hier
class TestBPRFHier:

    def _bp(self, pids=('baseline',), data=None, **gp_kw):
        m = make_model()
        bp = BPRF_hier(m, RNG.normal(size=(T, N)).astype('float32') if data is None else data,
                       noise_method='fit_normal')
        for p in pids:
            bp.h_add_param(p, 'gp', gp=make_gp(), **gp_kw)
        return m, bp

    def test_exact_hyperparameter_lookup(self):
        """'amplitude' must not pick up 'amplitude_sf1_*' hyperparameters"""
        m, bp = self._bp(pids=('amplitude', 'amplitude_sf1'))
        bp._update_gps_for_idx(np.arange(N))
        h = tf.constant(np.arange(len(bp.h_labels), dtype='float32')[None])
        kw = bp._gp_kwargs('amplitude', h)
        assert set(kw) == set(bp.h_gp_function['amplitude'].pids_inv)
        for k, v in kw.items():
            assert v.numpy()[0] == bp.h_labels[f'amplitude_{k}']

    def test_bijectors_chosen_on_gp_key(self):
        m, bp = self._bp(pids=('amplitude_sf1',))
        assert bp.h_bijector['amplitude_sf1_mfunc_mean'].name == 'identity'
        assert bp.h_bijector['amplitude_sf1_gpkk_l'].name == 'softplus'
        assert bp.h_prior_type['amplitude_sf1_mfunc_mean'] == 'none'
        assert bp.h_prior_type['amplitude_sf1_gpkk_l'] == 'LogNormal'

    def test_fixed_gp_hyperparameters_are_held_and_reported(self):
        m = make_model()
        bp = BPRF_hier(m, RNG.normal(size=(T, N)).astype('float32'), noise_method='fit_normal')
        gp = make_gp()
        gp.set_log_prob_fixed(**HP)
        bp.h_add_param('baseline', 'gp', gp=gp)
        h0 = pd.DataFrame([{k: 3.0 for k in bp.h_labels}])
        bp.fit_MAP_hier(init_pars=init_pars(m), h_init_pars=h0, num_steps=5, adam_kwargs=dict(learning_rate=0.1))
        for k, v in HP.items():
            assert np.isclose(bp.h_MAP_parameters[f'baseline_{k}'][0], v)

    def test_user_h_fixed_pars(self):
        m, bp = self._bp()
        h0 = pd.DataFrame([{k: 1.0 for k in bp.h_labels}])
        bp.fit_MAP_hier(init_pars=init_pars(m), h_init_pars=h0, num_steps=3,
                        h_fixed_pars={'baseline_gpkk_l': 2.5})
        assert np.isclose(bp.h_MAP_parameters['baseline_gpkk_l'][0], 2.5)
        assert bp.h_prior_type['baseline_gpkk_l'] == 'LogNormal'      # not permanently switched off

    def test_unfit_vertices_keep_init_and_unsorted_idx(self):
        m, bp = self._bp()
        init = init_pars(m)
        init['baseline'] = np.arange(N, dtype='float32')
        h0 = pd.DataFrame([{k: 1.0 for k in bp.h_labels}])
        idx = [30, 3, 17, 8]
        bp.fit_MAP_hier(init_pars=init, h_init_pars=h0, num_steps=2)       # warm-up trace on all vertices
        bp.fit_MAP_hier(init_pars=init, h_init_pars=h0, num_steps=0, idx=idx)
        out = bp.MAP_parameters
        rest = np.setdiff1d(np.arange(N), idx)
        np.testing.assert_allclose(out['baseline'].values[rest], init['baseline'].values[rest])
        np.testing.assert_allclose(out['baseline'].values[idx], init['baseline'].values[idx], rtol=1e-5)
        assert np.all(out['noise_scale'].values > 0)

    def test_subset_fit_then_full_fit(self):
        """the empirical-Bayes workflow: GP-only on a subset, then a full fit with the same GP objects"""
        m, bp = self._bp()
        h0 = pd.DataFrame([{k: 1.0 for k in bp.h_labels}])
        bp.fit_MAP_hier_GP_only(pid='baseline', pid_pars=init_pars(m), h_init_pars=h0, num_steps=3, idx=np.arange(20))
        bp.fit_MAP_hier(init_pars=init_pars(m), h_init_pars=bp.h_MAP_parameters, num_steps=3)
        bp.fit_MAP_hier_GP_only(pid='baseline', pid_pars=init_pars(m), h_init_pars=h0, num_steps=3, idx=np.arange(20, 40))
        assert np.all(np.isfinite(bp.MAP_parameters.values))

    def test_refit_with_different_fixed_pars(self):
        """transform methods must not keep the bijectors of their first trace"""
        m, bp = self._bp()
        bp.add_bijector('amplitude', 'softplus')
        h0 = pd.DataFrame([{k: 1.0 for k in bp.h_labels}])
        bp.fit_MAP_hier(init_pars=init_pars(m), h_init_pars=h0, num_steps=2, fixed_pars={'amplitude': 2.0})
        assert np.allclose(bp.MAP_parameters['amplitude'], 2.0)
        bp.add_bijector('amplitude', 'softplus')
        bp.fit_MAP_hier(init_pars=init_pars(m), h_init_pars=h0, num_steps=0)
        assert np.allclose(bp.MAP_parameters['amplitude'], 1.0, atol=1e-5)

    def test_unconstrained_gp_space(self):
        m = make_model()
        bp = BPRF_hier(m, RNG.normal(size=(T, N)).astype('float32'), noise_method='fit_normal')
        bp.add_bijector('amplitude', tfb_exp())
        bp.h_add_param('amplitude', 'gp', gp=make_gp(), gp_space='unconstrained')
        h0 = pd.DataFrame([{k: 1.0 for k in bp.h_labels}])
        bp.fit_MAP_hier(init_pars=init_pars(m), h_init_pars=h0, num_steps=3)
        assert np.all(np.isfinite(bp.MAP_parameters['amplitude'])) and np.all(bp.MAP_parameters['amplitude'] > 0)

    def test_gp_size_must_match_data(self):
        m = make_model()
        bp = BPRF_hier(m, RNG.normal(size=(T, N)).astype('float32'))
        with pytest.raises(ValueError):
            bp.h_add_param('baseline', 'gp', gp=make_gp(n=10, dX=DX[:10, :10]))


def tfb_exp():
    from tensorflow_probability import bijectors as tfb
    return tfb.Exp()
