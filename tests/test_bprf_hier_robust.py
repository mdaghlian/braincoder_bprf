"""Tests for the BPRF_hier guard rails (NaN-safe updates, clipping, best-state restore, early stopping,
constant-voxel masking), shared lengthscales, the MDS default and the 3-stage empirical-Bayes wrapper."""
import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from braincoder.bprf_GPs import GP, mds_embedding
from braincoder.bprf_hierarchical import BPRF_hier
from braincoder.models import CRFWithHRF
from braincoder.hrf import SPMHRFModel

N, T = 40, 60
RNG = np.random.default_rng(0)
XY = RNG.uniform(0, 5, (N, 2))
DX = np.sqrt(((XY[:, None] - XY[None]) ** 2).sum(-1))


def make_gp(**kw):
    gp = GP(n_vx=N)
    gp.add_xid_stationary_kernel('k', dXs=DX, **kw)
    return gp


def make_model():
    # blank blocks (NaN SF) so the amplitudes are not collinear with the baseline
    sf = np.repeat([0.3, np.nan, 3.0, np.nan, 0.3, 3.0], 10)
    con = np.repeat([1., 0., 1., 0., .1, .1], 10)
    return CRFWithHRF(SF_seq=sf, CON_seq=con, hrf_model=SPMHRFModel(tr=1.), crf_type='sqrt',
                      flexible_hrf_parameters=True)


def init_pars(m):
    p = pd.DataFrame(1.0, index=range(N), columns=m.parameter_labels + ['noise_scale']).astype('float32')
    p['hrf_delay'] = 6.0
    return p


def make_bp(data=None, pids=('baseline',), **bp_kw):
    m = make_model()
    data = RNG.normal(size=(T, N)).astype('float32') if data is None else data
    bp = BPRF_hier(m, data, noise_method='fit_normal', **bp_kw)
    for p in pids:
        bp.h_add_param(p, 'gp', gp=make_gp())
    return m, bp


def h_init(bp, val=1.0):
    return pd.DataFrame([{k: val for k in bp.h_labels}])


def run(bp, loss_fn, x0, num_steps, **kw):
    x = [tf.Variable(np.asarray(v, dtype='float32')) for v in x0]
    info = bp._run_MAP_optimizer(loss_fn(x), x, num_steps, 'test', **kw)
    return x, info


# ----------------------------------------------------------------------------- optimizer guard rails
class TestOptimizer:

    def test_nonfinite_loss_is_skipped_and_reverted(self):
        _, bp = make_bp()
        # minimum at 3, but the loss is NaN beyond 2.5: Adam with a big step jumps in there
        f = lambda x: lambda: tf.where(x[0] > 2.5, np.float32(np.nan), tf.square(x[0] - 3.0))
        x, info = run(bp, f, [0.0], 200, adam_kwargs=dict(learning_rate=1.0))
        assert info['n_nonfinite'] > 0
        assert np.isfinite(x[0].numpy()) and x[0].numpy() <= 2.5
        assert x[0].numpy() > 2.0                      # still made progress (smaller steps after reverts)

    def test_stops_after_max_nonfinite(self):
        _, bp = make_bp()
        f = lambda x: lambda: x[0] * np.float32(np.nan)
        x, info = run(bp, f, [1.0], 100, max_nonfinite=5)
        assert info['n_steps'] == 5 and info['n_nonfinite'] == 5
        assert x[0].numpy() == 1.0

    def test_nonfinite_gradient_entries_zeroed(self):
        _, bp = make_bp()
        # finite loss, but the gradient wrt x[1] is NaN (sqrt of a negative number in a dead branch)
        f = lambda x: lambda: tf.reduce_sum(tf.square(x[0] - 1.0)) + tf.where(
            x[1] > 0, tf.sqrt(x[1]), tf.zeros_like(x[1]))
        x, info = run(bp, f, [[5.0, 5.0], -1.0], 300, adam_kwargs=dict(learning_rate=0.1))
        np.testing.assert_allclose(x[0].numpy(), 1.0, atol=1e-2)
        assert x[1].numpy() == -1.0

    def test_global_norm_clipping(self):
        _, bp = make_bp()
        norms = []
        f = lambda x: lambda: 1e3 * tf.reduce_sum(tf.square(x[0])) + 1e3 * tf.square(x[1])
        x = [tf.Variable([3.0, -2.0]), tf.Variable(4.0)]
        bp._run_MAP_optimizer(f(x), x, 10, 'test', clipnorm=0.5,
                              step_callback=lambda s, l, g: norms.append(tf.linalg.global_norm(g).numpy()))
        assert len(norms) == 10 and max(norms) <= 0.5 + 1e-5

    def test_best_state_restored(self):
        _, bp = make_bp()
        f = lambda x: lambda: tf.square(x[0])
        # a large learning rate overshoots, so the last state is not the best
        x, info = run(bp, f, [5.0], 50, adam_kwargs=dict(learning_rate=3.0))
        assert np.isclose(x[0].numpy() ** 2, info['best_loss'])
        assert info['best_loss'] <= min(info['loss']) and info['best_loss'] <= info['final_loss']
        x2, info2 = run(bp, f, [5.0], 50, adam_kwargs=dict(learning_rate=3.0), restore_best=False)
        assert np.isclose(x2[0].numpy() ** 2, info2['final_loss'], rtol=1e-5)

    def test_early_stopping(self):
        _, bp = make_bp()
        f = lambda x: lambda: tf.constant(1.0) + 0.0 * x[0]
        x, info = run(bp, f, [1.0], 1000, patience=7)
        assert info['stopped_early'] and info['n_steps'] == 8
        x, info = run(bp, f, [1.0], 20)                          # patience=None: run all steps
        assert not info['stopped_early'] and info['n_steps'] == 20


# ----------------------------------------------------------------------------- constant voxels
class TestMasking:

    def _data(self):
        data = RNG.normal(size=(T, N)).astype('float32')
        data[:, 3] = 2.0            # constant
        data[5, 7] = np.nan         # non-finite
        return data

    def test_fit_map_hier_masks_constant_and_nan_voxels(self):
        m, bp = make_bp(data=self._data())
        init = init_pars(m)
        init['baseline'] = np.arange(N, dtype='float32')
        bp.fit_MAP_hier(init_pars=init, h_init_pars=h_init(bp), num_steps=5)
        np.testing.assert_array_equal(bp.masked_voxels, [3, 7])
        assert bp.n_vx_to_fit == N - 2
        out = bp.MAP_parameters
        assert np.all(np.isfinite(out.values))
        np.testing.assert_allclose(out['baseline'].values[[3, 7]], [3, 7])
        assert np.all(np.isfinite(bp.h_MAP_parameters.values))

    def test_mask_can_be_switched_off(self):
        data = RNG.normal(size=(T, N)).astype('float32')
        data[:, 3] = 2.0
        m, bp = make_bp(data=data)
        bp.fit_MAP_hier(init_pars=init_pars(m), h_init_pars=h_init(bp), num_steps=1, mask_constant=False)
        assert bp.n_vx_to_fit == N and len(bp.masked_voxels) == 0

    def test_per_vertex_fixed_pars_follow_mask(self):
        m, bp = make_bp(data=self._data())
        idx = np.arange(10)
        amp = np.arange(10, dtype='float32') + 1
        bp.fit_MAP_hier(init_pars=init_pars(m), h_init_pars=h_init(bp), num_steps=2, idx=idx,
                        fixed_pars={'amplitude': amp})
        kept = np.setdiff1d(idx, [3, 7])
        np.testing.assert_allclose(bp.MAP_parameters['amplitude'].values[kept], amp[kept])

    def test_gp_only_masks_nonfinite_pid_pars(self):
        m, bp = make_bp(data=self._data())
        pars = init_pars(m)
        pars['baseline'] = RNG.normal(size=N).astype('float32')
        pars.loc[11, 'baseline'] = np.nan
        bp.fit_MAP_hier_GP_only(pid='baseline', pid_pars=pars, h_init_pars=h_init(bp), num_steps=5)
        np.testing.assert_array_equal(bp.masked_voxels, [3, 7, 11])
        assert np.all(np.isfinite(bp.h_MAP_parameters.values))

    def test_all_masked_raises(self):
        m, bp = make_bp(data=np.ones((T, N), dtype='float32'))
        with pytest.raises(ValueError):
            bp.fit_MAP_hier(init_pars=init_pars(m), h_init_pars=h_init(bp), num_steps=1)


# ----------------------------------------------------------------------------- shared lengthscale
class TestSharedLengthscale:

    def test_labels_are_tied(self):
        m, bp = make_bp(pids=('baseline', 'amplitude'), shared_lengthscale=True)
        assert 'shared_gpkk_l' in bp.h_labels
        assert 'baseline_gpkk_l' not in bp.h_labels and 'amplitude_gpkk_l' not in bp.h_labels
        assert 'baseline_gpkk_v' in bp.h_labels and 'amplitude_gpkk_v' in bp.h_labels
        assert bp.h_prior_type['shared_gpkk_l'] == 'LogNormal'
        bp._update_gps_for_idx(np.arange(N))
        h = tf.constant(np.arange(len(bp.h_labels), dtype='float32')[None])
        lb, la = bp._gp_kwargs('baseline', h)['gpkk_l'], bp._gp_kwargs('amplitude', h)['gpkk_l']
        assert lb.numpy()[0] == la.numpy()[0] == bp.h_labels['shared_gpkk_l']

    def test_default_is_untied_and_per_param_override(self):
        m, bp = make_bp(pids=('baseline', 'amplitude'))
        assert 'baseline_gpkk_l' in bp.h_labels and 'shared_gpkk_l' not in bp.h_labels
        bp.h_add_param('hrf_delay', 'gp', gp=make_gp(), shared_lengthscale=True)
        assert 'shared_gpkk_l' in bp.h_labels

    def test_fit_with_shared_lengthscale(self):
        m, bp = make_bp(pids=('baseline', 'amplitude'), shared_lengthscale=True)
        bp.fit_MAP_hier(init_pars=init_pars(m), h_init_pars=h_init(bp), num_steps=3)
        bp.fit_MAP_hier_GP_only(pid=['baseline', 'amplitude'], pid_pars=init_pars(m) + RNG.normal(size=(N, 1)),
                                h_init_pars=h_init(bp), num_steps=3)
        assert 'shared_gpkk_l' in bp.h_MAP_parameters and np.all(np.isfinite(bp.h_MAP_parameters.values))

    def test_conflicting_fixed_lengthscales_raise(self):
        m = make_model()
        bp = BPRF_hier(m, RNG.normal(size=(T, N)).astype('float32'), shared_lengthscale=True)
        for p, l in [('baseline', 1.0), ('amplitude', 2.0)]:
            gp = make_gp()
            gp.set_log_prob_fixed(gpk_nugget=0.1, mfunc_mean=0.0, gpkk_l=l, gpkk_v=1.0)
            bp.h_add_param(p, 'gp', gp=gp)
        with pytest.raises(ValueError):
            bp.fit_MAP_hier(init_pars=init_pars(m), h_init_pars=h_init(bp), num_steps=1)


# ----------------------------------------------------------------------------- MDS default
class TestMDSDefault:

    def test_defaults(self):
        gp = GP(n_vx=N)
        assert gp.psd_control == 'euclidean' and gp.embedding_dim is None

    def test_geodesic_distances_give_psd_kernel(self):
        # arc lengths on a circle are not Euclidean distances; an RBF kernel on them is not PSD in general
        n = 60
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        d = np.abs(theta[:, None] - theta[None])
        d = np.minimum(d, 2 * np.pi - d)
        raw = GP(n_vx=n, psd_control='none')
        raw.add_xid_stationary_kernel('k', dXs=d)
        emb = GP(n_vx=n)
        emb.add_xid_stationary_kernel('k', dXs=d)
        hp = dict(gpk_nugget=0.0, mfunc_mean=0.0, gpkk_l=1.5, gpkk_v=1.0)
        K = lambda g: g._kernel(g.dXs['k'], tf.constant(1.5, tf.float64), tf.constant(1.0, tf.float64), 'RBF').numpy()
        assert np.linalg.eigvalsh(K(raw)).min() < -1e-6
        assert np.linalg.eigvalsh(K(emb)).min() > -1e-8
        assert np.isfinite(emb.return_log_prob(tf.zeros(n), **{**hp, 'gpk_nugget': 1e-3}).numpy())

    def test_euclidean_input_unchanged_and_coordinates_kept(self):
        assert np.allclose(make_gp().dXs['k'].numpy(), DX)
        gp = GP(n_vx=N)
        gp.add_xid_stationary_kernel('k', Xs=XY)
        assert 'k' in gp.Xs                         # no embedding for coordinates

    def test_all_positive_eigenvalues_scale_invariant(self):
        X3 = RNG.normal(size=(30, 3))
        D = np.sqrt(((X3[:, None] - X3[None]) ** 2).sum(-1))
        for scale in [1e-3, 1.0, 1e3]:
            assert mds_embedding(tf.constant(D * scale), embedding_dim=None).shape[1] == 3


# ----------------------------------------------------------------------------- empirical Bayes wrapper
class TestEmpiricalBayes:

    def _simulated(self, shared=False):
        m = make_model()
        truth = init_pars(m)
        truth['baseline'] = np.sin(XY[:, 0]) + np.cos(XY[:, 1])          # smooth over space
        truth['amplitude'] = 2.0 + 0.5 * np.sin(XY[:, 1])
        pred = m.predict(m.get_paradigm(m.paradigm), truth[m.parameter_labels], None).values
        noise_sd = 0.3 * (pred - pred.mean(0)).std()                       # SNR set by the HRF-scaled signal
        data = (pred + RNG.normal(scale=noise_sd, size=pred.shape)).astype('float32')
        data[:, 4] = 0.0                                                   # a constant voxel
        bp = BPRF_hier(m, data, noise_method='fit_normal', shared_lengthscale=shared)
        bp.h_add_param('baseline', 'gp', gp=make_gp())
        bp.h_add_param('amplitude', 'gp', gp=make_gp())
        init = truth[m.parameter_labels].copy()
        init['baseline'] = 0.0
        init['amplitude'] = 1.0
        return m, bp, truth, init

    def test_default_h_init_pars(self):
        m, bp, truth, _ = self._simulated()
        h = bp.h_default_init_pars(truth)
        assert list(h.columns) == list(bp.h_labels) and np.all(np.isfinite(h.values))
        assert np.isclose(h['baseline_mfunc_mean'][0], truth['baseline'].drop(4).mean(), rtol=1e-4)
        assert np.isclose(h['baseline_gpkk_l'][0], bp.h_gp_function['baseline'].median_dX['k'] / 2)

    def test_three_stage_recipe(self):
        m, bp, truth, init = self._simulated(shared=True)
        # amplitude is only identifiable with the HRF / sf1 gain held fixed
        fixed = {'hrf_delay': 6.0, 'hrf_dispersion': 1.0, 'amplitude_sf1': 1.0}
        out = bp.fit_MAP_empirical_bayes(
            init, fixed_pars=fixed, classical_kwargs=dict(max_n_iterations=300, min_n_iterations=50, learning_rate=0.05,
                                        progressbar=False),
            gp_only_kwargs=dict(num_steps=300, adam_kwargs=dict(learning_rate=0.05)),
            map_kwargs=dict(num_steps=200, adam_kwargs=dict(learning_rate=0.01)))
        assert set(bp.EB_stages) == {'classical', 'hyperparameters', 'MAP', 'h_MAP'}
        assert 'noise_scale' in out and np.all(out['noise_scale'] > 0)
        assert np.all(np.isfinite(out.values))
        np.testing.assert_array_equal(bp.masked_voxels, [4])
        for k, v in fixed.items():
            np.testing.assert_allclose(out[k], v)
        ok = np.arange(N) != 4
        for p in ['baseline', 'amplitude']:
            assert np.corrcoef(out[p][ok], truth[p][ok])[0, 1] > 0.8
        # hyperparameters held at their stage-2 values in stage 3; the user's GPs are not frozen
        h2, h3 = bp.EB_stages['hyperparameters'], bp.EB_stages['h_MAP']
        np.testing.assert_allclose(h3.values, h2.values, rtol=1e-5)
        assert not any(gp.is_fixed for gp in bp.h_gp_function.values())

    def test_joint_hyperparameters_and_no_classical_fit(self):
        m, bp, truth, init = self._simulated()
        init['noise_scale'] = 0.5
        bp.fit_MAP_empirical_bayes(init, classical_fit=False, fix_hyperparameters=False,
                                   gp_only_kwargs=dict(num_steps=20), map_kwargs=dict(num_steps=20))
        np.testing.assert_allclose(bp.EB_stages['classical']['noise_scale'], 0.5)   # given, so kept
        assert np.all(np.isfinite(bp.MAP_parameters.values))
