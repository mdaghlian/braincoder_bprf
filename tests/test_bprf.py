"""Tests for BPRF (Bayesian PRF) and BPRF_hier (hierarchical) fitting classes."""
import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import bijectors as tfb


# ---------------------------------------------------------------------------
# Shared test scaffold: tiny CSF model + synthetic data
# ---------------------------------------------------------------------------

N_T = 30
N_SF = 6
N_CON = 4
N_VX = 3


def _make_csf_model_and_data(rng=None):
    """Return a fitted ContrastSensitivity model + simulated data (DataFrame)."""
    if rng is None:
        rng = np.random.default_rng(0)
    from braincoder.models import ContrastSensitivity

    SFs  = np.tile(np.geomspace(0.5, 16, N_SF), N_CON).astype(np.float32)
    CONs = np.repeat(np.linspace(0.1, 1.0, N_CON), N_SF).astype(np.float32)

    true_pars = pd.DataFrame({
        'width_r':   [1.2, 1.0, 1.4],
        'SFp':       [2.5, 3.0, 2.0],
        'CSp':       [100., 150., 80.],
        'width_l':   [0.7, 0.6, 0.8],
        'crf_exp':   [2.0, 2.5, 1.5],
        'amplitude': [1.0, 1.2, 0.8],
        'baseline':  [0.0, 0.0, 0.0],
    }, dtype=np.float32)

    model = ContrastSensitivity(SF_seq=SFs, CON_seq=CONs, parameters=true_pars)
    predictions = model.predict(parameters=true_pars).values
    noise = rng.normal(0, 0.05, size=predictions.shape).astype(np.float32)
    data = pd.DataFrame(predictions + noise,
                        columns=[f'vx{i}' for i in range(N_VX)])
    return model, data, true_pars


@pytest.fixture(scope='module')
def csf_model_and_data():
    return _make_csf_model_and_data()


@pytest.fixture(scope='module')
def bprf_instance(csf_model_and_data):
    from braincoder.bprf import BPRF
    model, data, _ = csf_model_and_data
    return BPRF(model, data)


def _default_bounds():
    return {
        'width_r':   [0.1, 3.0],
        'SFp':       [0.5, 10.0],
        'CSp':       [10.,  500.],
        'width_l':   [0.1, 3.0],
        'crf_exp':   [0.5, 5.0],
        'amplitude': [0.1, 5.0],
        'baseline':  [-1., 1.],
        'noise_scale': [0.01, 2.0],
    }


def _default_init_pars(n_voxels=N_VX):
    return pd.DataFrame({
        'width_r':    np.ones(n_voxels, dtype=np.float32) * 1.0,
        'SFp':        np.ones(n_voxels, dtype=np.float32) * 3.0,
        'CSp':        np.ones(n_voxels, dtype=np.float32) * 100.0,
        'width_l':    np.ones(n_voxels, dtype=np.float32) * 0.7,
        'crf_exp':    np.ones(n_voxels, dtype=np.float32) * 2.0,
        'amplitude':  np.ones(n_voxels, dtype=np.float32) * 1.0,
        'baseline':   np.zeros(n_voxels, dtype=np.float32),
        'noise_scale': np.ones(n_voxels, dtype=np.float32) * 0.1,
    })


# ---------------------------------------------------------------------------
# BPRF initialisation tests
# ---------------------------------------------------------------------------

class TestBPRFInit:

    def test_noise_method_fit_normal(self, csf_model_and_data):
        from braincoder.bprf import BPRF
        model, data, _ = csf_model_and_data
        bprf = BPRF(model, data, noise_method='fit_normal')
        assert 'noise_scale' in bprf.model_labels
        assert 'noise_dof' not in bprf.model_labels

    def test_noise_method_fit_tdist(self, csf_model_and_data):
        from braincoder.bprf import BPRF
        model, data, _ = csf_model_and_data
        bprf = BPRF(model, data, noise_method='fit_tdist')
        assert 'noise_scale' in bprf.model_labels
        assert 'noise_dof' in bprf.model_labels

    def test_noise_method_fit_ar1(self, csf_model_and_data):
        from braincoder.bprf import BPRF
        model, data, _ = csf_model_and_data
        bprf = BPRF(model, data, noise_method='fit_ar1')
        assert 'noise_scale' in bprf.model_labels
        assert 'noise_ar1' in bprf.model_labels

    def test_noise_method_none(self, csf_model_and_data):
        from braincoder.bprf import BPRF
        model, data, _ = csf_model_and_data
        bprf = BPRF(model, data, noise_method='none')
        assert 'noise_scale' not in bprf.model_labels

    def test_invalid_noise_method(self, csf_model_and_data):
        from braincoder.bprf import BPRF
        model, data, _ = csf_model_and_data
        with pytest.raises(AssertionError):
            BPRF(model, data, noise_method='invalid')

    def test_n_voxels(self, bprf_instance):
        assert bprf_instance.n_voxels == N_VX

    def test_n_model_params(self, bprf_instance):
        from braincoder.models import ContrastSensitivity
        assert bprf_instance.n_model_params == len(ContrastSensitivity.parameter_labels)

    def test_default_priors_are_none(self, bprf_instance):
        from braincoder.bprf import PriorNone
        for p, prior in bprf_instance.p_prior.items():
            if 'noise' not in p:
                assert isinstance(prior, PriorNone)

    def test_default_bijectors_are_identity(self, csf_model_and_data):
        from braincoder.bprf import BPRF
        model, data, _ = csf_model_and_data
        bprf = BPRF(model, data, noise_method='none')
        for p, bij in bprf.p_bijector.items():
            assert isinstance(bij, tfb.Identity)


# ---------------------------------------------------------------------------
# Prior / bijector setup
# ---------------------------------------------------------------------------

class TestBPRFPriorSetup:

    def test_add_prior_normal(self, csf_model_and_data):
        from braincoder.bprf import BPRF, PriorNorm
        model, data, _ = csf_model_and_data
        bprf = BPRF(model, data, noise_method='none')
        bprf.add_prior('SFp', prior_type='normal', loc=3.0, scale=1.0)
        assert isinstance(bprf.p_prior['SFp'], PriorNorm)
        assert bprf.p_prior_type['SFp'] == 'normal'

    def test_add_prior_uniform(self, csf_model_and_data):
        from braincoder.bprf import BPRF, PriorUniform
        model, data, _ = csf_model_and_data
        bprf = BPRF(model, data, noise_method='none')
        bprf.add_prior('SFp', prior_type='uniform', low=0.5, high=10.0)
        assert isinstance(bprf.p_prior['SFp'], PriorUniform)

    def test_add_prior_fixed(self, csf_model_and_data):
        from braincoder.bprf import BPRF, PriorFixed
        model, data, _ = csf_model_and_data
        bprf = BPRF(model, data, noise_method='none')
        bprf.add_prior('baseline', prior_type='fixed', fixed_val=0.0)
        assert isinstance(bprf.p_prior['baseline'], PriorFixed)

    def test_add_prior_invalid_pid_is_noop(self, csf_model_and_data, capsys):
        from braincoder.bprf import BPRF
        model, data, _ = csf_model_and_data
        bprf = BPRF(model, data, noise_method='none')
        bprf.add_prior('nonexistent_param', prior_type='normal', loc=0.0, scale=1.0)
        out = capsys.readouterr().out
        assert 'error' in out.lower()

    def test_add_priors_from_bounds(self, csf_model_and_data):
        from braincoder.bprf import BPRF, PriorUniform, PriorFixed
        model, data, _ = csf_model_and_data
        bprf = BPRF(model, data, noise_method='none')
        bounds = {'SFp': [0.5, 10.0], 'baseline': [0.0, 0.0]}
        bprf.add_priors_from_bounds(bounds)
        assert isinstance(bprf.p_prior['SFp'], PriorUniform)
        assert isinstance(bprf.p_prior['baseline'], PriorFixed)

    def test_add_bijector_softplus(self, csf_model_and_data):
        from braincoder.bprf import BPRF
        model, data, _ = csf_model_and_data
        bprf = BPRF(model, data, noise_method='none')
        bprf.add_bijector('SFp', bijector_type='softplus')
        assert isinstance(bprf.p_bijector['SFp'], tfb.Softplus)

    def test_add_bijector_sigmoid(self, csf_model_and_data):
        from braincoder.bprf import BPRF
        model, data, _ = csf_model_and_data
        bprf = BPRF(model, data, noise_method='none')
        bprf.add_bijector('SFp', bijector_type='sigmoid', low=0.5, high=10.0)
        assert isinstance(bprf.p_bijector['SFp'], tfb.Sigmoid)

    def test_add_bijector_from_bounds(self, csf_model_and_data):
        from braincoder.bprf import BPRF
        model, data, _ = csf_model_and_data
        bprf = BPRF(model, data, noise_method='none')
        bounds = {'SFp': [0.5, 10.0], 'CSp': [10., 500.]}
        bprf.add_bijector_from_bounds(bounds)
        assert isinstance(bprf.p_bijector['SFp'], tfb.Sigmoid)
        assert isinstance(bprf.p_bijector['CSp'], tfb.Sigmoid)


# ---------------------------------------------------------------------------
# sort_parameters
# ---------------------------------------------------------------------------

class TestBPRFSortParameters:

    def test_sort_reorders_columns(self, bprf_instance):
        init = _default_init_pars()
        shuffled = init[list(reversed(init.columns))]
        sorted_pars = bprf_instance.sort_parameters(shuffled)
        expected_order = list(bprf_instance.model_labels.keys())
        assert list(sorted_pars.columns) == expected_order

    def test_sort_preserves_values(self, bprf_instance):
        init = _default_init_pars()
        sorted_pars = bprf_instance.sort_parameters(init)
        for col in init.columns:
            np.testing.assert_array_equal(init[col].values, sorted_pars[col].values)


# ---------------------------------------------------------------------------
# Parameter transform (forward/backward bijectors)
# ---------------------------------------------------------------------------

class TestBPRFParameterTransform:

    def test_transform_roundtrip(self, csf_model_and_data):
        from braincoder.bprf import BPRF
        model, data, _ = csf_model_and_data
        bprf = BPRF(model, data, noise_method='none')
        bounds = _default_bounds()
        bounds.pop('noise_scale')
        bprf.add_bijector_from_bounds(bounds)
        bprf.add_priors_from_bounds(bounds)
        bprf.idx_to_fit = list(range(N_VX))
        bprf.n_vx_to_fit = N_VX
        bprf.fixed_pars = {}
        bprf.prep_for_fitting()

        init = _default_init_pars()
        init = init.drop(columns=['noise_scale'])
        init = bprf.sort_parameters(init)
        constrained = tf.constant(init.values.astype(np.float32))

        # constrained → unconstrained → constrained should be identity
        unconstrained = bprf._bprf_transform_parameters_backward(constrained)
        recovered = bprf._bprf_transform_parameters_forward(unconstrained)
        np.testing.assert_allclose(constrained.numpy(), recovered.numpy(), rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# fit_MAP smoke test (very few steps, just checks it runs and returns output)
# ---------------------------------------------------------------------------

class TestBPRFFitMAP:

    def test_fit_map_runs_and_returns_dataframe(self, csf_model_and_data):
        from braincoder.bprf import BPRF
        model, data, true_pars = csf_model_and_data
        bprf = BPRF(model, data, noise_method='fit_normal')

        bounds = _default_bounds()
        bprf.add_priors_from_bounds(bounds)
        bprf.add_bijector_from_bounds(bounds)

        init = _default_init_pars()
        bprf.fit_MAP(init_pars=init, num_steps=3)

        assert isinstance(bprf.MAP_parameters, pd.DataFrame)
        assert bprf.MAP_parameters.shape == (N_VX, bprf.n_params)

    def test_fit_map_column_names(self, csf_model_and_data):
        from braincoder.bprf import BPRF
        model, data, _ = csf_model_and_data
        bprf = BPRF(model, data, noise_method='fit_normal')
        bounds = _default_bounds()
        bprf.add_priors_from_bounds(bounds)
        bprf.add_bijector_from_bounds(bounds)
        init = _default_init_pars()
        bprf.fit_MAP(init_pars=init, num_steps=2)

        expected_cols = set(bprf.model_labels.keys())
        assert set(bprf.MAP_parameters.columns) == expected_cols

    def test_fit_map_subset_of_voxels(self, csf_model_and_data):
        from braincoder.bprf import BPRF
        model, data, _ = csf_model_and_data
        bprf = BPRF(model, data, noise_method='fit_normal')
        bounds = _default_bounds()
        bprf.add_priors_from_bounds(bounds)
        bprf.add_bijector_from_bounds(bounds)
        init = _default_init_pars()
        bprf.fit_MAP(init_pars=init, num_steps=2, idx=[0])

        assert bprf.MAP_parameters.shape[0] == N_VX


# ---------------------------------------------------------------------------
# get_mcmc_summary (uses mock MCMC samples)
# ---------------------------------------------------------------------------

class TestBPRFMCMCSummary:

    def _inject_mock_samples(self, bprf):
        rng = np.random.default_rng(1)
        for vx in range(N_VX):
            samples = {p: rng.normal(1.0, 0.1, size=200).tolist()
                       for p in bprf.model_labels}
            bprf.mcmc_sampler[vx] = pd.DataFrame(samples)

    def test_get_mcmc_summary_shape(self, csf_model_and_data):
        from braincoder.bprf import BPRF
        model, data, _ = csf_model_and_data
        bprf = BPRF(model, data, noise_method='fit_normal')
        self._inject_mock_samples(bprf)
        bprf.get_mcmc_summary(burnin=50, pc_range=25)

        assert isinstance(bprf.mcmc_summary, pd.DataFrame)
        assert bprf.mcmc_summary.shape[0] == N_VX
        assert isinstance(bprf.mcmc_mean, pd.DataFrame)
        assert bprf.mcmc_mean.shape == (N_VX, bprf.n_params)

    def test_mcmc_summary_contains_expected_columns(self, csf_model_and_data):
        from braincoder.bprf import BPRF
        model, data, _ = csf_model_and_data
        bprf = BPRF(model, data, noise_method='fit_normal')
        self._inject_mock_samples(bprf)
        bprf.get_mcmc_summary(burnin=10, pc_range=25)

        for p in bprf.model_labels:
            for prefix in ('m_', 'q1_', 'q2_', 'uc_'):
                assert f'{prefix}{p}' in bprf.mcmc_summary.columns


# ---------------------------------------------------------------------------
# BPRF_hier tests
# ---------------------------------------------------------------------------

class TestBPRFHier:

    def test_init_inherits_bprf(self, csf_model_and_data):
        from braincoder.bprf_hierarchical import BPRF_hier
        from braincoder.bprf import BPRF
        model, data, _ = csf_model_and_data
        hier = BPRF_hier(model, data)
        assert isinstance(hier, BPRF)

    def test_h_add_param_normal_creates_loc_scale(self, csf_model_and_data):
        from braincoder.bprf_hierarchical import BPRF_hier
        model, data, _ = csf_model_and_data
        hier = BPRF_hier(model, data)
        hier.h_add_param('SFp', h_prior_to_apply='normal')
        assert 'SFp_loc' in hier.h_labels
        assert 'SFp_scale' in hier.h_labels

    def test_h_add_param_increments_labels(self, csf_model_and_data):
        from braincoder.bprf_hierarchical import BPRF_hier
        model, data, _ = csf_model_and_data
        hier = BPRF_hier(model, data)
        hier.h_add_param('SFp', h_prior_to_apply='normal')
        hier.h_add_param('CSp', h_prior_to_apply='normal')
        # 2 params × 2 (loc + scale) = 4 hierarchical labels
        assert len(hier.h_labels) == 4

    def test_h_add_param_sets_prior_to_none(self, csf_model_and_data):
        from braincoder.bprf_hierarchical import BPRF_hier
        model, data, _ = csf_model_and_data
        hier = BPRF_hier(model, data)
        hier.h_add_param('SFp', h_prior_to_apply='normal')
        # The direct prior for 'SFp' should be disabled
        assert hier.p_prior_type['SFp'] == 'none'

    def test_fit_map_hier_runs(self, csf_model_and_data):
        from braincoder.bprf_hierarchical import BPRF_hier
        model, data, _ = csf_model_and_data
        hier = BPRF_hier(model, data, noise_method='fit_normal')

        bounds = _default_bounds()
        hier.add_priors_from_bounds(bounds)
        hier.add_bijector_from_bounds(bounds)
        hier.h_add_param('SFp', h_prior_to_apply='normal')

        init = _default_init_pars()
        h_init = pd.DataFrame({'SFp_loc': [3.0], 'SFp_scale': [1.0]}, dtype=np.float32)

        hier.fit_MAP_hier(init_pars=init, h_init_pars=h_init, num_steps=3)

        assert isinstance(hier.MAP_parameters, pd.DataFrame)
        assert isinstance(hier.h_MAP_parameters, pd.DataFrame)
        assert 'SFp_loc' in hier.h_MAP_parameters.columns
        assert 'SFp_scale' in hier.h_MAP_parameters.columns
