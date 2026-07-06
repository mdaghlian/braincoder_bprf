"""Tests for ContrastSensitivity (CSF) and CompressiveSpatialGaussiansPRF2D (CSS) models."""
import numpy as np
import pandas as pd
import pytest
import tensorflow as tf


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

N_T = 40        # timepoints
N_SF = 8        # unique spatial frequencies
N_CON = 5       # unique contrast levels
N_VX = 4        # voxels


def _sf_con_sequences(rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    SFs = np.tile(np.geomspace(0.5, 16, N_SF), N_CON).astype(np.float32)
    CONs = np.repeat(np.linspace(0.1, 1.0, N_CON), N_SF).astype(np.float32)
    return SFs, CONs


# ---------------------------------------------------------------------------
# CSF fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sf_con():
    return _sf_con_sequences()


@pytest.fixture
def csf_parameters():
    return pd.DataFrame({
        'width_r':   np.array([1.2, 1.0, 1.4, 0.9], dtype=np.float32),
        'SFp':       np.array([2.5, 3.0, 2.0, 4.0], dtype=np.float32),
        'CSp':       np.array([100., 150., 80., 200.], dtype=np.float32),
        'width_l':   np.array([0.7, 0.6, 0.8, 0.5], dtype=np.float32),
        'crf_exp':   np.array([2.0, 2.5, 1.5, 3.0], dtype=np.float32),
        'amplitude': np.array([1.0, 1.2, 0.8, 1.5], dtype=np.float32),
        'baseline':  np.zeros(N_VX, dtype=np.float32),
    })


@pytest.fixture
def csf_model(sf_con, csf_parameters):
    from braincoder.models import ContrastSensitivity
    SFs, CONs = sf_con
    return ContrastSensitivity(
        SF_seq=SFs, CON_seq=CONs,
        parameters=csf_parameters,
    )


# ---------------------------------------------------------------------------
# ContrastSensitivity tests
# ---------------------------------------------------------------------------

class TestContrastSensitivity:

    def test_init_attributes(self, csf_model, sf_con):
        SFs, CONs = sf_con
        assert csf_model.stim_sequence.shape == (len(SFs), 2)
        assert set(csf_model.stim_sequence.columns) == {'SF', 'CON'}
        assert csf_model.n_SF == N_SF
        assert csf_model.n_CON == N_CON
        assert csf_model.parameter_labels == [
            'width_r', 'SFp', 'CSp', 'width_l', 'crf_exp', 'amplitude', 'baseline']

    def test_paradigm_shape(self, csf_model, sf_con):
        SFs, _ = sf_con
        assert csf_model.paradigm.shape == (len(SFs), 2)

    def test_basis_predictions_shape(self, csf_model, csf_parameters):
        # _basis_predictions expects (n_batches, n_timepoints, n_features)
        paradigm = tf.constant(csf_model.stim_sequence.values[np.newaxis, ...])  # (1, N_T, 2)
        params = tf.constant(csf_parameters.values[np.newaxis, ...])  # [1, n_vx, n_params]
        out = csf_model._basis_predictions(paradigm, params)
        n_stim = csf_model.stim_sequence.shape[0]
        assert out.shape == (1, n_stim, N_VX)

    def test_basis_predictions_non_negative(self, csf_model, csf_parameters):
        """CSF responses should be non-negative for positive amplitude and zero baseline."""
        paradigm = tf.constant(csf_model.stim_sequence.values[np.newaxis, ...])  # (1, N_T, 2)
        params = tf.constant(csf_parameters.values[np.newaxis, ...])
        out = csf_model._basis_predictions(paradigm, params).numpy()
        assert np.all(out >= 0)

    def test_get_csf_for_plot_shape(self, csf_model):
        SF_grid = np.geomspace(0.5, 16, 20, dtype=np.float32)
        CON_grid = np.array([0.5, 1.0], dtype=np.float32)
        csf, SFs_out, CONs_out = csf_model.get_csf_for_plot(SF_grid, CON_grid)
        n_pts = len(SF_grid) * len(CON_grid)
        assert csf.shape == (N_VX, n_pts)
        assert len(SFs_out) == n_pts
        assert len(CONs_out) == n_pts

    def test_transform_parameters_roundtrip(self, csf_model, csf_parameters):
        """Forward then backward transform should recover original parameters."""
        params = tf.constant(csf_parameters.values)
        fwd = csf_model._transform_parameters_forward(params)
        bwd = csf_model._transform_parameters_backward(fwd)
        np.testing.assert_allclose(params.numpy(), bwd.numpy(), rtol=1e-4, atol=1e-5)

    def test_bijectors_positive_constrained(self, csf_model, csf_parameters):
        """All softplus-transformed parameters should be positive."""
        params = tf.constant(csf_parameters.values)
        fwd = csf_model._transform_parameters_forward(params)
        fwd_np = fwd.numpy()
        # width_r, SFp, CSp, width_l, crf_exp, amplitude are all softplus-constrained
        for i in range(6):
            assert np.all(fwd_np[:, i] > 0), f"Parameter {i} not positive after transform"

    def test_csf_peak_at_sfp(self, sf_con, csf_parameters):
        """CSF should peak near SFp for each voxel."""
        from braincoder.models import ContrastSensitivity
        SFs, CONs = sf_con
        model = ContrastSensitivity(SF_seq=SFs, CON_seq=CONs, parameters=csf_parameters)
        SF_grid = np.geomspace(0.1, 20, 200, dtype=np.float32)
        CON_fixed = np.ones_like(SF_grid)
        csf, SF_out, _ = model.get_csf_for_plot(SF_grid, CON_fixed)
        peak_sfs = SF_out[np.argmax(csf, axis=1)]
        sfp_values = csf_parameters['SFp'].values
        # Peak should be within a factor of 3 of SFp
        np.testing.assert_array_less(np.abs(np.log(peak_sfs) - np.log(sfp_values)), np.log(3))

    def test_chung_legge_default_importable(self):
        from braincoder.models import Chung_Legge_default
        assert list(Chung_Legge_default.columns) == [
            'width_r', 'SFp', 'CSp', 'width_l', 'amplitude', 'baseline', 'crf_exp']


class TestContrastSensitivityWithHRF:

    def test_init(self, sf_con, csf_parameters):
        from braincoder.models import ContrastSensitivityWithHRF
        from braincoder.hrf import SPMHRFModel
        SFs, CONs = sf_con
        hrf = SPMHRFModel(tr=1.0)
        model = ContrastSensitivityWithHRF(
            SF_seq=SFs, CON_seq=CONs,
            parameters=csf_parameters,
            hrf_model=hrf,
        )
        assert hasattr(model, 'hrf_model')
        assert hasattr(model, 'parameter_labels')

    def test_prediction_shape(self, sf_con, csf_parameters):
        from braincoder.models import ContrastSensitivityWithHRF
        from braincoder.hrf import SPMHRFModel
        SFs, CONs = sf_con
        hrf = SPMHRFModel(tr=1.0)
        model = ContrastSensitivityWithHRF(
            SF_seq=SFs, CON_seq=CONs,
            parameters=csf_parameters,
            hrf_model=hrf,
        )
        paradigm = tf.constant(model.stim_sequence.values[np.newaxis, ...])  # (1, N_T, 2)
        params = tf.constant(csf_parameters.values[np.newaxis, ...])
        out = model._basis_predictions(paradigm, params)
        n_stim = model.stim_sequence.shape[0]
        assert out.shape == (1, n_stim, N_VX)


class TestContrastSensitivityExp:

    def test_init_and_is_subclass(self, sf_con, csf_parameters):
        from braincoder.models import ContrastSensitivityExp, ContrastSensitivityWithHRF
        assert issubclass(ContrastSensitivityExp, ContrastSensitivityWithHRF)


# ---------------------------------------------------------------------------
# CSS fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def image_paradigm_small():
    rng = np.random.default_rng(42)
    imgs = rng.uniform(0, 1, size=(N_T, 16, 16)).astype(np.float32)
    return imgs


@pytest.fixture
def css_parameters():
    return pd.DataFrame({
        'x':         np.array([0.0, 0.2, -0.2,  0.1], dtype=np.float32),
        'y':         np.array([0.0, 0.1,  0.1, -0.1], dtype=np.float32),
        'sd':        np.array([0.3, 0.25, 0.35, 0.3], dtype=np.float32),
        'baseline':  np.zeros(N_VX, dtype=np.float32),
        'amplitude': np.ones(N_VX, dtype=np.float32),
        'exponent':  np.array([0.5, 0.7, 0.3, 1.0], dtype=np.float32),
    })


@pytest.fixture
def gauss2d_parameters():
    return pd.DataFrame({
        'x':         np.array([0.0, 0.2, -0.2,  0.1], dtype=np.float32),
        'y':         np.array([0.0, 0.1,  0.1, -0.1], dtype=np.float32),
        'sd':        np.array([0.3, 0.25, 0.35, 0.3], dtype=np.float32),
        'baseline':  np.zeros(N_VX, dtype=np.float32),
        'amplitude': np.ones(N_VX, dtype=np.float32),
    })


@pytest.fixture
def css_model(image_paradigm_small, css_parameters):
    from braincoder.models import CompressiveSpatialGaussiansPRF2D
    return CompressiveSpatialGaussiansPRF2D(
        paradigm=image_paradigm_small,
        parameters=css_parameters,
    )


@pytest.fixture
def gauss2d_model(image_paradigm_small, gauss2d_parameters):
    from braincoder.models import GaussianPRF2D
    return GaussianPRF2D(
        paradigm=image_paradigm_small,
        parameters=gauss2d_parameters,
    )


# ---------------------------------------------------------------------------
# CompressiveSpatialGaussiansPRF2D (CSS) tests
# ---------------------------------------------------------------------------

class TestCompressiveSpatialGaussiansPRF2D:

    def test_parameter_labels(self, css_model):
        assert 'exponent' in css_model.parameter_labels
        assert css_model.parameter_labels == ['x', 'y', 'sd', 'baseline', 'amplitude', 'exponent']

    def test_init_grid_coordinates(self, css_model):
        gc = css_model.grid_coordinates
        assert gc.shape[1] == 2  # x and y columns

    def test_prediction_shape(self, css_model, css_parameters):
        preds = css_model.predict(parameters=css_parameters)
        assert preds.shape == (N_T, N_VX)

    def test_rf_shape(self, css_model):
        rf = css_model.get_rf(parameters=css_model.parameters)
        n_pixels = css_model.grid_coordinates.shape[0]
        assert rf.shape == (N_VX, n_pixels)

    def test_rf_non_negative(self, css_model):
        rf = css_model.get_rf(parameters=css_model.parameters)
        assert np.all(rf >= 0)

    def test_exponent_one_matches_gaussian(self, image_paradigm_small, css_parameters, gauss2d_parameters):
        """CSS with exponent=1 should produce the same predictions as GaussianPRF2D."""
        from braincoder.models import CompressiveSpatialGaussiansPRF2D, GaussianPRF2D

        css_pars_exp1 = css_parameters.copy()
        css_pars_exp1['exponent'] = 1.0

        css = CompressiveSpatialGaussiansPRF2D(paradigm=image_paradigm_small, parameters=css_pars_exp1)
        g2d = GaussianPRF2D(paradigm=image_paradigm_small, parameters=gauss2d_parameters)

        css_pred = css.predict(parameters=css_pars_exp1).values
        g2d_pred = g2d.predict(parameters=gauss2d_parameters).values

        np.testing.assert_allclose(css_pred, g2d_pred, rtol=1e-4, atol=1e-4)

    def test_simulate_shape(self, css_model, css_parameters):
        sim = css_model.simulate(parameters=css_parameters, noise=0)
        assert sim.shape == (N_T, N_VX)

    def test_exponent_reduces_response(self, image_paradigm_small, css_parameters):
        """Lower exponent should compress (reduce) peak response."""
        from braincoder.models import CompressiveSpatialGaussiansPRF2D

        pars_low = css_parameters.copy()
        pars_low['exponent'] = 0.3
        pars_high = css_parameters.copy()
        pars_high['exponent'] = 1.0

        model = CompressiveSpatialGaussiansPRF2D(paradigm=image_paradigm_small)
        pred_low  = model.predict(parameters=pars_low).values
        pred_high = model.predict(parameters=pars_high).values

        # Compressive exponent (<1) should reduce dynamic range
        assert pred_low.std() <= pred_high.std() + 1e-4

    def test_transformations_roundtrip(self, css_model, css_parameters):
        params = tf.constant(css_parameters.values)
        fwd = css_model._transform_parameters_forward(params)
        bwd = css_model._transform_parameters_backward(fwd)
        np.testing.assert_allclose(params.numpy(), bwd.numpy(), rtol=1e-4, atol=1e-5)
