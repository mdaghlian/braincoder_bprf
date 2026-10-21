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


# ---------------------------------------------------------------------------
# CRF / CRFWithHRF
# ---------------------------------------------------------------------------

def _crf_setup(crf_type='naka_rushton', with_blank=True):
    sf = np.tile(np.repeat([1., 4.], 4), 3).astype('float32')
    con = np.tile([.1, .3, .6, 1.], 6).astype('float32')
    if with_blank:
        sf[[0, 12]] = np.nan
        con[[0, 12]] = np.nan
    return sf, con


def _crf_params(model):
    """Absolute parameters: SF0 (bare names) and SF1 (_sf1) differ."""
    sf0 = {'amplitude': 2., 'baseline': 1., 'c50': .3, 'exponent': 2.}
    sf1 = {'amplitude': 3., 'baseline': 1.2, 'c50': .4, 'exponent': 1.5}
    vals = []
    for lab in model.parameter_labels:
        root = lab.split('_sf')[0]
        vals.append(sf1[root] if lab.endswith('_sf1') else sf0[root])
    return pd.DataFrame([vals], columns=model.parameter_labels).astype('float32')


class TestCRF:

    @pytest.mark.parametrize('crf_type', ['naka_rushton', 'sqrt', 'power', 'linear'])
    def test_labels_and_shape(self, crf_type):
        from braincoder.models import CRF
        sf, con = _crf_setup(crf_type)
        model = CRF(SF_seq=sf, CON_seq=con, crf_type=crf_type)
        assert model.n_sf == 2
        assert len(model.parameter_labels) == 2 * model.n_crf_params
        assert 'amplitude' in model.parameter_labels and 'baseline' in model.parameter_labels
        pred = model.predict(paradigm=model.paradigm, parameters=_crf_params(model))
        assert pred.shape == (len(sf), 1)
        assert np.isfinite(pred.values).all()

    def test_blank_timepoints_predict_baseline(self):
        from braincoder.models import CRF
        sf, con = _crf_setup()
        model = CRF(SF_seq=sf, CON_seq=con)
        params = _crf_params(model)
        pred = model.predict(paradigm=model.paradigm, parameters=params)
        blank = np.isnan(sf)
        np.testing.assert_allclose(pred.values[blank, 0], params['baseline'].iloc[0], rtol=1e-5)

    def test_matches_independent_naka_rushton(self):
        from braincoder.models import CRF
        sf, con = _crf_setup(with_blank=False)
        model = CRF(SF_seq=sf, CON_seq=con)
        p = _crf_params(model).iloc[0]
        pred = model.predict(paradigm=model.paradigm, parameters=_crf_params(model)).values[:, 0]
        m0 = sf == 1.
        c = con[m0]
        exp0 = p['amplitude'] * c**2 / (c**2 + .3**2) + p['baseline']
        np.testing.assert_allclose(pred[m0], exp0, rtol=1e-4)
        m1 = sf == 4.
        c = con[m1]
        exp1 = p['amplitude_sf1'] * c**1.5 / (c**1.5 + .4**1.5) + p['baseline_sf1']
        np.testing.assert_allclose(pred[m1], exp1, rtol=1e-4)

    def test_sf_parameters_are_independent(self):
        """Changing one SF's amplitude/baseline must not move the other SF."""
        from braincoder.models import CRF
        sf, con = _crf_setup(with_blank=False)
        model = CRF(SF_seq=sf, CON_seq=con)
        p = _crf_params(model)
        p2 = p.copy()
        p2['amplitude'] = 5.
        p2['baseline'] = 0.
        a = model.predict(paradigm=model.paradigm, parameters=p).values[:, 0]
        b = model.predict(paradigm=model.paradigm, parameters=p2).values[:, 0]
        np.testing.assert_allclose(a[sf == 4.], b[sf == 4.], rtol=1e-5)
        assert not np.allclose(a[sf == 1.], b[sf == 1.])

    def test_warns_on_unmatched_sf(self, caplog):
        import logging
        from braincoder.models import CRF
        sf, con = _crf_setup()
        model = CRF(SF_seq=sf, CON_seq=con)
        bad = model.paradigm.copy()
        bad.iloc[3, 0] = 2.5
        with caplog.at_level(logging.WARNING):
            model.predict(paradigm=bad, parameters=_crf_params(model))
        assert 'not among the fitted levels' in caplog.text

    def test_invalid_crf_type(self):
        from braincoder.models import CRF
        with pytest.raises(ValueError):
            CRF(SF_seq=np.array([1.]), CON_seq=np.array([.5]), crf_type='nope')

    def test_gradients_finite(self):
        from braincoder.models import CRF
        sf, con = _crf_setup(with_blank=False)
        model = CRF(SF_seq=sf, CON_seq=con)
        pt = tf.Variable(_crf_params(model).values[None])
        paradigm = model.paradigm.values[None].astype('float32')
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(model._basis_predictions(paradigm, pt))
        assert np.isfinite(tape.gradient(loss, pt).numpy()).all()


class TestCRFWithHRF:

    def _model(self, **kw):
        from braincoder.models import CRFWithHRF
        from braincoder.hrf import SPMHRFModel
        sf, con = _crf_setup()
        return CRFWithHRF(SF_seq=sf, CON_seq=con, hrf_model=SPMHRFModel(tr=1.0), **kw), sf

    def test_single_baseline_labels(self):
        model, _ = self._model()
        assert 'baseline' in model.parameter_labels
        assert 'baseline_sf1' not in model.parameter_labels
        assert 'amplitude_sf1' in model.parameter_labels

    def test_predict_shape_and_finite(self):
        model, sf = self._model()
        params = _crf_params(model)
        pred = model.predict(paradigm=model.paradigm, parameters=params)
        assert pred.shape == (len(sf), 1)
        assert np.isfinite(pred.values).all()

    def test_baseline_is_additive_and_amplitudes_are_pre_convolution(self):
        model, sf = self._model()
        p = _crf_params(model)
        base = model.predict(paradigm=model.paradigm, parameters=p).values[:, 0]

        p_b = p.copy()
        p_b['baseline'] += 2.
        shifted = model.predict(paradigm=model.paradigm, parameters=p_b).values[:, 0]
        np.testing.assert_allclose(shifted - base, 2., rtol=1e-4, atol=1e-4)

        # Tiny amplitudes -> response is just the baseline everywhere
        p_0 = p.copy()
        for lab in ('amplitude', 'amplitude_sf1'):
            p_0[lab] = 1e-6
        flat = model.predict(paradigm=model.paradigm, parameters=p_0).values[:, 0]
        np.testing.assert_allclose(flat, p['baseline'].iloc[0], atol=1e-3)

    def test_fit_recovers_simulated_response(self):
        from braincoder.models import CRF
        from braincoder.optimize import ParameterFitter
        rng = np.random.default_rng(1)
        sf, con = _crf_setup(crf_type='linear', with_blank=False)
        model = CRF(SF_seq=sf, CON_seq=con, crf_type='linear')
        true = pd.DataFrame([[2., 1., 3., 1.2]], columns=model.parameter_labels).astype('float32')
        data = model.predict(paradigm=model.paradigm, parameters=true)
        data = data + rng.normal(0, .01, data.shape).astype('float32')
        fitter = ParameterFitter(model, data, model.paradigm)
        init = pd.DataFrame([[1., 0., 1., 0.]], columns=model.parameter_labels).astype('float32')
        fitted = fitter.fit(init_pars=init, max_n_iterations=500, progressbar=False)
        pred = model.predict(paradigm=model.paradigm, parameters=fitted)
        assert np.corrcoef(pred.values[:, 0], data.values[:, 0])[0, 1] > 0.99


class TestOLSAmplitudes:

    def _simulate(self, model):
        true = _crf_params(model)
        data = model.predict(paradigm=model.paradigm, parameters=true)
        # Start from the true shape but wrong amplitudes/baselines
        init = true.copy()
        for lab in model.parameter_labels:
            if lab.startswith(('amplitude', 'baseline')):
                init[lab] = 1.
        return true, data, init

    def test_recovers_crf_with_per_sf_baselines(self):
        from braincoder.models import CRF
        from braincoder.optimize import ParameterFitter
        sf, con = _crf_setup()   # includes blank (NaN) timepoints
        model = CRF(SF_seq=sf, CON_seq=con)
        true, data, init = self._simulate(model)
        fitted = ParameterFitter(model, data, model.paradigm).ols_amplitudes(init)
        np.testing.assert_allclose(fitted.values, true.values, rtol=1e-3, atol=1e-3)

    def test_recovers_crf_with_hrf(self):
        from braincoder.models import CRFWithHRF
        from braincoder.hrf import SPMHRFModel
        from braincoder.optimize import ParameterFitter
        sf, con = _crf_setup()
        model = CRFWithHRF(SF_seq=sf, CON_seq=con, hrf_model=SPMHRFModel(tr=1.0))
        true, data, init = self._simulate(model)
        fitted = ParameterFitter(model, data, model.paradigm).ols_amplitudes(init)
        np.testing.assert_allclose(fitted.values, true.values, rtol=1e-3, atol=1e-3)

    def test_negative_amplitudes_clipped_and_betas_returned(self):
        from braincoder.models import CRF
        from braincoder.optimize import ParameterFitter
        sf, con = _crf_setup(with_blank=False)
        model = CRF(SF_seq=sf, CON_seq=con, crf_type='linear')
        true = pd.DataFrame([[2., 1., 3., 1.2]], columns=model.parameter_labels).astype('float32')
        data = 2. - model.predict(paradigm=model.paradigm, parameters=true)  # inverted response
        fitted, betas = ParameterFitter(model, data, model.paradigm).ols_amplitudes(
            true, return_betas=True, min_amplitude=1e-4)
        assert (betas[['amplitude', 'amplitude_sf1']].values < 0).all()
        np.testing.assert_allclose(fitted[['amplitude', 'amplitude_sf1']].values, 1e-4)

    def test_rejects_non_crf_model(self, csf_model, csf_parameters):
        from braincoder.optimize import ParameterFitter
        data = csf_model.predict(paradigm=csf_model.paradigm, parameters=csf_parameters)
        with pytest.raises(NotImplementedError):
            ParameterFitter(csf_model, data, csf_model.paradigm).ols_amplitudes(csf_parameters)


class TestParameterFitterNonFinite:

    def test_nonfinite_voxel_is_reported_and_isolated(self, caplog):
        import logging
        from braincoder.models import CRF
        from braincoder.optimize import ParameterFitter
        rng = np.random.default_rng(2)
        sf, con = _crf_setup(with_blank=False)
        model = CRF(SF_seq=sf, CON_seq=con, crf_type='linear')
        true = pd.DataFrame([[2., 1., 3., 1.2]] * 2, columns=model.parameter_labels).astype('float32')
        data = model.predict(paradigm=model.paradigm, parameters=true)
        data = data + rng.normal(0, .01, data.shape).astype('float32')
        init = pd.DataFrame([[1., 0., 1., 0.]] * 2, columns=model.parameter_labels).astype('float32')
        init.iloc[0, 0] = np.float32(3e38)  # overflows -> inf predictions for voxel 0 only

        fitter = ParameterFitter(model, data, model.paradigm)
        with caplog.at_level(logging.WARNING):
            fit = fitter.fit(init_pars=init, max_n_iterations=300, progressbar=False)

        assert np.isfinite(fit.values).all()
        assert 0 in list(fitter.nonfinite_voxels)
        assert 1 not in list(fitter.nonfinite_voxels)
        assert 'Non-finite' in caplog.text
        assert fitter.r2.loc[1] > 0.95  # the healthy voxel still fits


# ---------------------------------------------------------------------------
# CRFSharedExponent / CRFSharedExponentWithHRF
# ---------------------------------------------------------------------------

class TestCRFSharedExponent:

    def test_labels(self):
        from braincoder.models import CRFSharedExponent
        sf, con = _crf_setup()
        model = CRFSharedExponent(SF_seq=sf, CON_seq=con)
        assert model.crf_type == 'power'
        assert model.parameter_labels == [
            'amplitude', 'exponent', 'baseline', 'amplitude_sf1', 'baseline_sf1']

    def test_matches_power_law_with_shared_exponent(self):
        from braincoder.models import CRFSharedExponent
        sf, con = _crf_setup()
        model = CRFSharedExponent(SF_seq=sf, CON_seq=con)
        params = _crf_params(model)
        p = params.iloc[0]
        pred = model.predict(paradigm=model.paradigm, parameters=params).values[:, 0]
        for mask, amp, base in [(sf == 1., p['amplitude'], p['baseline']),
                                (sf == 4., p['amplitude_sf1'], p['baseline_sf1'])]:
            np.testing.assert_allclose(pred[mask], amp * con[mask]**p['exponent'] + base,
                                       rtol=1e-4)
        np.testing.assert_allclose(pred[np.isnan(sf)], p['baseline'], rtol=1e-5)

    def test_exponent_moves_all_sfs(self):
        from braincoder.models import CRFSharedExponent
        sf, con = _crf_setup(with_blank=False)
        model = CRFSharedExponent(SF_seq=sf, CON_seq=con)
        p = _crf_params(model)
        p2 = p.copy()
        p2['exponent'] = 1.
        a = model.predict(paradigm=model.paradigm, parameters=p).values[:, 0]
        b = model.predict(paradigm=model.paradigm, parameters=p2).values[:, 0]
        for level in (1., 4.):
            m = (sf == level) & (con < 1.)
            assert not np.allclose(a[m], b[m])

    def test_with_hrf(self):
        from braincoder.models import CRFSharedExponentWithHRF
        from braincoder.hrf import SPMHRFModel
        sf, con = _crf_setup()
        model = CRFSharedExponentWithHRF(SF_seq=sf, CON_seq=con,
                                         hrf_model=SPMHRFModel(tr=1.0))
        assert model.parameter_labels == [
            'amplitude', 'exponent', 'baseline', 'amplitude_sf1']
        p = _crf_params(model)
        base = model.predict(paradigm=model.paradigm, parameters=p).values[:, 0]
        assert np.isfinite(base).all()
        p['baseline'] += 2.
        shifted = model.predict(paradigm=model.paradigm, parameters=p).values[:, 0]
        np.testing.assert_allclose(shifted - base, 2., rtol=1e-4, atol=1e-4)
