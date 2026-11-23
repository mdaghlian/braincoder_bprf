"""Tests for braincoder HRF models and convolution utilities."""
import numpy as np
import pytest


TR = 1.5  # seconds, typical fMRI repetition time


# ---------------------------------------------------------------------------
# SPMHRFModel
# ---------------------------------------------------------------------------

class TestSPMHRFModel:

    @pytest.fixture
    def hrf_model(self):
        from braincoder.hrf import SPMHRFModel
        return SPMHRFModel(tr=TR)

    def test_get_hrf_returns_tensor(self, hrf_model):
        import tensorflow as tf
        hrf = hrf_model.get_hrf()
        assert isinstance(hrf, tf.Tensor)

    def test_hrf_length(self, hrf_model):
        """One sample per TR, from 0 up to time_length."""
        hrf = hrf_model.get_hrf().numpy()
        expected_len = len(np.arange(0., hrf_model.time_length, TR))
        assert hrf.shape[0] == expected_len

    @pytest.mark.parametrize('tr', [0.9, 1.0, 1.2, 1.5, 2.0])
    def test_matches_spm_canonical_hrf(self, tr):
        """Default parameters give SPM's double gamma: shapes 6 and 16, ratio 1/6, sampled every TR."""
        from braincoder.hrf import SPMHRFModel
        from scipy.stats import gamma
        model = SPMHRFModel(tr=tr)
        assert np.allclose(np.diff(model.time_stamps), tr, atol=1e-5)
        t = model.time_stamps.astype(np.float64)
        expected = gamma.pdf(t, 6.) - gamma.pdf(t, 16.) / 6.
        hrf = np.asarray(model.get_hrf()).ravel()
        assert np.corrcoef(hrf, expected)[0, 1] > 0.9999

    def test_default_hrf_peaks_near_5s(self):
        from braincoder.hrf import SPMHRFModel
        model = SPMHRFModel(tr=1.)
        hrf = np.asarray(model.get_hrf()).ravel()
        assert model.time_stamps[np.argmax(hrf)] == 5.

    def test_hrf_is_finite(self, hrf_model):
        hrf = hrf_model.get_hrf().numpy()
        assert np.all(np.isfinite(hrf))

    def test_hrf_peaks_in_plausible_range(self, hrf_model):
        """Peak of SPM HRF should be between 4–8 s after onset."""
        hrf = hrf_model.get_hrf().numpy()
        peak_time = hrf_model.time_stamps[np.argmax(hrf[:, 0])]
        assert 4.0 <= peak_time <= 9.0, f"HRF peak at {peak_time:.1f}s, expected 4–9s"

    def test_custom_delay_shifts_peak(self, hrf_model):
        """Larger delay parameter should shift the HRF peak later."""
        hrf_early = hrf_model.get_hrf(hrf_delay=4.).numpy()
        hrf_late = hrf_model.get_hrf(hrf_delay=7.).numpy()
        peak_early = hrf_model.time_stamps[np.argmax(hrf_early[:, 0])]
        peak_late = hrf_model.time_stamps[np.argmax(hrf_late[:, 0])]
        assert peak_late > peak_early, "Larger delay should produce later HRF peak"

    def test_parameter_labels(self, hrf_model):
        assert hrf_model.parameter_labels == ['hrf_delay', 'hrf_dispersion']



# ---------------------------------------------------------------------------
# spm_hrf function
# ---------------------------------------------------------------------------

class TestSpmHrfFunction:

    def test_output_shape(self):
        from braincoder.hrf import spm_hrf
        t = np.linspace(0.1, 30.0, 30, dtype=np.float32)[:, np.newaxis]
        hrf = spm_hrf(t).numpy()
        assert hrf.shape[0] == 30  # T timepoints preserved

    def test_output_finite(self):
        from braincoder.hrf import spm_hrf
        t = np.linspace(0.1, 30.0, 30, dtype=np.float32)[:, np.newaxis]
        hrf = spm_hrf(t).numpy()
        assert np.all(np.isfinite(hrf))

    def test_normalized(self):
        """spm_hrf output should sum to 1 (normalized)."""
        from braincoder.hrf import spm_hrf
        t = np.linspace(0.1, 30.0, 300, dtype=np.float32)[:, np.newaxis]
        hrf = spm_hrf(t).numpy()
        assert abs(hrf.sum() - 1.0) < 0.05


# ---------------------------------------------------------------------------
# GaussianPRFWithHRF uses SPMHRFModel end-to-end
# ---------------------------------------------------------------------------

class TestHRFConvolution:

    def test_hrf_convolution_changes_timecourse(self):
        """HRF-convolved predictions should differ from plain predictions."""
        import pandas as pd
        from braincoder.models import GaussianPRF, GaussianPRFWithHRF
        from braincoder.hrf import SPMHRFModel

        n_t = 60
        paradigm = np.linspace(-3, 3, n_t, dtype=np.float32)[:, np.newaxis]
        parameters = pd.DataFrame({
            'mu': [0.0], 'sd': [1.0], 'amplitude': [1.0], 'baseline': [0.0]
        }, dtype=np.float32)

        plain = GaussianPRF(paradigm=paradigm, parameters=parameters)
        hrf_model = SPMHRFModel(tr=TR)
        with_hrf = GaussianPRFWithHRF(
            paradigm=paradigm, parameters=parameters, hrf_model=hrf_model
        )

        pred_plain = plain.predict().values
        pred_hrf = with_hrf.predict().values
        assert not np.allclose(pred_plain, pred_hrf), \
            "HRF-convolved prediction should differ from plain prediction"

    def test_hrf_convolution_output_is_finite(self):
        import pandas as pd
        from braincoder.models import GaussianPRFWithHRF
        from braincoder.hrf import SPMHRFModel

        n_t = 60
        paradigm = np.linspace(-3, 3, n_t, dtype=np.float32)[:, np.newaxis]
        parameters = pd.DataFrame({
            'mu': [0.0], 'sd': [1.0], 'amplitude': [1.0], 'baseline': [0.0]
        }, dtype=np.float32)

        hrf_model = SPMHRFModel(tr=TR)
        model = GaussianPRFWithHRF(
            paradigm=paradigm, parameters=parameters, hrf_model=hrf_model
        )
        pred = model.predict()
        assert np.all(np.isfinite(pred.values))
