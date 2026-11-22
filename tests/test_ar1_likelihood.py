"""fit_ar1 likelihoods (BPRF and BPRF_hier) equal the exact stationary AR(1) Gaussian log-density."""
import types

import numpy as np
import pytest
import tensorflow as tf
from scipy.stats import multivariate_normal

from braincoder.bprf import BPRF
from braincoder.bprf_hierarchical import BPRF_hier


def _exact(r, s, phi):
    T = len(r)
    cov = s ** 2 / (1 - phi ** 2) * phi ** np.abs(np.subtract.outer(np.arange(T), np.arange(T)))
    return multivariate_normal(np.zeros(T), cov).logpdf(r)


@pytest.mark.parametrize("cls", [BPRF, BPRF_hier])
def test_ar1_matches_exact(cls):
    rng = np.random.default_rng(0)
    T, scales, phis = 120, np.array([0.5, 1.0, 2.0, 0.7]), np.array([0.0, 0.3, 0.9, -0.5])
    r = np.empty((T, len(phis)))
    r[0] = rng.normal(0, scales / np.sqrt(1 - phis ** 2))
    for t in range(1, T):
        r[t] = phis * r[t - 1] + rng.normal(0, scales)
    fake = types.SimpleNamespace(noise_method="fit_ar1", model_labels={"noise_scale": 0, "noise_ar1": 1})
    fn = cls._create_residual_ln_likelihood_fn(fake)
    out = fn(tf.constant(np.stack([scales, phis], 1), tf.float32), tf.constant(r, tf.float32)).numpy()
    exact = np.array([_exact(r[:, i], scales[i], phis[i]) for i in range(len(phis))])
    if cls is BPRF_hier:          # returns the total over voxels
        np.testing.assert_allclose(out, exact.sum(), rtol=1e-5)
    else:                         # returns one value per voxel
        np.testing.assert_allclose(out, exact, rtol=1e-5)
