"""
Spectrum preprocessing, CWT scalogram, and augmentation.

Pipeline for one raw spectrum:
  despike -> resample onto the common grid -> ALS baseline removal -> SNV normalize
Then the CWT scalogram is computed from the preprocessed spectrum.
"""
from dataclasses import asdict, dataclass

import numpy as np
import pywt
from scipy import sparse
from scipy.sparse.linalg import spsolve


@dataclass
class PreprocessConfig:
    shift_min: float = 200.0      # crop to the fingerprint region (cm^-1)
    shift_max: float = 1800.0
    n_points: int = 800           # common grid: 2 cm^-1 steps
    als_lambda: float = 1e5       # baseline smoothness
    als_p: float = 0.01           # baseline asymmetry
    despike_threshold: float = 6.0
    wavelet: str = 'mexh'         # Mexican hat: the standard wavelet for Raman peak detection
    n_scales: int = 48
    max_scale: float = 64.0       # in grid points (64 * 2 cm^-1 = 128 cm^-1 wide features)
    scalogram_pool: int = 4       # average-pool the shift axis 800 -> 200 for the 2D CNN

    def grid(self):
        return np.linspace(self.shift_min, self.shift_max, self.n_points)

    def scales(self):
        return np.geomspace(1.0, self.max_scale, self.n_scales)

    def to_dict(self):
        return asdict(self)


def despike(y, threshold):
    """Remove cosmic rays (Whitaker-Hayes): modified z-score of first differences, replace by local median."""
    d = np.diff(y)
    mad = np.median(np.abs(d - np.median(d))) or 1.0
    z = 0.6745 * (d - np.median(d)) / mad
    spikes = np.zeros(len(y), bool)
    spikes[1:] |= np.abs(z) > threshold
    spikes = np.convolve(spikes, np.ones(3), 'same') > 0  # also clean neighbours
    y = y.copy()
    for i in np.flatnonzero(spikes):
        window = np.arange(max(0, i - 5), min(len(y), i + 6))
        window = window[~spikes[window]]
        if len(window):
            y[i] = np.median(y[window])
    return y


def als_baseline(y, lam, p, n_iter=10):
    """Asymmetric least squares baseline (Eilers & Boelens 2005)."""
    n = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(n, n - 2))
    DDT = lam * D.dot(D.transpose())
    w = np.ones(n)
    for _ in range(n_iter):
        W = sparse.spdiags(w, 0, n, n)
        z = spsolve((W + DDT).tocsc(), w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def preprocess_spectrum(shift, intensity, cfg):
    """Raw export -> baseline-free, normalized spectrum on the common grid (length cfg.n_points)."""
    order = np.argsort(shift)
    shift, intensity = np.asarray(shift, float)[order], np.asarray(intensity, float)[order]
    if shift[0] > cfg.shift_min or shift[-1] < cfg.shift_max:
        raise ValueError(f'Spectrum covers {shift[0]:.0f}-{shift[-1]:.0f} cm^-1, '
                         f'needs {cfg.shift_min:.0f}-{cfg.shift_max:.0f} cm^-1')
    y = despike(intensity, cfg.despike_threshold)
    y = np.interp(cfg.grid(), shift, y)
    y = y - als_baseline(y, cfg.als_lambda, cfg.als_p)
    return ((y - y.mean()) / (y.std() or 1.0)).astype('float32')


def scalogram(spectrum, cfg):
    """CWT of the spectrum along the Raman-shift axis, as a (n_scales, n_points/pool) array.

    Computed as numbers, not a rendered picture, so no information is lost to colormaps or resizing.
    """
    coefs, _ = pywt.cwt(spectrum, cfg.scales(), cfg.wavelet)
    s = np.log1p(np.abs(coefs))
    s = s[:, : s.shape[1] // cfg.scalogram_pool * cfg.scalogram_pool]
    s = s.reshape(s.shape[0], -1, cfg.scalogram_pool).mean(axis=2)
    return ((s - s.mean()) / (s.std() or 1.0)).astype('float32')


def augment(spectrum, rng, max_shift_pts=2, noise=0.05):
    """Small calibration shift, intensity scaling, and noise on a preprocessed spectrum."""
    x = np.arange(len(spectrum))
    y = np.interp(x + rng.uniform(-max_shift_pts, max_shift_pts), x, spectrum)
    y = y * rng.uniform(0.9, 1.1) + rng.normal(0, noise, len(y))
    return ((y - y.mean()) / (y.std() or 1.0)).astype('float32')
