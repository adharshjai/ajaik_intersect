"""
Load Raman spectra, and generate a synthetic example dataset.

Each spectrum is stored the way most spectrometer software exports it: a CSV with
two columns, Raman shift (cm^-1) and intensity (counts). `labels.csv` lists every
file with its substance and the physical sample it was measured from.

The synthetic data is only for trying out the pipeline. Peak positions are rough
literature values and must not be used to identify real substances.
"""
import os

import numpy as np
import pandas as pd

# (center cm^-1, relative height, width cm^-1) per substance
SUBSTANCES = {
    'polystyrene':   [(620, .25, 8), (795, .1, 10), (1001, 1.0, 6), (1031, .35, 7), (1155, .2, 9),
                      (1450, .15, 12), (1583, .2, 8), (1602, .45, 8)],
    'ethanol':       [(433, .35, 14), (884, 1.0, 10), (1052, .5, 12), (1096, .45, 12), (1276, .15, 14),
                      (1454, .6, 14)],
    'acetaminophen': [(465, .3, 8), (651, .3, 8), (797, .5, 7), (858, .8, 7), (1168, .4, 8),
                      (1236, .55, 8), (1324, .65, 9), (1561, .45, 9), (1611, 1.0, 8), (1648, .45, 9)],
    'calcite':       [(282, .35, 12), (712, .3, 8), (1086, 1.0, 5), (1436, .05, 12), (1749, .05, 10)],
    'glucose':       [(422, .7, 10), (518, .55, 10), (838, .3, 10), (913, .35, 10), (1060, .6, 12),
                      (1125, 1.0, 12), (1365, .5, 14), (1460, .4, 14)],
    'cyclohexane':   [(384, .15, 8), (801, 1.0, 6), (1028, .45, 7), (1157, .1, 8), (1266, .3, 8),
                      (1444, .55, 10)],
    # Two hypothetical polymorphs of one compound: same peaks, a few shifted by 4-8 cm^-1
    # and different widths. This is the hard pair, where peak shape and width matter.
    'compound_form_I':  [(560, .4, 9), (740, .6, 8), (1010, 1.0, 7), (1180, .5, 9), (1330, .45, 10),
                         (1590, .8, 9)],
    'compound_form_II': [(560, .4, 12), (743, .6, 8), (1010, 1.0, 10), (1184, .5, 9), (1330, .45, 14),
                         (1593, .8, 9)],
}

CONTAMINATION_MAX = 0.25  # up to 25% of another substance's signal mixed in

SHIFT_RANGE = (150.0, 1900.0)  # what the simulated spectrometer records
N_PIXELS = 1024                # detector pixels


def _lorentzian(x, center, height, width):
    return height * (0.5 * width) ** 2 / ((x - center) ** 2 + (0.5 * width) ** 2)


def simulate_spectrum(substance, rng, calibration_offset=0.0):
    """One noisy measurement: peaks + fluorescence baseline + shot noise + occasional cosmic ray."""
    # Slightly non-linear pixel -> wavenumber axis, like a real CCD, plus a per-session calibration offset
    p = np.linspace(0, 1, N_PIXELS)
    shift = SHIFT_RANGE[0] + (SHIFT_RANGE[1] - SHIFT_RANGE[0]) * (p + 0.03 * p * (1 - p)) + calibration_offset

    def peaks(name, scale):
        out = np.zeros_like(shift)
        for center, height, width in SUBSTANCES[name]:
            out += _lorentzian(shift, center + rng.normal(0, 1.0),
                               scale * height * rng.uniform(0.8, 1.2), width * rng.uniform(0.9, 1.1))
        return out

    # Target substance plus a contaminant (impurity, container, substrate)
    other = rng.choice([s for s in SUBSTANCES if s != substance])
    signal = peaks(substance, 1.0) + peaks(other, rng.uniform(0, CONTAMINATION_MAX))

    # Fluorescence background: broad, often much stronger than the Raman peaks
    x = (shift - shift.min()) / np.ptp(shift)
    baseline = rng.uniform(1.0, 8.0) * np.exp(-((x - rng.uniform(0.3, 1.2)) ** 2) / rng.uniform(0.3, 1.5))
    baseline += rng.uniform(-0.5, 0.5) * x + rng.uniform(0, 1.0)

    counts = rng.uniform(80, 400) * (signal + baseline)  # low signal: short integration time
    counts = rng.poisson(np.clip(counts, 0, None)).astype(float) + rng.normal(0, 10, N_PIXELS)  # shot + read noise

    if rng.random() < 0.3:  # cosmic ray: 1-2 pixel spike
        i = rng.integers(5, N_PIXELS - 5)
        counts[i:i + rng.integers(1, 3)] += rng.uniform(500, 3000)
    return shift, counts


def generate_example_dataset(out_dir, samples_per_substance=6, spectra_per_sample=8, seed=0):
    """Several physical samples per substance, several spectra per sample, one CSV per spectrum."""
    rng = np.random.default_rng(seed)
    spectra_dir = os.path.join(out_dir, 'spectra')
    os.makedirs(spectra_dir, exist_ok=True)

    rows = []
    for substance in SUBSTANCES:
        for s in range(1, samples_per_substance + 1):
            sample_id = f'{substance}_S{s}'
            offset = rng.normal(0, 3.0)  # each sample measured in its own session
            for k in range(1, spectra_per_sample + 1):
                shift, counts = simulate_spectrum(substance, rng, offset)
                fname = f'{sample_id}_{k:02d}.csv'
                pd.DataFrame({'raman_shift_cm-1': shift.round(2), 'intensity': counts.round(1)}).to_csv(
                    os.path.join(spectra_dir, fname), index=False)
                rows.append({'file': fname, 'substance': substance, 'sample_id': sample_id})

    pd.DataFrame(rows).to_csv(os.path.join(out_dir, 'labels.csv'), index=False)
    print(f'Wrote {len(rows)} spectra ({len(SUBSTANCES)} substances) to {out_dir}')


def read_spectrum_csv(path):
    """Read a two-column (shift, intensity) export. Header row optional; ',', ';' or tab separated."""
    df = pd.read_csv(path, sep=None, engine='python', header=None, comment='#')
    df = df.apply(pd.to_numeric, errors='coerce').dropna()
    return df.iloc[:, 0].to_numpy(float), df.iloc[:, 1].to_numpy(float)


def load_dataset(data_dir):
    """Returns lists of (shift, intensity) arrays plus the labels table."""
    labels = pd.read_csv(os.path.join(data_dir, 'labels.csv'))
    spectra = [read_spectrum_csv(os.path.join(data_dir, 'spectra', f)) for f in labels['file']]
    return spectra, labels


if __name__ == '__main__':
    generate_example_dataset(os.path.join(os.path.dirname(__file__), '..', 'data', 'example'))
