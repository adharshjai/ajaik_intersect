# ajaiks_intersect

Raman spectroscopy substance classification with a **hybrid deep model** that looks at each spectrum two ways:
- the **raw spectrum** (intensity vs. Raman shift)
- its **CWT scalogram**, a wavelet transform along the Raman-shift axis

The trained model runs as a service on ORNL's [INTERSECT](https://intersect-architecture.readthedocs.io) platform for autonomous labs, built with the [INTERSECT-SDK](https://github.com/INTERSECT-SDK/python-sdk).

> **The included dataset is synthetic.** It exists only to exercise the pipeline until real spectrometer exports are available. Its peak positions are rough literature values; do not use it to identify real substances.

```
spectrometer export (.csv: shift, intensity)
        │
        ▼
 preprocess ─ despike ─ resample 200–1800 cm⁻¹ ─ ALS baseline ─ SNV
        │                                   │
        ▼                                   ▼
  1D CRNN branch                  CWT scalogram ─► 2D CNN branch
        └──────────────► fuse ◄─────────────┘
                           │
                           ▼
                substance + probabilities
                           │
                           ▼
          INTERSECT service (classify_spectrum)
```

## Why a scalogram for Raman?

The continuous wavelet transform is a standard tool for finding Raman peaks. Running it along the Raman-shift axis separates features by width:
- **small scales** capture the sharp peaks that identify a substance;
- **large scales** capture the broad fluorescence background.

The 2D CNN therefore sees each peak's width and shape at every scale. The 1D branch sees exact peak positions and relative intensities.

The scalogram is computed as a numeric array (48 scales × 200 points, Mexican-hat wavelet). It is not rendered as an image, so nothing is lost to colormaps or resizing.

## Repository layout

| Path | Purpose |
|---|---|
| `raman/data.py` | Reads spectrum CSVs; generates the synthetic example dataset |
| `raman/preprocess.py` | Despiking, ALS baseline removal, SNV, common grid, CWT scalogram, augmentation |
| `raman/model.py` | Hybrid model, the two single-branch ablations, PCA-LDA and SVM baselines |
| `raman/train.py` | Trains and compares every model on held-out samples, then saves the hybrid |
| `raman/bundle.py` | Saves and loads the model together with its preprocessing settings |
| `intersect_service/raman_service.py` | INTERSECT service, capability `RamanClassifier` |
| `intersect_service/raman_client.py` | Example client: sends CSV spectra, prints results |
| `data/example/` | Synthetic dataset: 8 substances × 6 samples × 8 spectra |
| `model_artifacts/` | Trained hybrid model, ready for the service |
| `results/` | Comparison table and confusion matrix |

## Quick start

```bash
pip install -r requirements.txt

python -m raman.data      # optional: regenerate data/example
python -m raman.train     # train + compare models, save model_artifacts/
```

Run as an INTERSECT service (needs Docker for the message broker):

```bash
docker compose -f intersect_service/docker-compose.yml up -d
python -m intersect_service.raman_service
python -m intersect_service.raman_client data/example/spectra/ethanol_S1_01.csv
```

On Apple Silicon, training and the service run on the CPU by default, because `tensorflow-metal` crashes on some 2D convolutions. Pass `--gpu` to `raman.train` to try the GPU.

## Using real data

1. Export each spectrum as a two-column file: Raman shift and intensity. Comma-, semicolon- and tab-separated files all work, and a header row is optional.
2. Put the files in `data/<name>/spectra/`, then write `data/<name>/labels.csv` with the columns `file,substance,sample_id`.
   - `sample_id` identifies the **physical sample**. Every spectrum from the same sample stays on the same side of the train/test split.
3. If your spectra don't cover 200–1800 cm⁻¹, adjust `PreprocessConfig` in `raman/preprocess.py`.
4. Train with `python -m raman.train --data-dir data/<name>`.

## Results on the synthetic dataset

Test set: 128 spectra from 16 held-out samples.

| Model | Test accuracy | Macro F1 |
|---|---|---|
| pca_lda | 0.977 | 0.967 |
| svm_rbf | 1.000 | 1.000 |
| spectrum_only | 0.883 | 0.860 |
| scalogram_only | 0.898 | 0.890 |
| hybrid | 0.844 | 0.827 |

What this shows:
- **Evaluation:** test samples are physical samples never seen in training. Splitting per spectrum instead leaks near-duplicates and inflates accuracy.
- **The classic SVM wins here.** With only ~200 training spectra, classic chemometrics is hard to beat, which is normal for small Raman datasets. The deep models need more data, or pre-training, before the hybrid can pay off.
- **Where the errors are:** mostly the two near-identical polymorphs (`compound_form_I` / `_II`). That pair is the case the scalogram branch is meant to help with.
- **Run-to-run variation:** the test set has only 16 samples, so accuracy moves several points between runs.

Always compare against these baselines before claiming the hybrid helps.

## INTERSECT interface

Service address: `ajaik.lab.raman.analysis.substance-classifier` (change it with CLI flags).

| Operation | Input | Output |
|---|---|---|
| `RamanClassifier.classify_spectrum` | `{raman_shift: [...], intensity: [...], spectrum_id?}` | substance, confidence, per-class probabilities |
| `RamanClassifier.get_model_info` | none | substances, required shift range, CWT settings |
| status | none | model loaded, spectra classified, last result |
| event `substance_identified` | none | broadcast after every classification |

In INTERSECT's design-pattern terms, this service is the **analyzer** (the "Orient" step of an OODA loop). It can grow into the following patterns:
- **Experiment Control:** acquire a spectrum, classify it automatically, log the result.
- **Experiment Steering:** when confidence is low, a controller re-measures with a longer integration time or higher laser power.
- **Design of Experiments:** the system chooses which sample or spot to measure next.

A spectrometer adapter, an INTERSECT Instrument Controller wrapping the Raman software, would complete the loop.

Broker settings come from `INTERSECT_BROKER_HOST`, `INTERSECT_BROKER_PORT`, `INTERSECT_BROKER_USERNAME` and `INTERSECT_BROKER_PASSWORD`. The defaults match `docker-compose.yml`.
