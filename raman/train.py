"""
Train and compare substance classifiers, then save the hybrid model for the INTERSECT service.

Compared on the same held-out test set:
  pca_lda, svm_rbf     classic chemometrics on the preprocessed spectrum
  svm_scalogram        SVM on the flattened CWT scalogram alone
  hybrid_svm           SVM on spectrum + scalogram together (the classic counterpart of the hybrid)
  spectrum_only        1D CRNN branch alone
  scalogram_only       2D CNN on the CWT scalogram alone
  hybrid               both branches fused (the model that gets deployed)

The split is by physical sample: every spectrum of a sample lands on the same side.
A random per-spectrum split leaks near-duplicate spectra into the test set and
inflates accuracy.

Run from the repository root:
    python -m raman.train --data-dir data/example
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm

from raman.bundle import DEFAULT_BUNDLE_DIR, save_bundle
from raman.data import load_dataset
from raman.model import build_model, classic_baselines, svm_variants
from raman.preprocess import PreprocessConfig, augment, preprocess_spectrum, scalogram


def group_split(y, groups, n_splits, seed):
    """One fold of a stratified, group-aware split: (train_idx, held_out_idx)."""
    return next(StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(y, y, groups))


def deep_inputs(kind, spectra, scalos):
    x = {}
    if kind in ('hybrid', 'spectrum_only'):
        x['spectrum_input'] = spectra[..., None]
    if kind in ('hybrid', 'scalogram_only'):
        x['scalogram_input'] = scalos[..., None]
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='data/example')
    ap.add_argument('--bundle-dir', default=DEFAULT_BUNDLE_DIR)
    ap.add_argument('--results-dir', default='results')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--augment-copies', type=int, default=10)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--gpu', action='store_true',
                    help='use the GPU (off by default: tensorflow-metal crashes on some 2D convolutions)')
    args = ap.parse_args()
    if not args.gpu:
        tf.config.set_visible_devices([], 'GPU')

    rng = np.random.default_rng(args.seed)
    tf.keras.utils.set_random_seed(args.seed)
    cfg = PreprocessConfig()

    raw, labels = load_dataset(args.data_dir)
    class_names = sorted(labels['substance'].unique())
    y = labels['substance'].map({c: i for i, c in enumerate(class_names)}).to_numpy()
    groups = labels['sample_id'].to_numpy()
    X = np.stack([preprocess_spectrum(s, i, cfg) for s, i in tqdm(raw, desc='Preprocessing')])

    # Held-out test samples, then validation samples from what is left
    trval, test = group_split(y, groups, n_splits=3, seed=args.seed)
    tr_rel, val_rel = group_split(y[trval], groups[trval], n_splits=4, seed=args.seed)
    train, val = trval[tr_rel], trval[val_rel]
    print(f'Spectra  train {len(train)} | val {len(val)} | test {len(test)}')
    print(f'Samples  train {len(set(groups[train]))} | val {len(set(groups[val]))} | test {len(set(groups[test]))}')

    # Augmented training set; scalograms computed for every spectrum
    X_train = np.concatenate([X[train]] + [np.stack([augment(s, rng) for s in X[train]])
                                           for _ in range(args.augment_copies)])
    y_train = np.tile(y[train], args.augment_copies + 1)
    S_train = np.stack([scalogram(s, cfg) for s in tqdm(X_train, desc='Scalograms')])
    S_val = np.stack([scalogram(s, cfg) for s in X[val]])
    S_test = np.stack([scalogram(s, cfg) for s in X[test]])

    results, preds = {}, {}
    for name, clf in classic_baselines().items():
        clf.fit(X[train], y[train])
        preds[name] = clf.predict(X[test])

    # SVMs that also see the scalogram. X_train starts with the un-augmented X[train],
    # so the first len(train) scalograms belong to it (classic models train without augmentation).
    for name, (features, clf) in svm_variants().items():
        clf.fit(features(X[train], S_train[:len(train)]), y[train])
        preds[name] = clf.predict(features(X[test], S_test))

    deployed = None
    for kind in ('spectrum_only', 'scalogram_only', 'hybrid'):
        print(f'\n--- Training {kind} ---')
        model = build_model(kind, len(class_names), cfg.n_points, S_train.shape[1:])
        model.fit(deep_inputs(kind, X_train, S_train), y_train,
                  validation_data=(deep_inputs(kind, X[val], S_val), y[val]),
                  epochs=args.epochs, batch_size=32, verbose=2,
                  callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=10, restore_best_weights=True),
                             tf.keras.callbacks.ReduceLROnPlateau('val_loss', factor=0.5, patience=5)])
        preds[kind] = model.predict(deep_inputs(kind, X[test], S_test), verbose=0).argmax(axis=1)
        if kind == 'hybrid':
            deployed = model

    for name, p in preds.items():
        results[name] = {'accuracy': accuracy_score(y[test], p), 'macro_f1': f1_score(y[test], p, average='macro')}

    # Report
    os.makedirs(args.results_dir, exist_ok=True)
    lines = ['| Model | Test accuracy | Macro F1 |', '|---|---|---|']
    lines += [f"| {n} | {r['accuracy']:.3f} | {r['macro_f1']:.3f} |" for n, r in results.items()]
    table = '\n'.join(lines)
    print('\n' + table)
    with open(os.path.join(args.results_dir, 'results.md'), 'w') as f:
        f.write(f'Test set: {len(test)} spectra from {len(set(groups[test]))} held-out samples.\n\n{table}\n')
    with open(os.path.join(args.results_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay(confusion_matrix(y[test], preds['hybrid'], labels=range(len(class_names))),
                           display_labels=class_names).plot(ax=ax, cmap='Blues', xticks_rotation=45, colorbar=False)
    ax.set_title('Hybrid model: held-out samples')
    fig.tight_layout()
    fig.savefig(os.path.join(args.results_dir, 'confusion_hybrid.png'), dpi=150)

    save_bundle(deployed, cfg, class_names, args.bundle_dir)
    print(f'\nSaved hybrid model to {args.bundle_dir}/ and results to {args.results_dir}/')


if __name__ == '__main__':
    main()
