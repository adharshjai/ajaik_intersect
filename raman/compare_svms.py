"""
Stress test for the SVMs: does adding the CWT scalogram help a classic model?

On clean held-out data every SVM can hit 100%, which says nothing. This repeats the
sample-grouped split several times and adds extra noise to the *test* spectra only,
simulating noisier field measurements than the model was trained on.

Runs in about a minute (no deep learning):
    python -m raman.compare_svms --data-dir data/example
"""
import argparse
import os

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from tqdm import tqdm

from raman.data import load_dataset
from raman.model import hybrid_svm_features
from raman.preprocess import PreprocessConfig, preprocess_spectrum, scalogram
from raman.train import group_split

FEATURES = {
    'svm_rbf': lambda X, S: X,
    'svm_scalogram': lambda X, S: S.reshape(len(S), -1),
    'hybrid_svm': hybrid_svm_features,
}


def renormalize(X):
    return (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='data/example')
    ap.add_argument('--results-dir', default='results')
    ap.add_argument('--repeats', type=int, default=10)
    ap.add_argument('--noise-levels', type=float, nargs='+', default=[0.0, 0.5, 1.0, 1.5])
    args = ap.parse_args()

    cfg = PreprocessConfig()
    raw, labels = load_dataset(args.data_dir)
    classes = sorted(labels['substance'].unique())
    y = labels['substance'].map({c: i for i, c in enumerate(classes)}).to_numpy()
    groups = labels['sample_id'].to_numpy()
    X = np.stack([preprocess_spectrum(s, i, cfg) for s, i in tqdm(raw, desc='Preprocessing')])
    S = np.stack([scalogram(s, cfg) for s in X])

    # acc[noise][model] -> list over repeats
    acc = {n: {m: [] for m in FEATURES} for n in args.noise_levels}
    for rep in tqdm(range(args.repeats), desc='Repeats'):
        rng = np.random.default_rng(rep)
        train, test = group_split(y, groups, n_splits=3, seed=rep)
        models = {m: SVC(kernel='rbf', C=10, gamma='scale').fit(f(X[train], S[train]), y[train])
                  for m, f in FEATURES.items()}
        for noise in args.noise_levels:
            X_test = renormalize(X[test] + rng.normal(0, noise, X[test].shape)).astype('float32')
            S_test = np.stack([scalogram(s, cfg) for s in X_test])
            for m, f in FEATURES.items():
                acc[noise][m].append(accuracy_score(y[test], models[m].predict(f(X_test, S_test))))

    header = '| Extra test noise | ' + ' | '.join(FEATURES) + ' |'
    lines = [header, '|' + '---|' * (len(FEATURES) + 1)]
    for noise in args.noise_levels:
        cells = [f'{np.mean(v):.3f} ± {np.std(v):.3f}' for v in acc[noise].values()]
        lines.append(f'| {noise:g} | ' + ' | '.join(cells) + ' |')
    table = '\n'.join(lines)
    print('\n' + table)

    os.makedirs(args.results_dir, exist_ok=True)
    with open(os.path.join(args.results_dir, 'svm_comparison.md'), 'w') as f:
        f.write(f'Mean ± std test accuracy over {args.repeats} sample-grouped splits. '
                f'Extra noise (std, in units of the normalized spectrum) is added to test spectra only.\n\n'
                f'{table}\n')


if __name__ == '__main__':
    main()
