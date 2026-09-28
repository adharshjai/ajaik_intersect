"""
Models: the hybrid CRNN + CNN, its two single-branch ablations, and classic baselines.
"""
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from tensorflow.keras import layers, models


def _spectrum_branch(n_points):
    """CRNN on the 1D spectrum: convolutions find peaks, the BiLSTM reads their order along the axis.

    A global max-pool of the conv features runs alongside the LSTM so strong peaks reach the
    classifier directly instead of having to survive the whole recurrent pass.
    """
    inp = layers.Input(shape=(n_points, 1), name='spectrum_input')
    x = inp
    for filters, k in [(32, 7), (64, 5), (64, 3)]:
        x = layers.Conv1D(filters, k, padding='same', activation='relu')(x)
        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.MaxPooling1D(2)(x)
    x = layers.concatenate([layers.Bidirectional(layers.LSTM(32))(x), layers.GlobalMaxPooling1D()(x)])
    return inp, layers.Dropout(0.3)(x)


def _scalogram_branch(scalogram_shape):
    """2D CNN on the CWT scalogram (scales x Raman shift)."""
    inp = layers.Input(shape=(*scalogram_shape, 1), name='scalogram_input')
    x = inp
    for filters in (16, 32, 64):
        x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.MaxPooling2D(2)(x)
    x = layers.GlobalAveragePooling2D()(x)
    return inp, layers.Dropout(0.3)(layers.Dense(64, activation='relu')(x))


def build_model(kind, n_classes, n_points, scalogram_shape, lr=1e-3):
    """kind: 'hybrid', 'spectrum_only' or 'scalogram_only'."""
    inputs, feats = [], []
    if kind in ('hybrid', 'spectrum_only'):
        i, f = _spectrum_branch(n_points)
        inputs.append(i); feats.append(f)
    if kind in ('hybrid', 'scalogram_only'):
        i, f = _scalogram_branch(scalogram_shape)
        inputs.append(i); feats.append(f)

    x = layers.concatenate(feats) if len(feats) > 1 else feats[0]
    x = layers.Dropout(0.4)(layers.Dense(64, activation='relu')(x))
    out = layers.Dense(n_classes, activation='softmax', name='substance')(x)

    model = models.Model(inputs, out, name=kind)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def classic_baselines():
    """Standard chemometrics baselines on the preprocessed spectrum."""
    return {
        'pca_lda': make_pipeline(PCA(n_components=20), LinearDiscriminantAnalysis()),
        'svm_rbf': SVC(kernel='rbf', C=10, gamma='scale'),
    }
