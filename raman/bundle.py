"""Save/load the trained hybrid model together with its preprocessing settings and class names."""
import json
import os

from raman.preprocess import PreprocessConfig

MODEL_FILENAME = 'raman_hybrid.keras'
METADATA_FILENAME = 'metadata.json'
DEFAULT_BUNDLE_DIR = 'model_artifacts'


def save_bundle(model, cfg, class_names, bundle_dir=DEFAULT_BUNDLE_DIR):
    os.makedirs(bundle_dir, exist_ok=True)
    model.save(os.path.join(bundle_dir, MODEL_FILENAME))
    with open(os.path.join(bundle_dir, METADATA_FILENAME), 'w') as f:
        json.dump({'preprocess': cfg.to_dict(), 'class_names': list(class_names)}, f, indent=2)


def load_bundle(bundle_dir=DEFAULT_BUNDLE_DIR):
    import tensorflow as tf

    model_path = os.path.join(bundle_dir, MODEL_FILENAME)
    meta_path = os.path.join(bundle_dir, METADATA_FILENAME)
    if not (os.path.exists(model_path) and os.path.exists(meta_path)):
        raise FileNotFoundError(f"No model bundle in '{bundle_dir}'. Run `python -m raman.train` first.")
    with open(meta_path) as f:
        meta = json.load(f)
    return tf.keras.models.load_model(model_path), PreprocessConfig(**meta['preprocess']), meta['class_names']
