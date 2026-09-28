"""
INTERSECT service exposing the Raman substance classifier (spectrum + CWT scalogram hybrid).

In INTERSECT pattern terms this is the *analyzer* (the "Orient" step of an OODA loop):
the spectrometer, or its adapter, sends a raw spectrum, and the service returns the
predicted substance with probabilities. Every result is also broadcast as a
`substance_identified` event, so a controller can react, e.g. re-measure with a longer
integration time when confidence is low.

Run from the repository root:
    python -m intersect_service.raman_service
"""
import argparse
import logging
import os
import threading
import time
from typing import ClassVar, Optional

import numpy as np
import tensorflow as tf
from pydantic import BaseModel, Field

from intersect_sdk import (
    HierarchyConfig,
    IntersectBaseCapabilityImplementation,
    IntersectCapabilityError,
    IntersectEventDefinition,
    IntersectService,
    IntersectServiceConfig,
    default_intersect_lifecycle_loop,
    intersect_message,
    intersect_status,
)

from raman.bundle import DEFAULT_BUNDLE_DIR, load_bundle
from raman.preprocess import preprocess_spectrum, scalogram

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# --- Message types (these become the service's published schema) ---
# =============================================================================

class RamanSpectrum(BaseModel):
    """One raw spectrum, exactly as exported by the spectrometer software."""
    raman_shift: list[float] = Field(description='Raman shift axis in cm^-1')
    intensity: list[float] = Field(description='Intensity (counts), same length as raman_shift')
    spectrum_id: Optional[str] = Field(default=None, description='Caller-chosen ID, echoed back')


class SubstanceClassification(BaseModel):
    spectrum_id: Optional[str]
    substance: str
    confidence: float = Field(description='Softmax probability of the predicted substance')
    probabilities: dict[str, float]
    inference_ms: float


class ModelInfo(BaseModel):
    substances: list[str]
    shift_range_cm: list[float] = Field(description='Spectra must cover at least this range')
    n_points: int
    wavelet: str
    n_scales: int


class ClassifierStatus(BaseModel):
    model_loaded: bool
    spectra_classified: int
    last_result: Optional[SubstanceClassification]


# =============================================================================
# --- Capability ---
# =============================================================================

class RamanClassifierCapability(IntersectBaseCapabilityImplementation):
    intersect_sdk_capability_name = 'RamanClassifier'
    intersect_sdk_events: ClassVar[dict[str, IntersectEventDefinition]] = {
        'substance_identified': IntersectEventDefinition(event_type=SubstanceClassification),
    }

    def __init__(self, bundle_dir=DEFAULT_BUNDLE_DIR):
        super().__init__()
        self.model, self.cfg, self.class_names = load_bundle(bundle_dir)
        self._lock = threading.Lock()
        self._count = 0
        self._last: Optional[SubstanceClassification] = None
        logger.info('Loaded Raman model from %s (%d substances)', bundle_dir, len(self.class_names))

    @intersect_status()
    def status(self) -> ClassifierStatus:
        return ClassifierStatus(model_loaded=True, spectra_classified=self._count, last_result=self._last)

    @intersect_message()
    def get_model_info(self) -> ModelInfo:
        """What the model can identify and what spectra it accepts."""
        return ModelInfo(
            substances=self.class_names,
            shift_range_cm=[self.cfg.shift_min, self.cfg.shift_max],
            n_points=self.cfg.n_points,
            wavelet=self.cfg.wavelet,
            n_scales=self.cfg.n_scales,
        )

    @intersect_message()
    def classify_spectrum(self, request: RamanSpectrum) -> SubstanceClassification:
        """Identify the substance from one raw Raman spectrum."""
        if len(request.raman_shift) != len(request.intensity):
            raise IntersectCapabilityError('raman_shift and intensity must have the same length')
        if len(request.raman_shift) < 50:
            raise IntersectCapabilityError('Spectrum has too few points')
        try:
            spectrum = preprocess_spectrum(np.array(request.raman_shift), np.array(request.intensity), self.cfg)
        except ValueError as e:
            raise IntersectCapabilityError(str(e)) from e

        with self._lock:
            start = time.perf_counter()
            probs = self.model.predict({
                'spectrum_input': spectrum[None, :, None],
                'scalogram_input': scalogram(spectrum, self.cfg)[None, ..., None],
            }, verbose=0)[0]
            idx = int(np.argmax(probs))
            result = SubstanceClassification(
                spectrum_id=request.spectrum_id,
                substance=self.class_names[idx],
                confidence=float(probs[idx]),
                probabilities={c: float(p) for c, p in zip(self.class_names, probs)},
                inference_ms=(time.perf_counter() - start) * 1000,
            )
            self._count += 1
            self._last = result

        self.intersect_sdk_emit_event('substance_identified', result)
        return result


# =============================================================================
# --- Entry point ---
# =============================================================================

def build_config(args):
    return IntersectServiceConfig(
        hierarchy=HierarchyConfig(
            organization=args.organization,
            facility=args.facility,
            system=args.system,
            subsystem=args.subsystem,
            service=args.service,
        ),
        brokers=[{
            'host': args.broker_host,
            'port': args.broker_port,
            'username': os.environ.get('INTERSECT_BROKER_USERNAME', 'intersect_username'),
            'password': os.environ.get('INTERSECT_BROKER_PASSWORD', 'intersect_password'),
            'protocol': 'mqtt5.0',
        }],
    )


def parse_args():
    p = argparse.ArgumentParser(description='INTERSECT service for Raman substance classification')
    p.add_argument('--bundle-dir', default=DEFAULT_BUNDLE_DIR)
    p.add_argument('--broker-host', default=os.environ.get('INTERSECT_BROKER_HOST', '127.0.0.1'))
    p.add_argument('--broker-port', type=int, default=int(os.environ.get('INTERSECT_BROKER_PORT', 1883)))
    p.add_argument('--organization', default='ajaik')
    p.add_argument('--facility', default='lab')
    p.add_argument('--system', default='raman')
    p.add_argument('--subsystem', default='analysis')
    p.add_argument('--service', default='substance-classifier')
    return p.parse_args()


if __name__ == '__main__':
    tf.config.set_visible_devices([], 'GPU')  # tensorflow-metal crashes on some 2D convolutions
    args = parse_args()
    service = IntersectService([RamanClassifierCapability(args.bundle_dir)], build_config(args))
    logger.info('Starting Raman classifier as %s.%s.%s.%s.%s, use Ctrl+C to exit.',
                args.organization, args.facility, args.system, args.subsystem, args.service)
    default_intersect_lifecycle_loop(service)
