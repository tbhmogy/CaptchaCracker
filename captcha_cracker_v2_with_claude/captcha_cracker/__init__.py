"""CaptchaCracker: Modern OCR library for captcha recognition using deep learning.

This package provides a modern, type-safe API for training and deploying
captcha recognition models using TensorFlow 2.18+ and Python 3.12+.

Examples:
    Basic prediction:
    >>> from captcha_cracker import CaptchaModel
    >>> model = CaptchaModel.load('weights.h5')
    >>> result = model.predict('captcha.png')
    >>> print(result)
    '023062'

    Training a new model:
    >>> from captcha_cracker import CaptchaTrainer
    >>> trainer = CaptchaTrainer(image_size=(200, 50))
    >>> trainer.load_data(['data/train/'])
    >>> trainer.train(epochs=100)
    >>> trainer.save('weights.h5')
"""

__version__ = "1.0.0"
__author__ = "CaptchaCracker Contributors"

# Import public API
from .exceptions import (
    CaptchaCrackerError,
    ConfigurationError,
    InvalidImageError,
    ModelLoadError,
    PredictionError,
    TrainingError,
)
from .model import CaptchaModel, PredictionResult
from .trainer import CaptchaTrainer

# Define public API
__all__ = [
    # Main classes
    "CaptchaModel",
    "CaptchaTrainer",
    "PredictionResult",
    # Exceptions
    "CaptchaCrackerError",
    "ModelLoadError",
    "InvalidImageError",
    "PredictionError",
    "TrainingError",
    "ConfigurationError",
    # Version
    "__version__",
]


def get_version() -> str:
    """Get the package version.

    Returns:
        Version string.
    """
    return __version__
