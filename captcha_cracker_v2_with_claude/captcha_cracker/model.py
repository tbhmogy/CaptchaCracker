"""CaptchaModel class for captcha recognition."""

import logging
from pathlib import Path
from typing import Any, Union

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from . import config
from .exceptions import InvalidImageError, ModelLoadError, PredictionError
from .preprocessing import (
    load_image_from_array,
    load_image_from_bytes,
    load_image_from_path,
)
from .utils import decode_predictions, get_prediction_model, setup_logger


class PredictionResult:
    """Result of a captcha prediction.

    Attributes:
        text: The predicted text.
        confidence: Confidence score (0-1). Currently placeholder.
    """

    def __init__(self, text: str, confidence: float = 1.0) -> None:
        """Initialize prediction result.

        Args:
            text: The predicted text.
            confidence: Confidence score (default: 1.0).
        """
        self.text = text
        self.confidence = confidence

    def __str__(self) -> str:
        """String representation."""
        return self.text

    def __repr__(self) -> str:
        """Detailed representation."""
        return f"PredictionResult(text='{self.text}', confidence={self.confidence:.3f})"


class CaptchaModel:
    """CaptchaCracker model for captcha recognition.

    This class provides a modern API for loading pre-trained captcha recognition
    models and making predictions on captcha images.

    The model uses a CNN-RNN architecture with CTC (Connectionist Temporal
    Classification) loss for sequence-to-sequence learning.

    Examples:
        Basic usage:
        >>> model = CaptchaModel.load('weights.h5')
        >>> result = model.predict('captcha.png')
        >>> print(result)
        '023062'

        With context manager:
        >>> with CaptchaModel.load('weights.h5') as model:
        ...     result = model.predict('captcha.png')
        ...     print(result)

        Batch prediction:
        >>> results = model.predict_batch(['img1.png', 'img2.png'])
        >>> for result in results:
        ...     print(result)

        With confidence scores:
        >>> result = model.predict('captcha.png', return_confidence=True)
        >>> print(f"Text: {result.text}, Confidence: {result.confidence}")

    Attributes:
        image_width: Width of input images.
        image_height: Height of input images.
        max_length: Maximum length of captcha text.
        characters: Set of characters that can be recognized.
        model: The full Keras model (with CTC loss).
        prediction_model: The inference-only model.
    """

    def __init__(
        self,
        model: keras.Model,
        prediction_model: keras.Model,
        char_to_num: layers.StringLookup,
        num_to_char: layers.StringLookup,
        image_width: int,
        image_height: int,
        max_length: int,
        characters: set[str],
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize CaptchaModel.

        Note: Users should use the `load()` class method instead of calling
        this constructor directly.

        Args:
            model: The full Keras model.
            prediction_model: The inference-only model.
            char_to_num: StringLookup layer for encoding.
            num_to_char: StringLookup layer for decoding.
            image_width: Input image width.
            image_height: Input image height.
            max_length: Maximum captcha text length.
            characters: Set of recognizable characters.
            logger: Optional logger instance.
        """
        self.model = model
        self.prediction_model = prediction_model
        self.char_to_num = char_to_num
        self.num_to_char = num_to_char
        self.image_width = image_width
        self.image_height = image_height
        self.max_length = max_length
        self.characters = characters
        self.logger = logger or setup_logger()

    @classmethod
    def load(
        cls,
        weights_path: Union[str, Path],
        image_size: tuple[int, int] = (config.DEFAULT_IMAGE_WIDTH, config.DEFAULT_IMAGE_HEIGHT),
        max_length: int = config.DEFAULT_MAX_LENGTH,
        characters: set[str] | None = None,
        log_level: int = logging.INFO,
    ) -> "CaptchaModel":
        """Load a pre-trained captcha recognition model.

        Args:
            weights_path: Path to the model weights file (.h5 or .keras).
            image_size: Tuple of (width, height) for input images (default: (200, 50)).
            max_length: Maximum length of captcha text (default: 6).
            characters: Set of characters to recognize (default: digits 0-9).
            log_level: Logging level (default: INFO).

        Returns:
            Loaded CaptchaModel instance.

        Raises:
            ModelLoadError: If the model cannot be loaded.

        Examples:
            >>> model = CaptchaModel.load('weights.h5')
            >>> model = CaptchaModel.load('weights.h5', image_size=(300, 60))
        """
        logger = setup_logger(level=log_level)
        logger.info(f"Loading model from {weights_path}")

        # Use default character set if not provided
        if characters is None:
            characters = config.DEFAULT_CHARACTERS

        image_width, image_height = image_size

        try:
            # Create character mapping layers
            char_to_num = layers.StringLookup(
                vocabulary=sorted(characters), num_oov_indices=0, mask_token=None
            )
            num_to_char = layers.StringLookup(
                vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
            )

            # Build model architecture
            from .utils import build_model

            model = build_model(image_width, image_height, len(characters))

            # Load weights
            model.load_weights(str(weights_path))

            # Create prediction model
            prediction_model = get_prediction_model(model)

            logger.info("Model loaded successfully")

            return cls(
                model=model,
                prediction_model=prediction_model,
                char_to_num=char_to_num,
                num_to_char=num_to_char,
                image_width=image_width,
                image_height=image_height,
                max_length=max_length,
                characters=characters,
                logger=logger,
            )

        except Exception as e:
            raise ModelLoadError(f"Failed to load model from {weights_path}: {e}") from e

    def predict(
        self,
        image: Union[str, Path, bytes, np.ndarray],
        return_confidence: bool = False,
    ) -> Union[str, PredictionResult]:
        """Predict captcha text from an image.

        Args:
            image: Input image. Can be:
                - Path to image file (str or Path)
                - Image bytes
                - Numpy array (H, W) or (H, W, 1) or (H, W, 3)
            return_confidence: If True, return PredictionResult with confidence score.
                             If False, return just the text string (default: False).

        Returns:
            Predicted text string, or PredictionResult if return_confidence=True.

        Raises:
            InvalidImageError: If the image cannot be loaded or processed.
            PredictionError: If prediction fails.

        Examples:
            >>> result = model.predict('captcha.png')
            >>> result = model.predict(image_bytes)
            >>> result = model.predict(numpy_array, return_confidence=True)
        """
        try:
            # Load and preprocess image based on input type
            if isinstance(image, (str, Path)):
                img_tensor = load_image_from_path(image, self.image_width, self.image_height)
            elif isinstance(image, bytes):
                img_tensor = load_image_from_bytes(image, self.image_width, self.image_height)
            elif isinstance(image, np.ndarray):
                img_tensor = load_image_from_array(image, self.image_width, self.image_height)
            else:
                raise InvalidImageError(
                    f"Unsupported image type: {type(image)}. "
                    "Expected str, Path, bytes, or np.ndarray."
                )

            # Add batch dimension: (width, height, 1) -> (1, width, height, 1)
            img_batch = tf.expand_dims(img_tensor, 0)

            # Predict
            predictions = self.prediction_model.predict(img_batch, verbose=0)

            # Decode predictions
            decoded_texts = decode_predictions(predictions, self.num_to_char, self.max_length)
            predicted_text = decoded_texts[0]

            # Return result
            if return_confidence:
                # TODO: Implement proper confidence calculation
                return PredictionResult(text=predicted_text, confidence=1.0)
            else:
                return predicted_text

        except InvalidImageError:
            raise
        except Exception as e:
            raise PredictionError(f"Prediction failed: {e}") from e

    def predict_batch(
        self,
        images: list[Union[str, Path, bytes, np.ndarray]],
        return_confidence: bool = False,
        batch_size: int = 32,
    ) -> list[Union[str, PredictionResult]]:
        """Predict captcha text for a batch of images.

        Args:
            images: List of images (paths, bytes, or arrays).
            return_confidence: If True, return PredictionResult objects (default: False).
            batch_size: Batch size for prediction (default: 32).

        Returns:
            List of predicted text strings or PredictionResult objects.

        Raises:
            InvalidImageError: If any image cannot be loaded or processed.
            PredictionError: If prediction fails.

        Examples:
            >>> results = model.predict_batch(['img1.png', 'img2.png', 'img3.png'])
            >>> for result in results:
            ...     print(result)
        """
        try:
            # Load and preprocess all images
            img_tensors = []
            for image in images:
                if isinstance(image, (str, Path)):
                    img_tensor = load_image_from_path(image, self.image_width, self.image_height)
                elif isinstance(image, bytes):
                    img_tensor = load_image_from_bytes(image, self.image_width, self.image_height)
                elif isinstance(image, np.ndarray):
                    img_tensor = load_image_from_array(
                        image, self.image_width, self.image_height
                    )
                else:
                    raise InvalidImageError(f"Unsupported image type at index {len(img_tensors)}")
                img_tensors.append(img_tensor)

            # Stack into batch
            img_batch = tf.stack(img_tensors, axis=0)

            # Predict
            predictions = self.prediction_model.predict(img_batch, batch_size=batch_size, verbose=0)

            # Decode predictions
            decoded_texts = decode_predictions(predictions, self.num_to_char, self.max_length)

            # Return results
            if return_confidence:
                return [PredictionResult(text=text, confidence=1.0) for text in decoded_texts]
            else:
                return decoded_texts

        except InvalidImageError:
            raise
        except Exception as e:
            raise PredictionError(f"Batch prediction failed: {e}") from e

    def predict_bytes(
        self, image_bytes: bytes, return_confidence: bool = False
    ) -> Union[str, PredictionResult]:
        """Predict captcha text from image bytes.

        Convenience method for predicting from bytes.

        Args:
            image_bytes: Image data as bytes.
            return_confidence: If True, return PredictionResult (default: False).

        Returns:
            Predicted text string or PredictionResult.
        """
        return self.predict(image_bytes, return_confidence=return_confidence)

    def predict_array(
        self, image_array: np.ndarray, return_confidence: bool = False
    ) -> Union[str, PredictionResult]:
        """Predict captcha text from a numpy array.

        Convenience method for predicting from numpy arrays.

        Args:
            image_array: Image as numpy array.
            return_confidence: If True, return PredictionResult (default: False).

        Returns:
            Predicted text string or PredictionResult.
        """
        return self.predict(image_array, return_confidence=return_confidence)

    def __enter__(self) -> "CaptchaModel":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager."""
        # Clean up resources if needed
        pass

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"CaptchaModel(image_size=({self.image_width}, {self.image_height}), "
            f"max_length={self.max_length}, num_characters={len(self.characters)})"
        )
