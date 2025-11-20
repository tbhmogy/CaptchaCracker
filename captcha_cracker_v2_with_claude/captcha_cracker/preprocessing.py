"""Image preprocessing utilities for CaptchaCracker."""

import io
from pathlib import Path
from typing import Union

import numpy as np
import tensorflow as tf
from PIL import Image

from .exceptions import InvalidImageError


def load_image_from_path(
    image_path: Union[str, Path], image_width: int, image_height: int
) -> tf.Tensor:
    """Load and preprocess an image from a file path.

    Args:
        image_path: Path to the image file.
        image_width: Target width for the image.
        image_height: Target height for the image.

    Returns:
        Preprocessed image tensor with shape (width, height, 1).

    Raises:
        InvalidImageError: If the image cannot be loaded or processed.
    """
    try:
        # Read image file
        img = tf.io.read_file(str(image_path))
        # Decode as grayscale
        img = tf.io.decode_png(img, channels=1)
        # Convert to float32 in [0, 1] range
        img = tf.image.convert_image_dtype(img, tf.float32)
        # Resize to target dimensions
        img = tf.image.resize(img, [image_height, image_width])
        # Transpose: (height, width, channels) -> (width, height, channels)
        # This makes the time dimension correspond to the width
        img = tf.transpose(img, perm=[1, 0, 2])
        return img
    except Exception as e:
        raise InvalidImageError(f"Failed to load image from {image_path}: {e}") from e


def load_image_from_bytes(
    image_bytes: bytes, image_width: int, image_height: int
) -> tf.Tensor:
    """Load and preprocess an image from bytes.

    Args:
        image_bytes: Image data as bytes.
        image_width: Target width for the image.
        image_height: Target height for the image.

    Returns:
        Preprocessed image tensor with shape (width, height, 1).

    Raises:
        InvalidImageError: If the image cannot be loaded or processed.
    """
    try:
        # Decode image (auto-detect format)
        img = tf.io.decode_image(image_bytes, channels=1)
        # Convert to float32 in [0, 1] range
        img = tf.image.convert_image_dtype(img, tf.float32)
        # Resize to target dimensions
        img = tf.image.resize(img, [image_height, image_width])
        # Transpose: (height, width, channels) -> (width, height, channels)
        img = tf.transpose(img, perm=[1, 0, 2])
        return img
    except Exception as e:
        raise InvalidImageError(f"Failed to load image from bytes: {e}") from e


def load_image_from_array(
    image_array: np.ndarray, image_width: int, image_height: int
) -> tf.Tensor:
    """Load and preprocess an image from a numpy array.

    Args:
        image_array: Image as numpy array. Can be (H, W), (H, W, 1), or (H, W, 3).
        image_width: Target width for the image.
        image_height: Target height for the image.

    Returns:
        Preprocessed image tensor with shape (width, height, 1).

    Raises:
        InvalidImageError: If the image cannot be processed.
    """
    try:
        # Convert to tensor
        img = tf.convert_to_tensor(image_array)

        # Handle different input shapes
        if len(img.shape) == 2:
            # (H, W) -> (H, W, 1)
            img = tf.expand_dims(img, -1)
        elif len(img.shape) == 3 and img.shape[-1] == 3:
            # (H, W, 3) -> (H, W, 1) - convert RGB to grayscale
            img = tf.image.rgb_to_grayscale(img)
        elif len(img.shape) != 3 or img.shape[-1] != 1:
            raise InvalidImageError(
                f"Invalid image shape: {img.shape}. Expected (H, W), (H, W, 1), or (H, W, 3)"
            )

        # Ensure float32 and normalize to [0, 1]
        if img.dtype != tf.float32:
            img = tf.image.convert_image_dtype(img, tf.float32)

        # Resize to target dimensions
        img = tf.image.resize(img, [image_height, image_width])
        # Transpose: (height, width, channels) -> (width, height, channels)
        img = tf.transpose(img, perm=[1, 0, 2])
        return img
    except Exception as e:
        raise InvalidImageError(f"Failed to load image from array: {e}") from e


def encode_training_sample(
    img_path: str,
    label: str,
    char_to_num: tf.keras.layers.StringLookup,
    image_width: int,
    image_height: int,
) -> dict[str, tf.Tensor]:
    """Encode a single training sample (image + label).

    Args:
        img_path: Path to the image file.
        label: Text label for the captcha.
        char_to_num: StringLookup layer to convert characters to integers.
        image_width: Target width for the image.
        image_height: Target height for the image.

    Returns:
        Dictionary with "image" and "label" tensors.
    """
    # Load and preprocess image
    img = load_image_from_path(img_path, image_width, image_height)

    # Convert label characters to integers
    label_encoded = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))

    return {"image": img, "label": label_encoded}
