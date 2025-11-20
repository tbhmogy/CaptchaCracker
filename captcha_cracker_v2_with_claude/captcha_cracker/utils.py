"""Utility functions for CaptchaCracker."""

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import tensorflow as tf
from tensorflow import keras

from .config import LOG_FORMAT


def setup_logger(name: str = "captcha_cracker", level: int = logging.INFO) -> logging.Logger:
    """Set up a logger with the specified name and level.

    Args:
        name: Logger name.
        level: Logging level (default: INFO).

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding multiple handlers if logger already exists
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)

    return logger


def split_data(
    images: np.ndarray,
    labels: np.ndarray,
    train_size: float = 0.9,
    shuffle: bool = True,
    random_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into training and validation sets.

    Args:
        images: Array of image paths.
        labels: Array of corresponding labels.
        train_size: Fraction of data to use for training (default: 0.9).
        shuffle: Whether to shuffle the data before splitting (default: True).
        random_seed: Random seed for reproducibility (default: None).

    Returns:
        Tuple of (x_train, x_valid, y_train, y_valid).
    """
    # Get the total size of the dataset
    size = len(images)

    # Make an indices array and shuffle it if required
    indices = np.arange(size)
    if shuffle:
        if random_seed is not None:
            np.random.seed(random_seed)
        np.random.shuffle(indices)

    # Get the size of training samples
    train_samples = int(size * train_size)

    # Split data into training and validation sets
    x_train = images[indices[:train_samples]]
    y_train = labels[indices[:train_samples]]
    x_valid = images[indices[train_samples:]]
    y_valid = labels[indices[train_samples:]]

    return x_train, x_valid, y_train, y_valid


def decode_predictions(
    predictions: np.ndarray,
    num_to_char: tf.keras.layers.StringLookup,
    max_length: int,
) -> list[str]:
    """Decode CTC predictions to text strings.

    Args:
        predictions: Model predictions with shape (batch_size, time_steps, num_classes).
        num_to_char: StringLookup layer to convert integers back to characters.
        max_length: Maximum length of the output text.

    Returns:
        List of decoded text strings.
    """
    # Get the length of each prediction sequence
    input_len = np.ones(predictions.shape[0]) * predictions.shape[1]

    # Use greedy decoding (choose the most likely character at each time step)
    # Updated API: tf.nn.ctc_greedy_decoder instead of keras.backend.ctc_decode
    decoded, _ = tf.nn.ctc_greedy_decoder(
        inputs=tf.transpose(predictions, perm=[1, 0, 2]),  # (time, batch, classes)
        sequence_length=tf.cast(input_len, dtype=tf.int32),
        merge_repeated=True,
    )

    # Get the first (and only) result from the decoder
    decoded_dense = tf.sparse.to_dense(decoded[0], default_value=-1)[:, :max_length]

    # Convert integers back to characters
    output_text = []
    for res in decoded_dense:
        # Add 1 to skip the blank token (CTC uses 0 for blank)
        res = tf.strings.reduce_join(num_to_char(res + 1)).numpy().decode("utf-8")
        output_text.append(res)

    return output_text


def build_model(
    image_width: int,
    image_height: int,
    num_characters: int,
    conv_filters: Sequence[int] = (32, 64),
    lstm_units: Sequence[int] = (128, 64),
    dropout_rate: float = 0.2,
    lstm_dropout: float = 0.25,
) -> keras.Model:
    """Build the CTC-based OCR model architecture.

    Args:
        image_width: Input image width.
        image_height: Input image height.
        num_characters: Number of unique characters in the vocabulary.
        conv_filters: Number of filters for each convolutional layer (default: [32, 64]).
        lstm_units: Number of units for each LSTM layer (default: [128, 64]).
        dropout_rate: Dropout rate for dense layers (default: 0.2).
        lstm_dropout: Dropout rate for LSTM layers (default: 0.25).

    Returns:
        Compiled Keras model.
    """
    from tensorflow.keras import layers

    from .layers import CTCLayer

    # Input layers
    input_img = layers.Input(
        shape=(image_width, image_height, 1), name="image", dtype="float32"
    )
    labels = layers.Input(name="label", shape=(None,), dtype="float32")

    # Convolutional blocks
    x = input_img
    for i, filters in enumerate(conv_filters):
        x = layers.Conv2D(
            filters,
            (3, 3),
            activation="relu",
            kernel_initializer="he_normal",
            padding="same",
            name=f"Conv{i+1}",
        )(x)
        x = layers.MaxPooling2D((2, 2), name=f"pool{i+1}")(x)

    # Reshape for RNN
    # After 2 max pooling layers with stride 2, the feature maps are 4x smaller
    new_shape = ((image_width // 4), (image_height // 4) * conv_filters[-1])
    x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = layers.Dense(64, activation="relu", name="dense1")(x)
    x = layers.Dropout(dropout_rate)(x)

    # Bidirectional LSTM layers
    for i, units in enumerate(lstm_units):
        x = layers.Bidirectional(
            layers.LSTM(units, return_sequences=True, dropout=lstm_dropout),
            name=f"bilstm{i+1}",
        )(x)

    # Output layer
    # +1 for CTC blank token
    x = layers.Dense(num_characters + 1, activation="softmax", name="dense2")(x)

    # CTC loss layer
    output = CTCLayer(name="ctc_loss")(labels, x)

    # Define the model
    model = keras.models.Model(inputs=[input_img, labels], outputs=output, name="captcha_model")

    return model


def get_prediction_model(model: keras.Model) -> keras.Model:
    """Extract the prediction-only model from a training model.

    Args:
        model: The full training model with CTC loss layer.

    Returns:
        A model that takes an image input and returns predictions.
    """
    return keras.models.Model(
        model.get_layer(name="image").input, model.get_layer(name="dense2").output
    )
