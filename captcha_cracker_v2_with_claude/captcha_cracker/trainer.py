"""CaptchaTrainer class for training captcha recognition models."""

import glob
import logging
import os
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from . import config
from .exceptions import ConfigurationError, TrainingError
from .preprocessing import encode_training_sample
from .utils import build_model, setup_logger, split_data


class CaptchaTrainer:
    """Trainer for captcha recognition models.

    This class handles the complete training pipeline including data loading,
    preprocessing, model building, and training.

    Examples:
        Basic training:
        >>> trainer = CaptchaTrainer(image_size=(200, 50))
        >>> trainer.load_data(['data/train/'])
        >>> trainer.train(epochs=100)
        >>> trainer.save('weights.h5')

        With custom model configuration:
        >>> trainer = CaptchaTrainer(
        ...     image_size=(200, 50),
        ...     model_config={
        ...         'conv_filters': [32, 64, 128],
        ...         'lstm_units': [256, 128],
        ...         'dropout_rate': 0.3
        ...     }
        ... )
        >>> trainer.load_data(['data/train_1/', 'data/train_2/'])
        >>> trainer.train(
        ...     epochs=100,
        ...     batch_size=32,
        ...     validation_split=0.1,
        ...     callbacks=['tensorboard', 'checkpoint']
        ... )

    Attributes:
        image_width: Width of input images.
        image_height: Height of input images.
        images: List of image file paths.
        labels: List of corresponding labels.
        characters: Set of unique characters in the dataset.
        max_length: Maximum length of captcha text.
        model: The Keras model (None until built).
    """

    def __init__(
        self,
        image_size: tuple[int, int] = (config.DEFAULT_IMAGE_WIDTH, config.DEFAULT_IMAGE_HEIGHT),
        model_config: dict[str, Any] | None = None,
        log_level: int = logging.INFO,
    ) -> None:
        """Initialize CaptchaTrainer.

        Args:
            image_size: Tuple of (width, height) for input images.
            model_config: Optional dictionary with model architecture settings:
                - conv_filters: List of filter counts for conv layers (default: [32, 64])
                - lstm_units: List of unit counts for LSTM layers (default: [128, 64])
                - dropout_rate: Dropout rate for dense layers (default: 0.2)
                - lstm_dropout: Dropout rate for LSTM layers (default: 0.25)
            log_level: Logging level (default: INFO).
        """
        self.logger = setup_logger(level=log_level)

        # Image dimensions
        self.image_width, self.image_height = image_size

        # Model configuration
        self.model_config = model_config or {}
        self.conv_filters = self.model_config.get("conv_filters", config.DEFAULT_CONV_FILTERS)
        self.lstm_units = self.model_config.get("lstm_units", config.DEFAULT_LSTM_UNITS)
        self.dropout_rate = self.model_config.get("dropout_rate", config.DEFAULT_DROPOUT_RATE)
        self.lstm_dropout = self.model_config.get("lstm_dropout", config.DEFAULT_LSTM_DROPOUT)

        # Data (to be loaded)
        self.images: list[str] = []
        self.labels: list[str] = []
        self.characters: set[str] = set()
        self.max_length: int = 0

        # Character mapping layers (to be created after loading data)
        self.char_to_num: layers.StringLookup | None = None
        self.num_to_char: layers.StringLookup | None = None

        # Model (to be built)
        self.model: keras.Model | None = None

        self.logger.info("CaptchaTrainer initialized")
        self.logger.info(f"Image size: ({self.image_width}, {self.image_height})")
        self.logger.info(f"Model config: {self.model_config}")

    def load_data(
        self,
        data_paths: list[str | Path],
        pattern: str = "*.png",
    ) -> None:
        """Load training data from directories.

        The function expects image files with filenames that contain the captcha
        text (e.g., '023062.png' for captcha '023062').

        Args:
            data_paths: List of directory paths containing training images.
            pattern: Glob pattern for image files (default: '*.png').

        Raises:
            ConfigurationError: If no images are found or data is invalid.

        Examples:
            >>> trainer.load_data(['data/train/'])
            >>> trainer.load_data(['data/train_1/', 'data/train_2/'], pattern='*.jpg')
        """
        self.logger.info(f"Loading data from {len(data_paths)} directories")

        all_images = []
        for data_path in data_paths:
            path_str = str(data_path)
            # Find all images matching the pattern
            images = sorted(glob.glob(os.path.join(path_str, pattern)))
            all_images.extend(images)
            self.logger.info(f"  {path_str}: {len(images)} images")

        if not all_images:
            raise ConfigurationError(
                f"No images found in {data_paths} with pattern '{pattern}'"
            )

        # Extract labels from filenames
        # Assumes filename format: 'label.extension' (e.g., '023062.png')
        self.images = all_images
        self.labels = [
            Path(img).stem for img in self.images  # Get filename without extension
        ]

        # Find unique characters
        self.characters = set(char for label in self.labels for char in label)

        # Find maximum label length
        self.max_length = max(len(label) for label in self.labels)

        self.logger.info(f"Loaded {len(self.images)} images")
        self.logger.info(f"Characters: {sorted(self.characters)}")
        self.logger.info(f"Max length: {self.max_length}")

        # Create character mapping layers
        self.char_to_num = layers.StringLookup(
            vocabulary=sorted(self.characters), num_oov_indices=0, mask_token=None
        )
        self.num_to_char = layers.StringLookup(
            vocabulary=self.char_to_num.get_vocabulary(), mask_token=None, invert=True
        )

    def build_model(self) -> keras.Model:
        """Build the model architecture.

        Returns:
            The built Keras model.

        Raises:
            ConfigurationError: If data hasn't been loaded yet.
        """
        if not self.characters:
            raise ConfigurationError("No data loaded. Call load_data() first.")

        self.logger.info("Building model...")

        self.model = build_model(
            image_width=self.image_width,
            image_height=self.image_height,
            num_characters=len(self.characters),
            conv_filters=self.conv_filters,
            lstm_units=self.lstm_units,
            dropout_rate=self.dropout_rate,
            lstm_dropout=self.lstm_dropout,
        )

        # Compile the model
        optimizer = keras.optimizers.Adam()
        self.model.compile(optimizer=optimizer)

        self.logger.info("Model built successfully")
        self.logger.info(f"Total parameters: {self.model.count_params():,}")

        return self.model

    def train(
        self,
        epochs: int = config.DEFAULT_EPOCHS,
        batch_size: int = config.DEFAULT_BATCH_SIZE,
        validation_split: float = config.DEFAULT_VALIDATION_SPLIT,
        callbacks: list[str | keras.callbacks.Callback] | None = None,
        early_stopping: bool = False,
        early_stopping_patience: int = 10,
        random_seed: int | None = None,
    ) -> keras.callbacks.History:
        """Train the model.

        Args:
            epochs: Number of training epochs (default: 100).
            batch_size: Batch size for training (default: 16).
            validation_split: Fraction of data to use for validation (default: 0.1).
            callbacks: List of callback names or Callback objects. Supported names:
                - 'tensorboard': TensorBoard logging
                - 'checkpoint': Save best model weights
                - 'reduce_lr': Reduce learning rate on plateau
            early_stopping: Whether to use early stopping (default: False).
            early_stopping_patience: Patience for early stopping (default: 10).
            random_seed: Random seed for reproducibility (default: None).

        Returns:
            Training history object.

        Raises:
            ConfigurationError: If data or model hasn't been set up.
            TrainingError: If training fails.

        Examples:
            >>> history = trainer.train(epochs=100, batch_size=16)
            >>> history = trainer.train(
            ...     epochs=100,
            ...     callbacks=['tensorboard', 'checkpoint'],
            ...     early_stopping=True
            ... )
        """
        if not self.images or not self.labels:
            raise ConfigurationError("No data loaded. Call load_data() first.")

        if self.model is None:
            self.logger.info("Model not built yet. Building now...")
            self.build_model()

        assert self.model is not None
        assert self.char_to_num is not None

        self.logger.info(f"Starting training for {epochs} epochs")
        self.logger.info(f"Batch size: {batch_size}")
        self.logger.info(f"Validation split: {validation_split}")

        try:
            # Split data
            x_train, x_valid, y_train, y_valid = split_data(
                np.array(self.images),
                np.array(self.labels),
                train_size=1 - validation_split,
                shuffle=True,
                random_seed=random_seed,
            )

            self.logger.info(f"Training samples: {len(x_train)}")
            self.logger.info(f"Validation samples: {len(x_valid)}")

            # Create datasets
            train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
            train_dataset = (
                train_dataset.map(
                    lambda img_path, label: encode_training_sample(
                        img_path, label, self.char_to_num, self.image_width, self.image_height
                    ),
                    num_parallel_calls=tf.data.AUTOTUNE,  # Updated API
                )
                .batch(batch_size)
                .prefetch(buffer_size=tf.data.AUTOTUNE)  # Updated API
            )

            validation_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
            validation_dataset = (
                validation_dataset.map(
                    lambda img_path, label: encode_training_sample(
                        img_path, label, self.char_to_num, self.image_width, self.image_height
                    ),
                    num_parallel_calls=tf.data.AUTOTUNE,  # Updated API
                )
                .batch(batch_size)
                .prefetch(buffer_size=tf.data.AUTOTUNE)  # Updated API
            )

            # Set up callbacks
            callback_list = self._setup_callbacks(callbacks, early_stopping, early_stopping_patience)

            # Train the model
            history = self.model.fit(
                train_dataset,
                validation_data=validation_dataset,
                epochs=epochs,
                callbacks=callback_list,
            )

            self.logger.info("Training completed successfully")
            return history

        except Exception as e:
            raise TrainingError(f"Training failed: {e}") from e

    def _setup_callbacks(
        self,
        callbacks: list[str | keras.callbacks.Callback] | None,
        early_stopping: bool,
        early_stopping_patience: int,
    ) -> list[keras.callbacks.Callback]:
        """Set up training callbacks.

        Args:
            callbacks: List of callback names or objects.
            early_stopping: Whether to add early stopping.
            early_stopping_patience: Patience for early stopping.

        Returns:
            List of Callback objects.
        """
        callback_list: list[keras.callbacks.Callback] = []

        if callbacks:
            for cb in callbacks:
                if isinstance(cb, str):
                    # Create callback from name
                    if cb == "tensorboard":
                        callback_list.append(
                            keras.callbacks.TensorBoard(log_dir="./logs", histogram_freq=1)
                        )
                    elif cb == "checkpoint":
                        callback_list.append(
                            keras.callbacks.ModelCheckpoint(
                                filepath="best_model.keras",
                                save_best_only=True,
                                monitor="val_loss",
                                verbose=1,
                            )
                        )
                    elif cb == "reduce_lr":
                        callback_list.append(
                            keras.callbacks.ReduceLROnPlateau(
                                monitor="val_loss", factor=0.5, patience=5, verbose=1
                            )
                        )
                    else:
                        self.logger.warning(f"Unknown callback name: {cb}")
                else:
                    # Add custom callback object
                    callback_list.append(cb)

        # Add early stopping if requested
        if early_stopping:
            callback_list.append(
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=early_stopping_patience,
                    restore_best_weights=True,
                    verbose=1,
                )
            )

        return callback_list

    def save(self, path: str | Path, save_format: Literal["h5", "keras"] = "h5") -> None:
        """Save the trained model weights.

        Args:
            path: Path to save the model weights.
            save_format: Format to save the model ('h5' or 'keras').

        Raises:
            ConfigurationError: If model hasn't been built.
            TrainingError: If saving fails.

        Examples:
            >>> trainer.save('weights.h5')
            >>> trainer.save('model.keras', save_format='keras')
        """
        if self.model is None:
            raise ConfigurationError("No model to save. Train the model first.")

        try:
            if save_format == "h5":
                self.model.save_weights(str(path))
            else:
                self.model.save(str(path))

            self.logger.info(f"Model saved to {path}")

        except Exception as e:
            raise TrainingError(f"Failed to save model: {e}") from e

    def evaluate(
        self, test_images: list[str], test_labels: list[str], batch_size: int = 16
    ) -> dict[str, float]:
        """Evaluate the model on test data.

        Args:
            test_images: List of test image paths.
            test_labels: List of corresponding labels.
            batch_size: Batch size for evaluation.

        Returns:
            Dictionary with evaluation metrics.

        Raises:
            ConfigurationError: If model hasn't been built.
        """
        if self.model is None:
            raise ConfigurationError("No model to evaluate. Train the model first.")

        assert self.char_to_num is not None

        # Create test dataset
        test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
        test_dataset = (
            test_dataset.map(
                lambda img_path, label: encode_training_sample(
                    img_path, label, self.char_to_num, self.image_width, self.image_height
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            .batch(batch_size)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )

        # Evaluate
        loss = self.model.evaluate(test_dataset)

        return {"loss": loss}

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"CaptchaTrainer(image_size=({self.image_width}, {self.image_height}), "
            f"num_samples={len(self.images)}, num_characters={len(self.characters)})"
        )
