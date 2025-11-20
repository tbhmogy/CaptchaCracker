"""Custom layers for CaptchaCracker models."""

from typing import Any

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class CTCLayer(layers.Layer):
    """CTC (Connectionist Temporal Classification) loss layer.

    This layer computes the CTC loss during training and returns predictions
    during inference.

    The CTC loss is used for sequence-to-sequence learning where the alignment
    between input and output is unknown. It's commonly used in OCR tasks.

    Examples:
        >>> ctc_layer = CTCLayer(name="ctc_loss")
        >>> output = ctc_layer(y_true, y_pred)

    Attributes:
        loss_fn: The CTC loss function from TensorFlow.
    """

    def __init__(self, name: str | None = None, **kwargs: Any) -> None:
        """Initialize the CTCLayer.

        Args:
            name: Optional name for the layer.
            **kwargs: Additional keyword arguments for the base Layer class.
        """
        super().__init__(name=name, **kwargs)
        # Use the updated TensorFlow 2.18 API instead of deprecated keras.backend
        self.loss_fn = tf.nn.ctc_loss

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Compute CTC loss and add it to the layer.

        During training, this computes the CTC loss and adds it to the model's
        total loss. During inference, it simply returns the predictions.

        Args:
            y_true: Ground truth labels. Shape: (batch_size, max_label_length)
            y_pred: Predicted logits. Shape: (batch_size, time_steps, num_classes)

        Returns:
            The predictions (y_pred) unchanged.
        """
        # Get batch size and sequence lengths
        batch_len = tf.cast(tf.shape(y_true)[0], dtype=tf.int32)
        input_length = tf.cast(tf.shape(y_pred)[1], dtype=tf.int32)
        label_length = tf.cast(tf.shape(y_true)[1], dtype=tf.int32)

        # Create tensors of the same length as batch for input and label lengths
        input_length = input_length * tf.ones(shape=(batch_len,), dtype=tf.int32)
        label_length = label_length * tf.ones(shape=(batch_len,), dtype=tf.int32)

        # Compute CTC loss
        # Note: TensorFlow 2.18 uses tf.nn.ctc_loss with different parameter order
        loss = self.loss_fn(
            labels=y_true,
            logits=y_pred,
            label_length=label_length,
            logit_length=input_length,
            logits_time_major=False,
            blank_index=-1,
        )

        # Add the loss to the layer
        self.add_loss(tf.reduce_mean(loss))

        # At test time, just return the computed predictions
        return y_pred

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration for serialization.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        return config
