"""Tests for utility functions."""

import numpy as np
import pytest
import tensorflow as tf

from captcha_cracker.utils import build_model, split_data


class TestUtils:
    """Test suite for utility functions."""

    def test_split_data_basic(self):
        """Test basic data splitting."""
        images = np.array(["img1", "img2", "img3", "img4", "img5"])
        labels = np.array(["001", "002", "003", "004", "005"])

        x_train, x_valid, y_train, y_valid = split_data(
            images, labels, train_size=0.8, shuffle=False
        )

        assert len(x_train) == 4
        assert len(x_valid) == 1
        assert len(y_train) == 4
        assert len(y_valid) == 1

    def test_split_data_with_shuffle(self):
        """Test data splitting with shuffling."""
        images = np.array([f"img{i}" for i in range(100)])
        labels = np.array([f"{i:03d}" for i in range(100)])

        x_train, x_valid, y_train, y_valid = split_data(
            images, labels, train_size=0.9, shuffle=True, random_seed=42
        )

        assert len(x_train) == 90
        assert len(x_valid) == 10
        # With shuffle, order should be different
        assert not np.array_equal(x_train[:10], images[:10])

    def test_split_data_reproducibility(self):
        """Test that random_seed gives reproducible splits."""
        images = np.array([f"img{i}" for i in range(100)])
        labels = np.array([f"{i:03d}" for i in range(100)])

        x_train1, _, _, _ = split_data(
            images, labels, train_size=0.9, shuffle=True, random_seed=42
        )
        x_train2, _, _, _ = split_data(
            images, labels, train_size=0.9, shuffle=True, random_seed=42
        )

        assert np.array_equal(x_train1, x_train2)

    def test_build_model_basic(self, sample_characters):
        """Test building a model with default parameters."""
        model = build_model(
            image_width=200, image_height=50, num_characters=len(sample_characters)
        )

        assert model is not None
        assert len(model.inputs) == 2  # image and label
        assert len(model.outputs) == 1

    def test_build_model_custom_config(self, sample_characters):
        """Test building a model with custom configuration."""
        model = build_model(
            image_width=200,
            image_height=50,
            num_characters=len(sample_characters),
            conv_filters=[16, 32],
            lstm_units=[64, 32],
            dropout_rate=0.3,
            lstm_dropout=0.2,
        )

        assert model is not None
        # Check that model has expected number of layers
        assert len(model.layers) > 0
