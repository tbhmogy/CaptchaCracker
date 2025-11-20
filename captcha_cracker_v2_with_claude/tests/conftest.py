"""Pytest configuration and fixtures."""

import numpy as np
import pytest
import tensorflow as tf


@pytest.fixture
def sample_image_array():
    """Create a sample image array for testing."""
    # Create a random grayscale image (50, 200)
    return np.random.rand(50, 200).astype(np.float32)


@pytest.fixture
def sample_image_tensor():
    """Create a sample image tensor for testing."""
    # Create a random image tensor (200, 50, 1)
    return tf.random.uniform((200, 50, 1), dtype=tf.float32)


@pytest.fixture
def sample_characters():
    """Sample character set for testing."""
    return {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}


@pytest.fixture
def sample_labels():
    """Sample labels for testing."""
    return ["012345", "123456", "234567", "345678"]
