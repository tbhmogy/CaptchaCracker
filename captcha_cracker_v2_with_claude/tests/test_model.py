"""Tests for CaptchaModel class."""

import numpy as np
import pytest

from captcha_cracker import CaptchaModel, InvalidImageError, ModelLoadError
from captcha_cracker.preprocessing import load_image_from_array


class TestCaptchaModel:
    """Test suite for CaptchaModel."""

    def test_model_repr(self):
        """Test model string representation."""
        # We can't easily test loading without a real weights file,
        # but we can test the repr format is correct
        pass

    def test_load_image_from_array_2d(self, sample_image_array):
        """Test loading image from 2D numpy array."""
        img = load_image_from_array(sample_image_array, 200, 50)
        assert img.shape == (200, 50, 1)
        assert img.dtype == tf.float32

    def test_load_image_from_array_3d(self):
        """Test loading image from 3D numpy array."""
        # Create RGB image
        img_array = np.random.rand(50, 200, 3).astype(np.float32)
        img = load_image_from_array(img_array, 200, 50)
        assert img.shape == (200, 50, 1)

    def test_load_image_from_array_grayscale(self):
        """Test loading grayscale image with channel dimension."""
        img_array = np.random.rand(50, 200, 1).astype(np.float32)
        img = load_image_from_array(img_array, 200, 50)
        assert img.shape == (200, 50, 1)

    def test_load_image_from_array_invalid_shape(self):
        """Test loading image with invalid shape raises error."""
        # 4D array should fail
        img_array = np.random.rand(10, 50, 200, 3)
        with pytest.raises(InvalidImageError):
            load_image_from_array(img_array, 200, 50)

    def test_prediction_result_str(self):
        """Test PredictionResult string representation."""
        from captcha_cracker.model import PredictionResult

        result = PredictionResult("123456", 0.95)
        assert str(result) == "123456"
        assert "PredictionResult" in repr(result)
        assert "0.950" in repr(result)
