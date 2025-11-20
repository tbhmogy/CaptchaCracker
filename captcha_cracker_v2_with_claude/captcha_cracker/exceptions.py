"""Custom exceptions for CaptchaCracker."""


class CaptchaCrackerError(Exception):
    """Base exception for CaptchaCracker."""

    pass


class ModelLoadError(CaptchaCrackerError):
    """Failed to load model weights."""

    pass


class InvalidImageError(CaptchaCrackerError):
    """Invalid image format or size."""

    pass


class PredictionError(CaptchaCrackerError):
    """Failed to predict captcha."""

    pass


class TrainingError(CaptchaCrackerError):
    """Failed to train model."""

    pass


class ConfigurationError(CaptchaCrackerError):
    """Invalid configuration."""

    pass
