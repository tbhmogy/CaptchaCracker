"""Configuration and constants for CaptchaCracker."""

from typing import Set

# Default image dimensions
DEFAULT_IMAGE_WIDTH = 200
DEFAULT_IMAGE_HEIGHT = 50

# Default character set (digits only)
DEFAULT_CHARACTERS: Set[str] = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}

# Default max length of captcha text
DEFAULT_MAX_LENGTH = 6

# Model architecture defaults
DEFAULT_CONV_FILTERS = [32, 64]
DEFAULT_LSTM_UNITS = [128, 64]
DEFAULT_DROPOUT_RATE = 0.2
DEFAULT_LSTM_DROPOUT = 0.25

# Training defaults
DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 100
DEFAULT_VALIDATION_SPLIT = 0.1
DEFAULT_TRAIN_SPLIT = 0.9

# Downsampling factor (Conv: 2, Pooling: 2)
DOWNSAMPLE_FACTOR = 4

# Logging
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
