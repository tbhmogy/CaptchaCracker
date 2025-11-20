# CaptchaCracker v1.0.0

Modern OCR library for captcha recognition using deep learning with TensorFlow 2.18+ and Python 3.12+.

## Features

- **Modern API**: Clean, intuitive API with type hints and docstrings
- **Easy to Use**: Load a model and predict in 2 lines of code
- **Flexible**: Support for file paths, bytes, and numpy arrays
- **Batch Processing**: Efficient batch prediction for multiple images
- **Context Manager**: Automatic resource management
- **Type Safe**: Full type annotations for better IDE support
- **Well Tested**: Comprehensive test suite with pytest
- **Production Ready**: Proper error handling and logging

## Installation

### Requirements
- Python 3.12 or higher
- TensorFlow 2.18.0 or higher

### Install from source

```bash
git clone <repository-url>
cd captcha_cracker_v2
pip install -e .
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

## Quick Start

### Prediction

```python
from captcha_cracker import CaptchaModel

# Load the model
model = CaptchaModel.load('model/weights_v2.h5')

# Predict from a file
result = model.predict('captcha.png')
print(result)  # '023062'

# Predict from bytes
with open('captcha.png', 'rb') as f:
    result = model.predict_bytes(f.read())

# Batch prediction
results = model.predict_batch(['img1.png', 'img2.png', 'img3.png'])
for result in results:
    print(result)
```

### With Context Manager

```python
with CaptchaModel.load('model/weights_v2.h5') as model:
    result = model.predict('captcha.png')
    print(result)
```

### With Confidence Scores

```python
result = model.predict('captcha.png', return_confidence=True)
print(f"Text: {result.text}")
print(f"Confidence: {result.confidence:.2f}")
```

### Training a New Model

```python
from captcha_cracker import CaptchaTrainer

# Initialize trainer
trainer = CaptchaTrainer(
    image_size=(200, 50),
    model_config={
        'conv_filters': [32, 64],
        'lstm_units': [128, 64],
        'dropout_rate': 0.2
    }
)

# Load training data
trainer.load_data(['data/train_numbers_only/'])

# Train the model
trainer.train(
    epochs=100,
    batch_size=16,
    validation_split=0.1,
    callbacks=['tensorboard', 'checkpoint'],
    early_stopping=True
)

# Save the model
trainer.save('model/weights_new.h5')
```

## API Reference

### CaptchaModel

Main class for loading pre-trained models and making predictions.

#### Methods

- `load(weights_path, image_size=(200, 50), max_length=6, characters=None, log_level=INFO)`: Load a pre-trained model
- `predict(image, return_confidence=False)`: Predict text from an image
- `predict_batch(images, return_confidence=False, batch_size=32)`: Batch prediction
- `predict_bytes(image_bytes, return_confidence=False)`: Predict from bytes
- `predict_array(image_array, return_confidence=False)`: Predict from numpy array

### CaptchaTrainer

Class for training new captcha recognition models.

#### Methods

- `__init__(image_size=(200, 50), model_config=None, log_level=INFO)`: Initialize trainer
- `load_data(data_paths, pattern='*.png')`: Load training data from directories
- `build_model()`: Build the model architecture
- `train(epochs=100, batch_size=16, validation_split=0.1, callbacks=None, early_stopping=False)`: Train the model
- `save(path, save_format='h5')`: Save the trained model
- `evaluate(test_images, test_labels, batch_size=16)`: Evaluate on test data

### Exceptions

- `CaptchaCrackerError`: Base exception
- `ModelLoadError`: Failed to load model
- `InvalidImageError`: Invalid image format or size
- `PredictionError`: Prediction failed
- `TrainingError`: Training failed
- `ConfigurationError`: Invalid configuration

## Migration from v0.x

### API Changes

| Old API (v0.0.7) | New API (v1.0.0) |
|------------------|------------------|
| `from CaptchaCracker import ApplyModel` | `from captcha_cracker import CaptchaModel` |
| `model = ApplyModel('weights.h5', (200, 50))` | `model = CaptchaModel.load('weights.h5')` |
| `result = model.predict('img.png')` | `result = model.predict('img.png')` |
| `result = model.predict_from_bytes(bytes)` | `result = model.predict_bytes(bytes)` |
| N/A | `results = model.predict_batch([...])` |
| N/A | `with CaptchaModel.load(...) as model:` |

### Key Improvements

1. **Class Names**: `ApplyModel` → `CaptchaModel` (more intuitive)
2. **Factory Pattern**: Use `.load()` class method instead of constructor
3. **Context Manager**: Support for `with` statement
4. **Batch Prediction**: New `predict_batch()` method
5. **Type Hints**: Full type annotations for all APIs
6. **Better Errors**: Clear exception types and messages
7. **Logging**: Proper logging instead of print statements

## Architecture

The model uses a CNN-RNN architecture with CTC loss:

1. **CNN Layers**: Extract visual features (32 and 64 filters)
2. **Bidirectional LSTM**: Process sequential information (128 and 64 units)
3. **CTC Loss**: Handle variable-length sequences without explicit alignment

```
Input (200x50x1)
  → Conv2D(32) → MaxPool
  → Conv2D(64) → MaxPool
  → Reshape
  → Dense(64) → Dropout
  → BiLSTM(128)
  → BiLSTM(64)
  → Dense(num_chars+1)
  → CTC Loss
```

## Development

### Setup Development Environment

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Run tests with coverage
pytest --cov=captcha_cracker --cov-report=html

# Format code
black captcha_cracker/
isort captcha_cracker/

# Type checking
mypy captcha_cracker/

# Linting
flake8 captcha_cracker/
pylint captcha_cracker/
```

### Project Structure

```
captcha_cracker_v2/
├── captcha_cracker/          # Main package
│   ├── __init__.py          # Public API exports
│   ├── model.py             # CaptchaModel class
│   ├── trainer.py           # CaptchaTrainer class
│   ├── layers.py            # CTCLayer
│   ├── preprocessing.py     # Image preprocessing
│   ├── utils.py             # Helper functions
│   ├── exceptions.py        # Custom exceptions
│   └── config.py            # Configuration constants
├── tests/                    # Test suite
│   ├── test_model.py
│   ├── test_trainer.py
│   └── ...
├── examples/                 # Example scripts
├── pyproject.toml           # Project configuration
├── requirements.txt         # Production dependencies
├── requirements-dev.txt     # Development dependencies
└── README.md               # This file
```

## Performance

| Metric | Value |
|--------|-------|
| Accuracy | 92%+ (on test set) |
| Inference Speed (CPU) | ~50ms per image |
| Inference Speed (GPU) | ~10ms per image |
| Model Size | ~20MB |

## Requirements

- Python >= 3.12
- TensorFlow >= 2.18.0, < 2.19.0
- Pillow >= 10.4.0
- NumPy >= 1.26.0, < 2.0.0

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Changelog

### [1.0.0] - 2025

#### Breaking Changes
- Python 3.12+ required (was 3.8)
- TensorFlow 2.18+ required (was 2.5)
- Complete API redesign (not backward compatible)
- Package name changed: `CaptchaCracker` → `captcha_cracker`

#### Added
- New `CaptchaModel` API with factory pattern
- Batch prediction support
- Context manager support
- Confidence scores in predictions
- Full type hints and docstrings
- Custom exception types
- Proper logging system
- Comprehensive test suite

#### Changed
- Replaced all deprecated TensorFlow APIs
- Modernized code structure
- Improved error handling
- Better documentation

#### Fixed
- Security vulnerabilities (Pillow upgrade)
- Memory leaks
- Edge cases in image preprocessing

## Support

For issues and questions, please open an issue on GitHub.

## Acknowledgments

Built with TensorFlow and Keras.
