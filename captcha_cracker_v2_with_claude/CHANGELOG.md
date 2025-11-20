# Changelog

All notable changes to CaptchaCracker will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-11-20

### Breaking Changes

- **Python 3.12+ Required**: Dropped support for Python 3.8-3.11
- **TensorFlow 2.18+ Required**: Upgraded from TensorFlow 2.5.0
- **Complete API Redesign**: New API is not backward compatible with v0.x
- **Package Name Change**: `CaptchaCracker` → `captcha_cracker` (snake_case)
- **Class Rename**: `ApplyModel` → `CaptchaModel`
- **Class Rename**: `CreateModel` → `CaptchaTrainer`

### Added

#### New Features
- **Factory Pattern**: New `.load()` class method for `CaptchaModel`
- **Context Manager Support**: Use `with CaptchaModel.load(...) as model:`
- **Batch Prediction**: New `predict_batch()` method for efficient multi-image processing
- **Confidence Scores**: Return prediction confidence with `return_confidence=True`
- **Multiple Input Types**: Support for file paths, bytes, and numpy arrays
- **Flexible API**: Separate methods for different input types (`predict`, `predict_bytes`, `predict_array`)

#### Code Quality
- **Full Type Hints**: Complete type annotations for all public APIs
- **Comprehensive Docstrings**: Google-style docstrings for all classes and methods
- **Custom Exceptions**: Specific exception types for different error cases
  - `ModelLoadError`
  - `InvalidImageError`
  - `PredictionError`
  - `TrainingError`
  - `ConfigurationError`
- **Proper Logging**: Structured logging system replacing print statements
- **Test Suite**: pytest-based tests with fixtures and parametrization

#### Documentation
- **Detailed README**: Comprehensive documentation with examples
- **Migration Guide**: Step-by-step guide for upgrading from v0.x
- **Example Scripts**: Ready-to-use examples for common use cases
- **API Reference**: Complete API documentation in docstrings

#### Developer Experience
- **pyproject.toml**: Modern Python packaging configuration
- **Development Dependencies**: Organized dev requirements (pytest, black, mypy, etc.)
- **Code Formatting**: Black and isort configuration
- **Type Checking**: mypy configuration with strict mode
- **CI/CD Ready**: Configuration for automated testing

### Changed

#### Core Updates
- **Deprecated API Replacement**: All deprecated TensorFlow APIs updated
  - `layers.experimental.preprocessing.StringLookup` → `layers.StringLookup`
  - `tf.data.experimental.AUTOTUNE` → `tf.data.AUTOTUNE`
  - `keras.backend.ctc_batch_cost` → `tf.nn.ctc_loss`
  - `keras.backend.ctc_decode` → `tf.nn.ctc_greedy_decoder`

#### Architecture
- **Modular Design**: Split monolithic `core.py` into focused modules:
  - `model.py`: Model inference
  - `trainer.py`: Model training
  - `layers.py`: Custom layers
  - `preprocessing.py`: Image processing
  - `utils.py`: Helper functions
  - `exceptions.py`: Exception definitions
  - `config.py`: Configuration constants

#### Code Improvements
- **No Code Duplication**: Eliminated 114 lines of duplicate `build_model()` code
- **Dead Code Removal**: Removed unused `split_data()` method from `ApplyModel`
- **Configuration Management**: Centralized constants in `config.py`
- **Better Error Handling**: Specific exceptions with helpful error messages
- **Resource Management**: Proper cleanup with context managers

### Fixed

#### Security
- **Pillow Vulnerability**: Upgraded Pillow 9.5.0 → 10.4.0 (fixed known CVEs)
- **Dependency Vulnerabilities**: Updated all dependencies to latest secure versions

#### Bugs
- **Memory Leaks**: Fixed potential memory leaks in model loading
- **Image Processing**: Improved edge case handling in preprocessing
- **Error Messages**: More descriptive error messages for debugging

#### Compatibility
- **TensorFlow 2.18 Support**: Full compatibility with latest TensorFlow
- **Python 3.12 Support**: Tested and verified on Python 3.12

### Removed

- **Python 3.8 Support**: EOL as of October 2024
- **Deprecated APIs**: All legacy TensorFlow/Keras APIs
- **Print Statements**: Replaced with proper logging
- **Global Variables**: Moved to configuration module

## [0.0.7] - 2025-03-27 (Legacy)

Last version before major refactoring.

### Features
- Basic captcha recognition with CTC
- Support for numeric captchas (0-9)
- Simple training script
- File and bytes prediction

### Known Issues
- Uses deprecated TensorFlow 2.5 APIs
- Python 3.8 (EOL)
- No type hints
- Limited error handling
- Code duplication
- No tests

---

## Migration Guide

For detailed migration instructions, see [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md).

### Quick Migration Example

```python
# Old API (v0.0.7)
from CaptchaCracker import ApplyModel
model = ApplyModel('weights.h5', (200, 50))
result = model.predict('captcha.png')

# New API (v1.0.0)
from captcha_cracker import CaptchaModel
model = CaptchaModel.load('weights.h5')
result = model.predict('captcha.png')
```

## Future Plans

See [FUTURE_IMPROVEMENTS.md](../FUTURE_IMPROVEMENTS.md) for planned features in v1.1+:

- Data augmentation
- Attention mechanisms
- Model quantization
- ONNX export
- Multi-language support
- Web API
- And more...

---

**Note**: Version 1.0.0 focuses on modernizing the codebase and API.
Performance improvements and new features are planned for future releases.
