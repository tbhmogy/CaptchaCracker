# Migration Guide: v0.0.7 → v1.0.0

This guide helps you migrate from CaptchaCracker v0.0.7 to v1.0.0.

## Overview

Version 1.0.0 is a **major breaking release** with a complete API redesign. The focus is on modernizing the codebase with:

- Python 3.12+ and TensorFlow 2.18+
- Type-safe, intuitive API
- Better error handling and logging
- Comprehensive documentation

## Breaking Changes Summary

| Aspect | v0.0.7 | v1.0.0 |
|--------|--------|--------|
| Python | 3.8.13 | 3.12+ |
| TensorFlow | 2.5.0 | 2.18.0+ |
| Package Name | `CaptchaCracker` | `captcha_cracker` |
| Main Class | `ApplyModel` | `CaptchaModel` |
| Training Class | `CreateModel` | `CaptchaTrainer` |
| Import Path | `from CaptchaCracker import` | `from captcha_cracker import` |

## Step-by-Step Migration

### 1. Update Python and Dependencies

#### Old Environment (v0.0.7)
```bash
python 3.8.13
tensorflow==2.5.0
pillow==9.5.0
numpy==1.19.5
```

#### New Environment (v1.0.0)
```bash
# Install Python 3.12
pyenv install 3.12.7
pyenv local 3.12.7

# Create new virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install new version
pip install -r requirements.txt
```

### 2. Update Import Statements

#### Old (v0.0.7)
```python
from CaptchaCracker import ApplyModel, CreateModel
```

#### New (v1.0.0)
```python
from captcha_cracker import CaptchaModel, CaptchaTrainer
```

### 3. Update Model Loading

#### Old (v0.0.7)
```python
# Constructor takes path and image size
model = ApplyModel('model/weights_v2.h5', (200, 50))
```

#### New (v1.0.0)
```python
# Use factory method .load()
model = CaptchaModel.load('model/weights_v2.h5', image_size=(200, 50))

# Or with default size (200, 50)
model = CaptchaModel.load('model/weights_v2.h5')
```

### 4. Update Prediction Code

#### Basic Prediction

**Old (v0.0.7)**
```python
result = model.predict('captcha.png')
print(result)  # '023062'
```

**New (v1.0.0)**
```python
# Same API for basic usage
result = model.predict('captcha.png')
print(result)  # '023062'

# Or get confidence score
result = model.predict('captcha.png', return_confidence=True)
print(f"Text: {result.text}, Confidence: {result.confidence}")
```

#### Predict from Bytes

**Old (v0.0.7)**
```python
result = model.predict_from_bytes(image_bytes)
```

**New (v1.0.0)**
```python
# Option 1: Use specific method (recommended)
result = model.predict_bytes(image_bytes)

# Option 2: Use general method
result = model.predict(image_bytes)
```

#### NEW: Batch Prediction

Not available in v0.0.7. New feature in v1.0.0:

```python
# Predict multiple images efficiently
results = model.predict_batch(['img1.png', 'img2.png', 'img3.png'])
for result in results:
    print(result)
```

#### NEW: Context Manager

Not available in v0.0.7. New feature in v1.0.0:

```python
# Automatic resource management
with CaptchaModel.load('weights.h5') as model:
    result = model.predict('captcha.png')
    print(result)
```

### 5. Update Training Code

#### Old (v0.0.7)
```python
from CaptchaCracker import CreateModel

# Initialize with image paths
train_images = sorted(glob.glob('data/train/*/*.png'))
model = CreateModel(train_images, img_width=200, img_height=50)

# Train
trained_model = model.train_model(epochs=100, earlystopping=True)

# Save
trained_model.save_weights('model/weights.h5')
```

#### New (v1.0.0)
```python
from captcha_cracker import CaptchaTrainer

# Initialize trainer
trainer = CaptchaTrainer(image_size=(200, 50))

# Load data from directories (more convenient)
trainer.load_data(['data/train/'])

# Train with more options
trainer.train(
    epochs=100,
    batch_size=16,
    validation_split=0.1,
    early_stopping=True,
    early_stopping_patience=10,
    callbacks=['tensorboard', 'checkpoint']
)

# Save (simpler)
trainer.save('model/weights.h5')
```

### 6. Update Error Handling

#### Old (v0.0.7)
```python
# Generic exceptions
try:
    result = model.predict('image.png')
except Exception as e:
    print(f"Error: {e}")
```

#### New (v1.0.0)
```python
# Specific exception types
from captcha_cracker import InvalidImageError, PredictionError

try:
    result = model.predict('image.png')
except InvalidImageError as e:
    print(f"Invalid image: {e}")
except PredictionError as e:
    print(f"Prediction failed: {e}")
```

### 7. Update Configuration

#### Old (v0.0.7)
```python
# Hard-coded values in code
model = ApplyModel(
    'weights.h5',
    (200, 50),
    max_length=6,
    characters={'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}
)
```

#### New (v1.0.0)
```python
# Use default config (same values)
model = CaptchaModel.load('weights.h5')

# Or customize
model = CaptchaModel.load(
    'weights.h5',
    image_size=(200, 50),
    max_length=6,
    characters={'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}
)
```

## Complete Migration Examples

### Example 1: Simple Prediction Script

**Before (v0.0.7)**
```python
from CaptchaCracker import ApplyModel

model = ApplyModel('model/weights_v2.h5', (200, 50))
result = model.predict('test.png')
print(result)
```

**After (v1.0.0)**
```python
from captcha_cracker import CaptchaModel

model = CaptchaModel.load('model/weights_v2.h5')
result = model.predict('test.png')
print(result)
```

### Example 2: Web Service

**Before (v0.0.7)**
```python
from flask import Flask, request
from CaptchaCracker import ApplyModel

app = Flask(__name__)
model = ApplyModel('weights.h5', (200, 50))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        image_bytes = request.files['image'].read()
        result = model.predict_from_bytes(image_bytes)
        return {'text': result}
    except Exception as e:
        return {'error': str(e)}, 500
```

**After (v1.0.0)**
```python
from flask import Flask, request
from captcha_cracker import CaptchaModel, InvalidImageError, PredictionError

app = Flask(__name__)
model = CaptchaModel.load('weights.h5')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        image_bytes = request.files['image'].read()
        result = model.predict_bytes(image_bytes, return_confidence=True)
        return {
            'text': result.text,
            'confidence': result.confidence
        }
    except InvalidImageError as e:
        return {'error': f'Invalid image: {e}'}, 400
    except PredictionError as e:
        return {'error': f'Prediction failed: {e}'}, 500
```

### Example 3: Training Script

**Before (v0.0.7)**
```python
import glob
from CaptchaCracker import CreateModel

# Collect training images
train_images = sorted(glob.glob('data/train/*/*.png'))

# Create and train
model = CreateModel(train_images, img_width=200, img_height=50)
trained_model = model.train_model(epochs=100, earlystopping=True)

# Save
trained_model.save_weights('weights.h5')
```

**After (v1.0.0)**
```python
from captcha_cracker import CaptchaTrainer

# Initialize trainer
trainer = CaptchaTrainer(image_size=(200, 50))

# Load data (much simpler)
trainer.load_data(['data/train/'])

# Train with more control
trainer.train(
    epochs=100,
    batch_size=16,
    validation_split=0.1,
    early_stopping=True,
    callbacks=['tensorboard', 'checkpoint']
)

# Save
trainer.save('weights.h5')
```

## Type Hints Migration

If you use type hints in your code, update them:

**Before (v0.0.7)**
```python
# No type hints in v0.0.7
def process_captcha(model, image_path):
    return model.predict(image_path)
```

**After (v1.0.0)**
```python
from pathlib import Path
from captcha_cracker import CaptchaModel

def process_captcha(model: CaptchaModel, image_path: str | Path) -> str:
    return model.predict(image_path)
```

## Testing Migration

Update your tests to use the new API:

**Before (v0.0.7)**
```python
import unittest
from CaptchaCracker import ApplyModel

class TestModel(unittest.TestCase):
    def test_predict(self):
        model = ApplyModel('weights.h5', (200, 50))
        result = model.predict('test.png')
        self.assertIsInstance(result, str)
```

**After (v1.0.0)**
```python
import pytest
from captcha_cracker import CaptchaModel

def test_predict():
    model = CaptchaModel.load('weights.h5')
    result = model.predict('test.png')
    assert isinstance(result, str)

def test_predict_with_confidence():
    model = CaptchaModel.load('weights.h5')
    result = model.predict('test.png', return_confidence=True)
    assert hasattr(result, 'text')
    assert hasattr(result, 'confidence')
```

## Common Issues and Solutions

### Issue 1: Model Weights Won't Load

**Problem**: Old `.h5` files may have compatibility issues with TensorFlow 2.18.

**Solution**:
```python
# Try loading with explicit configuration
model = CaptchaModel.load(
    'old_weights.h5',
    image_size=(200, 50),
    max_length=6,
    characters={'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}
)

# If that fails, you may need to retrain the model
```

### Issue 2: Import Errors

**Problem**: `ModuleNotFoundError: No module named 'CaptchaCracker'`

**Solution**: Update import statements from `CaptchaCracker` to `captcha_cracker` (lowercase, underscore).

### Issue 3: Different Predictions

**Problem**: Same image gives different predictions in v1.0.0.

**Solution**: This could be due to:
- Different preprocessing (unlikely, but check image sizes)
- Random seed differences (set `random_seed` if needed)
- Model weights need retraining with new TensorFlow version

## Getting Help

If you encounter issues during migration:

1. Check the [README.md](README.md) for API documentation
2. Look at the [examples/](examples/) directory for working code
3. Open an issue on GitHub with:
   - Your v0.0.7 code
   - What you tried in v1.0.0
   - Full error message

## Rollback Plan

If you need to rollback to v0.0.7:

```bash
# Reinstall old dependencies
pip install tensorflow==2.5.0 pillow==9.5.0 numpy==1.19.5

# Use old code
git checkout v0.0.7
```

## Benefits of Migration

After migration, you'll have:

✅ Modern Python 3.12 and TensorFlow 2.18
✅ Better error messages and debugging
✅ Type hints for IDE autocomplete
✅ Batch prediction for better performance
✅ Confidence scores for predictions
✅ Cleaner, more maintainable code
✅ Security updates (fixed Pillow CVEs)
✅ Better documentation
✅ Ready for future improvements (v1.1+)

## What's Next?

After migrating to v1.0.0, check out [FUTURE_IMPROVEMENTS.md](../FUTURE_IMPROVEMENTS.md) for planned features in upcoming releases:

- Data augmentation (v1.1)
- Model quantization (v1.2)
- Multi-language support (v2.0)
- And more...
