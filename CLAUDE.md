# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CaptchaCracker is a Python library for CAPTCHA image recognition using deep learning. The core architecture uses a CNN-RNN hybrid model with CTC (Connectionist Temporal Classification) loss to recognize digit sequences in CAPTCHA images. The model processes grayscale images through convolutional layers, then uses bidirectional LSTM layers for sequence recognition.

## Development Commands

### Environment Setup
```bash
# Install dependencies (TensorFlow 2.5.0 + Pillow)
pip install -r requirements.txt

# Install package in editable mode for development
pip install -e .
```

### Training
```bash
# Train model on prepared dataset
python train_model.py
```
- Expects labeled PNG files in `data/train_numbers_only/` and `data/train_numbers_only_2/`
- Filenames must match their content (e.g., `023062.png` contains "023062")
- Saves trained weights to `model/weights_v2.h5`
- Default: 100 epochs, 90/10 train/validation split, batch size 16

### Testing/Validation
```bash
# Download and predict CAPTCHA images (validates model)
python download_captcha.py
```
- Downloads CAPTCHAs from configured endpoint
- Runs prediction and renames files with predicted text
- Good for quick validation of model accuracy

## Architecture

### Core Components

**CaptchaCracker/core.py** contains three main classes:

1. **CTCLayer** (lines 14-33)
   - Custom Keras layer implementing CTC loss computation
   - Essential for training sequence recognition without pre-segmentation

2. **CreateModel** (lines 37-206)
   - Handles model training pipeline
   - Auto-extracts character vocabulary and max label length from training data
   - Architecture: Conv2D (32) → MaxPool → Conv2D (64) → MaxPool → Dense → 2x Bidirectional LSTM (128, 64) → Dense (softmax)
   - Downsampling factor: 4 (2x Conv pooling)

3. **ApplyModel** (lines 209-352)
   - Loads trained weights and performs inference
   - Supports both file path and bytes input
   - Uses greedy CTC decoding (beam search available but not implemented)

### Model Architecture Details

- **Input**: Grayscale images transposed to (width, height, channels) where width is the time dimension
- **Default dimensions**: 200x50 pixels
- **Preprocessing**: Images normalized to [0,1], resized, converted to grayscale, then transposed
- **CTC layer**: Enables training without explicit character segmentation in images
- **Character mapping**: StringLookup layers for character ↔ integer conversion

### Key Design Patterns

1. **Image preprocessing pipeline** (encode_single_sample):
   - Read → Decode PNG → Float32 conversion → Resize → Transpose
   - Transpose puts width as time dimension for LSTM processing

2. **Two-stage prediction**:
   - Full model used for training (includes CTC layer)
   - Prediction model extracts only image→dense2 path (excludes CTC)

3. **Character set auto-discovery**:
   - CreateModel extracts unique characters from training filenames
   - ApplyModel requires explicit character set (no auto-discovery during inference)

## Important Configuration Notes

### Model Parameters Must Match

When using ApplyModel for inference, ensure these match the training data:
- `img_width`, `img_height`: Must match training dimensions
- `max_length`: Must equal or exceed training data max label length
- `characters`: Must include all characters that appear in model output

Mismatch will cause prediction errors or incorrect decoding.

### Dataset Naming Convention

Training images **must** be named with their ground truth labels:
- Correct: `123456.png`, `098765.png`
- Wrong: `image_001.png`, `captcha_1.png`

The filename before `.png` is extracted as the label (line 46 in core.py).

### TensorFlow Version Pinning

The project requires TensorFlow 2.5.0 specifically:
- Uses deprecated `layers.experimental.preprocessing.StringLookup`
- Upgrading may require migration to `tf.keras.layers.StringLookup`
- Pillow pinned to 9.5.0 for compatibility

## Testing Strategy

No automated test suite exists. Validate changes by:

1. Train on small batch: Edit `train_model.py` to use fewer epochs/images
2. Check loss curve trends downward during training
3. Test inference with `ApplyModel.predict()` on held-out images
4. Compare predictions vs ground truth filenames
5. Document accuracy changes in PR descriptions

## Code Style

- Follow PEP 8: 4 spaces, snake_case functions, CamelCase classes
- Configuration constants at module top
- Korean comments acceptable (existing convention)
- Short docstrings for new preprocessing steps
- Descriptive tensor layer names (e.g., "Conv1", "pool2", "reshape")

## Git Workflow

- Commit format: `<type>: <summary>` (e.g., `fix: readme`, `feat: add beam search`)
- One logical change per commit
- Include reproduction steps in PR descriptions
- Document any new dependencies or environment changes
- Add sample predictions/screenshots for model changes

## Model Weights Management

- Canonical weights stored in `model/` directory
- Current versions: `weights.h5`, `weights_v2.h5`
- Do not commit experimental checkpoints (store locally or document download instructions)
- Weights files are ~1.7MB each (acceptable for git)

## External API Usage

`download_captcha.py` demonstrates inference workflow:
- Never hardcode API keys or tokens
- Use environment variables for endpoints
- Respect rate limits (current: 1 second delay between requests)
- Error handling for network failures required
