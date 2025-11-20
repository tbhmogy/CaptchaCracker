"""Example: Predict a single captcha image."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from captcha_cracker import CaptchaModel


def main():
    """Load model and predict a single image."""
    # Path to the model weights (adjust as needed)
    weights_path = "../model/weights_v2.h5"

    # Path to test image (adjust as needed)
    image_path = "captcha.png"

    print("Loading model...")
    model = CaptchaModel.load(weights_path)

    print(f"Predicting: {image_path}")
    result = model.predict(image_path, return_confidence=True)

    print(f"\nResult: {result.text}")
    print(f"Confidence: {result.confidence:.3f}")


if __name__ == "__main__":
    main()
