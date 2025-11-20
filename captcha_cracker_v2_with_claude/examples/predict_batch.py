"""Example: Batch prediction for multiple images."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from captcha_cracker import CaptchaModel


def main():
    """Load model and predict multiple images."""
    # Path to the model weights (adjust as needed)
    weights_path = "../model/weights_v2.h5"

    # List of images to predict (adjust as needed)
    image_paths = [
        "captcha1.png",
        "captcha2.png",
        "captcha3.png",
    ]

    print("Loading model...")
    with CaptchaModel.load(weights_path) as model:
        print(f"Predicting {len(image_paths)} images...")
        results = model.predict_batch(image_paths, return_confidence=True)

        print("\nResults:")
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result.text} (confidence: {result.confidence:.3f})")


if __name__ == "__main__":
    main()
