"""Example: Train a new captcha recognition model."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from captcha_cracker import CaptchaTrainer


def main():
    """Train a new model from scratch."""
    # Data directories (adjust as needed)
    data_dirs = [
        "../data/train_numbers_only/",
        # Add more directories if you have additional training data
    ]

    print("Initializing trainer...")
    trainer = CaptchaTrainer(
        image_size=(200, 50),
        model_config={
            "conv_filters": [32, 64],
            "lstm_units": [128, 64],
            "dropout_rate": 0.2,
            "lstm_dropout": 0.25,
        },
    )

    print("Loading training data...")
    trainer.load_data(data_dirs)

    print(f"Dataset: {len(trainer.images)} images")
    print(f"Characters: {sorted(trainer.characters)}")
    print(f"Max length: {trainer.max_length}")

    print("\nBuilding model...")
    trainer.build_model()

    print("\nStarting training...")
    history = trainer.train(
        epochs=100,
        batch_size=16,
        validation_split=0.1,
        callbacks=["tensorboard", "checkpoint"],
        early_stopping=True,
        early_stopping_patience=10,
    )

    print("\nTraining completed!")
    print(f"Final loss: {history.history['loss'][-1]:.4f}")
    print(f"Final val_loss: {history.history['val_loss'][-1]:.4f}")

    # Save the model
    output_path = "trained_model.h5"
    print(f"\nSaving model to {output_path}...")
    trainer.save(output_path)

    print("Done!")


if __name__ == "__main__":
    main()
