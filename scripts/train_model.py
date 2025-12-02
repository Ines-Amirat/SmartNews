"""
Entry point script to train the SmartNews model.

Usage:
    python -m scripts.train_model
(from the project root)
"""

from src.ml.training import train_and_save_model


def main():
    train_and_save_model()


if __name__ == "__main__":
    main()
