"""Train the classifier. Example: python train.py --head fastkan --seed 42."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from fair2026_lms_kan.cli_train import main


if __name__ == "__main__":
    main()
