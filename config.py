"""Central configuration for DermaScan AI."""

import torch
from pathlib import Path

# Paths
DATA_DIR = Path("data/HAM10000")
IMAGE_DIR = DATA_DIR.parent / "images"
METADATA_CSV = DATA_DIR / "HAM10000_metadata.csv"
MODEL_SAVE_DIR = Path("models")
OUTPUT_DIR = Path("outputs")

MODEL_SAVE_DIR.mkdir(exist_ok=True, parents=True)
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Dataset - 5 classes
SELECTED_CLASSES = ['mel', 'nv', 'bcc', 'akiec', 'bkl']
CLASS_NAMES = {
    'mel': 'Melanoma',
    'nv': 'Melanocytic Nevus',
    'bcc': 'Basal Cell Carcinoma',
    'akiec': 'Actinic Keratosis',
    'bkl': 'Benign Keratosis'
}
NUM_CLASSES = len(SELECTED_CLASSES)

# Model
MODEL_NAME = "efficientnet_b0"
PRETRAINED = True
FREEZE_BACKBONE_EPOCHS = 5

# Training
BATCH_SIZE = 16          # Reduce to 8 if memory errors
NUM_EPOCHS = 25
LEARNING_RATE = 1e-3
LEARNING_RATE_UNFROZEN = 1e-4
IMAGE_SIZE = 224

# Class weights (auto-calculated in dataset.py)
CLASS_WEIGHTS = None

# Augmentation
ROTATION_DEGREES = 30
COLOR_JITTER = 0.2

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Reproducibility
SEED = 42
