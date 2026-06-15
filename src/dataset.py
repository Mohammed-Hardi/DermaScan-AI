"""Data loading, preprocessing, and augmentation."""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import sys
import os
from pathlib import Path

# Add project root to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *


class SkinLesionDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.class_to_idx = {cls: idx for idx, cls in enumerate(SELECTED_CLASSES)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        self.image_dir = IMAGE_DIR
        self.image_index = self._build_image_index()

    def _build_image_index(self):
        image_index = {}
        if self.image_dir.exists():
            for root, _, files in os.walk(self.image_dir):
                for file in files:
                    stem = Path(file).stem
                    if stem not in image_index:
                        image_index[stem] = Path(root) / file
        return image_index

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = str(row['image_id'])
        img_path = self.image_index.get(image_id)
        if img_path is None:
            candidates = list(self.image_dir.glob(f"**/{image_id}.*"))
            if candidates:
                img_path = candidates[0]

        if img_path is None or not img_path.exists():
            raise FileNotFoundError(f"Image not found for ID: {image_id}")

        image = Image.open(img_path).convert("RGB")
        label = self.class_to_idx[row['dx']]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


def get_transforms(phase="train"):
    if phase == "train":
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=ROTATION_DEGREES),
            transforms.ColorJitter(
                brightness=COLOR_JITTER,
                contrast=COLOR_JITTER,
                saturation=COLOR_JITTER
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


def load_data():
    print("Loading dataset...")
    if not METADATA_CSV.exists():
        raise FileNotFoundError(f"Metadata file not found: {METADATA_CSV}")
    
    df = pd.read_csv(METADATA_CSV)
    df = df[df['dx'].isin(SELECTED_CLASSES)].copy()
    
    # Build a quick image index to filter out rows with missing files
    _image_index = {}
    if IMAGE_DIR.exists():
        for root, _, files in os.walk(IMAGE_DIR):
            for file in files:
                stem = Path(file).stem
                if stem not in _image_index:
                    _image_index[stem] = True
    
    before = len(df)
    df = df[df['image_id'].astype(str).isin(_image_index)].copy()
    dropped = before - len(df)
    if dropped > 0:
        print(f"WARNING: {dropped} rows dropped because their image files were not found on disk.")
        print(f"  (Only images found in '{IMAGE_DIR}' are used.)")
    
    if len(df) == 0:
        raise RuntimeError(
            f"No images found! Make sure the HAM10000 images are inside '{IMAGE_DIR}'."
        )
    
    print(f"Total images available: {len(df)}")
    print("Class distribution:")
    print(df['dx'].value_counts())
    
    train_df, temp_df = train_test_split(
        df, test_size=0.3, stratify=df['dx'], random_state=SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df['dx'], random_state=SEED
    )
    
    print(f"\nTrain: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    class_counts = train_df['dx'].value_counts().reindex(SELECTED_CLASSES)
    weights = 1.0 / class_counts.values
    weights = weights / weights.sum() * len(SELECTED_CLASSES)
    print(f"Class weights: {dict(zip(SELECTED_CLASSES, weights.round(3)))}")
    
    train_dataset = SkinLesionDataset(train_df, get_transforms("train"))
    val_dataset = SkinLesionDataset(val_df, get_transforms("val"))
    test_dataset = SkinLesionDataset(test_df, get_transforms("val"))
    
    weight_map = {cls: weights[idx] for cls, idx in train_dataset.class_to_idx.items()}
    sample_weights = train_df['dx'].map(weight_map).tolist()
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=sampler, 
        num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=0, pin_memory=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=0, pin_memory=False
    )
    
    return train_loader, val_loader, test_loader, train_dataset


if __name__ == "__main__":
    train_loader, val_loader, test_loader, dataset = load_data()
    print(f"\nClasses: {dataset.class_to_idx}")
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}, Labels: {labels}")
