"""Evaluation metrics and reporting."""

import torch
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from dataset import load_data
from model import create_model


def evaluate():
    train_loader, val_loader, test_loader, dataset = load_data()
    device = DEVICE
    model = create_model(freeze_backbone=False)
    checkpoint_path = MODEL_SAVE_DIR / "best_model.pth"
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())
    
    report = classification_report(
        all_labels,
        all_preds,
        labels=list(dataset.class_to_idx.values()),
        target_names=list(dataset.class_to_idx.keys()),
        zero_division=0,
        output_dict=True
    )
    cm = confusion_matrix(all_labels, all_preds, labels=list(dataset.class_to_idx.values()))
    
    results = {
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "accuracy": (
            sum(1 for i, j in zip(all_labels, all_preds) if i == j) / len(all_labels)
            if all_labels else 0.0
        )
    }
    
    pd.DataFrame(report).transpose().to_csv(OUTPUT_DIR / "classification_report.csv")
    pd.DataFrame(cm, index=dataset.class_to_idx.keys(), columns=dataset.class_to_idx.keys()).to_csv(OUTPUT_DIR / "confusion_matrix.csv")
    
    return results


if __name__ == "__main__":
    results = evaluate()
    print("Evaluation complete")
    print(f"Accuracy: {results['accuracy']:.4f}")
