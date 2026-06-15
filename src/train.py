"""Training loop."""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import json
import time
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from dataset import load_data
from model import create_model, unfreeze_backbone


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(loader), 100. * correct / total


def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(loader), 100. * correct / total


def main():
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    
    print("=" * 50)
    print("DermaScan AI - Training")
    print("=" * 50)
    
    train_loader, val_loader, test_loader, dataset = load_data()
    
    model = create_model(freeze_backbone=True)
    
    class_counts = dataset.df['dx'].value_counts().reindex(SELECTED_CLASSES).fillna(1)
    weights = 1.0 / class_counts.values
    weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    early_stopping = EarlyStopping(patience=7)
    best_val_acc = 0.0
    # Minimum val accuracy the model must reach before it is considered "properly trained".
    # Below this threshold the checkpoint exists but predict.py will warn the user.
    MIN_TRAINED_ACC = 30.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    start_time = time.time()
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 30)
        
        if epoch == FREEZE_BACKBONE_EPOCHS:
            model = unfreeze_backbone(model)
            optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE_UNFROZEN, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS - epoch)
            print(f"LR adjusted to {LEARNING_RATE_UNFROZEN}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_loader, criterion)
        scheduler.step()
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            is_well_trained = val_acc >= MIN_TRAINED_ACC
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_to_idx': dataset.class_to_idx,
                'is_well_trained': is_well_trained,
                'history': history
            }, MODEL_SAVE_DIR / "best_model.pth")
            # Write a separate status file so the app can check training quality
            import json as _json
            with open(MODEL_SAVE_DIR / "training_status.json", "w") as _f:
                _json.dump({
                    "best_val_acc": round(val_acc, 4),
                    "epoch": epoch + 1,
                    "is_well_trained": is_well_trained,
                    "min_required_acc": MIN_TRAINED_ACC
                }, _f, indent=2)
            print(f"[OK] Saved best model (val_acc: {val_acc:.2f}%)")
            if not is_well_trained:
                print(f"  [!] Below minimum accuracy threshold ({MIN_TRAINED_ACC}%) -- model not yet reliable")
        
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed/60:.1f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    with open(OUTPUT_DIR / "training_history.json", "w") as f:
        json.dump(history, f)


if __name__ == "__main__":
    main()
