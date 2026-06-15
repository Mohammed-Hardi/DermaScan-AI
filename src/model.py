"""Neural network architecture."""

import torch
import torch.nn as nn
from torchvision import models
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *


def create_model(num_classes=NUM_CLASSES, freeze_backbone=True):
    print(f"Creating {MODEL_NAME}...")
    
    if MODEL_NAME == "efficientnet_b0":
        weights = "IMAGENET1K_V1" if PRETRAINED else None
        model = models.efficientnet_b0(weights=weights)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)
    elif MODEL_NAME == "resnet50":
        weights = "IMAGENET1K_V1" if PRETRAINED else None
        model = models.resnet50(weights=weights)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {MODEL_NAME}")
    
    if freeze_backbone and PRETRAINED:
        for name, param in model.named_parameters():
            if 'classifier' not in name and 'fc' not in name:
                param.requires_grad = False
        print("Backbone frozen (classifier trainable only)")
    
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total:,} total, {trainable:,} trainable")
    
    return model.to(DEVICE)


def unfreeze_backbone(model):
    for param in model.parameters():
        param.requires_grad = True
    print("Backbone unfrozen for fine-tuning")
    return model


if __name__ == "__main__":
    model = create_model()
    print(f"\nModel ready on {DEVICE}")
