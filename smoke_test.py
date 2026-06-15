from pathlib import Path
import sys
import importlib

sys.path.insert(0, '.')
import config

print('Original PRETRAINED:', config.PRETRAINED)
# Patch config BEFORE importing model so the flag is actually respected
config.PRETRAINED = False
print('Temporarily set PRETRAINED to', config.PRETRAINED)

# Re-import model module AFTER patching so it picks up PRETRAINED=False
# (a plain `from src import model` caches the old binding)
import src.model as model_module
importlib.reload(model_module)

from src.dataset import load_data
train_loader, val_loader, test_loader, dataset = load_data()
print('Loaded datasets -> Train batches:', len(train_loader), 'Val batches:', len(val_loader), 'Test batches:', len(test_loader))
print('Sample class mapping:', dataset.class_to_idx)

# Create model with pretrained disabled (no weights download needed)
m = model_module.create_model()
print('Model created. Device info available via model.parameters()')
print('Smoke test completed successfully.')
