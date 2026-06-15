"""Single image prediction API."""

import json
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *


class DermaScanPredictor:
    def __init__(self, model_path=None):
        self.device = DEVICE
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(SELECTED_CLASSES)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        # Try to load checkpoint; if missing, create untrained model for demo
        checkpoint_path = Path(model_path) if model_path else MODEL_SAVE_DIR / "best_model.pth"
        if checkpoint_path.exists():
            self.model = self._load_checkpoint(checkpoint_path)
            # Cross-check training_status.json to confirm model trained well enough
            status_path = checkpoint_path.parent / "training_status.json"
            if status_path.exists():
                with open(status_path) as f:
                    status = json.load(f)
                self.is_trained = status.get("is_well_trained", True)
                if not self.is_trained:
                    print(
                        f"Warning: Model checkpoint found but validation accuracy "
                        f"({status.get('best_val_acc', '?')}%) is below the required "
                        f"threshold ({status.get('min_required_acc', '?')}%). "
                        "Predictions may be unreliable."
                    )
            else:
                # No status file — assume an older checkpoint is trained
                self.is_trained = True
        else:
            print(f"Warning: Model checkpoint not found at {checkpoint_path}")
            print("Creating untrained model for demo purposes (predictions will be random)")
            from model import create_model
            self.model = create_model(freeze_backbone=False)
            self.is_trained = False
    
    def _load_checkpoint(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        from model import create_model
        model = create_model(freeze_backbone=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
    
    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(image_tensor)
            probs = torch.softmax(output, dim=1)[0]
        
        predictions = {
            CLASS_NAMES[cls]: round(probs[i].item() * 100, 2)
            for i, cls in enumerate(SELECTED_CLASSES)
        }
        
        predicted_idx = torch.argmax(probs).item()
        predicted_class = self.idx_to_class[predicted_idx]
        
        return {
            'prediction': CLASS_NAMES[predicted_class],
            'confidence': round(probs.max().item() * 100, 2),
            'all_probabilities': predictions,
            'is_malignant': predicted_class in ['mel', 'bcc', 'akiec'],
            'is_trained': self.is_trained
        }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
    
    predictor = DermaScanPredictor()
    result = predictor.predict(sys.argv[1])
    
    print("\n" + "=" * 40)
    print("DERMASCAN AI PREDICTION")
    print("=" * 40)
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']}%")
    print(f"Malignant:  {'Yes ⚠️' if result['is_malignant'] else 'No ✅'}")
    print("\nAll probabilities:")
    for cls, prob in result['all_probabilities'].items():
        bar = "█" * int(prob / 5)
        print(f"  {cls:25s}: {prob:5.1f}% {bar}")
