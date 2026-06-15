"""Grad-CAM and explainability utilities."""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from model import create_model


def get_gradcam_heatmap(model, image_tensor, target_class):
    gradients = []
    activations = []
    
    def forward_hook(module, input, output):
        activations.append(output.detach())

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())
    
    target_layer = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module
    
    if target_layer is None:
        raise RuntimeError("No Conv2d layer found for Grad-CAM")
    
    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_backward_hook(backward_hook)
    
    model.zero_grad()
    output = model(image_tensor)
    loss = output[0, target_class]
    loss.backward(retain_graph=True)
    
    handle_forward.remove()
    handle_backward.remove()
    
    grads = gradients[0]
    acts = activations[0]
    pooled = grads.mean(dim=(2, 3), keepdim=True)
    weighted = acts * pooled
    heatmap = weighted.sum(dim=1).squeeze()
    heatmap = F.relu(heatmap)
    heatmap = heatmap - heatmap.min()
    heatmap = heatmap / (heatmap.max() + 1e-8)
    heatmap = heatmap.cpu().numpy()
    return heatmap


def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)


def explain(image_path, target_class=None):
    device = DEVICE
    model = create_model(freeze_backbone=False)
    checkpoint_path = MODEL_SAVE_DIR / "best_model.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    
    image_tensor = load_image(image_path).to(device)
    output = model(image_tensor)
    if target_class is None:
        target_class = output.argmax(dim=1).item()
    heatmap = get_gradcam_heatmap(model, image_tensor, target_class)
    return heatmap
