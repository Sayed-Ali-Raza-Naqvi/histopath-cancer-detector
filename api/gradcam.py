import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image


def load_model(weights_path: str, device: torch.device):
    model = models.efficientnet_b0(weights=None)


    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 1)
    )

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    return model


def get_gradcam_heatmap(
    model: nn.Module,
    image_tensor: torch.Tensor,
    raw_image: np.ndarray,
    device: torch.device
) -> np.ndarray:
    target_layer = [model.features[-1][0]]

    cam = GradCAM(model=model, target_layers=target_layer)
    grayscale_cam = cam(
        input_tensor=image_tensor.unsqueeze(0).to(device),
        targets=None
    )
    grayscale_cam = grayscale_cam[0]

    raw_float = raw_image.astype(np.float32) / 255.0
    heatmap = show_cam_on_image(raw_float, grayscale_cam, use_rgb=True)
    
    return heatmap


def run_inference_with_gradcam(
        pil_image: Image.Image,
        model: nn.Module,
        transform,
        device: torch.device,
        threshold: float = 0.4041
):
    raw_image = np.array(pil_image.convert("RGB"))
    image_tensor = transform(pil_image.convert("RGB"))

    with torch.no_grad():
        output = model(image_tensor.unsqueeze(0).to(device))
        probability = torch.sigmoid(output).item()

    prediction = "Tumor" if probability >= threshold else "Normal"
    confidence = probability if prediction == "Tumor" else 1 - probability
    heatmap = get_gradcam_heatmap(model, image_tensor, raw_image, device)

    return {
        "prediction": prediction,
        "tumor_probability": round(probability, 4),
        "confidence": round(confidence, 4),
        "heatmap": heatmap,
        "raw_image": raw_image
    }