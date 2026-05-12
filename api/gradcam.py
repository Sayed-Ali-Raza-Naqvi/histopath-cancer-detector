import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image

TARGET_SIZE = (96, 96)


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

    for param in model.parameters():
        param.requires_grad = True

    return model


def get_gradcam_heatmap(
    model: nn.Module,
    image_tensor: torch.Tensor,
    raw_image: np.ndarray,
    device: torch.device
) -> np.ndarray:
    target_layer = [model.features[-1]]
    cam = GradCAM(model=model, target_layers=target_layer)
    grayscale_cam = cam(
        input_tensor=image_tensor.unsqueeze(0).to(device),
        targets=None
    )
    grayscale_cam = grayscale_cam[0]

    # Ensure the raw image and CAM map match in size for overlay.
    cam_h, cam_w = grayscale_cam.shape
    print(
        f"Grad-CAM shapes: raw={raw_image.shape} cam={grayscale_cam.shape}",
        flush=True
    )
    if raw_image.shape[0] != cam_h or raw_image.shape[1] != cam_w:
        raw_image = cv2.resize(raw_image, (cam_w, cam_h), interpolation=cv2.INTER_LINEAR)
        print(
            f"Resized raw image to: {raw_image.shape}",
            flush=True
        )

    raw_float = raw_image.astype(np.float32) / 255.0
    try:
        heatmap = show_cam_on_image(raw_float, grayscale_cam, use_rgb=True)
    except Exception as e:
        raise RuntimeError(
            f"show_cam_on_image failed: raw={raw_image.shape} cam={grayscale_cam.shape} error={e}"
        )
    
    return heatmap


def run_inference_with_gradcam(
    pil_image: Image.Image,
    model: nn.Module,
    transform,
    device: torch.device,
    threshold: float = 0.4041
):
    pil_image = pil_image.convert("RGB")
    original_size = pil_image.size
    print(f"Original image size: {original_size}", flush=True)
    
    pil_image_resized = pil_image.resize((96, 96), Image.BILINEAR)
    raw_image = np.array(pil_image_resized)
    image_tensor = transform(pil_image_resized)

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
        "raw_image": raw_image,
    }