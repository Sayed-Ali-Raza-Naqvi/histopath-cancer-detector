import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

sys.path.append(os.path.dirname(__file__))
from gradcam import load_model, run_inference_with_gradcam

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_PATH = os.path.join(BASE_DIR, "model", "weights", "best_model.pth")
OUTPUT_DIR   = os.path.join(BASE_DIR, "assets", "samples")
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Device       : {DEVICE}")
print(f"Weights path : {WEIGHTS_PATH}")
print(f"Loading model...")

model = load_model(WEIGHTS_PATH, DEVICE)
print("Model loaded successfully")


def test_on_image(image_path: str, label: str):
    pil_image = Image.open(image_path).convert("RGB")
    result    = run_inference_with_gradcam(pil_image, model, transform, DEVICE)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    fig.suptitle(
        f"True: {label} | Predicted: {result['prediction']} | "
        f"Tumor prob: {result['tumor_probability']} | "
        f"Confidence: {result['confidence']}",
        fontsize=10
    )

    axes[0].imshow(result["raw_image"])
    axes[0].set_title("Original Patch")
    axes[0].axis("off")

    axes[1].imshow(result["heatmap"])
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

    output_name = f"gradcam_{label.lower()}_{os.path.basename(image_path)}.png"
    output_path = os.path.join(OUTPUT_DIR, output_name)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"\nImage     : {image_path}")
    print(f"True label: {label}")
    print(f"Predicted : {result['prediction']}")
    print(f"Tumor prob: {result['tumor_probability']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Saved     : {output_path}")


def extract_sample_patches():
    import h5py
    import numpy as np

    print("\nExtracting sample patches from test set...")

    image_path = os.path.join(BASE_DIR, "data", "pcam", "test_split.h5")
    label_path = os.path.join(
        BASE_DIR, "data", "Labels", "Labels",
        "camelyonpatch_level_2_split_test_y.h5"
    )

    with h5py.File(image_path, "r") as f:
        key    = list(f.keys())[0]
        images = f[key][:]

    with h5py.File(label_path, "r") as f:
        key    = list(f.keys())[0]
        labels = f[key][:].squeeze().astype(int)

    tumor_indices  = np.where(labels == 1)[0]
    normal_indices = np.where(labels == 0)[0]

    samples = []
    for idx in tumor_indices[:3]:
        img  = Image.fromarray(images[idx].astype(np.uint8))
        path = os.path.join(OUTPUT_DIR, f"sample_tumor_{idx}.png")
        img.save(path)
        samples.append((path, "Tumor"))

    for idx in normal_indices[:3]:
        img  = Image.fromarray(images[idx].astype(np.uint8))
        path = os.path.join(OUTPUT_DIR, f"sample_normal_{idx}.png")
        img.save(path)
        samples.append((path, "Normal"))

    print(f"Extracted 3 tumor + 3 normal patches to {OUTPUT_DIR}")
    return samples


if __name__ == "__main__":
    samples = extract_sample_patches()
    print("\nRunning Grad-CAM on all samples...")
    for image_path, label in samples:
        test_on_image(image_path, label)

    print(f"\nAll Grad-CAM outputs saved to: {OUTPUT_DIR}")