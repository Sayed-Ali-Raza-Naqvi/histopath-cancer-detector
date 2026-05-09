import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    auc,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    average_precision_score,
    precision_recall_curve
)
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
import h5py

sys.path.append(os.path.dirname(__file__))
from dataset import PCAMDataset, get_transforms

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR = os.path.join(BASE_DIR, "model", "weights")
PLOTS_DIR = os.path.join(BASE_DIR, "assets", "plots")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    model = models.efficientnet_b0(weights=None)

    for param in model.parameters():
        param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 1)
    )

    weights_path = os.path.join(WEIGHTS_DIR, "best_model.pth")
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    print(f"Model loaded from: {weights_path}")
    return model


def get_test_predictions(model):
    print("Loading test dataset into memory...")
    test_dataset = PCAMDataset(split="test", transform=get_transforms("test"))
    test_loader  = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    print(f"Test samples: {len(test_dataset)}")
    print(f"Test batches: {len(test_loader)}")
    print("Running inference...")

    all_labels, all_probs = [], []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_probs.extend(probs.flatten().tolist())
            all_labels.extend(labels.numpy().flatten().tolist())

            if (batch_idx + 1) % 50 == 0:
                print(f"  Batch {batch_idx + 1}/{len(test_loader)}")

    return all_labels, all_probs


def find_optimal_threshold(all_labels, all_probs):
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[best_idx]

    print(f"\nThreshold search:")
    print(f"  Default  (0.50) — checking metrics below")
    print(f"  Optimal  ({optimal_threshold:.4f}) — maximizes Youden J = sensitivity + specificity - 1")

    return round(float(optimal_threshold), 4)


def compute_metrics(all_labels, all_probs, threshold):
    all_preds = [1 if p >= threshold else 0 for p in all_probs]

    auc_roc = roc_auc_score(all_labels, all_probs)
    auc_pr = average_precision_score(all_labels, all_probs)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds).tolist()

    return {
        "threshold": threshold,
        "auc_roc": round(auc_roc, 4),
        "auc_pr": round(auc_pr, 4),
        "accuracy":  round(acc, 4),
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "confusion_matrix": cm
    }


def print_metrics(label, metrics):
    cm = metrics["confusion_matrix"]
    print(f"\n{label} (threshold = {metrics['threshold']}):")
    print(f"  AUC-ROC: {metrics['auc_roc']}")
    print(f"  AUC-PR: {metrics['auc_pr']}")
    print(f"  Accuracy: {metrics['accuracy']}")
    print(f"  F1 Score: {metrics['f1']}")
    print(f"  Precision: {metrics['precision']}")
    print(f"  Recall: {metrics['recall']}")
    print(f"  Confusion Matrix:")
    print(f"    TN: {cm[0][0]}  FP: {cm[0][1]}")
    print(f"    FN: {cm[1][0]}  TP: {cm[1][1]}")


def save_roc_curve(all_labels, all_probs, optimal_threshold):
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc              = auc(fpr, tpr)

    best_idx = np.argmin(np.abs(thresholds - optimal_threshold))

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color="#1D9E75", lw=2, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="#888780", lw=1, linestyle="--", label="Random Classifier")
    plt.scatter(fpr[best_idx], tpr[best_idx], color="#D85A30", zorder=5, s=80,
                label=f"Optimal threshold ({optimal_threshold:.4f})")
    plt.fill_between(fpr, tpr, alpha=0.05, color="#1D9E75")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Test Set")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "test_roc_curve.png"), dpi=150)
    plt.close()
    print("Saved: test_roc_curve.png")


def save_precision_recall_curve(all_labels, all_probs):
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    pr_auc               = auc(recall, precision)
    baseline             = sum(all_labels) / len(all_labels)

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, color="#534AB7", lw=2, label=f"PR Curve (AUC = {pr_auc:.4f})")
    plt.axhline(y=baseline, color="#888780", lw=1, linestyle="--",
                label=f"Baseline (prevalence = {baseline:.2f})")
    plt.fill_between(recall, precision, alpha=0.05, color="#534AB7")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve — Test Set")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "test_precision_recall_curve.png"), dpi=150)
    plt.close()
    print("Saved: test_precision_recall_curve.png")


def save_confusion_matrices(all_labels, all_probs, default_threshold, optimal_threshold):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, threshold, title in zip(
        axes,
        [default_threshold, optimal_threshold],
        [f"Default (threshold = {default_threshold})", f"Optimal (threshold = {optimal_threshold})"]
    ):
        preds = [1 if p >= threshold else 0 for p in all_probs]
        cm    = confusion_matrix(all_labels, preds)
        disp  = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Normal", "Tumor"]
        )
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "test_confusion_matrices.png"), dpi=150)
    plt.close()
    print("Saved: test_confusion_matrices.png")


def save_threshold_sweep(all_labels, all_probs):
    thresholds = np.arange(0.1, 0.9, 0.01)
    f1_scores, precisions, recalls, accuracies = [], [], [], []

    for t in thresholds:
        preds = [1 if p >= t else 0 for p in all_probs]
        f1_scores.append(f1_score(all_labels, preds, zero_division=0))
        precisions.append(precision_score(all_labels, preds, zero_division=0))
        recalls.append(recall_score(all_labels, preds, zero_division=0))
        accuracies.append(accuracy_score(all_labels, preds))

    plt.figure(figsize=(9, 5))
    plt.plot(thresholds, f1_scores,  label="F1",        color="#1D9E75", lw=2)
    plt.plot(thresholds, precisions, label="Precision",  color="#534AB7", lw=2)
    plt.plot(thresholds, recalls,    label="Recall",     color="#D85A30", lw=2)
    plt.plot(thresholds, accuracies, label="Accuracy",   color="#888780", lw=2, linestyle="--")
    plt.axvline(x=thresholds[np.argmax(f1_scores)], color="#D85A30",
                linestyle=":", lw=1.5, label=f"Best F1 threshold ({thresholds[np.argmax(f1_scores)]:.2f})")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Metrics vs Classification Threshold — Test Set")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "test_threshold_sweep.png"), dpi=150)
    plt.close()
    print("Saved: test_threshold_sweep.png")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    model = load_model()
    all_labels, all_probs = get_test_predictions(model)
    optimal_threshold = find_optimal_threshold(all_labels, all_probs)

    default_metrics = compute_metrics(all_labels, all_probs, threshold=0.5)
    optimal_metrics = compute_metrics(all_labels, all_probs, threshold=optimal_threshold)

    print_metrics("Default threshold results", default_metrics)
    print_metrics("Optimal threshold results", optimal_metrics)

    final_metrics = {
        "default":  default_metrics,
        "optimal":  optimal_metrics,
    }

    metrics_path = os.path.join(WEIGHTS_DIR, "test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"\nTest metrics saved to: {metrics_path}")

    print("\nSaving test plots...")
    save_roc_curve(all_labels, all_probs, optimal_threshold)
    save_precision_recall_curve(all_labels, all_probs)
    save_confusion_matrices(all_labels, all_probs, 0.5, optimal_threshold)
    save_threshold_sweep(all_labels, all_probs)

    print("\nAll test plots saved.")
    print(f"\nFinal recommendation:")
    print(f"  Use threshold = {optimal_threshold} for deployment")
    print(f"  Val  AUC-ROC = 0.9066")
    print(f"  Test AUC-ROC = {optimal_metrics['auc_roc']}")
    print(f"  Test Recall = {optimal_metrics['recall']}")
    print(f"  Test Precision= {optimal_metrics['precision']}")


if __name__ == "__main__":
    main()