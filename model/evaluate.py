import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    confusion_matrix,
    ConfusionMatrixDisplay
)

sys.path.append(os.path.dirname(__file__))

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "../model/weights")
PLOTS_DIR   = os.path.join(os.path.dirname(__file__), "../assets/plots")


def save_roc_curve(all_labels, all_probs):
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc     = auc(fpr, tpr)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color="#1D9E75", lw=2, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="#888780", lw=1, linestyle="--", label="Random Classifier")
    plt.fill_between(fpr, tpr, alpha=0.05, color="#1D9E75")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Histopathology Cancer Detector")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "roc_curve.png"), dpi=150)
    plt.close()
    print("Saved: roc_curve.png")


def save_precision_recall_curve(all_labels, all_probs):
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    pr_auc               = auc(recall, precision)
    baseline             = sum(all_labels) / len(all_labels)

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, color="#534AB7", lw=2, label=f"PR Curve (AUC = {pr_auc:.4f})")
    plt.axhline(y=baseline, color="#888780", lw=1, linestyle="--", label=f"Baseline (prevalence = {baseline:.2f})")
    plt.fill_between(recall, precision, alpha=0.05, color="#534AB7")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve — Histopathology Cancer Detector")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "precision_recall_curve.png"), dpi=150)
    plt.close()
    print("Saved: precision_recall_curve.png")


def save_confusion_matrix(all_labels, all_probs, threshold=0.5):
    all_preds = [1 if p >= threshold else 0 for p in all_probs]
    cm        = confusion_matrix(all_labels, all_preds)
    disp      = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Normal", "Tumor"]
    )

    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix (threshold = {threshold})")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()
    print("Saved: confusion_matrix.png")


def save_training_curves(history: list):
    epochs     = [h["epoch"]            for h in history]
    train_loss = [h["train_loss"]       for h in history]
    val_loss   = [h["val_loss"]         for h in history]
    train_auc  = [h["train"]["auc_roc"] for h in history]
    val_auc    = [h["val"]["auc_roc"]   for h in history]
    train_f1   = [h["train"]["f1-score"]      for h in history]
    val_f1     = [h["val"]["f1"]        for h in history]
    train_acc  = [h["train"]["accuracy"] for h in history]
    val_acc    = [h["val"]["accuracy"]   for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Training History", fontsize=14, fontweight="bold")

    axes[0, 0].plot(epochs, train_loss, label="Train", color="#1D9E75", marker="o", markersize=3)
    axes[0, 0].plot(epochs, val_loss,   label="Val",   color="#D85A30", marker="o", markersize=3)
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, train_auc, label="Train", color="#1D9E75", marker="o", markersize=3)
    axes[0, 1].plot(epochs, val_auc,   label="Val",   color="#D85A30", marker="o", markersize=3)
    axes[0, 1].set_title("AUC-ROC")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("AUC-ROC")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs, train_f1, label="Train", color="#1D9E75", marker="o", markersize=3)
    axes[1, 0].plot(epochs, val_f1,   label="Val",   color="#D85A30", marker="o", markersize=3)
    axes[1, 0].set_title("F1 Score")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("F1")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs, train_acc, label="Train", color="#1D9E75", marker="o", markersize=3)
    axes[1, 1].plot(epochs, val_acc,   label="Val",   color="#D85A30", marker="o", markersize=3)
    axes[1, 1].set_title("Accuracy")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Accuracy")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "training_curves.png"), dpi=150)
    plt.close()
    print("Saved: training_curves.png")


def save_all_plots(all_labels, all_probs, history):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    save_roc_curve(all_labels, all_probs)
    save_precision_recall_curve(all_labels, all_probs)
    save_confusion_matrix(all_labels, all_probs)
    save_training_curves(history)
    print(f"\nAll plots saved to: {PLOTS_DIR}/")


if __name__ == "__main__":
    history_path = os.path.join(WEIGHTS_DIR, "history.json")
    if not os.path.exists(history_path):
        print("No history.json found. Run train.py first.")
    else:
        with open(history_path, "r") as f:
            history = json.load(f)
        print("Loaded history.json — but labels/probs needed for curve plots.")
        save_training_curves(history)