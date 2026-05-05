import os
import json
import sys
import torch
import torch.nn as nn
from torchvision import models
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    average_precision_score
)
sys.path.append(os.path.dirname(__file__))
from dataset import get_dataloaders
from evaluate import save_all_plots


WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), '../model/weights')
EPOCHS = 15
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
SUBSET = 0.1
THRESHOLD = 0.5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_model():
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    
    for params in model.parameters():
        params.requires_grad = False
    
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 1)
    )

    return model.to(DEVICE)


def compute_metrics(all_labels, all_probs, threshold=THRESHOLD):
    all_preds = [1 if p >= threshold else 0 for p in all_probs]

    auc_roc = roc_auc_score(all_labels, all_probs)
    auc_pr = average_precision_score(all_labels, all_probs)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds).tolist()

    return {
        'auc_roc': round(auc_roc, 4),
        'auc_pr': round(auc_pr, 4),
        'accuracy': round(acc, 4),
        'f1_score': round(f1, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'confusion_matrix': cm
    }


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    all_labels, all_probs = [], []

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(DEVICE)
        labels = labels.float().unsqueeze(1).to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        probs = torch.sigmoid(outputs).detach().cpu().numpy()
        all_probs.extend(probs.flatten().tolist())
        all_labels.extend(labels.cpu().numpy().flatten().tolist())

        print(f'Batch {batch_idx+1}/{len(loader)} - Loss: {loss.item():.4f}', end='\r')

    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(all_labels, all_probs)

    return avg_loss, metrics


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_labels, all_probs = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.float().unsqueeze(1).to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(outputs).cpu().numpy()
            all_probs.extend(probs.flatten().tolist())
            all_labels.extend(labels.cpu().numpy().flatten().tolist())
        
    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(all_labels, all_probs)

    return avg_loss, metrics, all_labels, all_probs


def get_val_predictions(model, loader):
    model.eval()
    all_labels, all_probs = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.float().unsqueeze(1).to(DEVICE)

            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_probs.extend(probs.flatten().tolist())
            all_labels.extend(labels.cpu().numpy().flatten().tolist())
    
    return all_labels, all_probs


def print_metrics(split, loss, metrics):
    cm = metrics["confusion_matrix"]
    print(f"\n{split} Results:")
    print(f"  Loss      : {loss:.4f}")
    print(f"  AUC-ROC   : {metrics['auc_roc']}")
    print(f"  AUC-PR    : {metrics['auc_pr']}")
    print(f"  Accuracy  : {metrics['accuracy']}")
    print(f"  F1 Score  : {metrics['f1_score']}")
    print(f"  Precision : {metrics['precision']}")
    print(f"  Recall    : {metrics['recall']}")
    print(f"  \n\nConfusion Matrix:")
    print(f"    TN: {cm[0][0]}  FP: {cm[0][1]}")
    print(f"    FN: {cm[1][0]}  TP: {cm[1][1]}")


def main():
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    print(f"Device : {DEVICE}")
    print(f"Epochs : {EPOCHS} | Batch size: {BATCH_SIZE} | Subset: {SUBSET * 100:.0f}%\n")

    train_loader, val_loader = get_dataloaders(batch_size=BATCH_SIZE, subset_fraction=SUBSET)

    model = build_model()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_auc = 0.0
    history = []

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        print("-" * 30)

        train_loss, train_metrics = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_metrics, val_labels, val_probs = evaluate(model, val_loader, criterion)

        print_metrics("Train", train_loss, train_metrics)
        print_metrics("Val", val_loss, val_metrics)

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss,   4),
            "train": train_metrics,
            "val": val_metrics,
        })

        if val_metrics["auc_roc"] > best_auc:
            best_auc = val_metrics["auc_roc"]
            torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, "best_model.pth"))
            print(f"\n--> Best model saved (Val AUC-ROC: {best_auc:.4f})")

    with open(os.path.join(WEIGHTS_DIR, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best Val AUC-ROC: {best_auc:.4f}")
    print(f"Weights saved to: {WEIGHTS_DIR}/best_model.pth")
    print(f"History saved to: {WEIGHTS_DIR}/history.json")

    print("\nGenerating and saving plots...")
    val_labels, val_probs = get_val_predictions(model, val_loader)
    save_all_plots(val_labels, val_probs, history)


if __name__ == "__main__":
    main()