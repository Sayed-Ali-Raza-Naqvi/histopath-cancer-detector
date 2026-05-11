import streamlit as st
import requests
import json
import os
from PIL import Image

API_URL  = "http://localhost:8000/api/v1"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PLOTS_DIR = os.path.join(BASE_DIR, "assets", "plots")


def load_api_metrics():
    try:
        response = requests.get(f"{API_URL}/model-info", timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except requests.exceptions.ConnectionError:
        return None


def load_history():
    history_path = os.path.join(BASE_DIR, "model", "weights", "history.json")
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            return json.load(f)
    return None


st.title("Model Information")
st.markdown(
    "Performance metrics, training history, and evaluation plots "
    "for the EfficientNet-B0 histopathology cancer detector."
)
st.divider()

metrics = load_api_metrics()

if metrics is None:
    st.warning(
        "Could not fetch metrics from API. "
        "Make sure the FastAPI server is running on port 8000. "
        "Showing local metrics instead."
    )
    metrics_path = os.path.join(BASE_DIR, "model", "weights", "test_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            test_metrics = json.load(f)
        metrics = test_metrics.get("optimal", {})
    else:
        metrics = {}

st.subheader("Model Overview")
c1, c2 = st.columns(2)
with c1:
    st.markdown(f"**Architecture** : EfficientNet-B0 (fine-tuned)")
    st.markdown(f"**Dataset**      : PatchCamelyon — 327,680 H&E patches")
    st.markdown(f"**Task**         : Binary classification (Tumor / Normal)")
with c2:
    st.markdown(f"**Threshold**    : {metrics.get('threshold', 0.4041)}")
    st.markdown(f"**Explainability**: Grad-CAM")
    st.markdown(f"**Framework**    : PyTorch + FastAPI + Streamlit")

st.divider()
st.subheader("Test Set Performance")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("AUC-ROC",   metrics.get("test_auc_roc",  metrics.get("auc_roc",   "—")))
m2.metric("Accuracy",  metrics.get("test_accuracy", metrics.get("accuracy",  "—")))
m3.metric("F1 Score",  metrics.get("test_f1",       metrics.get("f1",        "—")))
m4.metric("Precision", metrics.get("test_precision",metrics.get("precision", "—")))
m5.metric("Recall",    metrics.get("test_recall",   metrics.get("recall",    "—")))

st.divider()
st.subheader("Training History")

history = load_history()
if history:
    epochs    = [h["epoch"]             for h in history]
    val_auc   = [h["val"]["auc_roc"]    for h in history]
    train_auc = [h["train"]["auc_roc"]  for h in history]
    val_f1    = [h["val"]["f1"]         for h in history]
    train_f1  = [h["train"]["f1"]       for h in history]
    val_loss  = [h["val_loss"]          for h in history]
    train_loss= [h["train_loss"]        for h in history]

    import pandas as pd

    tab1, tab2, tab3 = st.tabs(["AUC-ROC", "F1 Score", "Loss"])

    with tab1:
        auc_df = pd.DataFrame({
            "Epoch":      epochs,
            "Train AUC":  train_auc,
            "Val AUC":    val_auc,
        }).set_index("Epoch")
        st.line_chart(auc_df)

    with tab2:
        f1_df = pd.DataFrame({
            "Epoch":     epochs,
            "Train F1":  train_f1,
            "Val F1":    val_f1,
        }).set_index("Epoch")
        st.line_chart(f1_df)

    with tab3:
        loss_df = pd.DataFrame({
            "Epoch":      epochs,
            "Train Loss": train_loss,
            "Val Loss":   val_loss,
        }).set_index("Epoch")
        st.line_chart(loss_df)
else:
    st.info("history.json not found. Run training first.")

st.divider()
st.subheader("Evaluation Plots")

roc_path = os.path.join(PLOTS_DIR, "test_roc_curve.png")
pr_path  = os.path.join(PLOTS_DIR, "test_precision_recall_curve.png")
thr_path = os.path.join(PLOTS_DIR, "test_threshold_sweep.png")
cm_path  = os.path.join(PLOTS_DIR, "test_confusion_matrices.png")

# --- Row 1: ROC + PR (2 columns)
col1, col2 = st.columns(2)

with col1:
    st.markdown("**ROC Curve (Test Set)**")
    if os.path.exists(roc_path):
        st.image(Image.open(roc_path), use_column_width=True)
    else:
        st.warning("ROC curve not found")

with col2:
    st.markdown("**Precision-Recall Curve (Test Set)**")
    if os.path.exists(pr_path):
        st.image(Image.open(pr_path), use_column_width=True)
    else:
        st.warning("PR curve not found")

# --- Row 2: Centered Threshold Sweep
st.markdown("**Threshold Sweep**")

col_left, col_mid, col_right = st.columns([1, 2, 1])  # center emphasis

with col_mid:
    if os.path.exists(thr_path):
        st.image(Image.open(thr_path), use_column_width=True)
    else:
        st.warning("Threshold sweep not found")

# --- Row 3: Confusion Matrix full width
st.markdown("**Confusion Matrices**")

if os.path.exists(cm_path):
    st.image(Image.open(cm_path), use_column_width=True)
else:
    st.warning("Confusion matrix not found")


st.divider()
st.subheader("Grad-CAM Examples")

samples_dir = os.path.join(BASE_DIR, "assets", "samples")
if os.path.exists(samples_dir):
    tumor_samples  = sorted([
        f for f in os.listdir(samples_dir)
        if f.startswith("gradcam_tumor") and f.endswith(".png")
    ])[:2]
    normal_samples = sorted([
        f for f in os.listdir(samples_dir)
        if f.startswith("gradcam_normal") and f.endswith(".png")
    ])[:2]

    if tumor_samples:
        st.markdown("**Tumor predictions**")
        cols = st.columns(len(tumor_samples))
        for col, fname in zip(cols, tumor_samples):
            col.image(
                Image.open(os.path.join(samples_dir, fname)),
                use_column_width=True
            )

    if normal_samples:
        st.markdown("**Normal predictions**")
        cols = st.columns(len(normal_samples))
        for col, fname in zip(cols, normal_samples):
            col.image(
                Image.open(os.path.join(samples_dir, fname)),
                use_column_width=True
            )
else:
    st.info("No Grad-CAM samples found. Run api/test_gradcam.py first.")

st.divider()
st.subheader("Model Card")
st.markdown(
    """
    | Field | Details |
    |---|---|
    | **Intended use** | Research and educational purposes only |
    | **Input** | 96×96px RGB histopathology patch images |
    | **Output** | Binary classification with tumor probability and Grad-CAM |
    | **Training data** | PatchCamelyon — breast cancer lymph node patches |
    | **Known limitations** | Stain variation across hospitals may affect performance |
    | **Out-of-scope use** | Clinical diagnosis, real patient decisions |
    | **Fairness** | Not evaluated across demographic subgroups |
    """
)