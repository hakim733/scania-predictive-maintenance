# scania-predictive-maintenance
This work was developed for the IDA 2024 Scania Component X Predictive Maintenance Challenge, as part of the DT8059 course at Halmstad University.
---

## Methodology

### Problem Setup

- Predict the **Time-To-Event (TTE)** class based on the last 10 sensor readings for each vehicle.
- Original labels (Classes 0–4) were grouped into:
  - **Class 0**: Healthy (TTE > 48)
  - **Class 1**: Degrading (TTE 6–48, merged)
  - **Class 2**: Near failure (TTE ≤ 6)

### Architecture

- **GRU-based model**:
  - Input: Sequences of multivariate sensor readings + encoded vehicle specs
  - Layers: 1 GRU (hidden size = 128), followed by a fully connected output layer
  - Output: Class logits for 3 final categories (0, 1, 2)

### Handling Class Imbalance

- Merged rare classes to reduce sparsity
- Applied:
  - **Upsampling** of minority classes
  - **Class weighting** in the loss function
  - **Focal Loss** with γ=2.0 to focus on difficult examples

### Evaluation Metrics

- Accuracy
- Confusion Matrix
- Classification Report (precision, recall, F1)
- **Challenge-specific Cost Function** based on weighted penalties for misclassification

---

## Getting Started

### Requirements

- Python 3.10+
- PyTorch
- NumPy, Pandas, Scikit-learn
- Google Colab (recommended for resource capacity)

### Setup

```bash
pip install -r requirements.txt
