# Predictive Maintenance Pipeline

## Motivation

Unplanned equipment failures in industrial settings result in significant downtime, safety risks, and repair costs. Traditional scheduled maintenance can either be wasteful (too frequent) or risky (too infrequent). Our work addresses this challenge by forecasting time‑to‑failure from sensor streams and component specifications, enabling proactive interventions that minimize both unexpected outages and unnecessary service.

## Issue and Importance

* **High Costs**: Downtime can cost up to \$500k per hour in some industries.
* **Safety Concerns**: Sudden breakdowns on critical machinery can endanger personnel.
* **Resource Optimization**: Targeted maintenance reduces labor, parts, and operational disruptions.

Accurately predicting remaining life allows scheduling repairs precisely when needed—maximizing asset availability while controlling costs.

## Methods Under Investigation

We collaborate as a team of three, each proposing a distinct sequence-model approach:

1. **Bi-GRU with Attention** (Abdelhakim et al.)
2. **CNN-based Sequence Model**
3. **Transformer-based Sequence Model**

This README focuses on the Bi-GRU + Attention pipeline, with modular code to integrate the other methods seamlessly.

## Key Features

* **Six-bucket Time-to-Failure**: Labels range from *healthy* to *0–6h remaining* for fine-grained risk levels.
* **Cost-Sensitive Loss**: Early vs. late warning penalties encoded in a custom matrix.
* **Data Cleaning**: Forward/backward filling and NaN removal ensure robust inputs.
* **Sliding-Window Sequences**: Fixed-length windows capture temporal trends.
* **Bi-GRU + Attention**: Learns temporal dependencies and focuses on the most informative time steps.

## Pipeline Overview

```
flowchart TD
    A[Data Loading<br>(specs, ops, TTE/labels)] --> B[Cleaning<br>(ffill/bfill & drop NaNs)]
    B --> C[Labeling<br>(6-bucket class)]
    C --> D[Sequence Generation<br>(sliding windows)]
    D --> E[Model Training<br>(bi-GRU + Attention)]
    E --> F[Evaluation<br>(metrics & confusion matrix)]
```

## Project Structure

```
├── data/                   # Raw CSVs (specs, sensor readouts, labels)
├── notebooks/              # EDA and visualization notebooks
├── src/                    # Core code
│   ├── data_loader.py      # DataProcessor (merge, clean, label)
│   ├── model.py            # GRU+Attention architecture
│   ├── train.py            # Training & evaluation scripts
│   └── utils.py            # Helper functions
├── results/                # Figures, logs, and confusion matrices
├── README.md               # This file
└── references.bib          # Bibliography (Markdown links)
```

## Getting Started

1. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```
2. **Prepare data**: Place the SCANIA CSV files under `data/`.
3. **Train the model**:

   ```bash
   python src/train.py --config config/train_config.yaml
   ```
4. **Evaluate** on validation and test sets:

   ```bash
   python src/train.py --evaluate --checkpoint best_model.pth
   ```


## Methodology

1. **Data Loading & Cleaning**  
   Merge specifications, sensor streams, and TTE/label files by `vehicle_id`.  
   Forward-/back-fill missing readings and drop any remaining NaNs.

2. **Six-Bucket Labeling**  
   Compute `tte = length_of_study_time_step − time_step`, then bin into  
   `[-∞,0]→0`, `(0,6]→5`, `(6,12]→4`, `(12,24]→3`, `(24,48]→2`, `(48,∞)→1`.

3. **Sequence Generation**  
   Slide length-20 windows over each vehicle’s features; standardize across all windows.

4. **Model Architecture**  
   2-layer bidirectional GRU (hidden 256, dropout 0.3) + attention + MLP head (6 outputs).

5. **Training**  
   - Warm-start with plain cross-entropy to ensure stability.  
   - Fine-tune with cost-aware loss:  
     ```python
     CostAwareLoss = mean(CE * cost_matrix[true, pred])
     ```  
   - AdamW optimizer, ReduceLROnPlateau scheduler, gradient clipping, early stopping.

6. **Evaluation**  
   Six-class precision/recall/F1, 6×6 confusion matrix, and average cost per sample.

## Results

**Best Validation Accuracy:** 88.22 % (Epoch 5)

**Validation Metrics at Epoch 5:**

```text
Epoch 05 | Train Loss: 0.0947 | Val Loss: 0.7234 | Val Acc: 0.8822
                precision    recall  f1-score   support

**Classification Report (Test Set):

Healthy (tte ≤ 0)       0.98      0.99      0.98    108163
> 48h remaining         0.00      0.00      0.00       429
24–48h remaining        0.00      0.00      0.00       268
12–24h remaining        0.00      0.00      0.00       764
6–12h remaining         0.00      0.00      0.00      1259
0–6h remaining          0.00      0.00      0.00         0

accuracy                           0.96    110883
macro avg       0.16      0.16      0.16    110883
weighted avg    0.95      0.96      0.96    110883
**Confusion Matrix (Test Set):
[[106990    961    212      0      0      0]
 [   429      0      0      0      0      0]
 [   268      0      0      0      0      0]
 [   764      0      0      0      0      0]
 [  1192     56     11      0      0      0]
 [     0      0      0      0      0      0]]
```
# License
MIT License

Copyright (c) 2025 [Abdelhakim _Mraihi]
