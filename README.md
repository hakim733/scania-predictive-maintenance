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

```mermaid
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

## Results

* **Validation accuracy** peaks at \~88% after 5 epochs.
* **Average cost per sample** improves significantly over baseline once cost-sensitive training is enabled.
* Detailed confusion matrices and loss/accuracy plots are in `results/`.

## Next Steps

* Integrate CNN and Transformer methods in the same pipeline.
* Experiment with advanced regularization and ensembling.
* Deploy in a streaming environment for real-time alerts.

## License

This project is released under the MIT License.

## References

* [Deep Learning Series #13: GRU (Gated Recurrent Unit)](https://medium.com/@yashwanths_29644/deep-learning-series-13-gru-gated-recurrent-unit-7374776329c7)
* SCANIA Component X dataset and cost function (Scientific Data)
