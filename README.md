Project Title: Predictive Maintenance using Bi-GRU with Attention and Cost-Sensitive Learning

Overview
This repository implements a predictive maintenance pipeline for time-to-failure classification, based on multivariate sensor time-series and component specifications. We follow the SCANIA dataset and methodology, dividing remaining life into six buckets and training a bidirectional GRU with attention. A cost-sensitive loss is included to penalize early or late warnings differently.

Authors

Abdelhakim Mraihi (Algorithm: Bi-GRU + Attention)


Table of Contents

Project Structure

Requirements

Installation

Data Preparation

Pipeline Overview

Usage

Model Architecture

Training & Evaluation

Results

Licence

References

Project Structure

├── data/
│   ├── train_specifications.csv
│   ├── train_operational_readouts.csv
│   ├── train_tte.csv
│   ├── validation_specifications.csv
│   ├── validation_operational_readouts.csv
│   ├── validation_labels.csv
│   ├── test_specifications.csv
│   ├── test_operational_readouts.csv
│   └── test_labels.csv
├── notebooks/           # Jupyter notebooks for EDA and visualization
├── src/                 # Source code
│   ├── data_loader.py   # DataProcessor class
│   ├── model.py         # GRU+Attention model definition
│   ├── train.py         # Training & evaluation scripts
│   └── utils.py         # Helper functions
├── results/             # Generated figures and logs
├── README.md            # This file
└── references.bib       # Bibliography file

Requirements

Python 3.8+

pandas

numpy

torch

scikit-learn

joblib

Install via:

pip install -r requirements.txt

Data Preparation

Download the SCANIA dataset CSVs into data/ (specifications, operational readouts, TTE/labels).

Merge & Clean: The DataProcessor in src/data_loader.py handles merging specs, sensor readings, and labels, followed by forward/back-filling and dropping NaNs.

Pipeline Overview

See Figure \ref{fig:pipeline} in the paper for a high-level flow. Steps:

Load & merge raw CSVs

Clean missing values

Compute six time-to-failure buckets

Generate sliding-window sequences

Train Bi-GRU + Attention model

Evaluate with six-class metrics & cost-sensitive loss

Usage

Training

python src/train.py \
  --config config/train_config.yaml

Evaluation

python src/train.py --evaluate --checkpoint best_model.pth

Model Architecture

GRU: 2-layer bidirectional, hidden dim=256, dropout=0.3

Attention: learnable weights over time-steps

Classifier: MLP (64→6 logits)

Loss: Cross-Entropy (and optional CostAwareLoss)

Training & Evaluation

Hyperparameters: see config/train_config.yaml

Scheduler: ReduceLROnPlateau on validation loss

Early stopping: patience=5 epochs

Metrics: Classification report, confusion matrix, average cost per sample

Results

Peak validation accuracy: ~88 % at epoch 5 with CE loss

Stable separation of six buckets above chance (16.7 %)

Confusion matrices and cost curves in results/

Licence

MIT License

References

SCANIA Component X dataset and cost function. Scientific Data.

Abdelhakim et al., "Bi-GRU with Attention for Predictive Maintenance".

Yashwanth S., "Deep Learning Series #13: GRU (Gated Recurrent Unit)", Medium.\cite{yashwanth2019gru}

