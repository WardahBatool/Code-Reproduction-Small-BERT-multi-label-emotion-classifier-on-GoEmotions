
# Reproduction: Small-BERT multi-label emotion classifier on GoEmotions

Course: Deep Learning (AIC 401), Assignment 02, Bahria University Islamabad (BSCS 7A)
Students: Syedah Wardah Batool (01-134232-239), Husna Rahman (01-134232-217)

Reproduces: I. Bashynska, M. Sarafanov, O. Manikaeva, "Research and Development of a Modern Deep Learning
Model for Emotional Analysis Management of Text Data", Applied Sciences 14(5):1952, 2024.
DOI: https://doi.org/10.3390/app14051952   Authors' code (MIT): https://github.com/Shoomaher/sentiment-analysis

## Result (single-label test set, n = 4,666)
| Metric | Paper | Ours |
|---|---|---|
| F1 micro | 0.5835 | 0.5478 |
| F1 macro | 0.5070 | 0.4374 |
| AUC micro / macro | 0.9361 / 0.9136 | 0.9211 / 0.8833 |
Full table: comparison_table_final.csv. Per-class: per_class_final.csv. This is a close but not exact reproduction.

## Dataset
GoEmotions (Google Research): https://github.com/google-research/google-research/tree/master/goemotions
The notebook downloads train.tsv, dev.tsv and test.tsv into ./goemo with wget. Data is not stored in this repo.
Split logic adapted from the authors' utils.py (class-wise 85% split, single-label test, Neutral removed,
oversampling threshold 500, seed 42).

## How to run
1. Open notebook.ipynb in Google Colab with a GPU runtime (we used a Tesla T4).
2. Run in this order: setup cell, data cell, standardisation cell, training and evaluation cell.
   The first cells that use the Hugging Face "go_emotions" dataset are a preliminary run (see report).
3. Runtime: a few minutes. Outputs: comparison_table_final.csv, per_class_final.csv, baseline_final.pt,
   baseline_config.json, val/test probability .npy files, figures.
Install locally with: pip install -r requirements.txt

## Model and training settings (baseline_config.json)
Small BERT google/bert_uncased_L-2_H-128_A-2, Dropout 0.4, Dense 256 ReLU, Dropout 0.4, 27 sigmoid outputs,
BCE loss, AdamW lr 3e-4 (10% linear warm-up, linear decay), batch 64, up to 10 epochs, early stopping
patience 2, threshold 0.5, max length 128, weight decay 0.01, epsilon 1e-6, seed 42.
Final checkpoint = last epoch (7), chosen by validation macro-F1.

## Differences from the paper
PyTorch/Hugging Face instead of TensorFlow Hub; Python text standardisation with a shorter chat-word list and
no "/r" tag removal; unknown authors' seed; assumed weight decay, epsilon and max length; sentiment-level
metrics not reproduced.
