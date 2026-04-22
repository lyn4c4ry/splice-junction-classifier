# Splice Junction Gene Sequence Classifier

A machine learning project for classifying DNA splice junction sequences using Random Forest and MLP classifiers.

## Dataset
[Splice Junction Gene Sequences](https://www.kaggle.com/datasets/muhammetvarl/splicejunction-gene-sequences-dataset) — 3,186 samples, 180 binary features.

**Classes:**
- `EI` — Exon to Intron transition
- `IE` — Intron to Exon transition
- `N` — Neither (no splice junction)

## Results

| Model | F1 Score |
|-------|----------|
| Random Forest | **0.9487 ± 0.0101** |
| MLP Classifier | 0.9434 ± 0.0121 |

Evaluated with Stratified K-Fold Cross-Validation (k=5).

## Project Structure