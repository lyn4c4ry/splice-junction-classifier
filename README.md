# Splice Junction Gene Sequence Classifier

A machine learning project that classifies DNA splice junction sequences into three biological categories using Random Forest and MLP classifiers, evaluated with Stratified K-Fold Cross-Validation.

---

## What is a Splice Junction?

In molecular biology, splice junctions are boundaries between exons and introns in DNA sequences. Correctly identifying these junctions is critical for understanding gene expression and RNA processing.

**Classes:**
| Label | Meaning |
|-------|---------|
| EI | Exon to Intron transition |
| IE | Intron to Exon transition |
| N | Neither (no splice junction) |

---

## Dataset

[Splice Junction Gene Sequences](https://www.kaggle.com/datasets/muhammetvarl/splicejunction-gene-sequences-dataset) via Kaggle

- 3,186 samples
- 180 binary-encoded DNA position features
- Balanced EI/IE classes, N class is ~2x larger

---

## Results

Evaluated with **Stratified K-Fold Cross-Validation (k=5)**:

| Model | F1 Score (Weighted) |
|-------|---------------------|
| Random Forest | **0.9487 +/- 0.0101** |
| MLP Classifier | 0.9434 +/- 0.0121 |

Confusion matrices are saved in 
esults/confusion_matrices.png.

---

## Project Structure

\splice-junction-classifier/
├── data/
├── notebooks/
│   └── eda.ipynb           # Exploratory Data Analysis
├── results/
│   └── confusion_matrices.png
├── src/
│   ├── preprocess.py       # Dataset loading and inspection
│   ├── train.py            # Model training and comparison
│   └── evaluate.py         # Confusion matrix and classification report
├── .gitignore
├── requirements.txt
└── README.md
\
---

## Setup

\\ash
pip install -r requirements.txt
\
> Requires a Kaggle API key. Download \kaggle.json\ from your Kaggle account settings and place it in \~/.kaggle/\.

---

## Usage

\\ash
python src/preprocess.py   # Explore the dataset
python src/train.py        # Train and compare models
python src/evaluate.py     # Generate confusion matrices
\
---

## Tech Stack

- Python 3.11
- scikit-learn
- pandas
- matplotlib / seaborn
- kagglehub
