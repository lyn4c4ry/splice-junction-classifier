import pandas as pd
import kagglehub
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import numpy as np

# Load dataset dynamically via KaggleHub
path = kagglehub.dataset_download("muhammetvarl/splicejunction-gene-sequences-dataset")
df = pd.read_csv(os.path.join(path, "dna.csv"))

# Separate features and target
X = df.drop("class", axis=1)
y = df["class"]

# Stratified K-Fold cross-validation (5 folds)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define models to compare
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "MLP Classifier": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42)
}

# Train and evaluate each model
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=skf, scoring="f1_weighted")
    print(f"{name} -> F1: {scores.mean():.4f} +/- {scores.std():.4f}")