import pandas as pd
import kagglehub
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report

# Load dataset dynamically via KaggleHub
path = kagglehub.dataset_download("muhammetvarl/splicejunction-gene-sequences-dataset")
df = pd.read_csv(os.path.join(path, "dna.csv"))

# Separate features and target
X = df.drop("class", axis=1)
y = df["class"]

# Class labels: EI (exon->intron), IE (intron->exon), N (neither)
class_names = ["EI", "IE", "N"]

# Stratified K-Fold cross-validation (5 folds)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define models to evaluate
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "MLP Classifier": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42)
}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, (name, model) in zip(axes, models.items()):
    # Generate cross-validated predictions
    y_pred = cross_val_predict(model, X, y, cv=skf)

    # Plot confusion matrix
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(f"{name} - Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    # Print classification report
    print(f"\n{name}:\n{classification_report(y, y_pred, target_names=class_names)}")

plt.tight_layout()
# Create results directory if it doesn't exist
os.makedirs("results", exist_ok=True)
plt.savefig("results/confusion_matrices.png", dpi=150)
plt.show()
print("Saved: results/confusion_matrices.png")