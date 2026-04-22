import pandas as pd
import kagglehub
import os

# Load dataset dynamically via KaggleHub
path = kagglehub.dataset_download("muhammetvarl/splicejunction-gene-sequences-dataset")
df = pd.read_csv(os.path.join(path, "dna.csv"))

# Display basic dataset information
print(f"Dataset shape: {df.shape}")
print(f"\nColumn names:\n{df.columns.tolist()}")
print(f"\nFirst 5 rows:\n{df.head()}")

# Check for missing values
print(f"\nMissing values:\n{df.isnull().sum().sum()} total")

# Class distribution
# 1 = EI (exon -> intron), 2 = IE (intron -> exon), 3 = N (neither)
print(f"\nClass distribution:\n{df['class'].value_counts()}")