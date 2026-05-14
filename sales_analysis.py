import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("cleaned_dataset.csv")

# Display first 5 rows
print(df.head())

# Dataset information
print(df.info())

# Statistical summary
print(df.describe())

# Check missing values
print(df.isnull().sum())

# Correlation matrix
print(df.corr(numeric_only=True))

# Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Sales distribution
plt.figure(figsize=(8,5))
sns.histplot(df.select_dtypes(include=np.number).iloc[:,0], kde=True)
plt.title("Sales Distribution")
plt.show()