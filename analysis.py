import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
iris_data = pd.read_csv(
    'Iris-Data.csv',
    names=['attr1', 'attr2', 'attr3', 'attr4', 'class']
)

# Convert feature columns to numeric
feature_cols = ['attr1', 'attr2', 'attr3', 'attr4']
iris_data[feature_cols] = iris_data[feature_cols].apply(pd.to_numeric, errors='coerce')

# Drop rows with missing numeric values
iris_data = iris_data.dropna(subset=feature_cols)

# -----------------------------
# Task 1: Distribution by class
# -----------------------------
class_distribution = iris_data['class'].value_counts()
class_distribution.plot(kind='barh', color=['red', 'green', 'blue'])
plt.title('Distribution of Records for Each Class')
plt.xlabel('Number of Records')
plt.ylabel('Class')
plt.tight_layout()
plt.show()

# Split data by class
setosa_data = iris_data[iris_data['class'] == 'Iris-setosa'].copy()
versicolor_data = iris_data[iris_data['class'] == 'Iris-versicolor'].copy()
virginica_data = iris_data[iris_data['class'] == 'Iris-virginica'].copy()

# -------------------------------------------------------------
# Task 2: Correlation between attr1 and attr2 for Iris-setosa
# -------------------------------------------------------------
corr_np = np.corrcoef(setosa_data['attr1'], setosa_data['attr2'])[0, 1]
corr_pd = setosa_data[['attr1', 'attr2']].corr(method='pearson').iloc[0, 1]

plt.scatter(setosa_data['attr1'], setosa_data['attr2'])
plt.title(f'Iris-setosa: attr1 vs attr2 (Correlation: {corr_np:.4f})')
plt.xlabel('attr1')
plt.ylabel('attr2')
plt.tight_layout()
plt.show()

print(f'Correlation using NumPy (setosa attr1 vs attr2): {corr_np:.4f}')
print(f'Correlation using Pandas (setosa attr1 vs attr2): {corr_pd:.4f}')

# -------------------------------------------------------------------
# Task 3: Correlation between Iris-setosa attr1 and versicolor attr1
# -------------------------------------------------------------------
min_len = min(len(setosa_data), len(versicolor_data))
setosa_attr1 = setosa_data['attr1'].iloc[:min_len].reset_index(drop=True)
versicolor_attr1 = versicolor_data['attr1'].iloc[:min_len].reset_index(drop=True)

corr_setosa_versicolor = np.corrcoef(setosa_attr1, versicolor_attr1)[0, 1]

plt.scatter(setosa_attr1, versicolor_attr1)
plt.title(f'Setosa vs Versicolor: attr1 vs attr1 (Correlation: {corr_setosa_versicolor:.4f})')
plt.xlabel('attr1 (Setosa)')
plt.ylabel('attr1 (Versicolor)')
plt.tight_layout()
plt.show()

print(f'Correlation using NumPy (setosa attr1 vs versicolor attr1): {corr_setosa_versicolor:.4f}')

# -----------------------------------------------------------------------
# Task 4: Correlation between Iris-versicolor attr1 and virginica attr1
# -----------------------------------------------------------------------
min_len = min(len(versicolor_data), len(virginica_data))
versicolor_attr1 = versicolor_data['attr1'].iloc[:min_len].reset_index(drop=True)
virginica_attr1 = virginica_data['attr1'].iloc[:min_len].reset_index(drop=True)

corr_versicolor_virginica = np.corrcoef(versicolor_attr1, virginica_attr1)[0, 1]

plt.scatter(versicolor_attr1, virginica_attr1)
plt.title(f'Versicolor vs Virginica: attr1 vs attr1 (Correlation: {corr_versicolor_virginica:.4f})')
plt.xlabel('attr1 (Versicolor)')
plt.ylabel('attr1 (Virginica)')
plt.tight_layout()
plt.show()

print(f'Correlation using NumPy (versicolor attr1 vs virginica attr1): {corr_versicolor_virginica:.4f}')

# ---------------------------------------------------------
# Tasks 5-7: Correlations of attr1 with other attributes
# ---------------------------------------------------------
correlation_setosa = setosa_data[feature_cols].corr(method='pearson')['attr1'].drop('attr1')
correlation_versicolor = versicolor_data[feature_cols].corr(method='pearson')['attr1'].drop('attr1')
correlation_virginica = virginica_data[feature_cols].corr(method='pearson')['attr1'].drop('attr1')

print("\nCorrelations for Iris-setosa:")
print(correlation_setosa)

print("\nCorrelations for Iris-versicolor:")
print(correlation_versicolor)

print("\nCorrelations for Iris-virginica:")
print(correlation_virginica)

# ---------------------------------------------------------
# Task 8: Discussion
# ---------------------------------------------------------
print("\nDiscussion:")
print("Higher absolute correlation values indicate stronger linear relationships.")
print("Attributes with stronger correlations may be more useful for distinguishing flower classes,")
print("but correlation alone does not prove that a feature is the best classifier.")