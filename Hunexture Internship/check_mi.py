import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif

# Load a sample for speed
df = pd.read_csv('d:/Hunexture Internship/diversified_ecommerce_dataset.csv').sample(20000, random_state=42)

le = LabelEncoder()
target = le.fit_transform(df['Seasonality'])

# Encode categorical columns
cat_cols = ['Category', 'Customer Age Group', 'Customer Location', 'Customer Gender', 'Shipping Method', 'Product Name', 'Supplier ID']
X = df.drop(['Seasonality', 'Product ID'], axis=1)

for col in cat_cols:
    X[col] = le.fit_transform(X[col].astype(str))

# Mutual information
mi = mutual_info_classif(X, target, discrete_features='auto', n_neighbors=3, random_state=42)
mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)

print("Mutual Information Scores:")
print(mi_series)

# Check if there's any obvious logic like Price > threshold
print("\nChecking Price vs Seasonality Correlation:")
print(df.groupby('Seasonality')['Price'].describe())

# Check for target leak in Product Name or Product ID
df['Product_ID_Start'] = df['Product ID'].str[0]
print("\nSeasonality by Product ID start character:")
print(df.groupby('Product_ID_Start')['Seasonality'].value_counts(normalize=True).unstack().head())
