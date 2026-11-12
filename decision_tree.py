
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Bank Marketing Dataset
df = pd.read_csv("bank.csv", sep=";")

# Display first 5 rows
print("First 5 Rows:")
print(df.head())

# Dataset Information
print("\nDataset Information:")
df.info()

# Dataset Shape
print("\nDataset Shape:")
print(df.shape)

# Column Names
print("\nColumn Names:")
print(df.columns.tolist())


# Check Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Check Duplicate Records
print("\nDuplicate Records:")
print(df.duplicated().sum())

# Remove Duplicate Records
df = df.drop_duplicates()

# Convert Categorical Columns into Numerical Values
df = pd.get_dummies(df, drop_first=True, dtype=int)

# Display Processed Data
print("\nProcessed Dataset:")
print(df.head())

print("\nProcessed Dataset Shape:")
print(df.shape)


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Separate Features and Target
X = df.drop('y_yes', axis=1)
y = df['y_yes']

# Split Dataset into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create Decision Tree Model
model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=5,
    random_state=42
)

# Train the Model
model.fit(X_train, y_train)

print("\nModel Training Completed Successfully!")

# Make Predictions
y_pred = model.predict(X_test)

# Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy * 100, "%")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# Step 6: Graph Visualization

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.tree import plot_tree

# 1. Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 5))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=['No', 'Yes']
)
disp.plot(cmap='Blues', values_format='d')
plt.title("Decision Tree - Confusion Matrix")
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.show()

# 2. Decision Tree Visualization
plt.figure(figsize=(20, 10))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=['No', 'Yes'],
    filled=True,
    rounded=True,
    fontsize=7
)

plt.title("Decision Tree Classifier")
plt.savefig("decision_tree_graph.png", dpi=300, bbox_inches='tight')
plt.show()

print("\nGraph Visualization Completed Successfully!")
