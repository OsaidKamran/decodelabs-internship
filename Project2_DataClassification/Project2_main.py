import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, f1_score

# ==========================================
# INPUT: The Iris Benchmark
# ==========================================
# 1. Load the Iris dataset (150 Samples, 3 Classes, 4 Dimensions)
iris = load_iris()
X = iris.data    # Features: Sepal Length, Sepal Width, Petal Length, Petal Width
y = iris.target  # Classes: Setosa (0), Versicolor (1), Virginica (2)

# ==========================================
# PROCESS: Train-Test Split & Scaling
# ==========================================
# 2. Structural Integrity: The Split
# Splitting the data into 80% Training and 20% Testing sets with random shuffling.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, shuffle=True
)

# 3. The Gatekeeper Rule: Scaling
# Standardizing the features to have a Mean of 0 and a Variance of 1.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. The Algorithm: K-Nearest Neighbors
# The workflow dictates instantiating the model with K=5, fitting it, and predicting.
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_scaled, y_train)           # Memorize the map
predictions = model.predict(X_test_scaled)   # Apply logic

# ==========================================
# OUTPUT: Validation
# ==========================================
# 5. The Diagnostic Tool: Confusion Matrix
# Evaluates True Positives, True Negatives, False Positives (Type I), and False Negatives (Type II)
print("--- OUTPUT VALIDATION ---")
conf_matrix = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(conf_matrix)

# 6. Strategic Trade-Offs: F1 Score
# The harmonic mean balancing Precision and Recall
f1 = f1_score(y_test, predictions, average='weighted')
print(f"\nF1 Score: {f1:.4f}")
