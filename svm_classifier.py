import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Load CSV
data = pd.read_csv("runs/detect/train3/results.csv").fillna(0)

# Features
X = data[["metrics/precision(B)", "metrics/recall(B)"]].astype(float)

# Target (example: binary classification based on precision)
y = (data["metrics/precision(B)"] > 0.5).astype(int)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Train SVM
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

# Accuracy
print("SVM Accuracy:", svm.score(X_test, y_test))
