import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load Dataset
data = load_breast_cancer()
X, y = data.data, data.target

# 2. Preprocessing & Splitting (ML-4 requirement)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Supervised Models (ML-2 requirement: 2 distinct algorithms)
supervised_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
}

# Stratified K-Fold Evaluation (ML-4)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

fig, ax_roc = plt.subplots(figsize=(8, 6))

for name, model in models := supervised_models.items():
  # Cross-Validation Score
  cv_scores = cross_val_score(
      model, X_train_scaled, y_train, cv=skf, scoring="f1"
  )

  # Train and test evaluation
  model.fit(X_train_scaled, y_train)
  y_pred = model.predict(X_test_scaled)
  y_prob = model.predict_proba(X_test_scaled)[:, 1]

  acc = accuracy_score(y_test, y_pred)
  f1 = f1_score(y_test, y_pred)
  roc_auc = roc_auc_score(y_test, y_prob)
  cm = confusion_matrix(y_test, y_pred)

  print(f"\n--- {name} ---")
  print(f"Confusion Matrix:\n{cm}")
  print(f"CV F1 Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

  results.append({
      "Model": name,
      "Accuracy": acc,
      "F1-Score": f1,
      "ROC-AUC": roc_auc,
      "CV F1 (Mean)": cv_scores.mean(),
  })

  # ROC Curve Plotting
  fpr, tpr, _ = roc_curve(y_test, y_prob)
  ax_roc.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")

ax_roc.plot([0, 1], [0, 1], "k--", label="Random Guess")
ax_roc.set_title("Supervised Models ROC Curves", fontweight="bold")
ax_roc.set_xlabel("False Positive Rate", fontweight="bold")
ax_roc.set_ylabel("True Positive Rate", fontweight="bold")
ax_roc.legend()
ax_roc.grid(True, linestyle="--", alpha=0.5)
plt.savefig("comparison_roc.png")

# 4. Unsupervised Technique (ML-3 requirement: K-Means clustering comparison)
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled_full := scaler.fit_transform(X))
print(
    "\nUnsupervised K-Means clustering complete. Clusters mapped to evaluate"
    " alignment with target classes."
)

# 5. Summary Table DataFrame
df_results = pd.DataFrame(results)
print("\n=== Final Model Comparison Table ===")
print(df_results.to_string(index=False))
