import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

# 1. Create a synthetic classification dataset with Class Imbalance (e.g., 90% negative, 10% positive)
X, y = make_classification(
    n_samples=2000,
    n_features=20,
    weights=[0.9, 0.1],
    flip_y=0.01,
    random_state=42,
)

# Hold-out test split for confusion matrix and curve plotting
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 2. Define 2 Classifiers to Compare
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
}

# 3. Implement Stratified k-Fold Cross-Validation (k=5)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("--- Stratified 5-Fold Cross-Validation Results ---")
for name, model in models.items():
  cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="f1")
  print(
      f"{name}: Mean F1 = {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})"
  )

# 4. Train models on training set for curve generation & confusion matrices
fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 6))

for name, model in models.items():
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  y_prob = model.predict_proba(X_test)[:, 1]

  # Confusion Matrix evaluation
  cm = confusion_matrix(y_test, y_pred)
  print(f"\nConfusion Matrix for {name}:\n", cm)

  # ROC Curve & AUC
  fpr, tpr, _ = roc_curve(y_test, y_prob)
  auc_score = roc_auc_score(y_test, y_prob)
  ax_roc.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.3f})")

  # Precision-Recall Curve
  precision, recall, _ = precision_recall_curve(y_test, y_prob)
  pr_auc = auc(recall, precision)
  ax_pr.plot(recall, precision, label=f"{name} (PR-AUC = {pr_auc:.3f})")

# Finalize ROC Plot
ax_roc.plot([0, 1], [0, 1], "k--", label="Random Guess")
ax_roc.set_title("ROC Curves", fontweight="bold")
ax_roc.set_xlabel("False Positive Rate", fontweight="bold")
ax_roc.set_ylabel("True Positive Rate", fontweight="bold")
ax_roc.legend()
ax_roc.grid(True, linestyle="--", alpha=0.5)

# Finalize Precision-Recall Plot
ax_pr.set_title("Precision-Recall Curves", fontweight="bold")
ax_pr.set_xlabel("Recall", fontweight="bold")
ax_pr.set_ylabel("Precision", fontweight="bold")
ax_pr.legend()
ax_pr.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("evaluation_curves.png")
print(
    "\nEvaluation curves saved successfully as 'evaluation_curves.png'!"
)
