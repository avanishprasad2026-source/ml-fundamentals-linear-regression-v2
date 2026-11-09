import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier  # Or use Decision Tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# 1. Load a non-trivial real dataset
data = load_breast_cancer()
X, y = data.data, data.target

# 2. Consistent Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Preprocessing: Apply StandardScaler (vital for KNN, SVM, Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Define 5 Distinct Supervised Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Support Vector Machine": SVC(probability=True),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Naive Bayes": GaussianNB(),
}

# 5. Train and Evaluate each model consistently
results = []

for name, model in models.items():
  # Use scaled data for distance/linear models, unscaled works fine for trees but scaled is consistent here
  if name in ["Decision Tree", "Naive Bayes"]:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
  else:
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

  acc = accuracy_score(y_test, y_pred)
  f1 = f1_score(y_test, y_pred, average="weighted")

  results.append({"Model": name, "Accuracy": acc, "F1-Score": f1})
  print(f"Finished evaluating: {name}")

# 6. Build a comparison table ranking models
df_results = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
print("\n--- Model Comparison Ranking ---")
print(df_results.to_string(index=False))

# 7. Produce a comparison bar chart
plt.figure(figsize=(10, 6))
bars = plt.bar(
    df_results["Model"],
    df_results["Accuracy"],
    color=["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"],
)
plt.xlabel("Supervised Models", fontweight="bold")
plt.ylabel("Accuracy Score", fontweight="bold")
plt.title(
    "Comparative Study of 5 Supervised Learning Algorithms",
    fontsize=14,
    fontweight="bold",
)
plt.ylim(0.8, 1.0)
plt.xticks(rotation=20)

for bar in bars:
  yval = bar.get_height()
  plt.text(
      bar.get_x() + bar.get_width() / 2.0,
      yval - 0.02,
      f"{yval:.3f}",
      ha="center",
      va="bottom",
      color="white",
      fontweight="bold",
  )

plt.tight_layout()
plt.savefig("comparison_chart.png")
print("\nComparison chart saved successfully as 'comparison_chart.png'!")
