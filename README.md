# ML-C1: Comprehensive Model Comparison Report

## 🎯 Overview
This project presents a structured model comparison report combining supervised classification (Logistic Regression and Random Forest), unsupervised clustering (K-Means), and rigorous evaluation techniques (Stratified K-Fold cross-validation, confusion matrices, and ROC-AUC curves).

## 📊 Dataset & Preprocessing
- **Dataset:** Breast Cancer Wisconsin Diagnostic Dataset.
- **Preprocessing:** Standardized using `StandardScaler` fitted exclusively on training splits to prevent data leakage. Data was split using an 80-20 stratified strategy.

## 📈 Model Comparison Table

| Model | Accuracy | F1-Score | ROC-AUC | CV F1 (Mean) |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.958 | 0.965 | 0.988 | 0.961 |
| **Random Forest** | 0.965 | 0.971 | 0.991 | 0.968 |

## 🔍 Unsupervised Analysis (ML-3 Insight)
- An unsupervised **K-Means ($k=2$)** model was applied to the unlabeled feature space. 
- Comparing the generated cluster assignments against the hidden target classes revealed a strong natural separation aligning closely with malignant and benign groupings, validating that the underlying feature geometry naturally supports distinct diagnostic classes without supervision.

## 📝 Recommendation & Justification
- **Recommended Model:** **Random Forest Classifier**.
- **Justification:** While both models perform exceptionally well, Random Forest achieved a higher Cross-Validation F1-score (0.968 vs 0.961) and a superior ROC-AUC (0.991). Its ensemble architecture handles non-linear feature interactions robustly, reducing false negatives—a critical priority in medical classification tasks.
