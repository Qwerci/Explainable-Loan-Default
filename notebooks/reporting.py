import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score

# Paths
model_path = r"C:\Users\theow\Documents\Project\Explainable-Loan-Default\models\best_xgb_model.pkl"
X_test_data_path = r"C:\Users\theow\Documents\Project\Explainable-Loan-Default\data\processed\X_test.csv"
y_test_data_path = r"C:\Users\theow\Documents\Project\Explainable-Loan-Default\data\processed\y_test.csv"
shap_values_path = r"C:\Users\theow\Documents\Project\Explainable-Loan-Default\results\shap_values.csv"
report_dir = r"C:\Users\theow\Documents\Project\Explainable-Loan-Default\reports"
os.makedirs(report_dir, exist_ok=True)

# Load model and test data
model = joblib.load(model_path)
X_test = pd.read_csv(X_test_data_path)
y_test = pd.read_csv(y_test_data_path)

# Predict and calculate metrics
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# Load SHAP values
shap_df = pd.read_csv(shap_values_path)

# Exclude the 'income_bracket' column from the SHAP values DataFrame for calculation
shap_df_numeric = shap_df.drop(columns=['income_bracket'])

# Calculate top features
top_features = shap_df_numeric.abs().mean().sort_values(ascending=False).head(5)

# Generate model card
model_card = f"""
Model Card: Loan Default Prediction Model

1. Model Overview:
   - Model Type: XGBoost Classifier
   - Dataset: Loan Applicant Data
   - Features Used: {X_test.shape[1]}
   - Target: Loan Default (0 = Non-default, 1 = Default)

2. Performance Metrics:
   - Accuracy: {accuracy:.2f}
   - ROC AUC: {roc_auc:.2f}
   - Precision: {precision:.2f}
   - Recall: {recall:.2f}

3. Top 5 Important Features (SHAP-based):
""" + "\n".join([f"   - {f}" for f in top_features.index]) + """

4. Explainability Insights:
   - SHAP shows that factors like income, credit history, and loan amount are major drivers.
   - Interpretability tools suggest increased risk with higher loan-to-income ratios.

5. Limitations:
   - Potential bias if dataset is unbalanced.
   - Assumes linearity in some financial relationships.

6. Income Brackets:
   - The dataset includes the following income brackets: {shap_df['income_bracket'].unique()}

"""

# Save model card
with open(os.path.join(report_dir, "model_card.txt"), "w") as f:
    f.write(model_card)

print("Reporting complete. Model card saved to results/model_card.txt")
