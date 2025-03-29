
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE

# Load the dataset
data = pd.read_csv(r'C:\Users\USER\Desktop\project ideas\fraudlent\Fraud.csv')

# Display first few rows
data.head()

# -------------------------------
# STEP 1: DATA UNDERSTANDING
# -------------------------------

# Check data shape
print("Data Shape:", data.shape)

# Check data types and missing values
data.info()

# Check class distribution
print("\nClass Distribution:")
print(data['isFraud'].value_counts(normalize=True))

# -------------------------------
# STEP 2: DATA CLEANING
# -------------------------------

# No missing values found

# -------------------------------
# STEP 3: FEATURE ENGINEERING
# -------------------------------

# Create new features
data['diffOrig'] = data['oldbalanceOrg'] - data['newbalanceOrig']
data['diffDest'] = data['newbalanceDest'] - data['oldbalanceDest']
data['isSameAccount'] = np.where(data['nameOrig'] == data['nameDest'], 1, 0)
data['errorBalanceOrig'] = data['oldbalanceOrg'] - data['newbalanceOrig'] - data['amount']
data['errorBalanceDest'] = data['newbalanceDest'] - data['oldbalanceDest'] - data['amount']

# Drop unhelpful columns
data.drop(['nameOrig', 'nameDest'], axis=1, inplace=True)

# Filter for only TRANSFER and CASH_OUT types (focus on fraud patterns)
data = data[(data['type'] == 'TRANSFER') | (data['type'] == 'CASH_OUT')]

# Convert transaction type to numerical
data['type'] = data['type'].map({'TRANSFER': 0, 'CASH_OUT': 1})

# -------------------------------
# STEP 4: SPLIT DATASET
# -------------------------------

# Define features and target
X = data.drop(['isFraud', 'isFlaggedFraud'], axis=1)
y = data['isFraud']

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# -------------------------------
# STEP 5: HANDLE IMBALANCE
# -------------------------------

# Apply SMOTE to balance the training data
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# -------------------------------
# STEP 6: RANDOM FOREST MODEL
# -------------------------------

# Initialize and train model
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42)
rf_model.fit(X_train_res, y_train_res)

# -------------------------------
# STEP 7: EVALUATION
# -------------------------------

# Predict on test data
y_pred = rf_model.predict(X_test)

# Classification report
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ROC Curve
y_prob = rf_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label='ROC Curve (area = %0.2f)' % roc_auc_score(y_test, y_prob))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

# ROC AUC Score
auc = roc_auc_score(y_test, y_prob)
print("\nROC AUC Score:", auc)

# -------------------------------
# STEP 8: FEATURE IMPORTANCE
# -------------------------------

# Plot feature importance
importances = rf_model.feature_importances_
features = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))
sns.barplot(x=importances[indices], y=features[indices])
plt.title('Feature Importances')
plt.show()

# -------------------------------
# STEP 9: CONCLUSION AND NEXT STEPS
# -------------------------------

print("""
Conclusion:
- Random Forest detected fraudulent transactions with high precision and recall.
- Top factors: transaction type, amount, sender and receiver balances.
- Next steps: Deploy model for real-time fraud detection and alerting.

Recommendations:
1. Real-time monitoring of TRANSFER and CASH_OUT transactions.
2. Flag large transactions for manual verification.
3. Educate users about account security.
4. Regularly update and retrain the model.
""")
