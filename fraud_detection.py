# ==============================
# Fraud Detection System
# ==============================

# 1. IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# ==============================
# 2. LOAD DATASET
# ==============================
print("Starting Fraud Detection Project...")

data = pd.read_csv("creditcard.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", data.shape)
print(data.head())

# ==============================
# 3. FRAUD VS NON-FRAUD ANALYSIS
# ==============================
fraud_counts = data['Class'].value_counts()
print("\nFraud vs Non-Fraud count:")
print(fraud_counts)

fraud_percentage = (fraud_counts[1] / fraud_counts.sum()) * 100
print(f"\nFraud transactions percentage: {fraud_percentage:.4f}%")

# ==============================
# 4. VISUALIZE CLASS IMBALANCE
# ==============================
plt.figure(figsize=(6, 4))
sns.countplot(x='Class', data=data)
plt.title("Fraud vs Non-Fraud Transactions")
plt.xlabel("Transaction Type (0 = Normal, 1 = Fraud)")
plt.ylabel("Count")
plt.show()

# ==============================
# 5. FEATURE SCALING
# ==============================
scaler = StandardScaler()

data['scaled_amount'] = scaler.fit_transform(data[['Amount']])
data['scaled_time'] = scaler.fit_transform(data[['Time']])

# Drop original Amount and Time columns
data = data.drop(['Amount', 'Time'], axis=1)

# ==============================
# 6. FEATURE / TARGET SPLIT
# ==============================
X = data.drop('Class', axis=1)
y = data['Class']

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)

# ==============================
# 7. TRAIN-TEST SPLIT (STRATIFIED)
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

print("\nTraining set class distribution:")
print(y_train.value_counts())

print("\nTesting set class distribution:")
print(y_test.value_counts())

# ==============================
# 8. TRAIN LOGISTIC REGRESSION
# ==============================
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

print("\nLogistic Regression model trained successfully.")

# ==============================
# 9. PREDICTION
# ==============================
y_pred_lr = lr_model.predict(X_test)

# ==============================
# 10. MODEL EVALUATION
# ==============================
print("\nConfusion Matrix (Logistic Regression):")
print(confusion_matrix(y_test, y_pred_lr))

print("\nClassification Report (Logistic Regression):")
print(classification_report(y_test, y_pred_lr))

# ==============================
# 11. SAMPLE PREDICTION (SIMULATION)
# ==============================
sample_transaction = X_test.iloc[0].values.reshape(1, -1)
prediction = lr_model.predict(sample_transaction)

print("\nSample Transaction Prediction:")
if prediction[0] == 1:
    print("🚨 Fraudulent Transaction")
else:
    print("✅ Legitimate Transaction")




from sklearn.ensemble import RandomForestClassifier

# ==============================
# 12. TRAIN RANDOM FOREST
# ==============================
rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

rf_model.fit(X_train, y_train)
print("\nRandom Forest model trained.")

# ==============================
# 13. RANDOM FOREST PREDICTIONS
# ==============================
y_pred_rf = rf_model.predict(X_test)

# ==============================
# 14. RANDOM FOREST EVALUATION
# ==============================
print("\nConfusion Matrix (Random Forest):")
print(confusion_matrix(y_test, y_pred_rf))

print("\nClassification Report (Random Forest):")
print(classification_report(y_test, y_pred_rf))




#compare

print("\n=== MODEL COMPARISON SUMMARY ===")
print("Logistic Regression:")
print(classification_report(y_test, y_pred_lr))

print("\nRandom Forest:")
print(classification_report(y_test, y_pred_rf))
print("\nProject execution completed.")