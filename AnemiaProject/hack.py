# ==========================================
# ANEMIA PREDICTION PROJECT
# File: hack.py
# Dataset: anemia.csv
# ==========================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# ==========================================
# 1. LOAD DATASET
# ==========================================

file_path = r"D:\AnemiaProject\anemia.csv"
data = pd.read_csv(file_path)

print("Dataset Loaded Successfully\n")
print(data.head())

# ==========================================
# 2. DEFINE FEATURES AND TARGET
# ==========================================

X = data[['Gender', 'Hemoglobin', 'MCH', 'MCHC', 'MCV']]
y = data['Result']

# ==========================================
# 3. SPLIT DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================================
# 4. FEATURE SCALING
# ==========================================

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================================
# 5. TRAIN MODEL
# ==========================================

model = LogisticRegression()
model.fit(X_train, y_train)

# ==========================================
# 6. MODEL EVALUATION
# ==========================================

y_pred = model.predict(X_test)

print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ==========================================
# 7. SAVE MODEL
# ==========================================

joblib.dump(model, "anemia_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel Saved Successfully!")

# ==========================================
# 8. ADVANCED HEALTH ANALYSIS
# ==========================================

print("\n======================================")
print("        🩸 HEMOSCAN AI REPORT 🩸")
print("======================================")

gender = int(input("Gender (0=Female, 1=Male): "))
hemoglobin = float(input("Hemoglobin (g/dL): "))
mch = float(input("MCH (pg): "))
mchc = float(input("MCHC (g/dL): "))
mcv = float(input("MCV (fL): "))

# Prepare input
new_data = np.array([[gender, hemoglobin, mch, mchc, mcv]])
new_data_scaled = scaler.transform(new_data)

prediction = model.predict(new_data_scaled)[0]
probability = model.predict_proba(new_data_scaled)[0][1]  # probability of anemia

print("\n--------------------------------------")
print("            ANALYSIS REPORT")
print("--------------------------------------")

print(f"Anemia Probability: {round(probability*100,2)}%")

# Severity based on Hemoglobin
if hemoglobin >= 13:
    severity = "Normal"
elif 10 <= hemoglobin < 13:
    severity = "Mild Anemia"
elif 7 <= hemoglobin < 10:
    severity = "Moderate Anemia"
else:
    severity = "Severe Anemia"

# MCV classification
if mcv < 80:
    mcv_type = "Microcytic (Possible Iron Deficiency)"
elif 80 <= mcv <= 100:
    mcv_type = "Normocytic"
else:
    mcv_type = "Macrocytic"

# Final result
if prediction == 1:
    status = "⚠️ Patient HAS Anemia"
else:
    status = "✅ Patient DOES NOT have Anemia"

print("\nFinal Prediction:")
print(status)

print("\nSeverity Level Based on Hemoglobin:")
print(severity)

print("\nRed Blood Cell Type (Based on MCV):")
print(mcv_type)

# Overall Health Score
health_score = 100 - (probability * 100)
print(f"\nEstimated Health Score: {round(health_score,2)} / 100")

print("\n======================================")
print("  ⚕️ This is an AI-assisted prediction.")
print("  Please consult a medical professional.")
print("======================================")