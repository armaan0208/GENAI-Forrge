# ==========================================
# 🩸 HEMOSCAN AI – Clinical Edition (No Graphs)
# ==========================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from datetime import datetime

# ==========================================
# 1. LOAD DATASET
# ==========================================

file_path = r"D:\AnemiaProject\anemia.csv"
data = pd.read_csv(file_path)

print("\n✅ Dataset Loaded Successfully")

# ==========================================
# 2. TRAIN MODEL
# ==========================================

X = data[['Gender', 'Hemoglobin', 'MCH', 'MCHC', 'MCV']]
y = data['Result']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))

print("📊 Model Accuracy:", round(accuracy * 100, 2), "%")

# ==========================================
# 3. PATIENT INPUT
# ==========================================

print("\n======================================")
print("        🩸 HEMOSCAN AI REPORT 🩸")
print("======================================")

gender = int(input("Gender (0=Female, 1=Male): "))
hemoglobin = float(input("Hemoglobin (g/dL): "))
mch = float(input("MCH (pg): "))
mchc = float(input("MCHC (g/dL): "))
mcv = float(input("MCV (fL): "))

new_data = np.array([[gender, hemoglobin, mch, mchc, mcv]])
new_scaled = scaler.transform(new_data)

prediction = model.predict(new_scaled)[0]
probability = model.predict_proba(new_scaled)[0][1]

# ==========================================
# 4. RISK & SEVERITY ANALYSIS
# ==========================================

if probability < 0.3:
    risk = "LOW RISK"
elif probability < 0.7:
    risk = "MODERATE RISK"
else:
    risk = "HIGH RISK"

if hemoglobin >= 13:
    severity = "Normal"
elif 10 <= hemoglobin < 13:
    severity = "Mild Anemia"
elif 7 <= hemoglobin < 10:
    severity = "Moderate Anemia"
else:
    severity = "Severe Anemia"

if mcv < 80:
    rbc_type = "Microcytic (Iron Deficiency Pattern)"
elif 80 <= mcv <= 100:
    rbc_type = "Normocytic"
else:
    rbc_type = "Macrocytic (B12/Folate Pattern)"

health_score = 100 - (probability * 100)

# ==========================================
# 5. PRINT TERMINAL SUMMARY
# ==========================================

print("\n------------- RESULT SUMMARY -------------")
print("Anemia Probability:", round(probability * 100, 2), "%")
print("Risk Category:", risk)
print("Severity Level:", severity)
print("RBC Classification:", rbc_type)
print("Health Score:", round(health_score, 2), "/100")

# ==========================================
# 6. GENERATE PROFESSIONAL PDF
# ==========================================

def generate_pdf():
    filename = "HemoScan_Clinical_Report.pdf"
    doc = SimpleDocTemplate(filename, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("HEMOSCAN AI - CLINICAL SCREENING REPORT", styles['Heading1']))
    elements.append(Spacer(1, 0.3 * inch))

    elements.append(Paragraph(f"Report Generated On: {datetime.now()}", styles['Normal']))
    elements.append(Spacer(1, 0.3 * inch))

    patient_data = [
        ["Parameter", "Value"],
        ["Gender", "Male" if gender == 1 else "Female"],
        ["Hemoglobin (g/dL)", hemoglobin],
        ["MCH (pg)", mch],
        ["MCHC (g/dL)", mchc],
        ["MCV (fL)", mcv],
    ]

    result_data = [
        ["Analysis Metric", "Result"],
        ["Anemia Probability", f"{round(probability*100,2)}%"],
        ["Risk Category", risk],
        ["Severity Level", severity],
        ["RBC Classification", rbc_type],
        ["Health Score", f"{round(health_score,2)}/100"]
    ]

    table1 = Table(patient_data)
    table2 = Table(result_data)

    table1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('GRID', (0,0), (-1,-1), 1, colors.grey)
    ]))

    table2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('GRID', (0,0), (-1,-1), 1, colors.grey)
    ]))

    elements.append(Paragraph("Patient Input Parameters", styles['Heading2']))
    elements.append(Spacer(1, 0.2 * inch))
    elements.append(table1)

    elements.append(Spacer(1, 0.4 * inch))
    elements.append(Paragraph("AI Analysis Results", styles['Heading2']))
    elements.append(Spacer(1, 0.2 * inch))
    elements.append(table2)

    elements.append(Spacer(1, 0.4 * inch))
    elements.append(Paragraph("Disclaimer: This AI system provides screening assistance only. Please consult a certified medical professional for diagnosis.", styles['Normal']))

    doc.build(elements)

    print("\n📄 Professional PDF Generated:", filename)

generate_pdf()

print("\n======================================")
print("⚕ HemoScan AI Screening Complete")
print("======================================")