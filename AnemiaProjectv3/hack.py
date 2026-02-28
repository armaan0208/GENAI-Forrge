# ==========================================
# 🩸 HEMOSCAN AI – Advanced Clinical Edition
# ==========================================

from flask import Flask, render_template, request, send_file
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER
from datetime import datetime

app = Flask(__name__)

# ===============================
# LOAD DATASET
# ===============================

file_path = r"D:\AnemiaProjectv3\anemia.csv"
data = pd.read_csv(file_path)

X = data[['Gender', 'Hemoglobin', 'MCH', 'MCHC', 'MCV']]
y = data['Result']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ===============================
# PROFESSIONAL PDF GENERATOR
# ===============================

def generate_pdf(patient_info, report_data, recommendation):

    filename = "HemoScan_Report.pdf"
    doc = SimpleDocTemplate(filename, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()

    # Header
    header = Table([["HEMOSCAN AI - CLINICAL SCREENING REPORT"]])
    header.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.darkred),
        ('TEXTCOLOR', (0,0), (-1,-1), colors.white),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTSIZE', (0,0), (-1,-1), 18),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ('TOPPADDING', (0,0), (-1,-1), 10),
    ]))

    elements.append(header)
    elements.append(Spacer(1, 15))
    elements.append(Paragraph(f"Generated On: {datetime.now()}", styles['Normal']))
    elements.append(Spacer(1, 15))

    # Patient Info
    elements.append(Paragraph("<b>Patient Information</b>", styles['Heading2']))
    elements.append(Spacer(1, 10))

    patient_table = Table(patient_info)
    patient_table.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 1, colors.grey),
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey)
    ]))

    elements.append(patient_table)
    elements.append(Spacer(1, 20))

    # Clinical Data
    elements.append(Paragraph("<b>Clinical Analysis</b>", styles['Heading2']))
    elements.append(Spacer(1, 10))

    table = Table(report_data)
    table.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 1, colors.grey),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.whitesmoke, colors.lightgrey])
    ]))

    elements.append(table)
    elements.append(Spacer(1, 20))

    # Recommendation
    elements.append(Paragraph("<b>Diet & Precaution Recommendation</b>", styles['Heading2']))
    elements.append(Spacer(1, 10))
    elements.append(Paragraph(recommendation, styles['Normal']))
    elements.append(Spacer(1, 20))

    elements.append(Paragraph(
        "Disclaimer: This AI-generated report is for screening purposes only. "
        "Consult a certified medical professional for diagnosis.",
        styles['Normal']
    ))

    doc.build(elements)
    return filename

# ===============================
# ROUTES
# ===============================

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    name = request.form['name']
    age = request.form['age']
    gender = int(request.form['gender'])
    hemoglobin = float(request.form['hemoglobin'])
    mch = float(request.form['mch'])
    mchc = float(request.form['mchc'])
    mcv = float(request.form['mcv'])

    new_data = np.array([[gender, hemoglobin, mch, mchc, mcv]])
    new_scaled = scaler.transform(new_data)
    probability = model.predict_proba(new_scaled)[0][1]

    # Reference ranges
    hb_range = (13.5, 17.5) if gender == 1 else (12, 15.5)
    mcv_range = (80, 100)
    mch_range = (27, 33)
    mchc_range = (32, 36)

    def check(value, normal):
        if value < normal[0]:
            return "Low"
        elif value > normal[1]:
            return "High"
        else:
            return "Normal"

    hb_status = check(hemoglobin, hb_range)
    mcv_status = check(mcv, mcv_range)
    mch_status = check(mch, mch_range)
    mchc_status = check(mchc, mchc_range)

    # Severity
    if hemoglobin >= hb_range[0]:
        severity = "Normal"
    elif hemoglobin >= 10:
        severity = "Mild Anemia"
    elif hemoglobin >= 7:
        severity = "Moderate Anemia"
    else:
        severity = "Severe Anemia"

    # Diet suggestion
    if mcv < 80:
        recommendation = "Increase iron-rich foods like spinach, lentils, red meat. Take Vitamin C to improve absorption."
    elif mcv > 100:
        recommendation = "Increase Vitamin B12 & folate sources like eggs, dairy, leafy vegetables."
    else:
        recommendation = "Maintain balanced diet and schedule regular health checkups."

    health_score = round(100 - (probability * 100), 2)

    patient_info = [
        ["Field", "Details"],
        ["Name", name],
        ["Age", age],
        ["Gender", "Male" if gender == 1 else "Female"]
    ]

    report_data = [
        ["Metric", "Value", "Normal Range"],
        ["Hemoglobin", hemoglobin, f"{hb_range[0]} - {hb_range[1]}"],
        ["MCV", mcv, "80 - 100"],
        ["MCH", mch, "27 - 33"],
        ["MCHC", mchc, "32 - 36"],
        ["Anemia Probability", f"{round(probability*100,2)}%", "-"],
        ["Severity", severity, "-"],
        ["Health Score", f"{health_score}/100", "-"]
    ]

    generate_pdf(patient_info, report_data, recommendation)

    return render_template(
        "index.html",
        probability=round(probability*100,2),
        severity=severity,
        health_score=health_score,
        recommendation=recommendation,
        hb_status=hb_status,
        mcv_status=mcv_status,
        mch_status=mch_status,
        mchc_status=mchc_status,
        name=name,
        age=age
    )

@app.route('/download')
def download():
    return send_file("HemoScan_Report.pdf", as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)