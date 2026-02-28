# ==========================================
# 🩸 HEMOSCAN AI – Advanced Clinical Edition
# Folder: AnemiaProjectv4
# ==========================================

from flask import Flask, render_template, request, send_file
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from datetime import datetime
import random

app = Flask(__name__)

# ===============================
# LOAD & TRAIN MODEL
# ===============================

file_path = r"D:\AnemiaProjectv4\anemia.csv"   # 🔥 UPDATED PATH
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
# PDF GENERATOR
# ===============================

def generate_pdf(patient_info, report_data, recommendation):

    filename = "HemoScan_Report.pdf"
    doc = SimpleDocTemplate(filename, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()

    # HEADER
    header = Table([["AI HEMOSCAN AI - CLINICAL SCREENING REPORT"]])
    header.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#1f4e79")),
        ('TEXTCOLOR', (0,0), (-1,-1), colors.white),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTSIZE', (0,0), (-1,-1), 18),
        ('BOTTOMPADDING', (0,0), (-1,-1), 12),
        ('TOPPADDING', (0,0), (-1,-1), 12),
    ]))

    elements.append(header)
    elements.append(Spacer(1, 15))

    report_id = f"HS-{datetime.now().strftime('%Y%m%d')}-{random.randint(100,999)}"

    elements.append(Paragraph(
        f"<b>Report ID:</b> {report_id} | "
        f"<b>Generated On:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        styles['Normal']
    ))
    elements.append(Spacer(1, 20))

    # PATIENT INFO
    elements.append(Paragraph("<b>PATIENT INFORMATION</b>", styles['Heading3']))
    elements.append(Spacer(1, 8))

    patient_table = Table(patient_info, colWidths=[2*inch, 3*inch])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#d9e1f2")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.whitesmoke, colors.lightgrey])
    ]))

    elements.append(patient_table)
    elements.append(Spacer(1, 20))

    # CLINICAL ANALYSIS
    elements.append(Paragraph("<b>CLINICAL ANALYSIS</b>", styles['Heading3']))
    elements.append(Spacer(1, 8))

    clinical_table = Table(report_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
    clinical_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#305496")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('ALIGN', (1,1), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.whitesmoke, colors.HexColor("#e9f1f7")])
    ]))

    elements.append(clinical_table)
    elements.append(Spacer(1, 20))

    # RECOMMENDATION
    elements.append(Paragraph("<b>DIET & PRECAUTIONS</b>", styles['Heading3']))
    elements.append(Spacer(1, 10))

    rec_table = Table([[recommendation]], colWidths=[6*inch])
    rec_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#d9ead3")),
        ('BOX', (0,0), (-1,-1), 1, colors.green),
    ]))

    elements.append(rec_table)
    elements.append(Spacer(1, 20))

    elements.append(Paragraph(
        "<font size=9>Disclaimer: AI-based screening tool. Consult doctor for diagnosis.</font>",
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

    if hemoglobin >= hb_range[0]:
        severity = "Normal"
    elif hemoglobin >= 10:
        severity = "Mild Anemia"
    elif hemoglobin >= 7:
        severity = "Moderate Anemia"
    else:
        severity = "Severe Anemia"

    if mcv < 80:
        recommendation = "Increase iron-rich foods like spinach, lentils, red meat. Add Vitamin C."
    elif mcv > 100:
        recommendation = "Increase Vitamin B12 & folate intake like eggs, dairy, leafy vegetables."
    else:
        recommendation = "Maintain balanced diet and regular checkups."

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