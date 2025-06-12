<<<<<<< Updated upstream
from flask import Flask, render_template, request
from model import calculate_tdee, recommend_meals

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        age = int(request.form["age"])
        weight = float(request.form["weight"])
        height = float(request.form["height"])
        gender = request.form["gender"]
        activity = request.form["activity"]
        goal = request.form["goal"]
=======
from flask import Flask, render_template, request, jsonify, redirect, url_for

import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Path model
MODEL_DIR = 'model'
MODEL_NAME = 'obesity_knn_pipeline.pkl'  # hapus titik (.) setelah 'obesity'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.pkl')

# Load model dan encoder
model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

# Load data untuk ambil nilai dropdown
df = pd.read_csv('data/obesity.csv')

# Ambil opsi unik untuk dropdown dari dataset
dropdown_options = {
    'Gender': sorted(df['Gender'].unique()),
    'CALC': sorted(df['CALC'].unique()),
    'FAVC': sorted(df['FAVC'].unique()),
    'SCC': sorted(df['SCC'].unique()),
    'SMOKE': sorted(df['SMOKE'].unique()),
    'family_history_with_overweight': sorted(df['family_history_with_overweight'].unique()),
    'CAEC': sorted(df['CAEC'].unique()),
    'MTRANS': sorted(df['MTRANS'].unique()),
}


@app.route('/start')
def start():
    return render_template('start.html')

@app.route('/')
def home():
    return redirect(url_for('start'))  # Arahkan ke /start

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        try:
            # Ambil nilai dari form
            input_data = {
                'Age': int(request.form['Age']),
                'Gender': request.form['Gender'],
                'Height': float(request.form['Height']),
                'Weight': float(request.form['Weight']),
                'CALC': request.form['CALC'],
                'FAVC': request.form['FAVC'],
                'FCVC': float(request.form['FCVC']),
                'NCP': float(request.form['NCP']),
                'SCC': request.form['SCC'],
                'SMOKE': request.form['SMOKE'],
                'CH2O': float(request.form['CH2O']),
                'family_history_with_overweight': request.form['family_history_with_overweight'],
                'FAF': float(request.form['FAF']),
                'TUE': float(request.form['TUE']),
                'CAEC': request.form['CAEC'],
                'MTRANS': request.form['MTRANS']
            }
>>>>>>> Stashed changes

        tdee = calculate_tdee(gender, weight, height, age, activity, goal)
        meals = recommend_meals(tdee)

        return render_template("index.html", tdee=tdee, meals=meals)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
