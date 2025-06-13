from flask import Flask, render_template, request
import joblib
import numpy as np
import json

app = Flask(__name__)

# Load trained model
model = joblib.load('model.pkl')

# Load metrics (especially MAE)
with open('metrics.json', 'r') as f:
    metrics = json.load(f)
    mae = metrics.get('mae', 0)

# Feature names sesuai urutan input
feature_names = [
    'Age',
    'Diabetes',
    'BloodPressureProblems',
    'AnyTransplants',
    'AnyChronicDiseases',
    'Height',
    'Weight',
    'KnownAllergies',
    'HistoryOfCancerInFamily',
    'NumberOfMajorSurgeries'
]

# Fitur yang dianggap bisa dikendalikan
controllable_features = [
    # 'Diabetes',
    'BloodPressureProblems',
    # 'AnyTransplants',
    # 'AnyChronicDiseases',
    'Weight',
    # 'KnownAllergies',
    # 'NumberOfMajorSurgeries'
]


def simulate_savings(user_input, model, predicted_price):
        recommendations = []
        for i, feature in enumerate(feature_names):
            if feature in controllable_features and user_input[i] == 1:
                simulated_input = user_input.copy()
                simulated_input[i] = 0
                simulated_price = model.predict([simulated_input])[0]
                saving = predicted_price - simulated_price
                if saving > 0:
                    readable = feature.replace("BloodPressureProblems", "Tekanan Darah Tinggi")
                                    #   .replace("AnyChronicDiseases", "Penyakit Kronis") \
                                    #   .replace("AnyTransplants", "Riwayat Transplantasi") \
                                    #   .replace("KnownAllergies", "Alergi") \
                                    #   .replace("Diabetes", "Diabetes") \
                                    #   .replace("NumberOfMajorSurgeries", "Operasi Besar")
                    recommendations.append({
                        'type': feature,
                        'title': f"Kurangi Risiko: {readable}",
                        'description': f"Dengan mengatasi masalah {readable}, Anda bisa menghemat hingga ${saving:,.2f} per tahun.",
                        'potential_saving': saving
                    })

        
        # Tambahkan simulasi berat badan berdasarkan BMI
        idx_weight = feature_names.index("Weight")
        idx_height = feature_names.index("Height")
        weight = user_input[idx_weight]
        height_cm = user_input[idx_height]
        height_m = height_cm / 100
        bmi = weight / (height_m ** 2)

        # BMI Sehat = 18.5 - 24.9
        if bmi > 24.9:
            target_weight = 24.9 * (height_m ** 2)
            simulated_input = user_input.copy()
            simulated_input[idx_weight] = round(target_weight)
            simulated_price = model.predict([simulated_input])[0]
            saving = predicted_price - simulated_price
            if saving > 0:
                recommendations.append({
                    'type': 'bmi_decrease',
                    'title': 'Turunkan Berat Badan untuk Optimalkan Premi',
                    'description': (
                        f"BMI Anda saat ini adalah {bmi:.1f}. Jika Anda menurunkan berat ke {round(target_weight)} kg "
                        f"(BMI sehat 24.9), premi Anda bisa turun hingga **${saving:,.2f}** per tahun."
                    ),
                    'potential_saving': saving
                })

        elif bmi < 18.5:
            target_weight = 18.5 * (height_m ** 2)
            simulated_input = user_input.copy()
            simulated_input[idx_weight] = round(target_weight)
            simulated_price = model.predict([simulated_input])[0]
            saving = predicted_price - simulated_price
            if saving > 0:
                recommendations.append({
                    'type': 'bmi_increase',
                    'title': 'Naikkan Berat Badan ke BMI Sehat',
                    'description': (
                        f"BMI Anda saat ini adalah {bmi:.1f}. Jika Anda menaikkan berat ke {round(target_weight)} kg "
                        f"(BMI sehat 18.5), premi Anda bisa turun hingga **${saving:,.2f}** per tahun."
                    ),
                    'potential_saving': saving
                })


        recommendations.sort(key=lambda x: x['potential_saving'], reverse=True);
        return recommendations


# WEBSITE ROUTE (buat form GET/POST)
@app.route('/')
def home():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from the form
        input_features = [
            int(request.form['Age']),
            int(request.form['Diabetes']),
            int(request.form['BloodPressureProblems']),
            int(request.form['AnyTransplants']),
            int(request.form['AnyChronicDiseases']),
            int(request.form['Height']),
            int(request.form['Weight']),
            int(request.form['KnownAllergies']),
            int(request.form['HistoryOfCancerInFamily']),
            int(request.form['NumberOfMajorSurgeries'])
        ]

        # Convert to 2D array as model expects
        input_array = np.array([input_features])

        # Make prediction
        predicted_value = model.predict(input_array)[0]

        # Make the estimation range based on MAE
        lower_bound = max(0, predicted_value - mae)
        upper_bound = predicted_value + mae

        prediction_result = (
            f"Estimated Premium Price: ${predicted_value:,.2f}<br>"
            f"Estimated Range: ${lower_bound:,.2f} - ${upper_bound:,.2f}"
        )

        # Simulasikan potensi penghematan
        savings = simulate_savings(input_features, model, predicted_value)

        return render_template('form.html', prediction=prediction_result, recommendations=savings)
    
    except Exception as e:
        return render_template('form.html', prediction=f"Error: {str(e)}")


if __name__ == '__main__':
    app.run(debug=True)
