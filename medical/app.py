from flask import Flask, render_template, request
import joblib
import numpy as np
import json
import os # Import os for path joining

app = Flask(__name__)

# Define paths for model and metrics
MODEL_PATH = 'model.pkl'
METRICS_PATH = 'metrics.json'

# --- Load Model and Metrics ---
# Check if model.pkl exists
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file '{MODEL_PATH}' not found. Please ensure your train_model.py script has run successfully.")
    # Exit or handle gracefully if running locally
    # For production, you might want to raise an exception or serve an error page
    model = None # Set model to None to prevent app from crashing immediately
else:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")

# Check if metrics.json exists
if not os.path.exists(METRICS_PATH):
    print(f"Error: Metrics file '{METRICS_PATH}' not found. Defaulting MAE to 0.")
    mae = 0
else:
    with open(METRICS_PATH, 'r') as f:
        metrics = json.load(f)
        mae = metrics.get('mae', 0)
    print(f"Metrics loaded successfully from {METRICS_PATH}. MAE: {mae:.2f}")


# Feature names sesuai urutan input (HARUS SAMA DENGAN URUTAN SAAT TRAINING!)
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

# Fitur yang dianggap bisa dikendalikan dan terjemahannya
controllable_features_map = {
    'Diabetes': 'Diabetes',
    'BloodPressureProblems': 'Tekanan Darah Tinggi',
    'AnyChronicDiseases': 'Penyakit Kronis',
    'Weight': 'Berat Badan', # Berat badan adalah kategori khusus untuk BMI
    'KnownAllergies': 'Alergi',
    'NumberOfMajorSurgeries': 'Operasi Besar'
    # 'AnyTransplants' biasanya tidak bisa dikendalikan secara langsung
    # 'HistoryOfCancerInFamily' juga tidak bisa dikendalikan
}


def simulate_savings(user_input_list, model, predicted_price):
    recommendations = []
    
    # Simulate for binary controllable features
    for i, feature in enumerate(feature_names):
        if feature in controllable_features_map and user_input_list[i] == 1 and feature != 'Weight':
            simulated_input = user_input_list.copy()
            simulated_input[i] = 0 # Change 1 to 0 for simulation
            
            simulated_price = model.predict(np.array([simulated_input]))[0]
            saving = predicted_price - simulated_price
            
            if saving > 0:
                readable_name = controllable_features_map[feature]
                recommendations.append({
                    'type': feature,
                    'title': f"Kurangi Risiko: {readable_name}",
                    'description': (
                        f"Dengan mengatasi masalah {readable_name}, Anda berpotensi menghemat hingga "
                        f"<strong>${saving:,.2f}</strong> per tahun. Konsultasi dengan profesional kesehatan Anda."
                    ),
                    'potential_saving': saving
                })

    # Simulate for BMI based on Weight and Height
    idx_weight = feature_names.index("Weight")
    idx_height = feature_names.index("Height")
    
    weight = user_input_list[idx_weight]
    height_cm = user_input_list[idx_height]

    height_m = height_cm / 100
    
    bmi = weight / (height_m ** 2) if height_m != 0 else 0 # Handle division by zero
    
    if bmi > 24.9: # Overweight or Obese
        target_weight = 24.9 * (height_m ** 2)
        simulated_input = user_input_list.copy()
        simulated_input[idx_weight] = round(target_weight)
        
        simulated_price = model.predict(np.array([simulated_input]))[0]
        saving = predicted_price - simulated_price
        
        if saving > 0:
            recommendations.append({
                'type': 'bmi_decrease',
                'title': 'Turunkan Berat Badan ke BMI Sehat',
                'description': (
                    f"BMI Anda saat ini adalah <strong>{bmi:.1f}</strong>. Jika Anda menurunkan berat ke **{round(target_weight)} kg** "
                    f"(mencapai BMI sehat 24.9), premi Anda bisa turun hingga **${saving:,.2f}** per tahun. "
                    f"Fokus pada pola makan seimbang dan aktivitas fisik."
                ),
                'potential_saving': saving
            })
    elif bmi < 18.5 and bmi > 0: # Underweight (and not zero height scenario)
        target_weight = 18.5 * (height_m ** 2)
        simulated_input = user_input_list.copy()
        simulated_input[idx_weight] = round(target_weight)
        
        simulated_price = model.predict(np.array([simulated_input]))[0]
        saving = predicted_price - simulated_price
        
        if saving > 0:
            recommendations.append({
                'type': 'bmi_increase',
                'title': 'Naikkan Berat Badan ke BMI Sehat',
                'description': (
                    f"BMI Anda saat ini adalah **{bmi:.1f}**. Jika Anda menaikkan berat ke **{round(target_weight)} kg** "
                    f"(mencapai BMI sehat 18.5), premi Anda bisa turun hingga **${saving:,.2f}** per tahun. "
                    f"Konsultasikan dengan ahli gizi untuk rencana penambahan berat badan yang sehat."
                ),
                'potential_saving': saving
            })

    recommendations.sort(key=lambda x: x['potential_saving'], reverse=True)
    return recommendations


# --- WEBSITE ROUTES ---
@app.route('/')
def home():
    # Pass an empty dictionary for initial form rendering
    return render_template('form.html', form_data={})

@app.route('/predict', methods=['POST'])
def predict():
    # Handle case where model might not be loaded
    if model is None:
        return render_template('form.html', error="Model belum siap. Mohon hubungi administrator.", form_data=request.form)

    try:
        # Get input values from the form
        # Store them in a dictionary keyed by feature_name for easy access in template
        input_form_data = {name: int(request.form[name]) for name in feature_names}
        input_features_list = [input_form_data[name] for name in feature_names]

        # Convert to 2D array as model expects
        input_array = np.array([input_features_list])

        # Make prediction
        predicted_value = model.predict(input_array)[0]

        # Make the estimation range based on MAE
        lower_bound = max(0, predicted_value - mae)
        upper_bound = predicted_value + mae

        prediction_html = (
            f"<strong>Premi Asuransi yang Diprediksi: <span style='color: var(--primary-color);'>${predicted_value:,.2f}</span></strong><br>"
            f"Rentang Estimasi: <span style='color: #666;'>${lower_bound:,.2f} - ${upper_bound:,.2f}</span>"
        )

        # Simulasikan potensi penghematan
        savings = simulate_savings(input_features_list, model, predicted_value)

        return render_template('form.html', 
                               prediction=prediction_html, 
                               recommendations=savings, 
                               form_data=input_form_data) # Pass form data back
    
    except ValueError:
        # Handle cases where input is not an integer or is out of expected range (e.g., empty string)
        return render_template('form.html', error="Input tidak valid. Pastikan semua nilai diisi dengan angka yang benar (misal: 0 atau 1 untuk Ya/Tidak).", form_data=request.form)
    except KeyError as ke:
        # Handle cases where a form field might be missing (e.g., if HTML form is modified)
        return render_template('form.html', error=f"Input tidak lengkap: '{str(ke)}' tidak ditemukan. Mohon lengkapi semua kolom.", form_data=request.form)
    except Exception as e:
        # Catch any other unexpected errors
        return render_template('form.html', error=f"Terjadi kesalahan saat memproses: {str(e)}. Mohon coba lagi nanti.", form_data=request.form)


if __name__ == '__main__':
    # Adjust debug mode based on environment
    app.run(debug=True) # Set to False in production