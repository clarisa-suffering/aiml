from flask import Flask, render_template, request
import joblib
import numpy as np
import json
import os # Import os for path joining
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64 # encode gambar menjadi string
from io import BytesIO #menyimpan gambar ke memori
import pandas as pd
from sklearn.decomposition import PCA

app = Flask(__name__)

# Define paths for model and metrics
MODEL_PATH = 'model.pkl'
METRICS_PATH = 'metrics.json'

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

model = None
mae = 0
average_feature_values = {}
baseline_premi_value = 0

# --- Load Model and Metrics ---
# Check if model.pkl exists
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file '{MODEL_PATH}' not found. Please ensure your train_model.py script has run successfully.")
    # Exit or handle gracefully if running locally
    # For production, you might want to raise an exception or serve an error page
else:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")

# Check if metrics.json exists
if not os.path.exists(METRICS_PATH):
    print(f"Error: Metrics file '{METRICS_PATH}' not found. Defaulting MAE to 0.")
    mae = 0

    # set default 0
    for name in feature_names:
        average_feature_values[name] = 0
else:
    with open(METRICS_PATH, 'r') as f:
        metrics = json.load(f)
        mae = metrics.get('mae', 0)
        # Load average_feature_values dari JSON
        loaded_avg_values = metrics.get('average_feature_values', {})
        for name in feature_names:
            if name in loaded_avg_values:
                average_feature_values[name] = loaded_avg_values[name]
            else:
                # Fallback if average_feature_values key is missing inside metrics.json
                average_feature_values[name] = 0

    print(f"Metrics loaded successfully from {METRICS_PATH}. MAE: {mae:.2f}")
    print(f"Average feature values loaded: {average_feature_values}")

if model is not None and average_feature_values: # Only calculate if model and avg_values are loaded
    try:
        baseline_input_list = [average_feature_values.get(name, 0) for name in feature_names]
        baseline_input_array = np.array([baseline_input_list])
        baseline_premi_value = model.predict(baseline_input_array)[0]
        print(f"Calculated baseline premium: ${baseline_premi_value:,.2f}")
    except Exception as e:
        print(f"Warning: Could not calculate baseline premium at startup: {e}")
        baseline_premi_value = 0 # Fallback if error occurs

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

# --- Load KMeans Model and Scaler ---
KMEANS_MODEL_PATH = 'kmeans_model.pkl'
SCALER_PATH = 'scaler_kmeans.pkl'

kmeans_model = None
scaler_kmeans = None

if os.path.exists(KMEANS_MODEL_PATH) and os.path.exists(SCALER_PATH):
    kmeans_model = joblib.load(KMEANS_MODEL_PATH)
    scaler_kmeans = joblib.load(SCALER_PATH)
    print("KMeans model dan scaler loaded.")
else:
    print("Model clustering tidak ditemukan.")

def create_feature_contribution_plot(user_input_list, model, feature_names, average_feature_values_dict, current_prediction):
    contributions = {}

    # Readable feature names for plot
    readable_feature_names = {
        'Age': 'Usia',
        'Diabetes': 'Diabetes',
        'BloodPressureProblems': 'Tekanan Darah Tinggi',
        'AnyTransplants': 'Transplantasi',
        'AnyChronicDiseases': 'Penyakit Kronis',
        'Height': 'Tinggi',
        'Weight': 'Berat Badan',
        'KnownAllergies': 'Alergi',
        'HistoryOfCancerInFamily': 'Riwayat Kanker Keluarga',
        'NumberOfMajorSurgeries': 'Jumlah Operasi Besar'
    }

    
    controllable_features_map = {
        'Diabetes': 'Diabetes',
        'BloodPressureProblems': 'Tekanan Darah Tinggi',
        'AnyChronicDiseases': 'Penyakit Kronis',
        'KnownAllergies': 'Alergi',
        'NumberOfMajorSurgeries': 'Operasi Besar'
    }

    for i, feature in enumerate(feature_names):
        if feature in controllable_features_map and user_input_list[i] == 1: # Only if user has the problem
            simulated_input = user_input_list.copy()
            simulated_input[i] = 0 # Simulate improvement (1 -> 0)
            
            simulated_price = model.predict(np.array([simulated_input]))[0]
            
            contribution_value = current_prediction - simulated_price
            
            if contribution_value > 0: # Only show if it actually increases premium
                contributions[feature] = contribution_value
        
        # === Logic for BMI (Weight) ===
        elif feature == 'Weight': # Handle BMI separately as it's continuous
            idx_weight = feature_names.index("Weight")
            idx_height = feature_names.index("Height")
            
            weight = user_input_list[idx_weight]
            height_cm = user_input_list[idx_height]
            height_m = height_cm / 100
            bmi = weight / (height_m ** 2) if height_m != 0 else 0
            
            if bmi > 24.9: # Overweight or Obese
                target_weight = 24.9 * (height_m ** 2)
                simulated_input = user_input_list.copy()
                simulated_input[idx_weight] = round(target_weight)
                
                simulated_price = model.predict(np.array([simulated_input]))[0]
                bmi_contribution = current_prediction - simulated_price
                if bmi_contribution > 0:
                    contributions['Weight'] = bmi_contribution # Use 'Weight' as feature name for consistency
            elif bmi < 18.5 and bmi > 0: # Underweight
                target_weight = 18.5 * (height_m ** 2)
                simulated_input = user_input_list.copy()
                simulated_input[idx_weight] = round(target_weight)
                
                simulated_price = model.predict(np.array([simulated_input]))[0]
                bmi_contribution = current_prediction - simulated_price
                if bmi_contribution > 0: # If going to healthy weight increases premium, don't show as a positive "impact"
                    contributions['Weight'] = bmi_contribution
    
    # 1. For controllable factors (if problematic), show their potential increase
    # 2. For uncontrollable factors, show their impact relative to baseline (or average user)
    
    # Re-introducing baseline for uncontrollable factors or those not explicitly handled above
    baseline_input_list_full = [average_feature_values_dict.get(name, 0) for name in feature_names]
    baseline_prediction_full = model.predict(np.array([baseline_input_list_full]))[0]

    for i, feature in enumerate(feature_names):
        # Skip features already handled as 'controllable and problematic' or 'Weight' for BMI
        if feature not in controllable_features_map and feature != 'Weight': 
            temp_input_list = baseline_input_list_full.copy()
            temp_input_list[i] = user_input_list[i]
            
            temp_prediction = model.predict(np.array([temp_input_list]))[0]
            
            # Contribution relative to baseline
            contribution_value = temp_prediction - baseline_prediction_full
            
            # Only add if it has a noticeable impact (e.g., above a small threshold)
            if abs(contribution_value) > 10: # bisa di-adjust
                contributions[feature] = contribution_value

    # Prepare data for plotting
    plot_data = [(readable_feature_names.get(f, f), c) for f, c in contributions.items()]
    
    sorted_plot_data = sorted(plot_data, key=lambda item: item[1], reverse=True) # Sort by value (positive first)

    features_for_plot = [item[0] for item in sorted_plot_data]
    values_for_plot = [item[1] for item in sorted_plot_data]

    # Create the plot (rest of the plotting code remains similar)
    fig, ax = plt.subplots(figsize=(12, max(7, len(features_for_plot) * 0.7)))

    colors = []
    colors = []
    for v in values_for_plot:
        if v >= 0:
            colors.append('#F44336') # Merah untuk dampak positif (menaikkan premi)
        else:
            colors.append('#4CAF50') # Hijau untuk dampak negatif (menurunkan premi)

    plt.barh(features_for_plot, values_for_plot, color=colors)
    plt.xlabel('Dampak pada Premi ($)')
    plt.title('Dampak Faktor Terhadap Premi Anda (dari Kondisi Saat Ini)') # Updated title
    plt.axvline(0, color='grey', linewidth=0.8)
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#F44336', label='Menaikkan Premi Anda'),
        Patch(facecolor='#4CAF50', label='Menurunkan Premi Anda')
    ]
    ax.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(1, -0.2), ncol=2, frameon=False)

    plt.tight_layout()

    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return plot_url

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
                        f"<strong>${saving:,.2f}</strong> per tahun. Konsultasikan dengan profesional kesehatan Anda."
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
                    f"BMI Anda saat ini adalah <strong>{bmi:.1f}</strong>. Jika Anda menurunkan berat ke <strong>{round(target_weight)} kg</strong> "
                    f"(mencapai BMI sehat 24.9), premi Anda bisa turun hingga <strong>${saving:,.2f}</strong> per tahun. "
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
                    f"BMI Anda saat ini adalah <strong>{bmi:.1f}</strong>. Jika Anda menaikkan berat ke <strong>{round(target_weight)} kg</strong> "
                    f"(mencapai BMI sehat 18.5), premi Anda bisa turun hingga <strong>${saving:,.2f}</strong> per tahun. "
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
    
    if not average_feature_values:
         return render_template('form.html', error="Kesalahan konfigurasi: Average feature values tidak termuat. Mohon jalankan ulang train_model.py.", form_data=request.form)

    try:
        # Get input values from the form
        # Store them in a dictionary keyed by feature_name for easy access in template
        input_form_data = {name: int(request.form[name]) for name in feature_names}
        input_features_list = [input_form_data[name] for name in feature_names]

        # Scaling fitur untuk clustering
        user_input_scaled = scaler_kmeans.transform(pd.DataFrame([input_features_list], columns=feature_names))
        cluster = kmeans_model.predict(user_input_scaled)[0]

        cluster_labels = {
            0: 'Premium Plan (Risiko Tinggi)',
            1: 'Moderate Plan (Risiko Sedang)',
            2: 'Basic Plan (Risiko Rendah)'
        }
        # Kamu bisa atur ulang mapping ini setelah melihat hasil analisis
        cluster_label = cluster_labels[cluster]

        if not average_feature_values or baseline_premi_value == 0:
         return render_template('form.html', error="Kesalahan konfigurasi: Data baseline belum termuat lengkap. Mohon jalankan ulang train_model.py dan pastikan tidak ada error di startup app.py.", form_data=request.form)
        
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
            f"<hr style='border-top: 1px dashed #ccc; margin: 15px 0;'> Premi Baseline (Rata-rata/Tanpa Risiko Tambahan): <strong>${baseline_premi_value:,.2f}</strong>"
        )
        
        feature_plot_url = create_feature_contribution_plot(input_features_list, model, feature_names, average_feature_values, predicted_value)
        
        # Simulasikan potensi penghematan
        savings = simulate_savings(input_features_list, model, predicted_value)

        return render_template('form.html', 
                               prediction=prediction_html, 
                               recommendations=savings, 
                               form_data=input_form_data,
                               feature_plot=feature_plot_url,
                               risk_segment=cluster_label)
    
    except ValueError:
        # Handle cases where input is not an integer or is out of expected range (e.g., empty string)
        return render_template('form.html', error="Input tidak valid. Pastikan semua nilai diisi dengan angka yang benar (misal: 0 atau 1 untuk Ya/Tidak).", form_data=request.form)
    except KeyError as ke:
        # Handle cases where a form field might be missing (e.g., if HTML form is modified)
        return render_template('form.html', error=f"Input tidak lengkap: '{str(ke)}' tidak ditemukan. Mohon lengkapi semua kolom.", form_data=request.form)
    except Exception as e:
        # Catch any other unexpected errors
        return render_template('form.html', error=f"Terjadi kesalahan saat memproses: {str(e)}. Mohon coba lagi nanti.", form_data=request.form)
    

# CLUSTER VISUALIZATION ROUTE
kmeans_model = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler_kmeans.pkl')
data = pd.read_csv('medical/data/Medicalpremium.csv')  

@app.route('/cluster-visual')
def cluster_visual():
    # Ambil fitur untuk clustering
    features = [
        'Age', 'Diabetes', 'BloodPressureProblems', 'AnyTransplants',
        'AnyChronicDiseases', 'Height', 'Weight', 'KnownAllergies',
        'HistoryOfCancerInFamily', 'NumberOfMajorSurgeries'
    ]
    X = data[features]
    X_scaled = scaler.transform(X)

    # Prediksi cluster
    clusters = kmeans_model.predict(X_scaled)

    # PCA untuk 2D visualisasi
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Plot scatter dengan warna cluster
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=clusters, cmap='Set1', alpha=0.6)
    plt.title('Visualisasi Clustering KMeans (2D PCA)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(scatter, label='Cluster')
    plt.tight_layout()

    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    # Statistik cluster
    cluster_counts = pd.Series(clusters).value_counts().sort_index()
    cluster_means = data.groupby(clusters)[features].mean()

    return render_template('cluster_visual.html',
                           plot_url=plot_url,
                           cluster_counts=cluster_counts,
                           cluster_means=cluster_means)
    
if __name__ == '__main__':
    # Adjust debug mode based on environment
    app.run(debug=True) # Set to False in production