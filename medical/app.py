from flask import Flask, render_template, request
import joblib
import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import pandas as pd
from sklearn.decomposition import PCA

app = Flask(__name__)

#path untuk model dan metric
MODEL_PATH = 'model.pkl'
METRICS_PATH = 'metrics.json'
KMEANS_MODEL_PATH = 'kmeans_model.pkl'
SCALER_PATH = 'scaler_kmeans.pkl'

#feature names sesuai urutan input (harus sama dengan urutan saat training)
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
best_case_premi_value = 0
best_case_input_list = []
kmeans_model = None
scaler_kmeans = None

# --- Load Model dan Metrics ---
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file '{MODEL_PATH}' not found. Please ensure your train_model.py script has run successfully.")
else:
    try:
        model = joblib.load(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model from {MODEL_PATH}: {e}")
        model = None

if not os.path.exists(METRICS_PATH):
    print(f"Error: Metrics file '{METRICS_PATH}' not found. Defaulting MAE to 0.")
    mae = 0
    for name in feature_names:
        average_feature_values[name] = 0
else:
    try:
        with open(METRICS_PATH, 'r') as f:
            metrics = json.load(f)
            mae = metrics.get('mae', 0)
            loaded_avg_values = metrics.get('average_feature_values', {})
            for name in feature_names:
                if name in loaded_avg_values:
                    average_feature_values[name] = loaded_avg_values[name]
                else:
                    #jika average_feature_values key tidak ada, maka di-set 0
                    average_feature_values[name] = 0
            print(f"Metrics loaded successfully from {METRICS_PATH}. MAE: {mae:.2f}")
            print(f"Average feature values loaded: {average_feature_values}")
    except Exception as e:
        print(f"Error loading metrics from {METRICS_PATH}: {e}")
        mae = 0
        average_feature_values = {name: 0 for name in feature_names}


if model is not None and average_feature_values:
    try:
        best_case_input_list = []
        for name in feature_names:
            if name == 'Age':
                best_case_input_list.append(18) #contoh usia 18 sebagai yang terbaik
            elif name in ['Diabetes', 'BloodPressureProblems', 'AnyTransplants',
                          'AnyChronicDiseases', 'KnownAllergies', 'HistoryOfCancerInFamily',
                          'NumberOfMajorSurgeries']:
                best_case_input_list.append(0) #tidak ada masalah/riwayat
            elif name == 'Height':
                best_case_input_list.append(average_feature_values.get('Height', 170)) #contoh default tinggi 170 cm
            elif name == 'Weight':
                best_height_cm = average_feature_values.get('Height', 170)
                best_height_m = best_height_cm / 100
                ideal_weight = 22.0 * (best_height_m ** 2)
                best_case_input_list.append(round(ideal_weight))
            else:
                best_case_input_list.append(average_feature_values.get(name, 0))
        
        best_case_input_array = np.array([best_case_input_list])
        best_case_premi_value = model.predict(best_case_input_array)[0]
        print(f"Calculated best-case premium: ${best_case_premi_value:,.2f}")

    except Exception as e:
        print(f"Warning: Could not calculate best-case premium at startup: {e}")
        best_case_premi_value = 0
        best_case_input_list = [0] * len(feature_names) 


# --- Load KMeans Model dan Scaler ---
if os.path.exists(KMEANS_MODEL_PATH) and os.path.exists(SCALER_PATH):
    try:
        kmeans_model = joblib.load(KMEANS_MODEL_PATH)
        scaler_kmeans = joblib.load(SCALER_PATH)
        print("KMeans model dan scaler loaded.")
    except Exception as e:
        print(f"Error loading KMeans model or scaler: {e}")
        kmeans_model = None
        scaler_kmeans = None
else:
    print("Model clustering atau scaler tidak ditemukan.")

#fitur yang bisa dikontrol dan namanya dalam bahasa indonesia
controllable_features_map = {
    'Weight': 'Berat Badan',
}

# --- Hypothetical Long-Term Projection Function ---
def predict_long_term_premium(user_current_age, user_current_premium, years=5):
    #asumsi fixed annual age-related premium increase percentage (misal 0.5% per tahun) dan general inflation rate (misal 2% per tahun)
    age_factor_increase = 0.005
    inflation_factor = 0.02

    projected_premium = user_current_premium
    for _ in range(years):
        projected_premium *= (1 + age_factor_increase)
        projected_premium *= (1 + inflation_factor)

    return projected_premium

# --- FEATURE CONTRIBUTION PLOT FUNCTION ---
def create_feature_contribution_plot(user_input_list, model, feature_names, average_feature_values_dict, current_prediction, best_case_input_list, best_case_premi_value):
    contributions = {}

    readable_feature_names = {
        'Age': 'Usia',
        'Diabetes': 'Diabetes',
        'BloodPressureProblems': 'Tekanan Darah Tinggi',
        'AnyTransplants': 'Transplantasi',
        'AnyChronicDiseases': 'Penyakit Kronis',
        'Height': 'Tinggi Badan', 
        'Weight': 'Berat Badan', 
        'KnownAllergies': 'Alergi',
        'HistoryOfCancerInFamily': 'Riwayat Kanker Keluarga',
        'NumberOfMajorSurgeries': 'Jumlah Operasi Besar'
    }

    #loop per fitur untuk menghitung kontribusi tiap kolom terhadap best case
    for i, feature in enumerate(feature_names):
        #buat temporary input list dari best_case_input_list
        temp_input_list = best_case_input_list.copy()
        
        #ganti fitur sekarang dengan input user
        temp_input_list[i] = user_input_list[i] 
        
        #prediksi menggunakan input yang dimodifikasi
        temp_prediction = model.predict(np.array([temp_input_list]))[0]
        
        #kontribusi = selisih antara prediction dan overall best_case_premi_value
        contribution_value = temp_prediction - best_case_premi_value
        
        #hanya tambahkan kontribusi yang signifikan (misalnya, abs > $10)
        if abs(contribution_value) > 10:
            contributions[feature] = contribution_value

    #siapkan data, sort by impact
    plot_data = [(readable_feature_names.get(f, f), c) for f, c in contributions.items()]
    plot_data = sorted(plot_data, key=lambda item: item[1], reverse=True)

    features_for_plot = [item[0] for item in plot_data]
    values_for_plot = [item[1] for item in plot_data]

    fig, ax = plt.subplots(figsize=(12, max(7, len(features_for_plot) * 0.7)))

    colors = ['#F44336' if v >= 0 else '#4CAF50' for v in values_for_plot]

    plt.barh(features_for_plot, values_for_plot, color=colors)
    plt.xlabel('Dampak pada Premi ($) dari Kondisi Ideal') 
    plt.title('Seberapa Banyak Setiap Faktor Menaikkan Premi Anda dari Kondisi Ideal') 
    plt.axvline(0, color='grey', linewidth=0.8)
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#F44336', label='Menaikkan Premi Anda (dari Ideal)'), 
        Patch(facecolor='#4CAF50', label='Menurunkan Premi Anda (dari Ideal)') 
    ]
    ax.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(1, -0.2), ncol=2, frameon=False)

    plt.tight_layout()

    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return plot_url

# --- SIMULATE SAVINGS ---
def simulate_savings(user_input_list, model, predicted_price):
    recommendations = []
    
    temp_controllable_features_map = {
        # 'Diabetes': 'Diabetes',
        # 'BloodPressureProblems': 'Tekanan Darah Tinggi',
        # 'AnyChronicDiseases': 'Penyakit Kronis',
        # 'KnownAllergies': 'Alergi',
        # 'NumberOfMajorSurgeries': 'Operasi Besar'
    } 
    
    for i, feature in enumerate(feature_names):
        if feature in temp_controllable_features_map and user_input_list[i] == 1:
            simulated_input = user_input_list.copy()
            simulated_input[i] = 0 #ganti 1 menjadi 0 untuk simulasi
            
            simulated_price = model.predict(np.array([simulated_input]))[0]
            saving = predicted_price - simulated_price
            
            if saving > 0:
                readable_name = temp_controllable_features_map[feature]
                recommendations.append({
                    'type': feature,
                    'title': f"Kurangi Risiko: {readable_name}",
                    'description': (
                        f"Dengan mengatasi masalah {readable_name}, Anda berpotensi menghemat hingga "
                        f"<strong>${saving:,.2f}</strong> per tahun. Konsultasikan dengan profesional kesehatan Anda."
                    ),
                    'potential_saving': saving
                })

    #simulasi BMI berdasarkan height dan weight
    idx_weight = feature_names.index("Weight")
    idx_height = feature_names.index("Height")
    
    weight = user_input_list[idx_weight]
    height_cm = user_input_list[idx_height]

    height_m = height_cm / 100
    
    bmi = weight / (height_m ** 2) if height_m != 0 else 0
    
    if bmi > 24.9: #overweight
        target_weight = 22.0 * (height_m ** 2) #target BMI sehat (tengah-tengah dari rentang 18,5 - 25)
        simulated_input = user_input_list.copy()
        simulated_input[idx_weight] = round(target_weight)
        
        simulated_price = model.predict(np.array([simulated_input]))[0]
        saving = predicted_price - simulated_price
        
        if saving > 0:
            recommendations.append({
                'type': 'bmi_decrease',
                'title': 'Turunkan Berat Badan ke BMI Sehat',
                'description': (
                    f"BMI Anda saat ini adalah <strong>{bmi:.1f}</strong>. Jika Anda menurunkan berat ke sekitar <strong>{round(target_weight)} kg</strong> "
                    f"(mencapai BMI sehat), premi Anda bisa turun hingga <strong>${saving:,.2f}</strong> per tahun. "
                    f"Fokus pada pola makan seimbang dan aktivitas fisik."
                ),
                'potential_saving': saving
            })
    elif bmi < 18.5 and bmi > 0: #underweight
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
                    f"BMI Anda saat ini adalah <strong>{bmi:.1f}</strong>. Jika Anda menaikkan berat ke sekitar <strong>{round(target_weight)} kg</strong> "
                    f"(mencapai BMI sehat), premi Anda bisa turun hingga <strong>${saving:,.2f}</strong> per tahun. "
                    f"Konsultasikan dengan ahli gizi untuk rencana penambahan berat badan yang sehat."
                ),
                'potential_saving': saving
            })
    
    recommendations.sort(key=lambda x: x['potential_saving'], reverse=True)
    return recommendations


# --- WEBSITE ROUTES ---
@app.route('/')
def home():
    return render_template('form.html', form_data={})

@app.route('/predict', methods=['POST'])
def predict():
    global best_case_premi_value, best_case_input_list

    if model is None:
        return render_template('form.html', error="Model belum siap. Mohon hubungi administrator.", form_data=request.form)
    
    if not average_feature_values or best_case_premi_value == 0 or not best_case_input_list:
        return render_template('form.html', error="Kesalahan konfigurasi: Data best-case belum termuat lengkap. Mohon jalankan ulang train_model.py dan pastikan tidak ada error di startup app.py.", form_data=request.form)

    try:
        input_form_data = {}
        for name in feature_names:
            if name in request.form and request.form[name].isdigit():
                input_form_data[name] = int(request.form[name])
            else:
                raise KeyError(f"Input tidak lengkap atau tidak valid untuk kolom: {name}")

        input_features_list = [input_form_data[name] for name in feature_names]

        #scaling fitur untuk clustering
        cluster_label = "Kategori risiko tidak tersedia (model clustering tidak dimuat)."
        if kmeans_model and scaler_kmeans:
            user_input_scaled = scaler_kmeans.transform([input_features_list])
            cluster = kmeans_model.predict(user_input_scaled)[0]

            cluster_labels = {
                0: 'Premium Plan (Risiko Tinggi)',
                1: 'Moderate Plan (Risiko Sedang)',
                2: 'Basic Plan (Risiko Rendah)'
            }
            cluster_label = cluster_labels.get(cluster, "Unknown Risk")

        #convert ke 2D array
        input_array = np.array([input_features_list])

        #buat prediction
        predicted_value = model.predict(input_array)[0]

        #buat estimasi range berdasarkan MAE
        lower_bound = max(0, predicted_value - mae)
        upper_bound = predicted_value + mae

        prediction_html = (
            f"<strong>Premi Asuransi yang Diprediksi: <span style='color: var(--primary-color);'>${predicted_value:,.2f}</span></strong><br>"
            f"Rentang Estimasi: <span style='color: #666;'>${lower_bound:,.2f} - ${upper_bound:,.2f}</span>"
            f"<hr style='border-top: 1px dashed #ccc; margin: 15px 0;'>Premi Kondisi Ideal: <strong>${best_case_premi_value:,.2f}</strong>"
        )
        
        #pass best_case_input_list dan best_case_premi_value ke plotting function
        feature_plot_url = create_feature_contribution_plot(input_features_list, model, feature_names, average_feature_values, predicted_value, best_case_input_list, best_case_premi_value)
        
        #simulasikan potensi penghematan
        savings = simulate_savings(input_features_list, model, predicted_value)

        # --- Generate Detailed Health Insights ---
        user_health_insights = []

        #insight fitur spesifik berdasarkan current input_form_data
        if input_form_data.get('Diabetes') == 1:
            user_health_insights.append("<strong>Diabetes:</strong> Kondisi ini meningkatkan risiko komplikasi kesehatan serius seperti penyakit jantung, ginjal, dan saraf. Asuransi mempertimbangkan ini karena potensi biaya pengobatan jangka panjang.")
        if input_form_data.get('BloodPressureProblems') == 1:
            user_health_insights.append("<strong>Masalah Tekanan Darah Tinggi:</strong> Hipertensi adalah faktor risiko utama untuk stroke, serangan jantung, dan gagal ginjal. Pengendalian tekanan darah sangat penting untuk mengurangi risiko premi.")
        if input_form_data.get('AnyChronicDiseases') == 1:
            user_health_insights.append("<strong>Penyakit Kronis:</strong> Memiliki penyakit kronis (selain diabetes/tekanan darah tinggi yang sudah terpisah) berarti kebutuhan perawatan dan pengobatan yang berkelanjutan, yang meningkatkan beban premi.")
        if input_form_data.get('AnyTransplants') == 1:
            user_health_insights.append("<strong>Riwayat Transplantasi Organ:</strong> Ini menunjukkan kondisi kesehatan yang sangat serius di masa lalu dan kebutuhan pengobatan imunosupresif atau pemantauan seumur hidup, yang sangat memengaruhi premi.")
        if input_form_data.get('KnownAllergies') == 1:
            user_health_insights.append("<strong>Alergi:</strong> Tergantung tingkat keparahannya, alergi dapat memerlukan penanganan medis rutin atau darurat, yang dapat sedikit memengaruhi premi.")
        if input_form_data.get('HistoryOfCancerInFamily') == 1:
            user_health_insights.append("<strong>Riwayat Kanker dalam Keluarga:</strong> Ini menandakan adanya kecenderungan genetik terhadap kanker. Meskipun bukan jaminan, risiko pribadi Anda dinilai lebih tinggi.")
        if input_form_data.get('NumberOfMajorSurgeries') > 0:
            user_health_insights.append(f"<strong>Riwayat {input_form_data.get('NumberOfMajorSurgeries')} Operasi Besar:</strong> Jumlah operasi besar di masa lalu menunjukkan riwayat kesehatan yang kompleks dan potensi komplikasi atau kebutuhan perawatan pasca-operasi.")

        #BMI Insight
        idx_weight = feature_names.index("Weight")
        idx_height = feature_names.index("Height")
        weight = input_features_list[idx_weight]
        height_cm = input_features_list[idx_height]
        height_m = height_cm / 100
        bmi = weight / (height_m ** 2) if height_m != 0 else 0

        if bmi >= 30:
            user_health_insights.append(f"<strong>Indeks Massa Tubuh (BMI) Anda ({bmi:.1f}) menunjukkan Obesitas:</strong> Obesitas secara signifikan meningkatkan risiko penyakit jantung, diabetes tipe 2, stroke, dan jenis kanker tertentu. Mengelola berat badan sangat penting untuk kesehatan dan premi.")
        elif bmi >= 25 and bmi < 30:
            user_health_insights.append(f"<strong>Indeks Massa Tubuh (BMI) Anda ({bmi:.1f}) menunjukkan Kelebihan Berat Badan:</strong> Kelebihan berat badan meningkatkan risiko tekanan darah tinggi, diabetes, dan masalah persendian. Menurunkan berat badan ke rentang sehat dapat mengurangi risiko Anda.")
        elif bmi > 0 and bmi < 18.5:
            user_health_insights.append(f"<strong>Indeks Massa Tubuh (BMI) Anda ({bmi:.1f}) menunjukkan Berat Badan Kurang:</strong> Berat badan kurang juga dapat mengindikasikan masalah kesehatan tertentu atau kekurangan gizi yang bisa memengaruhi premi.")
        else:
            user_health_insights.append(f"<strong>Indeks Massa Tubuh (BMI) Anda ({bmi:.1f}) berada dalam rentang sehat.</strong> Ini adalah indikator kesehatan yang baik dan berkontribusi positif pada premi Anda.")

        #age insight
        user_health_insights.append(f"<strong>Usia Anda ({input_form_data.get('Age')} tahun):</strong> Seiring bertambahnya usia, risiko kesehatan cenderung meningkat secara alami, yang merupakan faktor penting dalam perhitungan premi asuransi.")
        
        # --- Long-Term Projection Calculation ---
        long_term_projection_text = None
        if model:
            projected_5_year_premium = predict_long_term_premium(
                user_current_age=input_form_data.get('Age'),
                user_current_premium=predicted_value,
                years=5
            )
            long_term_projection_text = (
                f"Premi yang Diproyeksikan dalam 5 Tahun: <strong>${projected_5_year_premium:,.2f}</strong> "
            )

        return render_template('form.html', 
                                prediction=prediction_html, 
                                recommendations=savings, 
                                form_data=input_form_data,
                                feature_plot=feature_plot_url,
                                risk_segment=cluster_label,
                                user_health_insights=user_health_insights,
                                long_term_projection=long_term_projection_text)
    
    except ValueError:
        return render_template('form.html', error="Input tidak valid. Pastikan semua nilai diisi dengan angka yang benar (misal: 0 atau 1 untuk Ya/Tidak, atau sesuai rentang).", form_data=request.form)
    except KeyError as ke:
        return render_template('form.html', error=f"Input tidak lengkap: '{str(ke)}' tidak ditemukan. Mohon lengkapi semua kolom.", form_data=request.form)
    except Exception as e:
        return render_template('form.html', error=f"Terjadi kesalahan saat memproses: {str(e)}. Mohon coba lagi nanti.", form_data=request.form)
    
#CLUSTERRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR
try:
    kmeans_model = joblib.load('kmeans_model.pkl')
    scaler = joblib.load('scaler_kmeans.pkl')
    data = pd.read_csv('medical/data/Medicalpremium.csv') 
    print("KMeans model, scaler, and data for visualization loaded.")
except Exception as e:
    print(f"Error loading KMeans visual components: {e}")
    kmeans_model = None
    scaler = None
    data = None

@app.route('/cluster-visual')
def cluster_visual():
    if kmeans_model is None or scaler is None or data is None:
        return render_template('error.html', message="Clustering visualization components not loaded. Please check server logs.")

    features = [
        'Age', 'Diabetes', 'BloodPressureProblems', 'AnyTransplants',
        'AnyChronicDiseases', 'Height', 'Weight', 'KnownAllergies',
        'HistoryOfCancerInFamily', 'NumberOfMajorSurgeries'
    ]
    X = data[features]
    X_scaled = scaler.transform(X)

    clusters = kmeans_model.predict(X_scaled)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

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

    cluster_counts = pd.Series(clusters).value_counts().sort_index()
    cluster_means = data.groupby(clusters)[features].mean()

    return render_template('cluster_visual.html',
                            plot_url=plot_url,
                            cluster_counts=cluster_counts,
                            cluster_means=cluster_means)
    
if __name__ == '__main__':
    app.run(debug=True)