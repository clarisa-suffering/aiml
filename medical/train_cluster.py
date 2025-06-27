import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

# Load data
data = pd.read_csv('medical/data/Medicalpremium.csv')  # Ganti dengan path data kamu

# Fitur yang dipakai
features = [
    'Age', 'Diabetes', 'BloodPressureProblems', 'AnyTransplants',
    'AnyChronicDiseases', 'Height', 'Weight', 'KnownAllergies',
    'HistoryOfCancerInFamily', 'NumberOfMajorSurgeries'
]

X = data[features]

# Scaling fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Clustering dengan KMeans (3 cluster)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# Simpan model dan scaler ke file
joblib.dump(kmeans, 'kmeans_model.pkl')
joblib.dump(scaler, 'scaler_kmeans.pkl')

# Prediksi cluster untuk data asli (scaled)
data['cluster'] = kmeans.predict(X_scaled)

# Tampilkan rata-rata fitur per cluster
print("Rata-rata fitur per cluster:")
print(data.groupby('cluster')[features].mean())

# Tampilkan jumlah data per cluster
print("\nJumlah data per cluster:")
print(data['cluster'].value_counts())
