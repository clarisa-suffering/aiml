import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

data = pd.read_csv('medical/data/Medicalpremium.csv')

features = [
    'Age', 'Diabetes', 'BloodPressureProblems', 'AnyTransplants',
    'AnyChronicDiseases', 'Height', 'Weight', 'KnownAllergies',
    'HistoryOfCancerInFamily', 'NumberOfMajorSurgeries'
]
X = data[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

joblib.dump(kmeans, 'kmeans_model.pkl')
joblib.dump(scaler, 'scaler_kmeans.pkl')

data['cluster'] = kmeans.predict(X_scaled)

print("Rata-rata fitur per cluster:")
print(data.groupby('cluster')[features].mean())

print("\nJumlah data per cluster:")
print(data['cluster'].value_counts())

cluster_summary = data.groupby('cluster').mean()

print(cluster_summary)
cluster_summary.to_csv('cluster_summary.csv')