import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import os
import numpy as np

DATA_DIR = 'data'
MODEL_DIR = 'model'
DATASET_NAME = 'obesity.csv'
MODEL_NAME = 'obesity_knn_pipeline.pkl'
DATASET_PATH = os.path.join(DATA_DIR, DATASET_NAME)
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

# Fungsi untuk melatih model menggunakan metode KNN untuk klasifikasi tingkat obesitas
def train_and_save_knn_model():
    # memastikan directory ada
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    try:
        df = pd.read_csv(DATASET_PATH)
        print(f"Dataset '{DATASET_NAME}' berhasil dimuat. Jumlah baris: {df.shape[0]}, kolom: {df.shape[1]}.")
        print("Nama kolom:", df.columns.tolist())
    except FileNotFoundError:
        print(f"Error: File '{DATASET_NAME}' tidak ditemukan di '{DATASET_PATH}'.")
        print("Pastikan dataset ada di folder 'data/' dan bernama 'obesity.csv'")
        return
    except Exception as e:
        print(f"Terjadi kesalahan saat memuat dataset: {e}")
        return
    
    # DATA PREPROCESSING
    # tambahkan kolom BMI
    if 'BMI' not in df.columns:
        if 'Height' in df.columns and 'Weight' in df.columns:
            # memastikan Height dan Weight numerik sebelum perhitungan BMI
            df['Height'] = pd.to_numeric(df['Height'], errors='coerce')
            df['Weight'] = pd.to_numeric(df['Weight'], errors='coerce')
            
            # hitung BMI hanya jika Height valid (tidak nol atau NaN)
            valid_height_mask = df['Height'].notna() & (df['Height'] != 0)
            df['BMI'] = np.nan # Inisialisasi kolom BMI dengan NaN
            df.loc[valid_height_mask, 'BMI'] = df.loc[valid_height_mask, 'Weight'] / (df.loc[valid_height_mask, 'Height']**2)
            
            if df['BMI'].isnull().any():
                bmi_median = df['BMI'].median()
                df['BMI'] = df['BMI'].fillna(bmi_median)
                print(f"Kolom 'BMI' berhasil ditambahkan dan nilai NaN diisi dengan median: {bmi_median:.2f}.")
            else:
                print("Kolom 'BMI' berhasil ditambahkan. Tidak ada nilai NaN yang diisi.")
        else:
            print("Warning: Kolom 'Height' atau 'Weight' tidak ditemukan. Tidak dapat menghitung BMI.")
    else:
        print("Kolom 'BMI' sudah ada di dataset.")

    # Tentukan final_numerical_features dan final_categorical_features
    # *** PERUBAHAN DI SINI: 'Weight' TIDAK DIHAPUS lagi dari final_numerical_features ***
    final_numerical_features = ['BMI', 'Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    # Filter fitur numerik yang benar-benar ada di DataFrame
    final_numerical_features = [f for f in final_numerical_features if f in df.columns] 
    # Pastikan BMI tidak hilang jika Height/Weight tidak ada
    if 'BMI' not in df.columns and 'BMI' in final_numerical_features:
        final_numerical_features.remove('BMI')

    target_column = 'NObeyesdad'
    categorical_features = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
    # Filter fitur kategorikal yang benar-benar ada di DataFrame
    final_categorical_features = [f for f in categorical_features if f in df.columns]

    X = df[final_numerical_features + final_categorical_features].copy()
    Y = df[target_column].copy()

    # encode label target Y
    label_encoder = LabelEncoder()
    Y_encoded = label_encoder.fit_transform(Y)
    print(f"Label target berhasil di-encode: {list(label_encoder.classes_)}")

    # --- PENANGANAN MISSING VALUES (SETELAH BMI DIHITUNG) ---
    print("\n--- Penanganan Missing Values ---")
    for col in final_numerical_features:
        X[col] = pd.to_numeric(X[col], errors='coerce') # Pastikan numerik
        if X[col].isnull().any():
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val)
            print(f"Mengisi nilai hilang di '{col}' dengan median ({median_val}).")

    for col in final_categorical_features:
        if X[col].isnull().any():
            mode_val = X[col].mode()[0]
            X[col] = X[col].fillna(mode_val)
            print(f"Mengisi nilai hilang di '{col}' dengan modus ('{mode_val}').")

    print("\n--- Analisis Korelasi Fitur Numerik dengan Target ---")
    X_num_final = X[final_numerical_features]

    # Gabungkan fitur numerik dan target untuk korelasi
    temp_corr_df = pd.concat([X_num_final, pd.Series(Y_encoded, name='Target_Encoded', index=X_num_final.index)], axis=1)

    correlations = temp_corr_df.corr()['Target_Encoded'].drop('Target_Encoded').sort_values(ascending=False)
    print("Korelasi fitur numerik dengan target (semakin tinggi, semakin berpengaruh):")
    print(correlations)

    print("\nKorelasi antar fitur numerik (untuk cek redundansi):")
    print(X_num_final.corr())

    # --- PEMERIKSAAN DISTRIBUSI KELAS (setelah preprocessing) ---
    print("\n--- Pemeriksaan Distribusi Kelas NObeyesdad (setelah preprocessing) ---")
    
    # Pastikan 'Insufficient_Weight' ada di kelas-kelas yang di-encode sebelum mencoba mengaksesnya
    if 'Insufficient_Weight' in label_encoder.classes_:
        insufficient_weight_encoded_value = label_encoder.transform(['Insufficient_Weight'])[0]
        count_insufficient_weight = (Y_encoded == insufficient_weight_encoded_value).sum()
        total_rows_after_preprocessing = X.shape[0]
        percentage_insufficient_weight = (count_insufficient_weight / total_rows_after_preprocessing) * 100
        
        print(f"Jumlah data dengan 'Insufficient_Weight': {count_insufficient_weight} baris")
        print(f"Persentase data dengan 'Insufficient_Weight': {percentage_insufficient_weight:.2f}% dari total {total_rows_after_preprocessing} baris.")
    else:
        print("Kelas 'Insufficient_Weight' tidak ditemukan di target NObeyesdad.")
        
    print("\nDistribusi lengkap kelas NObeyesdad (setelah encoding dan preprocessing):")
    y_original_after_preprocessing = label_encoder.inverse_transform(Y_encoded)
    print(pd.Series(y_original_after_preprocessing).value_counts())
    print("--- Akhir Pemeriksaan Distribusi Kelas ---")

    # pipeline preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), final_numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), final_categorical_features)
        ],
        remainder='drop'
    )

    # pipeline model KNN
    knn_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', KNeighborsClassifier())
    ])

    # coba berbagai nilai k, dari 1 sampai 20
    # coba penggunaan weights dan metric yang berbeda
    param_grid = {
        'classifier__n_neighbors': range(1, 21),
        'classifier__weights': ['uniform', 'distance'],
        'classifier__metric': ['euclidean', 'manhattan']
    }

    # bagi dataset menjadi 80% data training dan 20% test data
    X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(X, Y_encoded, test_size=0.2, random_state=42, stratify=Y_encoded)
    
    # mencari hyperparameter terbaik untuk metode KNN
    print("\nMencari hyperparameter terbaik untuk KNN...")
    grid_search = GridSearchCV(knn_pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train_encoded)

    print(f"Nilai K terbaik: {grid_search.best_params_['classifier__n_neighbors']}")
    print(f"Weights terbaik: {grid_search.best_params_['classifier__weights']}")
    print(f"Metric terbaik: {grid_search.best_params_['classifier__metric']}")
    print(f"Akurasi terbaik pada cross validation: {grid_search.best_score_:.4f}")
    best_knn_pipeline = grid_search.best_estimator_
    
    # evaluasi model terbaik pada data latih
    y_pred_encoded = best_knn_pipeline.predict(X_test) # Prediksi dalam bentuk encoded
    
    # konversi kembali prediksi dan label tes ke label asli untuk laporan
    y_test_original = label_encoder.inverse_transform(y_test_encoded)
    y_pred_original = label_encoder.inverse_transform(y_pred_encoded)

    print("\n--- Hasil Evaluasi Model KNN pada Data Testing ---")
    print(f"Akurasi: {accuracy_score(y_test_original, y_pred_original):.4f}")
    print("\nLaporan Klasifikasi:")
    print(classification_report(y_test_original, y_pred_original))
    print("\nConfusion Matrix:")
    
    # confusion matrix menggunakan label asli yang sudah diurutkan
    sorted_labels = label_encoder.classes_ 
    cm = confusion_matrix(y_test_original, y_pred_original, labels=sorted_labels)
    cm_df = pd.DataFrame(cm, index=[f'Actual: {l}' for l in sorted_labels], columns=[f'Predicted: {l}' for l in sorted_labels])
    print(cm_df)

    # simpan pipeline terbaik (termasuk preprocessor dan classifier) dan label encoder untuk NObeyesdad
    joblib.dump(best_knn_pipeline, MODEL_PATH)
    joblib.dump(label_encoder, os.path.join(MODEL_DIR, 'label_encoder.pkl')) # Simpan juga LabelEncoder
    print(f"Model terbaik berhasil disimpan sebagai '{MODEL_NAME}' di '{MODEL_DIR}/'.")
    print(f"LabelEncoder berhasil disimpan sebagai 'label_encoder.pkl' di '{MODEL_DIR}/'.")

if __name__ == '__main__':
    train_and_save_knn_model()