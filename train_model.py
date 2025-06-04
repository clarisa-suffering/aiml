# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Load dataset
df = pd.read_csv("healthy_meal_plans.csv")

# 2. Pilih fitur (X) dan label (y)
#   Misal kita prediksi kolom 'gluten_free' (0/1)
X = df[["calories", "prep_time", "protein", "fat", "carbs", 
        "vegan", "vegetarian", "keto", "paleo", "mediterranean"]]
y = df["gluten_free"]

# 3. Split data menjadi train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Buat dan latih model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 5. Evaluasi singkat (opsional)
score = clf.score(X_test, y_test)
print(f"Accuracy di test set: {score:.2f}")

# 6. Simpan model ke file .pkl
joblib.dump(clf, "model_gluten_free.pkl")
print("Model tersimpan sebagai model_gluten_free.pkl")
