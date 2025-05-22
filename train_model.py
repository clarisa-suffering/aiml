import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load CSV
try:
    df = pd.read_csv('meals.csv')
except FileNotFoundError:
    print("Error: The file 'meals.csv' was not found.")
    exit()

# Check for required columns
required_columns = ["Calories (kcal)", "Protein (g)", "Carbohydrates (g)", "Fat (g)", "Fiber (g)", "Sugars (g)", "Sodium (mg)", "Cholesterol (mg)", "Water_Intake (ml)", "Food_Item"]
if not all(col in df.columns for col in required_columns):
    print("Error: The CSV file must contain the following columns:", required_columns)
    exit()

# Features & label
X = df[["Calories (kcal)", "Protein (g)", "Carbohydrates (g)", "Fat (g)", "Fiber (g)", "Sugars (g)", "Sodium (mg)", "Cholesterol (mg)", "Water_Intake (ml)"]]
y = df["Food_Item"]

# Encode label
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42)

# Model
model = DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Save model
with open("meal_model.pkl", "wb") as f:
    pickle.dump((model, le, scaler, df), f)

print("Model berhasil dilatih dan disimpan.")
