import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

df = pd.read_csv('healthy_meal_plans.csv')

# Encode diet tags sebagai fitur
diet_cols = ['vegan','vegetarian','keto','paleo','gluten_free','mediterranean']

X = df[['calories','protein','fat','carbs'] + diet_cols]
y = df['meal_name']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42)

model = DecisionTreeClassifier(max_depth=5, min_samples_leaf=2, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le.classes_))

with open("meal_model.pkl", "wb") as f:
    pickle.dump((model, le, scaler, df), f)

print("Model trained and saved.")
