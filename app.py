import pandas as pd
import numpy as np
import pickle
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# Load model and tools
with open("meal_model.pkl", "rb") as f:
    model, le, scaler, df = pickle.load(f)

HISTORY_FILE = "meal_history.csv"

# Initialize history file if it doesn't exist
if not os.path.exists(HISTORY_FILE):
    pd.DataFrame(columns=["User_ID", "Food_Item"]).to_csv(HISTORY_FILE, index=False)

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.json
    user_id = data["User_ID"]
    input_data = [[
        data['Calories'],
        data['Protein (g)'],
        data['Carbohydrates (g)'],
        data['Fat (g)'],
        data['Fiber (g)'],
        data['Sugars (g)'],
        data['Sodium (mg)'],
        data['Cholesterol (mg)'],
        data['Water_Intake (ml)']
    ]]

    # Standardize input
    input_scaled = scaler.transform(input_data)

    # Predict probabilities
    probs = model.predict_proba(input_scaled)[0]
    sorted_indices = np.argsort(probs)[::-1]

    # Load history and filter
    history_df = pd.read_csv(HISTORY_FILE)
    seen = history_df[history_df["User_ID"] == user_id]["Food_Item"].tolist()

    recommendations = []
    for idx in sorted_indices:
        food = le.inverse_transform([idx])[0]
        if food not in seen:
            row = df[df["Food_Item"] == food].iloc[0]
            recommendations.append({
                "Food_Item": food,
                "Category": row["Category"],
                "Meal_Type": row["Meal_Type"]  # Include meal type
            })
        if len(recommendations) == 3:
            break

    # Save history
    new_history = pd.DataFrame([{"User_ID": user_id, "Food_Item": r["Food_Item"]} for r in recommendations])
    new_history.to_csv(HISTORY_FILE, mode="a", header=False, index=False)

    return jsonify(recommendations)

if __name__ == "__main__":
    app.run(debug=True)
