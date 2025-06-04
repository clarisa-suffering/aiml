from flask import Flask, request, jsonify
import pandas as pd
import pickle
import numpy as np
from flask_cors import CORS
from sklearn.cluster import KMeans

app = Flask(__name__)
CORS(app)

# Load the model, scaler, and dataframe
with open("meal_model.pkl", "rb") as f:
    model, le, scaler, df = pickle.load(f)

diet_cols = ['vegan','vegetarian','keto','paleo','gluten_free','mediterranean']
nutrient_cols = ['calories','protein','fat','carbs']

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    target_calories = float(data.get('target_calories', 2000))
    meal_count = int(data.get('meal_count', 3))
    preferences = data.get('preferences', {})

    # Filter meals by preferences
    filtered_df = df.copy()
    for diet in diet_cols:
        if preferences.get(diet, 0) == 1:
            filtered_df = filtered_df[filtered_df[diet] == 1]

    if len(filtered_df) < meal_count:
        return jsonify({"error": "Not enough meals match preferences."}), 400

    # Scale nutrient features for clustering
    X_cluster = filtered_df[nutrient_cols]
    X_scaled = scaler.transform(filtered_df[nutrient_cols + diet_cols])[:, :4]  # only nutrients

    # Perform KMeans clustering
    k = min(meal_count, len(filtered_df))  # number of clusters = meals requested or max possible
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    filtered_df['cluster'] = clusters

    # Select one representative meal from each cluster
    selected_meals = []
    for cluster_id in range(k):
        cluster_meals = filtered_df[filtered_df['cluster'] == cluster_id]
        # Choose the one closest to target_calories / meal_count
        cluster_meals['score'] = abs(cluster_meals['calories'] - target_calories / meal_count)
        best = cluster_meals.nsmallest(1, 'score')
        selected_meals.append(best)

    # Combine results
    meal_plan_df = pd.concat(selected_meals)
    meal_plan = meal_plan_df[['meal_name', 'calories', 'protein', 'fat', 'carbs']].to_dict(orient='records')

    return jsonify({
        "meal_plan": meal_plan,
        "total_calories": sum([m['calories'] for m in meal_plan])
    })

if __name__ == '__main__':
    app.run(debug=True)
