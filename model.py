import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors

# Load meal dataset
df = pd.read_csv("healthy_meal_plans.csv")

# Select numeric features
features =  ["calories", "prep_time", "protein", "fat", "carbs", 
        "vegan", "vegetarian", "keto", "paleo", "mediterranean"]
X = df[features]

# Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train KNN model
knn = NearestNeighbors(n_neighbors=3, metric='euclidean')
knn.fit(X_scaled)

def calculate_tdee(gender, weight, height, age, activity_level, goal):
    # BMR (Mifflin-St Jeor)
    if gender == 'male':
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161

    activity_factors = {
        'low': 1.2,
        'moderate': 1.55,
        'high': 1.9
    }
    tdee = bmr * activity_factors[activity_level]

    if goal == 'lose':
        tdee -= 500
    elif goal == 'gain':
        tdee += 300

    return round(tdee)

def recommend_meals(target_cal):
    # Assume 30% breakfast, 40% lunch, 30% dinner
    targets = {
        "breakfast": target_cal * 0.3,
        "lunch": target_cal * 0.4,
        "dinner": target_cal * 0.3
    }

    recommendations = {}
    for meal_time, cal in targets.items():
        input_vector = np.array([[cal, 20, 15, 50]])  # rough target
        input_vector_scaled = scaler.transform(input_vector)
        distances, indices = knn.kneighbors(input_vector_scaled)
        recommended = df.iloc[indices[0]].sample(1)
        recommendations[meal_time] = recommended.iloc[0]['meal_name']

    return recommendations
