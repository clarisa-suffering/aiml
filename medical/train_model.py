import pandas as pd
import numpy as np
import joblib
import json

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression

#load csv
df = pd.read_csv('medical/data/Medicalpremium.csv')

#cek data
print("Initial shape:", df.shape)
#cek null values
print("Checking for null values:\n", df.isnull().sum())
#cek tipe data
print("Checking data types:\n", df.dtypes)

#drop duplicate
df = df.drop_duplicates()

#split x y
#kolom target y =PremiumPrice, kolom lain jadi x
X = df.drop(columns=['PremiumPrice'])
y = df['PremiumPrice']

#80% buat training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#evaluate error
def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return {'model': model, 'mae': mae, 'mse': mse, 'r2': r2}

results = {}


#Linear Regression sebagai baseline
print("\nTraining Linear Regression...")
lr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])
lr_pipeline.fit(X_train, y_train)
results['LinearRegression'] = evaluate_model("Linear Regression", lr_pipeline, X_test, y_test)


#random forest pakai GridSearchCV
print("\n Tuning Random Forest...")
rf_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor(random_state=42))
])
rf_param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [10, None],
    'model__min_samples_split': [2, 10],
    'model__min_samples_leaf': [1, 4]
}
rf_grid = GridSearchCV(rf_pipeline, rf_param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
rf_grid.fit(X_train, y_train)
results['RandomForest'] = evaluate_model("Random Forest", rf_grid.best_estimator_, X_test, y_test)

#XGBoost pakai GridSearchCV
print("\n Tuning XGBoost...")
xgb_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', XGBRegressor(objective='reg:squarederror', random_state=42))
])
xgb_param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [3, 5],
    'model__learning_rate': [0.05, 0.1]
}
xgb_grid = GridSearchCV(xgb_pipeline, xgb_param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
xgb_grid.fit(X_train, y_train)
results['XGBoost'] = evaluate_model("XGBoost", xgb_grid.best_estimator_, X_test, y_test)

#CatBoost pakai GridSearchCV
print("\n Tuning CatBoost...")
cat_model = CatBoostRegressor(verbose=0, random_state=42)
cat_param_grid = {
    'iterations': [100, 200],
    'depth': [4, 6],
    'learning_rate': [0.05, 0.1]
}
cat_grid = GridSearchCV(cat_model, cat_param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
cat_grid.fit(X_train, y_train)
results['CatBoost'] = evaluate_model("CatBoost", cat_grid.best_estimator_, X_test, y_test)


print("\n---  Model Comparison After Hyperparameter Tuning ---")
for name, metrics in results.items():
    print(f"\n{name} Results:")
    print(f"MAE : {metrics['mae']:.2f}")
    print(f"MSE : {metrics['mse']:.2f}")
    print(f"R²  : {metrics['r2']:.4f}")

#save model terbaik
best_model_name = min(results, key=lambda k: results[k]['mae'])
best_model = results[best_model_name]['model']
joblib.dump(best_model, 'model.pkl')

print(f"\n Best Model: {best_model_name} (saved as model.pkl)")


#save metrics
best_metrics = {
    'mae': results[best_model_name]['mae'],
    'mse': results[best_model_name]['mse'],
    'r2': results[best_model_name]['r2']
}

#analisa dataset untuk average feature values
#pakai median untuk fitur numerik dan mode untuk fitur categorical
calculated_average_feature_values = {}
for col in X_train.columns:
    if col in ['Age', 'Height', 'Weight', 'NumberOfMajorSurgeries']:
        calculated_average_feature_values[col] = round(X_train[col].median())
    elif col in ['Diabetes', 'BloodPressureProblems', 'AnyTransplants', 'AnyChronicDiseases', 'KnownAllergies', 'HistoryOfCancerInFamily']:
        calculated_average_feature_values[col] = 0
    else:
        calculated_average_feature_values[col] = round(X_train[col].mean())

best_metrics['average_feature_values'] = calculated_average_feature_values

with open('metrics.json', 'w') as f:
    json.dump(best_metrics, f, indent=4)

print(f"\n Saved metrics for {best_model_name} and calculated average feature values to metrics.json")
