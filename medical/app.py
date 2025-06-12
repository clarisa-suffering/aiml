from flask import Flask, render_template, request
import joblib
import numpy as np
import json

app = Flask(__name__)

# Load trained model
model = joblib.load('model.pkl')

# Load metrics (especially MAE)
with open('metrics.json', 'r') as f:
    metrics = json.load(f)
    mae = metrics.get('mae', 0)


# WEBSITE ROUTE (buat form GET/POST)
@app.route('/')
def home():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from the form
        input_features = [
            int(request.form['Age']),
            int(request.form['Diabetes']),
            int(request.form['BloodPressureProblems']),
            int(request.form['AnyTransplants']),
            int(request.form['AnyChronicDiseases']),
            int(request.form['Height']),
            int(request.form['Weight']),
            int(request.form['KnownAllergies']),
            int(request.form['HistoryOfCancerInFamily']),
            int(request.form['NumberOfMajorSurgeries'])
        ]

        # Convert to 2D array as model expects
        input_array = np.array([input_features])

        # Make prediction
        predicted_value = model.predict(input_array)[0]

        # Make the estimation range based on MAE
        lower_bound = max(0, predicted_value - mae)
        upper_bound = predicted_value + mae

        prediction_result = (
            f"Estimated Premium Price: ${predicted_value:,.2f}<br>"
            f"Estimated Range: ${lower_bound:,.2f} - ${upper_bound:,.2f}"
        )

        return render_template('form.html', prediction=prediction_result)
    
    except Exception as e:
        return render_template('form.html', prediction=f"Error: {str(e)}")


if __name__ == '__main__':
    app.run(debug=True)
