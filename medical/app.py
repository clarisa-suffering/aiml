from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load('model.pkl')

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
        prediction = model.predict(input_array)[0]

        return render_template('form.html', prediction=f"Estimated Premium Price: ${prediction:,.2f}")
    except Exception as e:
        return render_template('form.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
