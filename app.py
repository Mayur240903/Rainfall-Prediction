from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
model = pickle.load(open("rf_rainfall_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from the form
        features = [float(request.form[key]) for key in ['pressure', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']]
        
        # Scale the input features
        features_scaled = scaler.transform([features])
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        
        # Convert prediction to meaningful output
        prediction_result = "Yes, it will rain." if prediction == 1 else "No, it will not rain."
        
        return render_template('result.html', prediction_result=prediction_result)
    except Exception as e:
        return render_template('result.html', prediction_result=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
