from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained Random Forest model
model = pickle.load(open('model/rf_rainfall_model.pkl', 'rb'))
scaler = StandardScaler()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        features = [float(request.form[key]) for key in ['pressure', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'windspeed', 'winddirection']]
        
        # Convert input to numpy array and scale it
        input_data = np.array(features).reshape(1, -1)
        input_scaled = scaler.fit_transform(input_data)
        
        # Predict
        prediction = model.predict(input_scaled)[0]
        result = 'Yes, it will rain!' if prediction == 1 else 'No, it will not rain.'
        
        return render_template('index.html', prediction=result)
    except Exception as e:
        return render_template('index.html', prediction=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
