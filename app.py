from flask import Flask, request, jsonify, send_file  
import joblib
import numpy as np
import os

# Load saved model and scaler
model = joblib.load("Fish Model.pkl")
scaler = joblib.load("Fish scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Initialize Flask app
app = Flask(__name__, template_folder="index.html")

# Serve the HTML frontend
@app.route('/')
def home():
     return send_file('index.html')   # This serves the HTML file

# Prediction API
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from request
        data = request.get_json()
        features = [data['features']]
        
        # Scale the input features
        scaled_features = scaler.transform(features)
        
        # Make a prediction
        species_pred = model.predict(scaled_features)[0]
        species_name = label_encoder.inverse_transform([species_pred])[0]

        return jsonify({"predicted_species": species_name})
    
    except Exception as e:
        return jsonify({"error": str(e)})

# Run Flask inside Jupyter Notebook
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
