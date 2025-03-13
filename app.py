from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained model
model = joblib.load("model.pkl")  # Ensure model.pkl exists in the same directory

# Home route (optional)
@app.route("/")
def home():
    return render_template("index.html")  # Ensure Lab4.html is in the correct location

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Parse incoming request
        data = request.get_json()
        features = np.array(data["features"]).reshape(1, -1)  # Convert to 2D array

        # Make prediction
        predicted_weight = model.predict(features)[0]

        # Return JSON response
        return jsonify({"predicted_weight": round(predicted_weight, 2)})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
