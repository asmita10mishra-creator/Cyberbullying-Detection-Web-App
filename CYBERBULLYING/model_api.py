from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import re

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load trained model and vectorizer
model = joblib.load("cyberbullying_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Clean text before sending to model
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Flask route to handle POST + OPTIONS (CORS)
@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze():
    if request.method == 'OPTIONS':
        return '', 200  # CORS preflight

    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    clean = clean_text(text)
    vectorized = vectorizer.transform([clean])
    prediction = model.predict(vectorized)[0]

    return jsonify({"prediction": int(prediction)})

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, port=5000)
