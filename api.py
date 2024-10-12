from linear_regression import SimpleLinearRegression
from flask import Flask, request, jsonify, session
import numpy as np
import pickle
from uuid import uuid4

app = Flask(__name__)
app.secret_key = 'supersecret' 

@app.before_request
def ensure_session_id():
    """Ensure each request has a unique session ID."""
    if 'id' not in session:
        session['id'] = str(uuid4()) 
@app.route('/train', methods=['POST'])
def train():
    """Train the model for the current session."""
    data = request.get_json()
    X = np.array(data['X'])
    y = np.array(data['y'])

    try:
        model = SimpleLinearRegression()
        model.least_squares(X, y) 
        session['model'] = pickle.dumps(model).decode('latin1')
        return jsonify({"message": "Model trained successfully"}), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

@app.route('/predict', methods=['POST'])
def predict():
    """Make a prediction using the current session's model."""
    if 'model' not in session:
        return jsonify({"error": "No model found for this session"}), 400

    model = pickle.loads(session['model'].encode('latin1'))
    data = request.get_json()
    X = np.array(data['X'])

    try:
        prediction = model.predict(X).tolist()
        return jsonify({"prediction": prediction}), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

@app.route('/coefficients', methods=['GET'])
def coefficients():
    """Get the coefficients for the current session's model."""
    if 'model' not in session:
        return jsonify({"error": "No model found for this session"}), 400

    model = pickle.loads(session['model'].encode('latin1'))

    try:
        angular_coefficient = model.get_angular_coefficient()
        linear_coefficient = model.get_linear_coefficient()
        return jsonify({
            "angular_coefficient": angular_coefficient,
            "linear_coefficient": linear_coefficient
        }), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
