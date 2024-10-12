from linear_regression import SimpleLinearRegression
from flask import Flask, request, jsonify, session
import numpy as np
from uuid import uuid4

app = Flask(__name__)
app.secret_key = 'supersecret'  

models = {}

@app.route('/train', methods=['POST'])
def train():
    """
    Endpoint to train the model.
    Expects JSON input with 'X' and 'y' lists.

    Example:
    {
        "X": [1, 2, 3, 4, 5],
        "y": [2, 3, 4, 5, 6]
    }
    """
    session_id = session.get('id', str(uuid4()))
    session['id'] = session_id

    data = request.get_json()
    X = np.array(data['X'])
    y = np.array(data['y'])

    try:
        model = SimpleLinearRegression()
        model.least_squares(X, y)  
        models[session_id] = model 
        return jsonify({"message": "Model trained successfully"}), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to predict using the model.
    Expects JSON input with 'X' list.

    Example:
    {
        "X": [1]
    }
    """
    session_id = session.get('id')
    if session_id not in models:
        return jsonify({"error": "No model found for this session"}), 400

    model = models[session_id]  
    data = request.get_json()
    X = np.array(data['X'])

    try:
        prediction = model.predict(X).tolist()  
        return jsonify({"prediction": prediction}), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

@app.route('/coefficients', methods=['GET'])
def coefficients():
    """
    Endpoint to get the coefficients of the model.

    Example response:
    {
        "angular_coefficient": 1.0,
        "linear_coefficient": 1.0
    }
    """
    session_id = session.get('id')
    if session_id not in models:
        return jsonify({"error": "No model found for this session"}), 400

    model = models[session_id] 

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
