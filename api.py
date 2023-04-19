from flask import Flask, request, jsonify
import random
from Linear_Regression import linear_regression

app = Flask(__name__)

@app.route('/linear-regression', methods=['POST'])
def run_linear_regression():
    data = request.get_json()

    features = data.get('features')
    labels = data.get('labels')
    learning_rate = data.get('learning_rate', 0.01)
    epochs = data.get('epochs', 1000) 

    if features is None or labels is None:
        return jsonify(error="Missing required parameters."), 400

    price_per_room, base_price = linear_regression(features, labels, learning_rate, epochs)
    return jsonify(price_per_room=price_per_room, base_price=base_price)

if __name__ == '__main__':
    app.run(debug=True)
