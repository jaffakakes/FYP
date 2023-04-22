from flask import Flask, request, jsonify
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
def get_polynomial_regression_formula(polynomial_regression):
    coef = polynomial_regression.named_steps['linearregression'].coef_
    intercept = polynomial_regression.named_steps['linearregression'].intercept_
    return intercept, coef

@app.route('/polynomial-regression', methods=['POST'])
def run_polynomial_regression():
    data = request.get_json()

    mileage = np.array(data.get('mileage'))
    cost = np.array(data.get('cost'))
    degree = data.get('degree', 2)

    if mileage is None or cost is None:
        return jsonify(error="Missing required parameters."), 400

    mileage = mileage.reshape(-1, 1)
    X_train, _, y_train, _ = train_test_split(mileage, cost, test_size=0.2, random_state=42)

    polynomial_regression = make_pipeline(
        PolynomialFeatures(degree, include_bias=False),
        LinearRegression()
    )

    # Train the model
    polynomial_regression.fit(X_train, y_train)

    # Get the intercept and coefficients
    intercept, coefficients = get_polynomial_regression_formula(polynomial_regression)

    return jsonify(intercept=intercept, coefficients=coefficients.tolist())

if __name__ == '__main__':
    app.run(debug=True)