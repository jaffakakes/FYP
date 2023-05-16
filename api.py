from flask import Flask, request, jsonify
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from flask_cors import CORS
from Linear_Regression import get_polynomial_regression_formula,linear_regression

app = Flask(__name__)
CORS(app)

@app.route('/polynomial-regression', methods=['POST'])
def run_polynomial_regression():
    
    intercept, coefficients = get_polynomial_regression_formula()

    return jsonify(intercept=intercept, coefficients=coefficients.tolist())

@app.route('/linear-regression', methods=['POST'])
def run_linear_regression():
    price_per_mileage, base_price = linear_regression()

    return jsonify(price_per_mileage=price_per_mileage, base_price=base_price)

if __name__ == '__main__':
    app.run(debug=True)