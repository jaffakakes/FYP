import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# We set the random seed in order to always get the same results.
np.random.seed(42)

# def square_trick(base_price, price_per_mileage, mileage, price, learning_rate):
#     predicted_price = base_price + price_per_mileage * mileage
#     price_per_mileage += learning_rate * mileage * (price - predicted_price)
#     base_price += learning_rate * (price - predicted_price)
#     return price_per_mileage, base_price



# def linear_regression(mileage, cost, learning_rate=1, epochs=1000):
#     price_per_mileage = random.random()
#     base_price = random.random()
#     for epoch in range(epochs):
#         i = random.randint(0, len(mileage)-1)
#         current_mileage = mileage[i]
#         current_cost = cost[i]

#         price_per_mileage, base_price = square_trick(base_price, price_per_mileage, current_mileage, current_cost, learning_rate=learning_rate)
     
#         # Debugging print statements
#         print(f'Epoch: {epoch}, Mileage: {current_mileage}, Cost: {current_cost}, Price per mileage: {price_per_mileage}, Base price: {base_price}')

#     return price_per_mileage, base_price


def get_polynomial_regression_formula(polynomial_regression):
    coef = polynomial_regression.named_steps['linearregression'].coef_
    intercept = polynomial_regression.named_steps['linearregression'].intercept_
    return intercept, coef

degree = 2
polynomial_regression = make_pipeline(
    PolynomialFeatures(degree, include_bias=False),
    LinearRegression()
)

# Train the model


# Get the intercept and coefficients
intercept, coefficients = get_polynomial_regression_formula(polynomial_regression)

print("Intercept:", intercept)
print("Coefficients:", coefficients)



