import numpy as np, random
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd

# We set the random seed in order to always get the same results.


def square_trick(base_price, price_per_mileage, mileage, price, learning_rate):
    predicted_price = base_price + price_per_mileage * mileage
    price_per_mileage += learning_rate * mileage * (price - predicted_price)
    base_price += learning_rate * (price - predicted_price)
    return price_per_mileage, base_price





df = pd.read_csv('BOOK2.csv')
df['Cost'] = df['Cost'].replace({'£': '', ',': ''}, regex=True).astype(float)
mileage = np.array(df['Mileage']).reshape(-1, 1)
cost = np.array(df['Cost'])



def linear_regression(learning_rate=0.0000000001, epochs=1000):
    price_per_mileage = np.random.random()
    base_price =np.random.random()
    for epoch in range(epochs):
        i = random.randint(0, len(mileage)-1)
        current_mileage = mileage[i]
        current_cost = cost[i]

        price_per_mileage, base_price = square_trick(base_price, price_per_mileage, current_mileage, current_cost, learning_rate=learning_rate)
     
        # Debugging print statements
        print(f'Epoch: {epoch}, Mileage: {current_mileage}, Cost: {current_cost}, Price per mileage: {price_per_mileage}, Base price: {base_price}')

    return price_per_mileage, base_price







degree = 2
X_train, _, y_train, _ = train_test_split(mileage, cost, test_size=0.2, random_state=42)
polynomial_regression = make_pipeline(
    PolynomialFeatures(degree, include_bias=False),
    LinearRegression()
)
polynomial_regression.fit(X_train, y_train)

def get_polynomial_regression_formula():
    coef = polynomial_regression.named_steps['linearregression'].coef_
    intercept = polynomial_regression.named_steps['linearregression'].intercept_
    return intercept, coef