import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score
from data import generate_xor_data
from visual import plot_2d_data

X, y=generate_xor_data(n=200)

plot_2d_data(X,y,title="XOR Data (Original Space)")

linear_model=LogisticRegression()
linear_model.fit(X, y)
y_pred_linear=linear_model.predict(X)
linear_accuracy=accuracy_score(y,y_pred_linear)
print(f"Linear Model Accuracy on XOR Data: {linear_accuracy:.2f}")
plot_2d_data(X, y_pred_linear, title="XOR Data-Linear Model Predictions")
