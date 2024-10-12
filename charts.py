import matplotlib.pyplot as plt
import numpy as np
import io
import base64

def plot_regression_line(X, y, a, b):
    """
    Plots the regression line based on the model's coefficients.

    Args:
    X (list): List of X values (independent variable).
    y (list): List of y values (dependent variable).
    a (float): Angular coefficient (slope).
    b (float): Linear coefficient (intercept).

    Returns:
    str: Base64 string of the plot image.
    """
    plt.figure()
    plt.scatter(X, y, color='blue', label='Training Data')

    X_line = np.linspace(min(X), max(X), 100)
    y_line = a * X_line + b

    plt.plot(X_line, y_line, color='red', label=f'Regression Line: y = {a:.2f}x + {b:.2f}')
    plt.xlabel('X (Independent Variable)')
    plt.ylabel('y (Dependent Variable)')
    plt.title('Linear Regression Model')
    plt.legend()
    plt.grid(True)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)

    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return img_str
