import numpy as np


class SimpleLinearRegression:
    """
    Simple Linear Regression model

    Attributes:
    a (float): angular coefficient
    b (float): linear coefficient

    Methods:
    least_squares(X: np.ndarray, y: np.ndarray) -> None: calculates the angular and linear coefficients
    predict(X: np.ndarray) -> np.ndarray: predicts the output based on the input
    get_angular_coefficient() -> float: returns the angular coefficient
    get_linear_coefficient() -> float: returns the linear coefficient

    """
    def __init__(self):
        """
        Initializes the SimpleLinearRegression class
        """
        self.a: float = None
        self.b: float = None
    
    def least_squares(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Calculates the angular and linear coefficients

        Args:
        X (np.ndarray): input data
        y (np.ndarray): output data

        Returns:
        None

        Example:

        >>>> X = np.array([1, 2, 3, 4, 5])
        >>>> y = np.array([2, 3, 4, 5, 6])
        >>>> model = SimpleLinearRegression()
        >>>> model.least_squares(X, y)

        Raises:
        ValueError: if input and output data have different lengths
        """

        if len(X) != len(y):
            raise ValueError("Input and output data must have the same length")

        n: int = len(X)
        x_sum: float = np.sum(X)
        y_sum: float = np.sum(y)
        x_square: float = np.sum(X**2)
        y_square: float = np.sum(y**2)
        xy_sum: float = np.sum(X*y)

        self.a = (n*xy_sum - x_sum*y_sum) / (n*x_square - x_sum**2)
        self.b = (y_sum - self.a*x_sum) / n

    def check_train(self) -> bool:
        """
        Checks if the model has been trained

        Returns:
        bool: True if the model has been trained, False otherwise

        Example:

        >>>> model.check_train()

        """

        if self.a == 0 and self.b == 0:
            return False
        return True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the output based on the input

        Args:
        X (np.ndarray): input data

        Returns:
        np.ndarray: predicted output

        Example:

        >>>> X = np.array([7])
        >>>> model.predict(X)

        Raises:
        ValueError: if input data is not a numpy array
        ValueError: if input data has more than one element
        ValueError: if model has not been trained yet
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("Input data must be a numpy array")
        
        if len(X) >1:
            raise ValueError("Input data must be a numpy array with only one element")
        
        if self.check_train() == False:
            raise ValueError("Model has not been trained yet")
        

        return self.a*X + self.b
    
    def get_angular_coefficient(self) -> float:
        """
        Returns the angular coefficient

        Returns:
        float: angular coefficient

        Example:

        >>>> model.get_angular_coefficient()

        Raises:
        ValueError: if model has not been trained yet
        """

        if self.check_train() == False:
            raise ValueError("Model has not been trained yet")

        return self.a
    
    def get_linear_coefficient(self) -> float:
        """
        Returns the linear coefficient

        Returns:
        float: linear coefficient

        Example:

        >>>> model.get_linear_coefficient()

        Raises:
        ValueError: if model has not been trained yet


        """

        if self.check_train() == False:
            raise ValueError("Model has not been trained yet")

        return self.b



if __name__ == "__main__":
    X = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 3, 4, 5, 6])
    model = SimpleLinearRegression()
    model.least_squares(X, y)
    print(model.predict(np.array([7])))
    print(model.get_angular_coefficient())
    print(model.get_linear_coefficient())
    print(model.check_train())
    print(model.predict(np.array([7])))

    # Test if model has been trained
    model2 = SimpleLinearRegression()
    print(model2.check_train())

    # Test if model has been trained
    model3 = SimpleLinearRegression()
    model3.least_squares(X, y)
    print(model3.check_train())

    # Test if input data is a numpy array
    try:
        model.predict([7])
    except ValueError as e:
        print(e)

    # Test if input and output data have the same length
    try:
        model.least_squares(np.array([1, 2, 3, 4]), np.array([2, 3, 4, 5, 6]))
    except ValueError as e:
        print(e)

    # Test if model has not been trained yet
    try:
        model.predict(np.array([7]))
    except ValueError as e:
        print(e)

    # Test if model has not been trained yet
    try:
        model.get_angular_coefficient()
    except ValueError as e:
        print(e)

    # Test if model has not been trained yet
    try:
        model.get_linear_coefficient()
    except ValueError as e:
        print(e)