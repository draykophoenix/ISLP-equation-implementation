from typing import Callable
import numpy.typing as npt

def mse(x: npt.ArrayLike, y: npt.ArrayLike, f_hat: Callable[[npt.ArrayLike], float]) -> float:
    '''
    Calculates the Mean Squared Error for a given predictor function.
    ## Parameters
    x : array-like 
        An array of observations.
    y : array-like 
        An array of response variables.
    f_hat : callable
        A function that predicts `y_i`, given `x_i`.

    ## Returns
    MSE : float
        A value for the mean squared error of the predictor on the supplied data.
    '''

    n = len(y)

    return 1/n * sum( (y[i] - f_hat(x[i]))**2  for i in range(n))