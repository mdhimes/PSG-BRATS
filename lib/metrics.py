import numpy as np


def susemiehl(true, pred, fiducial):
    """
    Computes the median of the top 10% of errors, as described in Susemiehl et al 2023.

    Inputs
    ------
    true: array. True values.
    pred: array. Interpolated values.
    fiducial: array.  The nominal spectrum provided to make the grid.
    """
    err = (true - pred)**2
    dstar = np.median(np.sort(err)[..., -int(err.shape[-1]/10):], axis=-1)**0.5
    return dstar / np.maximum(true.mean(-1), fiducial.mean())


def maxape(true, pred):
    """
    Computes the maximum absolute percent error between two arrays.

    Inputs
    ------
    true: array. True values.
    pred: array. Interpolated values.

    Outputs
    -------
    maxerr: float. Maximum interpolation error within the grid.
    """
    return np.abs((true - pred) / true).max(-1)


def sigrej_maxape(true, pred, sig=5):
    """
    Computes the sigma-rejected maximum absolute percent error between two arrays.

    Inputs
    ------
    true: array. True values.
    pred: array. Interpolated values.
    sig : float. Number of standard deviations from the median to reject.
                 Default: 5

    Outputs
    -------
    maxerr: float. Maximum interpolation error within the grid.
    """
    err = np.abs((true - pred) / true)
    if not np.all(err==0):
        med = np.median(err)
        std = err.std()
        err[(err <= med - sig*std) * (err >= med + sig*std)] = 0
        return err.max(-1)
    else:
        return np.zeros(true.shape[:-1], dtype=float)


def mse(true, pred):
    """
    Computes the mean square error between two arrays.

    Inputs
    ------
    true: array. True values.
    pred: array. Interpolated values.

    Outputs
    -------
    maxerr: float. Calcualted RMSE.
    """
    return ((true - pred)**2).sum(-1) / true.shape[0]


