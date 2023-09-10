# file: dataprepRC.py
import warnings
import numpy as np
import pdb

def lag_data(X, lag, **kwargs):
    """
    Inputs:
    X: a num_dimensions X sample size numpy array of data

    """
    times = False
    for arg in kwargs:
        if arg == 'times':
            t = kwargs[arg]
            times = True
        else:
            warnings.warn("Unrecognized keyword argument passed to lag_data.  Argument ignored.")
    if len(X.shape) == 2:
        terminus = X.shape[1] - lag + 1
    elif len(X.shape) == 1:
        terminus = len(X) - lag + 1
    tmp = []
    for ii in range(lag):
        if len(X.shape) == 2:
            tmp.append(X[:, ii:(terminus + ii)])
        elif len(X.shape) == 1:
            tmp.append(X[ii:(terminus + ii)].reshape(1,-1))
    Xlagged = np.concatenate(tmp, axis=0)
    if times:
        if len(t.shape) ==1:
            t = t[:-lag]
        else:
            t = t[:, :-lag]
        return Xlagged, t
    else:
        return Xlagged


def unlag_data(Xlagged, lag):
    """
    Inputs:
    Xlagged: data in the format returned by lag_data

    Outputs:
    X : data in time series without lag
    """
    dims = int(Xlagged.shape[0] / lag)
    start = Xlagged[:dims, :]
    end = Xlagged[-dims:, -lag+1:]
    return np.concatenate([start, end], axis=1)


def train_test(X, train_proportion=0.9):
    """
    """
    cut = int(train_proportion * X.shape[0])
    train_u = X[:cut, :][:-1,:]
    train_y = X[:cut, :][1:,:]
    y_init = X[cut, :].reshape(-1,1)
    test_y = X[cut+1:,:]

    return train_u, train_y, test_y, y_init


