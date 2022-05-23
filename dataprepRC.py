# file: dataprepRC.py
import warnings
import numpy as np

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

    terminus = X.shape[1] - lag
    tmp = []
    for ii in range(lag):
        if len(X.shape) == 1:
            tmp.append(X[:, ii:(terminus + ii)].reshape(1,-1))
        else:
            tmp.append(X[:, ii:(terminus + ii)])
    Xlagged = np.concatenate(tmp, axis=0)
    if times:
        if len(t.shape) ==1:
            t = t[:-lag]
        else:
            t = t[:, :-lag]
        return Xlagged, t
    else:
        return Xlagged


def train_test(X, train_proportion=0.9):
    """
    """
    cut = int(train_proportion * X.shape[0])
    train_u = X[:cut, :][:-1,:]
    train_y = X[:cut, :][1:,:]
    test_y = X[cut:,:]

    return train_u, train_y, test_y


