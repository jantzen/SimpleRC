from __future__ import division
import numpy as np
from scipy.integrate import ode
from scipy.stats import linregress
import matplotlib.pyplot as plt
from simpleRC import *
from dataprepRC import *
import os



def main(plots=False, noise=False, gpu=False):

    def lorenz96(t, x, arg1):
        """ Based on model at https://en.wikipedia.org/wiki/Lorenz_96_model
        """
        """Lorenz 96 model with constant forcing"""
        N = arg1[0]
        F = arg1[1]
        # Setting up vector
        d = np.zeros(N)
        # Loops over indices (with operations and Python underflow indexing handling edge cases)
        for i in range(N):
            d[i] = (x[(i + 1) % N] - x[i - 2]) * x[i - 1] - x[i] + F
        return d

    # parameters
    N = 5  # Number of variables
    F = 8.  # Forcing

    # time points
    t0 = 0.
    t1 = 110.
    resolution = 1. * 10**5
    dt = (t1 - t0) / resolution

    x0 = F * np.ones(N)  # Initial state (equilibrium)
    x0[0] += 0.01  # Add small perturbation to the first variable
    # solve ODE at each timestep
    r = ode(lorenz96).set_integrator('lsoda')
    r.set_initial_value(x0, t0).set_f_params([N, F])
    X = []
    t = []
    while r.successful() and r.t < t1:
        t.append(r.t)
        tmp = r.integrate(r.t+dt)
        X.append(tmp.reshape(-1,1))
    X = np.concatenate(X, axis=1)
    t = np.array(t)
    
    c = 0.3
    if noise:
        X += c * np.random.standard_normal(X.shape)

    X = X.T
    # drop the first 10 percent of the simulation 
    X = X[int(X.shape[0] * 0.1):, :]
    t = t[int(t.shape[0] * 0.1):]
    plt.plot(t, X[:,0])
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(X[:, 0], X[:, 1], X[:, 2])

    # prep training and test data
    cut = int(0.97 * X.shape[0])
    train_u = X[:cut, :][:-1,:]
    train_y = X[:cut, :][1:,:]
    test_y = X[cut+1:,:]
    U_init = X[cut, :].reshape(-1, 1)

    # setup an RC
    print("Setting up RC...")
    nn = 600
    sparsity = 0.5
    g = 0.2 # increase with increasing sparsity
    print("Training to forecast future states...")
    nu = no = X.shape[1]
    rc_predict = simpleRC(nu, nn, no, sparsity=sparsity, gpu=gpu)
    rc_predict.train(train_u, train_y, gamma=g, settling_steps=100)
    preds = rc_predict.predict(train_u, settling_steps=100)
    error = np.sqrt(np.mean((train_y[100:,:] - preds)**2))
    print("Error on training set: {}".format(error))
#    U_init = test_y[0,:].reshape(-1,1)
    steps = test_y.shape[0]
    preds = rc_predict.project(U_init, steps)
    error = np.sqrt(np.mean((test_y - preds)**2))
    print("Error on test set: {}".format(error))

    try:
        if plots:
            plt.figure()
            for ii in range(N):
                plt.plot(t[cut+1:], test_y[:,ii], 'bo')
                plt.plot(t[cut+1:], preds[:,ii], 'r-')
            plt.legend(('true','predicted'))
    except Exception as e:
        print("An exception occurred attempting to plot.")
        print(e)
 
    plt.show()

if __name__ == '__main__':
#    main(plots=True, gpu=True, lag=5, noise=True)
    main(plots=True, gpu=True, noise=False)
