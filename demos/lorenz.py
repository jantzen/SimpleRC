# lorenz.py

from __future__ import division
import numpy as np
from scipy.integrate import ode
from scipy.stats import linregress
import matplotlib.pyplot as plt
from simpleRC import *
import os
import pdb

def main(plots=False, noise=False, partial=False, gpu=False):

    # open file for saving output
    f = open('./output/lorenz_output', 'w')

    # check to see if data already exists
    if os.path.exists('../data/lorenz_data.npy'):
        f.write('Loading preexisting data.')
        print('Loading preexisting data.')
        data = np.load('../data/lorenz_data.npy')
        t = data[0, :]
        X = data[1:, :]

    else:
        # set up Lorenz system in the chaotic regime
        beta = 8./3.
        rho = 28.
        sigma = 10.
    
        def model(t, X, arg1):
            beta = arg1[0]
            rho = arg1[1]
            sigma = arg1[2]
            x = X[0]
            y = X[1] 
            z = X[2]
            dX = np.zeros_like(X)
            
            dx = sigma * (y - x)
            dy = x * (rho - z) - y
            dz = x * y - beta * z
            dX = np.array([dx, dy, dz])
            return dX
    
        # initial condition
        X0 = np.array([10., 10., 10.])
    
        # time points
        t0 = 0.
        t1 = 100.
        resolution = 10**5
        dt = (t1 - t0) / resolution
    
        # solve ODE at each timestep
        r = ode(model).set_integrator('lsoda')
        r.set_initial_value(X0, t0).set_f_params([beta, rho, sigma])
        X = []
        t = []
        while r.successful() and r.t < t1:
            t.append(r.t)
            tmp = r.integrate(r.t+dt)
            X.append(tmp.reshape(-1,1))
    
        X = np.concatenate(X, axis=1)
        t = np.array(t).reshape(1,-1)
   
        c = 0.5
        if noise:
            X += c * np.random.standard_normal(X.shape)
        np.save('../data/lorenz_data.npy', np.concatenate([t, X], axis=0), allow_pickle=False)

    X = X.T
    plt.plot(t.flatten(), X[:,0])
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(X[:, 0], X[:, 1], X[:, 2])

    # prepare training and test data
    cut = int(0.95 * X.shape[0])
    train_u = X[:cut, :][:-1,:]
    train_y = X[:cut, :][1:,:]
    test_y = X[cut+1:,:]
    U_init = X[cut, :].reshape(-1, 1)

    # setup an RC
    print("Setting up RC...")
    f.write("Setting up RC...\n")
    nn = 600
    sparsity = 0.5
    g = 0.2 # increase with increasing sparsity
    print("Training to forecast future states...")
    f.write("Training to forecast future states...\n")
    nu = no = X.shape[1]
    rc_predict = simpleRC(nu, nn, no, sparsity=sparsity, gpu=gpu)
    rc_predict.train(train_u, train_y, gamma=g, settling_steps=100)
    preds = rc_predict.predict(train_u, settling_steps=100)
    error = np.sqrt(np.mean((train_y[100:,:] - preds)**2))
    print("Error on training set: {}".format(error))
    f.write("Error on training set: {}\n".format(error))
    steps = test_y.shape[0]
    preds = rc_predict.project(U_init, steps)
    error = np.sqrt(np.mean((test_y - preds)**2))
    print("Error on test set: {}".format(error))
    f.write("Error on test set: {}\n".format(error))

    try:
        if plots:
            plt.figure()
            for ii in range(3):
                plt.plot(t.flatten()[cut+1:], test_y[:,ii], 'bo')
                plt.plot(t.flatten()[cut+1:], preds[:,ii], 'r-')
            plt.legend(('true','predicted'))
    except Exception as e:
        print("An exception occurred attempting to plot.")
        print(e)

    # Save predictions for plotting
    pred_data = np.concatenate([t.flatten()[cut+1:].reshape(-1,1), test_y, preds], axis=1)
    np.save('../data/lorenz_predictions.npy', pred_data)

    f.close()
    plt.show()


if __name__ == "__main__":
#    main(plots=True, noise=False)
    main(plots=True, noise=False, gpu=True)
#    main(plots=True, noise=True, gpu=True)
#    main(plots=True, noise=True)
