# duffing.py

from __future__ import division
import numpy as np
from scipy.integrate import ode
from scipy.stats import linregress
import matplotlib.pyplot as plt
from simpleRC import *

def main(plots=False, noise=False, partial=False, gpu=False):

    # open file for saving output
    f = open('./output/duffing_output', 'w')

    # set up Duffing system in the chaotic regime
    alpha = -1.
    beta = 1.
#    gamma = 0.2
#    gamma = 0.28
#    gamma = 0.37
    gamma = 0.5
    delta = 0.3
    omega = 1.2

    def model(t, X, arg1):
        alpha = arg1[0]
        beta = arg1[1]
        gamma = arg1[2]
        delta = arg1[3]
        omega = arg1[4]
        x = X[0]
        y = X[1] # y = x_dot
        dX = np.zeros_like(X)
        
        dx = y
        dy = gamma * np.cos(omega * t) - delta * y - alpha * x - beta * x**3
        dX = np.array([dx, dy])
        return dX

    # initial condition
    X0 = np.array([1., 0.])

    # time points
    t0 = 0.
    t1 = 400.
    resolution = 2 * 10**4
    dt = (t1 - t0) / resolution

    # solve ODE at each timestep
    r = ode(model).set_integrator('lsoda')
    r.set_initial_value(X0, t0).set_f_params([alpha, beta, gamma, delta, omega])
    X = []
    t = []
    while r.successful() and r.t < t1:
        t.append(r.t)
        tmp = r.integrate(r.t+dt)
        X.append(tmp.reshape(-1,1))

    X = np.concatenate(X, axis=1)
    t = np.array(t).reshape(1,-1)
#    forcing = np.cos(omega * t)
#    t_norm = t / np.max(t)
#    X = np.concatenate([t_norm, X], axis=0)
#    X = np.concatenate([forcing, X], axis=0)
    X = X[:, int(50 / dt) :]
    t= t[:, int(50 / dt) :]
    X = X.T
    t = t.T
    c = 0.1
    if noise:
        X += c * np.random.random_sample(X.shape)

    if plots:
        plt.plot(t, X[:,0])
        plt.figure()
        plt.plot(X[:, 0], X[:, 1])

    # prepare training and test data
#    x = X.T
#    tmp = []
#    lag = 1000 
#    terminus = X.shape[1] - lag
#    for ii in range(lag):
#        # save only coordinate, not derivative
#        tmp.append(X[0, ii:(terminus + ii)].reshape(1,-1))
#    tmp = np.concatenate(tmp, axis=0)
#    x = tmp.T
#    t = t[:, :-lag]

    # data for predicting the future
    cut = int(0.95 * X.shape[0])
    train_u = X[:cut, :][:-1,:]
    train_y = X[:cut, :][1:,:]
    test_y = X[cut+1:,:]
    U_init = X[cut, :].reshape(-1, 1)

    # setup an RC
    print("Setting up RC...")
    f.write("Setting up RC...\n")
    nn = 500
    sparsity = 0.5
    g = 0.2 # increase with increasing sparsity
    nu = no = X.shape[1]
    ss = 500
    print("Training to forecast future states...")
    f.write("Training to forecast future states...\n")
    #rc_predict = simpleRC(lag, nn, lag, sparsity=sparsity, mode='recurrent_forced',
    #        gpu=True)
    #rc_predict.train(train_u, train_y, gamma=g)
    rc_predict = simpleRC(nu, nn, no, sparsity=sparsity, gpu=gpu)
    rc_predict.train(train_u, train_y, gamma=g, settling_steps=500)
    #preds = rc_predict.predict(train_u)
    preds = rc_predict.predict(train_u, settling_steps=500)
    #error = np.sqrt(np.mean((train_y - preds)**2))
    error = np.sqrt(np.mean((train_y[ss:,:] - preds)**2))
    print("Error on training set: {}".format(error))
    f.write("Error on training set: {}\n".format(error))
#    U_init = test_y[0,:].reshape(-1,1)
#    print(U_init)
    steps = test_y.shape[0]
    preds = rc_predict.project(U_init, steps)
#    preds = rc_predict.run(U_init, steps)
    error = np.sqrt(np.mean((test_y - preds)**2))
    print("Error on test set: {}".format(error))
    f.write("Error on test set: {}\n".format(error))

#    if plots:
#        plt.figure()
#        for ii in range(1):
##            plt.plot(X[0, cut:], test_y[:,ii], 'bo')
##            plt.plot(X[0, cut:], preds[:,ii], 'r-')
#            plt.plot(t[0, cut:], test_y[:,ii], 'bo')
#            plt.plot(t[0, cut:], preds[:,ii], 'r-')
#        plt.legend(('true','predicted'))
#    f.close()
#    plt.show()

    try:
        if plots:
            plt.figure()
            for ii in range(test_y.shape[1]):
                plt.plot(t[cut+1:], test_y[:,ii], 'bo')
                plt.plot(t[cut+1:], preds[:,ii], 'r-')
            plt.legend(('true','predicted'))
    except Exception as e:
        print("An exception occurred attempting to plot.")
        print(e)
    f.close()
    plt.show()



if __name__ == "__main__":
#    main(plots=True, noise=False)
    main(plots=True, noise=False, gpu=True)
    main(plots=True, noise=True, gpu=True)
