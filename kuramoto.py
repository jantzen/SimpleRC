from __future__ import division
import numpy as np
from scipy.integrate import ode
from scipy import zeros_like
from scipy.stats import linregress
import matplotlib.pyplot as plt
from simpleRC import *

def main(plots=False, noise=False, lag=10, fore=2., animate=True):

    # open file for saving output
    f = open('kuramoto_output', 'w')

    # set up Kuramoto systems

    N = 6
    K = 0.01

    omega= np.random.rand(N) * 2. * np.pi

    def model(t, theta, arg1):
        K = arg1[0]
        omega = arg1[1]
        dtheta = zeros_like(theta)
        sum1 = [0.] * N
        for ii in range(N):
            for jj in range(N):
                sum1[ii] += np.sin(theta[jj] - theta[ii])
                
            dtheta[ii] = omega[ii] + (K / N) * sum1[ii]
        
        return dtheta

    # initial condition
    theta0 = np.linspace(0.1, 2., N)

    # time points
    t0 = 0.
    t1 = 100.
    resolution = 10000
    dt = (t1 - t0) / resolution

    # solve ODE at each timestep
    r = ode(model).set_integrator('lsoda')
    r.set_initial_value(theta0, t0).set_f_params([K, omega])
    x = []
    t = []
    raw_theta_untrans1 = []
    while r.successful() and r.t < t1:
        t.append(r.t)
        tmp = r.integrate(r.t+dt)
        raw_theta_untrans1.append(tmp.reshape(-1,1))
        x.append(np.array([np.cos(tmp), np.sin(tmp)]).reshape(-1,1))

    x = np.concatenate(x, axis=1)

    c = 0.2

    if noise:
        x += c * np.random.random_sample(x.shape)

    # construct plot
    if plots:
        for ii in range(2*N):
            plt.plot(t, x[ii,:])

    # prepare training and test data
    x = x.T

    # data for predicting the past (checking the RC's memory)
    cut = int(0.8 * x.shape[0])
    past_train_u = x[:cut, :][lag:,:]
    past_train_y = x[:cut, :][:-lag,:]
    past_test_u = x[cut:,:][lag:,:]
    past_test_y = x[cut:,:][:-lag,:]

    # data for predicting the future
    stop = int(fore/dt)
    train_u = x[:cut, :][:-stop,:]
    train_y = x[:cut, :][stop:,:]
    test_u = x[cut:,:][:-stop,:]
    test_y = x[cut:,:][stop:,:]

    # setup an RC
    print("Setting up RC...")
    f.write("Setting up RC...\n")
    nn = 49
    sparsity = 0.5
    gamma = 0.1
    rc_past = simpleRC(2*N, nn, 2*N, sparsity=sparsity)
    print("Training to 'predict' past values...")
    f.write("Training to 'predict' past values...\n")
    rc_past.train(past_train_u, past_train_y, gamma=gamma)
    preds = rc_past.predict(past_train_u)
    error = np.sqrt(np.mean((past_train_y - preds)**2))
    print("Error on training set: {}".format(error))
    f.write("Error on training set: {}\n".format(error))
    preds = rc_past.predict(past_test_u)
    error = np.sqrt(np.mean((past_test_y - preds)**2))
    print("Error on test set: {}".format(error))
    f.write("Error on test set: {}\n".format(error))

    print("Setting up RC...")
    f.write("Setting up RC...\n")
    print("Training to forecast future states...")
    f.write("Training to forecast future states...\n")
    rc_predict = simpleRC(2*N, nn, 2*N, sparsity=sparsity)
    if animate:
        print("Building animation while training...")
        rc_predict.visual_train(train_u, train_y, gamma=gamma,
            filename='kuramoto_viz.mp4')
    else:
        rc_predict.train(train_u, train_y, gamma=gamma)
    preds = rc_predict.predict(train_u)
    error = np.sqrt(np.mean((train_y - preds)**2))
    print("Error on training set: {}".format(error))
    f.write("Error on training set: {}\n".format(error))
    preds = rc_predict.predict(test_u)
    error = np.sqrt(np.mean((test_y - preds)**2))
    print("Error on test set: {}".format(error))
    f.write("Error on test set: {}\n".format(error))

    if plots:
        plt.figure()
        for ii in range(2*N):
            plt.plot(t[cut:][stop:], test_y[:,ii], 'bo')
            plt.plot(t[cut:][stop:], preds[:,ii], 'ro')

    plt.show()



if __name__ == "__main__":
    main(plots=True, noise=False, fore=10., animate=False)
