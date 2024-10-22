# kuramoto.py

from __future__ import division
import numpy as np
from scipy.integrate import ode
#from scipy import zeros_like
from scipy.stats import linregress
import matplotlib.pyplot as plt
from simpleRC import *
from dataprepRC import *
import os

def main(plots=False, noise=False, animate=True, partial=False):

    # open file for saving output
    if noise:
        f = open('./output/kuramoto_output_noisy', 'w')
    else:
        f = open('./output/kuramoto_output', 'w')

    # set up Kuramoto systems
    N = 6
    K = 0.01

    omega= np.random.rand(N) * 2. * np.pi

    def model(t, theta, arg1):
        K = arg1[0]
        omega = arg1[1]
        dtheta = np.zeros_like(theta)
        sum1 = [0.] * N
        for ii in range(N):
            for jj in range(N):
                sum1[ii] += np.sin(theta[jj] - theta[ii])
                
            dtheta[ii] = omega[ii] + (K / N) * sum1[ii]
        
        return dtheta

    if not noise and os.path.isfile('../data/kuramoto_data.npy'):
        print("Using previously generated data...")
        f.write("Using previously generated data.\n")
        data = np.load('../data/kuramoto_data.npy')
        t = data[0, :]
        x = data[1:,:]
    elif noise and os.path.isfile('../data/kuramoto_data_noisy.npy'):
        print("Using previously generated noisy data...")
        f.write("Using previously generated noisy data.\n")
        data = np.load('../data/kuramoto_data_noisy.npy')
        t = data[0, :]
        x = data[1:,:]
    else:
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

        c = 0.05

        data_filename = 'kuramoto_data.npy'

        if noise:
            x += c * np.random.random_sample(x.shape)
            data_filename = 'kuramoto_data_noisy.npy'

        data = np.concatenate([np.array(t).reshape(1,-1), x], axis=0)

        np.save('../data/' + data_filename, data)

    # prepare training and test data
    x = x.T

    # data for predicting the future
    cut = int(0.95 * x.shape[0])
    train_u = x[:cut, :][:-1,:]
    train_y = x[:cut, :][1:,:]
    test_y = x[cut+1:,:]
    U_init = x[cut, :].reshape(-1, 1)

    # setup an RC
    print("Setting up RC...")
    f.write("Setting up RC...\n")
    nn = 100
    sparsity = 0.5
    gamma = 0.1
    print("Setting up RC...")
    f.write("Setting up RC...\n")
    print("Training to forecast future states...")
    f.write("Training to forecast future states...\n")
    rc_predict = simpleRC(2*N, nn, 2*N, sparsity=sparsity)
    rc_predict.train(train_u, train_y, gamma=gamma, settling_steps=100)
    U_trunc, preds = rc_predict.predict(train_u, settling_steps=100)
    error = np.sqrt(np.mean((train_y[100:,:] - preds)**2))
    print("Error on training set: {}".format(error))
    f.write("Error on training set: {}\n".format(error))
    steps = test_y.shape[0]
    rc_predict.train(train_u, train_y, gamma=gamma, settling_steps=100)
    preds = rc_predict.project(U_init, steps)
    error = np.sqrt(np.mean((test_y - preds)**2))
    print("Error on test set: {}".format(error))
    f.write("Error on test set: {}\n".format(error))

    if plots:
        plt.figure()
        for ii in range(2*N):
            plt.plot(t[cut+1:], test_y[:,ii], 'bo')
            plt.plot(t[cut+1:], preds[:,ii], 'r-')
        plt.legend(('true','predicted'))
    f.close()
    plt.show()


if __name__ == "__main__":
    main(plots=True, noise=False, animate=False)
    main(plots=True, noise=True, animate=False)
