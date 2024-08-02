# mechanical_systems_demo.py

import numpy as np
import matplotlib.pyplot as plt
from simpleRC import *
import os
import pandas as pd
import pdb

def main(plots=False, noise=False, partial=False, gpu=False):
    # open file for saving output
    f = open('meche_output', 'w')

    # Double linear (two masses on springs constrained to move in one dimension)
    # open data file
    data = pd.read_csv("./data/real_double_linear_h_1.txt", skiprows=1, sep='\s+', header=0, usecols=[1, 2, 3, 4, 5], names=('t', 'x1', 'x2', 'v1','v2'))
#    data = pd.read_csv("./data/real_double_linear_h_1.txt", skiprows=1, sep='\s+', header=0, usecols=[1, 2, 3], names=('t', 'x1', 'x2'))
#    data = pd.read_csv("./data/double_linear_h_1.txt", skiprows=1, sep='\s+', header=0, usecols=[1, 2, 3, 4, 5], names=('t', 'x1', 'x2', 'v1','v2'))
    data_np = data.to_numpy()
    t = data_np[:,0]
    X = data_np[:,1:]
    ax = plt.figure()
    for ii in range(X.shape[1]):
        plt.plot(t.flatten(), X[:,ii])
#    plt.legend(['x1', 'x2', 'v1', 'v2'])
    plt.legend(['x1', 'x2'])

    # prepare training and test data
    train_frac = 0.95
    cut = int(train_frac * X.shape[0])
    train_u = X[:cut, :][:-1,:]
    train_y = X[:cut, :][1:,:]
    test_y = X[cut+1:,:]
    U_init = X[cut, :].reshape(-1, 1)

    # setup an RC
    ss = 100 # settling steps
    print("Setting up RC...")
    f.write("Setting up RC...\n")
    nn = 600
    sparsity = 0.5
    g = 0.2 # increase with increasing sparsity
    print("Training to forecast future states...")
    f.write("Training to forecast future states...\n")
    nu = no = X.shape[1]
    rc = simpleRC(nu, nn, no, sparsity=sparsity, gpu=gpu)
    rc.train(train_u, train_y, gamma=g, settling_steps=ss)
    preds = rc.predict(train_u, settling_steps=ss)
    error = np.sqrt(np.mean((train_y[ss:,:] - preds)**2))
    print("Error on training set: {}".format(error))
    f.write("Error on training set: {}\n".format(error))
    steps = test_y.shape[0]
    preds = rc.project(U_init, steps)
    error = np.sqrt(np.mean((test_y - preds)**2))
    print("Error on test set: {}".format(error))
    f.write("Error on test set: {}\n".format(error))

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

    # Double pendulum
    # open data file
#    data = pd.read_csv("./data/real_double_pend_h_1.txt", skiprows=1, sep='\s+',header=0,names=('trial', 't','o1','o2'), usecols=[0,1,2,3])
    data = pd.read_csv("./data/real_double_pend_h_1.txt", skiprows=1, sep='\s+',header=0,names=('trial', 't','o1','o2', 'w1', 'w2'), usecols=[0,1,2,3,4,5])
#    data = pd.read_csv("./data/double_pend_h_1.txt", skiprows=1, sep='\s+',header=0,names=('trial', 't','o1','o2', 'w1', 'w2'), usecols=[0,1,2,3,4,5])
    data_np = data[data['trial'] == 0].to_numpy()
    # this dataset contains two trials -- use only the first
    t = data_np[:,1]
    X = data_np[:,2:]
    ax = plt.figure()
    for ii in range(X.shape[1]):
        plt.plot(t.flatten(), X[:,ii])
#    plt.legend(['o1', 'o2', 'w1', 'w2'])
    plt.legend(['o1', 'o2'])

    # prepare training and test data
    cut = int(train_frac * X.shape[0])
    train_u = X[:cut, :][:-1,:]
    train_y = X[:cut, :][1:,:]
    test_y = X[cut+1:,:]
    U_init = X[cut, :].reshape(-1, 1)

    # retraining the RC
    print("Training to forecast future states...")
    f.write("Training to forecast future states...\n")
    rc.train(train_u, train_y, gamma=g, settling_steps=ss)
    preds = rc.predict(train_u, settling_steps=ss)
    error = np.sqrt(np.mean((train_y[ss:,:] - preds)**2))
    print("Error on training set: {}".format(error))
    f.write("Error on training set: {}\n".format(error))
    steps = test_y.shape[0]
    preds = rc.project(U_init, steps)
    error = np.sqrt(np.mean((test_y - preds)**2))
    print("Error on test set: {}".format(error))
    f.write("Error on test set: {}\n".format(error))

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
#    main(plots=True, noise=True, gpu=True)
#    main(plots=True, noise=True)
