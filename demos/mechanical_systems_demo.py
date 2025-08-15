# mechanical_systems_demo.py

import numpy as np
import matplotlib.pyplot as plt
from simpleRC import *
import os
import pandas as pd
import importlib.util
import pdb

def main(plots=False, noise=False, partial=False, gpu=False):
    # open file for saving output
    f = open('./output/meche_output', 'w')

    # Double linear (two masses on springs constrained to move in one dimension)
    # open data file
    data = pd.read_csv("../data/real_double_linear_h_1.txt", skiprows=1, sep='\s+', header=0, usecols=[1, 2, 3, 4, 5], names=('t', 'x1', 'x2', 'v1','v2'))
    data_np = data.to_numpy()
    t = data_np[:,0]
    X = data_np[:,1:]
    ax = plt.figure()
    for ii in range(X.shape[1]):
        plt.plot(t.flatten(), X[:,ii])
    plt.legend(['x1', 'x2', 'v1', 'v2'])
#    plt.legend(['x1', 'x2'])

    # prepare training and test data
    train_frac = 0.92
    cut = int(train_frac * X.shape[0])
    train_u = X[:cut, :][:-1,:]
    train_y = X[:cut, :][1:,:]
    test_y = X[cut+1:,:]
    U_init = X[cut, :].reshape(-1, 1)

    # setup an RC
    ss = 30 # settling steps
    print("Setting up RC...")
    f.write("Setting up RC...\n")
    nn = 100
    sparsity = 0.5
    g = 0.4 # increase with increasing sparsity
    print("Training to forecast future states...")
    f.write("Training to forecast future states...\n")
    nu = no = X.shape[1]
    rc = simpleRC(nu, nn, no, sparsity=sparsity, gpu=gpu)
    if os.path.exists('./output/W_res.npy'):
        print("Loading previously saved reservoir...")
        f.write("Loading previously saved rexervoir...\n")
        W_temp = np.load('./output/W_res.npy')
        if gpu:
            try:
                importlib.util.find_spec('torch')
                found = True
            except ImportError:
                errmsg = "The PyTorch module (torch) was not found. Restricted to CPU methods."
                warnings.warn(errmsg)
                found = False
            if found:
                import torch
                torch.set_default_dtype(torch.float64)
            rc.W_res = torch.from_numpy(W_temp).to(rc.device)
        else:
            rc.W_res = W_temp
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

    # Save predictions for plotting
    pred_data = np.concatenate([t.flatten()[cut+1:].reshape(-1,1), test_y, preds], axis=1)
    np.save('../data/mech_preds_double_linear.npy', pred_data)


    # Double pendulum
    # open data file
    data = pd.read_csv("../data/real_double_pend_h_1.txt", skiprows=1, sep='\s+',header=0,names=('trial', 't','o1','o2', 'w1', 'w2'), usecols=[0,1,2,3,4,5])
    data_np = data[data['trial'] == 0].to_numpy()
    # this dataset contains two trials -- use only the first
    t = data_np[:,1]
    X = data_np[:,2:]
    ax = plt.figure()
    for ii in range(X.shape[1]):
        plt.plot(t.flatten(), X[:,ii])
    plt.legend(['o1', 'o2', 'w1', 'w2'])
#    plt.legend(['o1', 'o2'])

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
 
    
    # Save predictions for plotting
    pred_data = np.concatenate([t.flatten()[cut+1:].reshape(-1,1), test_y, preds], axis=1)
    np.save('../data/mech_preds_double_pend.npy', pred_data)

    f.close()
    plt.show()

    ans = input("Save this reservoir? (y / n)  :  ")
    if ans == 'y' or 'Y':
        if gpu:
            out = rc.Wres.cpu()
        else:
            out = rc.Wres
        np.save('./output/W_res.npy', out)


if __name__ == "__main__":
#    main(plots=True, noise=False)
    main(plots=True, noise=False, gpu=True)
#    main(plots=True, noise=True, gpu=True)
#    main(plots=True, noise=True)
