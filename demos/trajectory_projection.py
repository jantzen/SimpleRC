# trajectory_projection.py

import numpy as np
import matplotlib.pyplot as plt
from simpleRC import *
import os
import pandas as pd
import importlib.util
import pdb

def main(plots=False, noise=False, partial=False, gpu=False):
    # open file for saving output
    f = open('./output/trajectory_projection_output', 'w')

    # Build a parabolic trajectory
    t = np.linspace(-1,1,1000).reshape(-1,1)
    X = -t ** 2 + 1 
    plt.plot(t, X)

    # prepare training and test data
    train_frac = 0.8
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
    if os.path.exists('./output/proj_res.npy'):
        print("Loading previously saved reservoir...")
        f.write("Loading previously saved rexervoir...\n")
        W_temp = np.load('./output/proj_res.npy')
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
    test_y_truncated, preds = rc.predict(test_y, 30)
    error = np.sqrt(np.mean((test_y_truncated[1:,:] - preds[:-1,:])**2))
    print("Error on test set: {}".format(error))
    print("Relative error on test set: {}".format(error/(np.max(test_y_truncated)-np.min(test_y_truncated))))
    f.write("Error on test set: {}\n".format(error))

    try:
        if plots:
            plt.figure()
            plt.plot(t[cut+1+31:], test_y_truncated[1:], 'bo')
            plt.plot(t[cut+1+31:], preds[:-1], 'r-')
            plt.legend(('true','predicted'))
    except Exception as e:
        print("An exception occurred attempting to plot.")
        print(e)

    # Save predictions for plotting
    pred_data = np.concatenate([t.flatten()[cut+1+30:].reshape(-1,1),
                                test_y_truncated, preds], axis=1)
    np.save('../data/traj_proj_rc.npy', pred_data)

    # Repeat with an overtly representational method
    preds = []
    for ii in range(2, test_y_truncated.shape[0]):

        dt = t[1] - t[0]

        # estimate first derivative
        d1 = test_y_truncated[ii] - test_y_truncated[ii-1]
        d2 = test_y_truncated[ii-1] - test_y_truncated[ii-2]
            
        # estimate second derivative
        dd = d1 - d2

        # extrapolate
        preds.append(test_y_truncated[ii] + d1 * dt + 0.5 * dd * dt**2)

    preds = np.array(preds)

    error = np.sqrt(np.mean((test_y_truncated[3:] - preds[:-1])**2))
    print("Error on test set: {}".format(error))
    print("Relative error on test set: {}".format(error/(np.max(test_y_truncated)-np.min(test_y_truncated))))
    f.write("Error on test set: {}\n".format(error))



    try:
        if plots:
            plt.figure()
            plt.plot(t[cut+1+33:], test_y_truncated[3:], 'bo')
            plt.plot(t[cut+1+33:], preds[:-1], 'r-')
            plt.legend(('true','predicted'))
    except Exception as e:
        print("An exception occurred attempting to plot.")
        print(e)
# 
#    
#    # Save predictions for plotting
#    pred_data = np.concatenate([t.flatten()[cut+1:].reshape(-1,1), test_y, preds], axis=1)
#    np.save('../data/mech_preds_double_pend.npy', pred_data)
#
#    f.close()
    plt.show()

    ans = input("Save this reservoir? (y / n)  :  ")
    if ans == 'y' or 'Y':
        if gpu:
            out = rc.Wres.cpu()
        else:
            out = rc.Wres
        np.save('./output/proj_res.npy', out)


if __name__ == "__main__":
#    main(plots=True, noise=False)
    main(plots=True, noise=False, gpu=True)
#    main(plots=True, noise=True, gpu=True)
#    main(plots=True, noise=True)
