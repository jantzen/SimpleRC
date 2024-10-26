# file: projection.py

from simpleRC import *
import numpy as np
import pdb

def main():
    rc = simpleRC(2, 3, 2, sparsity=0.5)
    # drive the RC for a while
    t = np.linspace(0., 6280., 5000)
    tmp = np.concatenate([np.sin(t).reshape(-1,1),
                              np.cos(0.7*t).reshape(-1,1)], axis=1)
    U_train = tmp[:-1,:]
    y_train = tmp[1:, :]
    rc.train(U_train, y_train)

#    # pick a random seed
#    rc.y = np.random.rand(2, 1)
    y = []
    x = []
    for ii in range(1000):
        rc.update(rc.y)
        x.append(rc.x.T)
        y.append(rc.y.T)
    y = np.concatenate(y, axis=0)
    x = np.concatenate(x, axis=0)
    
    ax1 = plt.figure().add_subplot(projection='3d')
    ax1.plot(x[:, 0], x[:, 1], x[:, 2], 'b.')
    ax2 = plt.figure().add_subplot()
    ax2.plot(y[:, 0], y[:, 1], 'b.')

    plt.show()

    # prompt and save data
    ans = input("Save this data? (y/n) \n")
    if ans == 'y' or ans == 'Y':
        data = np.concatenate([x, y], axis=1)
        np.save('../data/projection_demo.npy', data)

if __name__ == '__main__':
    main()
