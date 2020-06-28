# file: simpleRC.py

import numpy as np
import warnings
import matplotlib.pyplot as plt
import pdb

class simpleRC( object ):

    def __init__(self,
            nu, # size of input layer
            nn, # size of the reservoir
            no,  # size of the output layer
            sparsity=0.1 # fraction of connections to make
            ):
        self.nu = nu
        self.nn = nn
        self.no = no

        # set input weights (nn X nu)
        self.Win = np.random.normal(size=(nn, nu + 1))
        
        # set reservoir connections and weights (nn X nn)
        edge_matrix = np.random.choice([0, 1], size=(nn, nn), 
                p=[1 - sparsity, sparsity])
        self.Wres = np.random.normal(size=(nn, nn)) * edge_matrix

        # check spectral radius and rescale
        w, v = np.linalg.eig(self.Wres)
        radius = np.abs(np.max(w))
        if radius > 1:
            print("Rescaling weight matrix to reduce spectral radius.")
            self.Wres = self.Wres / (1.1 * radius)
        # verify
        w, v = np.linalg.eig(self.Wres)
        radius = np.abs(np.max(w))
        if radius > 1:
            warnings.warn("Spectral radius still greater than 1.")

        # set output weights (no X (nu + nn + 1))
        self.Wout = np.random.normal(size=(no, nu + nn + 1))

        # initialize reservoir activations
        self.x = np.zeros((nn, 1))

        # initialize output
        self.y = np.zeros((no, 1))
        

    def update(self, u):
        if not u.shape == (self.nu, 1):
            raise ValueError(
                    "Expected input dims: {}, Received: {}".format(
                (self.nu, 1), u.shape))
        # insert value for bias
        u_bias = np.vstack((np.ones((1,1)), u))

        # updates without leaky integration in this version
        self.x = np.tanh(np.dot(self.Win, u_bias) + np.dot(self.Wres, self.x))
        self.y = np.dot(self.Wout, np.vstack((u_bias, self.x)))


    def zero_in_out(self):
        # initialize reservoir activations
        self.x = np.zeros((self.nn, 1))
        # initialize output
        self.y = np.zeros((self.no, 1))


    def predict(self, U):
        """ Inputs:
                U: an ss X nu array where ss is the sample size 
            Ouptput:
                out: an ss X no.
        """
        self.zero_in_out()
        steps = U.shape[0]
        out = []
        for ii in range(steps):
            tmp = U[ii,:].reshape(-1,1)
            self.update(tmp)
            out.append(self.y.T)
        return(np.concatenate(out, axis=0))


    def visual_train(self, U, y, gamma=0.5):
        """ Trains with ridge regression (see Lukusvicius, jaeger, and
        Schrauwen). Returns internal states for animation.
        Inputs:
            U: an ss X nu array where ss is the sample size 
            y: an ss X no array of target outputs.
        """
        self.zero_in_out()
        # Build concatenated matrices
        X = []
        Y = y.T
        steps = U.shape[0]
        # figure out how many zeros are needed to pad self.x to make a square
        tmp = np.max(self.x.shape)
        ns = int(np.ceil(np.sqrt(tmp)))
        pad_length = ns ** 2 - tmp
        # prep for saving plots
        ims = []
        x_pad = np.concatenate([self.x.reshape(1,-1),
            np.zeros((1,pad_length))], axis=1)
        im = plt.imshow(x_pad.reshape(ns,ns), animated=True)
        ax = plt.gca()
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ims.append([im])
        
        for ii in range(steps):
            tmp = U[ii,:].reshape(-1,1)
            self.update(tmp)
            x_pad = np.concatenate([self.x.reshape(1,-1),
                np.zeros((1,pad_length))], axis=1)
            im = plt.imshow((x_pad.reshape(ns,ns) + 1.) / 2., animated=True, cmap='plasma')
            ims.append([im])
            X.append(np.vstack((np.ones((1,1)), tmp, self.x)))
        X = np.concatenate(X, axis=1)
        I = np.identity(X.shape[0])
        self.Wout = np.dot(np.dot(Y, X.T), np.linalg.inv(np.dot(X, X.T) +
            gamma**2 * I))

        return ims


    def train(self, U, y, gamma=0.5):
        """ Trains with ridge regression (see Lukusvicius, jaeger, and
        Schrauwen). 
        Inputs:
            U: an ss X nu array where ss is the sample size 
            y: an ss X no array of target outputs.
        """
        self.zero_in_out()
        # Build concatenated matrices
        X = []
        Y = y.T
        steps = U.shape[0]
        for ii in range(steps):
            tmp = U[ii,:].reshape(-1,1)
            self.update(tmp)
            X.append(np.vstack((np.ones((1,1)), tmp, self.x)))
        X = np.concatenate(X, axis=1)
        I = np.identity(X.shape[0])
        self.Wout = np.dot(np.dot(Y, X.T), np.linalg.inv(np.dot(X, X.T) +
            gamma**2 * I))
 
