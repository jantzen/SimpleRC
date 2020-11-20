# file: StochasticRC.py

from simpleRC import simpleRC
import numpy as np

class stochasticRC( simpleRC ):

    def __init__(self,
            nu, # size of input layer
            nn, # size of the reservoir
            no,  # size of the output layer
            sparsity=0.1, # fraction of connections to make
            ns=0 # number of stochastic nodes
            ):

        simpleRC.__init__(self, nu, nn, no, sparsity=sparsity)
        self.ns = ns


    def update(self, u):
        if not u.shape == (self.nu, 1):
            raise ValueError(
                    "Expected input dims: {}, Received: {}".format(
                (self.nu, 1), u.shape))
        # insert value for bias
        u_bias = np.vstack((np.ones((1,1)), u))

        # updates without leaky integration in this version
        noise = np.random.normal(size=(self.ns, 1))
        pad = np.zeros((self.nn - self.ns, 1))
        delta = np.concatenate([noise, pad], axis=0)
        self.x = np.tanh(np.dot(self.Win, u_bias) + np.dot(self.Wres, self.x +
            delta))
        self.y = np.dot(self.Wout, np.vstack((u_bias, self.x)))



