# file: metric.py
from simpleRC import *
import numpy as np
import pdb

class RC_metrics( object ):
    """ A helper class containing various methods
        for measuring the capacity and performance 
        of reservoir computers.
    """

    def __init__(self, rc, sample_size=1000):
        self._rc = rc
        self._sample_size = sample_size

    def cost(self, z_u, preds, times=None):
        num = np.sum((preds - z_u) ** 2)
        den = npm.sum((z_u - np.mean(z_u)) ** 2)
        return num / den


    def memory(self,
                  T=5, # maximum delay
                  settling_steps=None,
                  train_frac = .75
                  ):
        if settling_steps is None:
            settling_steps = 2 * T
        if settling_steps < T:
            raise ValueError("settling_steps must be greater than T")

        # generate data and the matched time-delayed sequences
        rng = np.random.default_rng(42)
        u = rng.random(self._sample_size + T)
        z = []
        split = round(train_frac * len(u))
        train_u = u[:split].reshape(-1,1)
        test_u = u[split:].reshape(-1,1)
        c = []

        for ii in range(T):
#            z.append(np.concatenate((np.array(ii * [u[0]]), u[:-(ii+1)])))
            # construct the target
            z = np.concatenate((np.array(ii * [u[0]]), u[:-(ii+1)]))
            # split the z-data into test and train
            train_z = z[:split].reshape(-1,1)
            test_z = z[split:].reshape(-1,1)

            # train the RC
            self._rc.train(train_u, train_z, settling_steps=settling_steps)

            # test the RC
            preds = self._rc.predict(test_u, settling_steps=settling_steps)
            num = np.sum((preds[1] - preds[0]) ** 2)
            den = np.sum((preds[0] - np.mean(preds[0])) ** 2)
            c.append(num / den)
            pdb.set_trace()

        return c
            

        return u, z
                

    def nonlinearity_Legendre(self,
        K # highest order of Legendre polynomial
                              ):
        pass
