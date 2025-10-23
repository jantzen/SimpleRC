# file: simpleRC.py

import numpy as np
import warnings
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import imp
try:
    imp.find_module('torch')
    found = True
except ImportError:
    errmsg = "The PyTorch module (torch) was not found. Restricted to CPU methods."
    warnings.warn(errmsg)
    found = False
if found:
    import torch
    torch.set_default_dtype(torch.float64)
import pdb


class simpleRC( object ):

    def __init__(self,
            nu, # size of input layer
            nn, # size of the reservoir
            no,  # size of the output layer
            sparsity=0.1, # fraction of connections to make
            gpu=False, # indicates whether to use gpu for computation
            rescale_radius=True
            ):
        self.nu = nu
        self.nn = nn
        self.no = no
        self.gpu = gpu

        if self.gpu:
            if not torch.cuda.is_available():
                self.gpu = False
                errmsg = "No gpu available. Reverting to CPU method."
                warnings.warn(errmsg)
            else:
                print("Using gpu for computation.")
                self.device = torch.device("cuda")

        if self.gpu:
            # set input weights (nn X nu)
            self.Win = torch.normal(0., 1., (nn, nu + 1)).to(self.device)
            
            # set reservoir connections and weights (nn X nn)
            edge_matrix = np.random.choice([0, 1], size=(nn, nn), 
                    p=[1 - sparsity, sparsity])
            tmp = np.random.normal(size=(nn, nn)) * edge_matrix
            self.Wres = torch.from_numpy(tmp).to(self.device)

            if rescale_radius:
                # check spectral radius and rescale
                w, v = torch.linalg.eig(self.Wres)
                radius = torch.max(torch.abs(w))
                if radius > 1:
                    print("Rescaling weight matrix to reduce spectral radius.")
                    self.Wres = self.Wres / (1.1 * radius)
                # verify
                w, v = torch.linalg.eig(self.Wres)
                radius = torch.max(torch.abs(w))
                if radius > 1:
                    warnings.warn("Spectral radius still greater than 1.")

            # set output weights (no X (nu + nn + 1))
            self.Wout = torch.normal(0., 1., (no, nu + nn + 1)).to(self.device)

            # initialize reservoir activations
            self.x = torch.zeros((nn, 1)).to(self.device)

            # initialize output
            self.y = torch.zeros((no, 1)).to(self.device)
 
        else:
            # set input weights (nn X nu)
            self.Win = np.random.normal(size=(nn, nu + 1))
            
            # set reservoir connections and weights (nn X nn)
            edge_matrix = np.random.choice([0, 1], size=(nn, nn), 
                    p=[1 - sparsity, sparsity])
            self.Wres = np.random.normal(size=(nn, nn)) * edge_matrix

            # check spectral radius and rescale
            w, v = np.linalg.eig(self.Wres)
            radius = np.max(np.abs(w))
            if radius > 1:
                print("Rescaling weight matrix to reduce spectral radius.")
                self.Wres = self.Wres / (1.1 * radius)
            # verify
            w, v = np.linalg.eig(self.Wres)
            radius = np.max(np.abs(w))
            if radius > 1:
                warnings.warn("Spectral radius still greater than 1.")

            # set output weights (no X (nu + nn + 1))
            self.Wout = np.random.normal(size=(no, nu + nn + 1))

            # initialize reservoir activations
            self.x = np.zeros((nn, 1))

            # initialize output
            self.y = np.zeros((no, 1))

    def update(self, u):
        if self.gpu:
            if not u.shape == (self.nu, 1):
                raise ValueError(
                        "Expected input dims: {}, Received: {}".format(
                    (self.nu, 1), u.shape))
            if isinstance(u, np.ndarray):
                u = torch.from_numpy(u).to(self.device)
            # insert value for bias
            u_bias = torch.vstack((torch.ones((1,1)).to(self.device), u))

            # updates without leaky integration in this version
            self.x = torch.tanh(torch.matmul(self.Win, u_bias) + torch.matmul(self.Wres, self.x))
            self.y = torch.matmul(self.Wout, torch.vstack((u_bias, self.x)))
        else:
            if not u.shape == (self.nu, 1):
                raise ValueError(
                        "Expected input dims: {}, Received: {}".format(
                    (self.nu, 1), u.shape))
            # insert value for bias
            u_bias = np.vstack((np.ones((1,1)), u))

            # updates without leaky integration in this version
            self.x = np.tanh(np.dot(self.Win, u_bias) + np.dot(self.Wres, self.x))
            self.y = np.dot(self.Wout, np.vstack((u_bias, self.x)))


    def zero_activations(self):
        if self.gpu:
            # initialize reservoir activations
            self.x = torch.zeros((self.nn, 1)).to(self.device)
            # initialize output
            self.y = torch.zeros((self.no, 1)).to(self.device)

        else:
            # initialize reservoir activations
            self.x = np.zeros((self.nn, 1))
            # initialize output
            self.y = np.zeros((self.no, 1))


    def predict(self, U, settling_steps=None):
        """ Inputs:
                U: an ss X nu array where ss is the sample size 
                settling_steps: number of steps to ignore (to allow
                    network to stabilize)
            Ouptput:
                U_truncated: an (ss - settling_steps) X nu array
                preds: an (ss - settling_steps) X no.
        """
        steps = U.shape[0]
        out = []
        if settling_steps is None:
            settling_steps = int(0.1 * steps)
        elif settling_steps > steps:
            raise ValueError(
                    "Cannot set settling_steps to greater than the sample size."
                    )
        for ii in range(steps):
            tmp = U[ii,:].reshape(-1,1)
            self.update(tmp)
            if self.gpu:
                out.append(self.y.T.cpu().numpy())
            else:
                out.append(self.y.T)
        preds = np.concatenate(out[settling_steps:], axis=0)
        U_truncated = U[settling_steps:,:]

        return(U_truncated, preds)


    def project(self, U_init, steps):
        """ Runs the RC on its own predictions (from its current state) for
            a number of iterations equal to 'steps'.
        """
        # verify that this RC produces output of the same dimension as input
        self.update(U_init)
        tmp = self.y
        if not tmp.shape == U_init.shape:
            raise ValueError("The output dimensions do not match input.")
        out = [tmp]
        for ii in range(steps):
            tmp = out[ii]
            self.update(tmp)
            out.append(self.y)
        if self.gpu:
            tmp = torch.cat(out, axis=1).T[:-1,:]
            return(tmp.cpu().numpy())
        else:
            return(np.concatenate(out, axis=1).T[:-1,:])


    def train(self, U, y, gamma=0.5, settling_steps=None, zero_activations=True):
        """ Trains with ridge regression (see Lukusvicius, jaeger, and
        Schrauwen). 
        Inputs:
            U: an ss X nu array where ss is the sample size 
            y: an ss X no array of target outputs.
            gamma: learning bias
            settling_steps: number of steps to ignore (to allow
                network to stabilize)
            zero_activations: indicates whether or not to zero the activations of the
                current RC before training. This is important for reproducibility
        """
        # Build concatenated matrices
        steps = U.shape[0]
        if settling_steps is None:
            settling_steps = int(0.1 * steps)
        if zero_activations:
            self.zero_activations() 
        X = []
        Y = y[settling_steps:,:].T
        for ii in range(steps):
            tmp = U[ii,:].reshape(-1,1)
            self.update(tmp)
            if ii >= settling_steps:
                if self.gpu:
                    X.append(np.vstack((np.ones((1,1)), tmp, self.x.cpu().numpy())))
                else:
                    X.append(np.vstack((np.ones((1,1)), tmp, self.x)))
        X = np.concatenate(X, axis=1)
        I = np.identity(X.shape[0])
        if self.gpu:
            X = torch.from_numpy(X).to(self.device)
            I = torch.from_numpy(I).to(self.device)
            Y = torch.from_numpy(Y).to(self.device)
            self.Wout = torch.matmul(torch.matmul(Y, X.T),
                    torch.linalg.inv(torch.matmul(X, X.T) +
                gamma**2 * I))
        else:
            self.Wout = np.dot(np.dot(Y, X.T), np.linalg.inv(np.dot(X, X.T) +
                gamma**2 * I))

    def classifier_train(self, U, y, gamma=0.5, settling_steps=10):
        """ Trains with ridge regression (see Lukusvicius, jaeger, and
        Schrauwen) for classification problems. 
        Inputs:
            U: an ss X nu array where ss is the sample size 
            y: an ss X no array of target outputs.
            gamma: learning bias
            settling_steps: number of steps to allow 
                network to stabilize
        """
        steps = U.shape[0]
        X = []
        Y = y.T
        for ii in range(steps):
            # zero the activations
            self.zero_activations()
            # inject the input
            tmp = U[ii,:].reshape(-1,1)
            self.update(tmp)
            # let the reservoir evolve
            for jj in range(settling_steps):
                self.update(np.zeros((U.shape[1], 1)))
            # read and save the reservoir state
            if self.gpu:
                X.append(np.vstack((np.ones((1,1)), tmp, self.x.cpu().numpy())))
            else:
                X.append(np.vstack((np.ones((1,1)), tmp, self.x)))
        X = np.concatenate(X, axis=1)
        I = np.identity(X.shape[0])
        if self.gpu:
            X = torch.from_numpy(X).to(self.device)
            I = torch.from_numpy(I).to(self.device)
            Y = torch.from_numpy(Y).to(self.device)
            self.Wout = torch.matmul(torch.matmul(Y, X.T),
                    torch.linalg.inv(torch.matmul(X, X.T) +
                gamma**2 * I))
        else:
            self.Wout = np.dot(np.dot(Y, X.T), np.linalg.inv(np.dot(X, X.T) +
                gamma**2 * I))

    def classify(self, U, settling_steps=10, discretize=False):
        """ Inputs:
                U: an ss X nu array where ss is the sample size 
                settling_steps: number of steps to allow
                        network to stabilize
            discretize: switch to round outputs of network to represent
            discrete classes
            Output:
                U: U
                preds: an ss X no.
        """ 
        steps = U.shape[0]
        out = []
        for ii in range(steps):
            tmp = U[ii,:].reshape(-1,1)
            # zero the activations
            self.zero_activations()
            # inject the input
            tmp = U[ii,:].reshape(-1,1)
            self.update(tmp)
            # let the reservoir evolve
            for jj in range(settling_steps):
                self.update(np.zeros((U.shape[1], 1)))
            # read the output
            if self.gpu:
                out.append(self.y.T.cpu().numpy())
            else:
                out.append(self.y.T)
        if discretize:
            preds = np.round(np.concatenate(out, axis=0))
        else:
            preds = np.concatenate(out, axis=0)

        return(U, preds)


