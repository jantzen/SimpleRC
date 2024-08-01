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


    def zero_weights(self):
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

#    def run_os(self, U_init, steps):
#        """ Runs the RC on its own predictions for steps.
#
#            Inputs:
#                U: a 1 x nu array
#            Output:
#                out: an steps x nu array
#        """
#        # verify that this RC produces output of the same dimension as input
#        self.zero_in_out() 
#        self.update(U_init)
#        tmp = self.y
#        if not tmp.shape == U_init.shape:
#            raise ValueError("The output dimensions do not match input.")
#        out = [tmp]
#        for ii in range(steps):
#            self.zero_in_out() # reset prior to each input
#            tmp = out[ii]
#            self.update(tmp)
#            out.append(self.y)
#        if self.gpu:
#            tmp = torch.cat(out, axis=1).T[:-1,:]
#            return(tmp.cpu().numpy())
#        else:
#            return(np.concatenate(out, axis=1).T[:-1,:])
#
#
#    def run_rf(self, U_init, steps):
#        """ Runs the RC on its own predictions for steps.
#
#            Inputs:
#                U: a 1 x nu array
#            Output:
#                out: an steps x nu array
#        """
#        # verify that this RC produces output of the same dimension as input
#        self.zero_in_out() # reset only once
#        self.update(U_init)
#        tmp = self.y
#        if not tmp.shape == U_init.shape:
#            raise ValueError("The output dimensions do not match input.")
#        out = [tmp]
#        for ii in range(steps):
#            tmp = out[ii]
#            self.update(tmp)
#            out.append(self.y)
#        if self.gpu:
#            tmp = torch.cat(out, axis=1).T[:-1,:]
#            return(tmp.cpu().numpy())
#        else:
#            return(np.concatenate(out, axis=1).T[:-1,:])
#
#
#    def run(self, U_init, steps):
#        """ Runs the RC on its own predictions for steps.
#
#            Inputs:
#                U: a 1 x nu array
#            Output:
#                out: an steps x nu array
#        """
#        if self.mode == 'onestep':
#            return self.run_os(U_init, steps)
##        if self.mode == 'recurrent':
##            return self.run_r(U_init, steps)
#        if self.mode == 'recurrent_forced':
#            return self.run_rf(U_init, steps)
# 
#
#    def visual_train(self, U, y, gamma=0.5, filename=None):
#        """ Trains with ridge regression (see Lukusvicius, jaeger, and
#        Schrauwen). Returns internal states for animation.
#        Inputs:
#            ax: axes object for drawing
#            U: an ss X nu array where ss is the sample size 
#            y: an ss X no array of target outputs.
#        """
#        # Build concatenated matrices
#        X = []
#        Y = y.T
#        steps = U.shape[0]
#        # figure out how many zeros are needed to pad self.x to make a square
#        tmp = np.max(self.x.shape)
#        ns = int(np.ceil(np.sqrt(tmp)))
#        pad_length = ns ** 2 - tmp
#        # prep for saving plots
#        fg = plt.figure()
#        ax = plt.axes()
#        ims = []
#        x_pad = np.concatenate([self.x.reshape(1,-1),
#            np.zeros((1,pad_length))], axis=1)
#        im = ax.imshow(x_pad.reshape(ns,ns), animated=True, cmap='plasma',
#                vmin=-1., vmax=1., interpolation='None')
##        ax = plt.gca()
#        ax.xaxis.set_visible(False)
#        ax.yaxis.set_visible(False)
#        cbar = ax.figure.colorbar(im, ax=ax)
#        cbar.ax.set_ylabel("Activation", rotation=-90, va="bottom")
#        ims.append([im])
#        
#        for ii in range(steps):
#            self.zero_in_out()
#            tmp = U[ii,:].reshape(-1,1)
#            self.update(tmp)
#            x_pad = np.concatenate([self.x.reshape(1,-1),
#                np.zeros((1,pad_length))], axis=1)
#            im = ax.imshow((x_pad.reshape(ns,ns) + 1.) / 2., animated=True,
#                    cmap='plasma', vmin=-1., vmax=1., interpolation='None')
#            ims.append([im])
#            X.append(np.vstack((np.ones((1,1)), tmp, self.x)))
#        X = np.concatenate(X, axis=1)
#        I = np.identity(X.shape[0])
#        self.Wout = np.dot(np.dot(Y, X.T), np.linalg.inv(np.dot(X, X.T) +
#            gamma**2 * I))
#
#        print("Building animation...")
#        ani = animation.ArtistAnimation(fg, ims, interval=33, repeat_delay=500, blit=True)
#        if filename is not None:
#            print("Saving animation...")
#            ani.save(filename)
#            return ani
#        else:
#            return ani
#
#
#    def train_os(self, U, y, gamma):
#        """ Trains with ridge regression (see Lukusvicius, jaeger, and
#        Schrauwen). 
#        Inputs:
#            U: an ss X nu array where ss is the sample size 
#            y: an ss X no array of target outputs.
#        """
#        # Build concatenated matrices
#        X = []
#        Y = y.T
#        steps = U.shape[0]
#        for ii in range(steps):
#            self.zero_in_out()
#            tmp = U[ii,:].reshape(-1,1)
#            self.update(tmp)
#            if self.gpu:
#                X.append(np.vstack((np.ones((1,1)), tmp, self.x.cpu().numpy())))
#            else:
#                X.append(np.vstack((np.ones((1,1)), tmp, self.x)))
#        X = np.concatenate(X, axis=1)
#        I = np.identity(X.shape[0])
#        if self.gpu:
#            X = torch.from_numpy(X).to(self.device)
#            I = torch.from_numpy(I).to(self.device)
#            Y = torch.from_numpy(Y).to(self.device)
#            self.Wout = torch.matmul(torch.matmul(Y, X.T),
#                    torch.linalg.inv(torch.matmul(X, X.T) +
#                gamma**2 * I))
#        else:
#            self.Wout = np.dot(np.dot(Y, X.T), np.linalg.inv(np.dot(X, X.T) +
#                gamma**2 * I))
#
#
#    def train_rf(self, U, y, gamma):
#        """ Trains with ridge regression (see Lukusvicius, jaeger, and
#        Schrauwen). 
#        Inputs:
#            U: an ss X nu array where ss is the sample size 
#            y: an ss X no array of target outputs.
#        """
#        # Build concatenated matrices
#        X = []
#        Y = y.T
#        steps = U.shape[0]
#        self.zero_in_out() # reset only once
#        for ii in range(steps):
#            tmp = U[ii,:].reshape(-1,1)
#            self.update(tmp)
#            if self.gpu:
#                X.append(np.vstack((np.ones((1,1)), tmp, self.x.cpu().numpy())))
#            else:
#                X.append(np.vstack((np.ones((1,1)), tmp, self.x)))
#        X = np.concatenate(X, axis=1)
#        I = np.identity(X.shape[0])
#        if self.gpu:
#            X = torch.from_numpy(X).to(self.device)
#            I = torch.from_numpy(I).to(self.device)
#            Y = torch.from_numpy(Y).to(self.device)
#            self.Wout = torch.matmul(torch.matmul(Y, X.T),
#                    torch.linalg.inv(torch.matmul(X, X.T) +
#                gamma**2 * I))
#        else:
#            self.Wout = np.dot(np.dot(Y, X.T), np.linalg.inv(np.dot(X, X.T) +
#                gamma**2 * I))
#
#
#    def train(self, U, y, gamma=0.5):
#        """ Trains with ridge regression (see Lukusvicius, jaeger, and
#        Schrauwen). 
#        Inputs:
#            U: an ss X nu array where ss is the sample size 
#            y: an ss X no array of target outputs.
#        """
# 
#        if self.mode == 'onestep':
#            return self.train_os(U, y, gamma=gamma)
##        if self.mode == 'recurrent':
##            return self.train_r(U, y, gamma=gamma)
#        if self.mode == 'recurrent_forced':
#            return self.train_rf(U, y, gamma=gamma)

    def train(self, U, y, gamma=0.5, settling_steps=None, zero_weights=False):
        """ Trains with ridge regression (see Lukusvicius, jaeger, and
        Schrauwen). 
        Inputs:
            U: an ss X nu array where ss is the sample size 
            y: an ss X no array of target outputs.
            gamma: learning bias
            settling_steps: number of steps to ignore (to allow
                network to stabilize)
            zero_weights: indicates whether or not to zero the weights of the
                current RC before training. This is important for reproducibility
        """
        # Build concatenated matrices
        steps = U.shape[0]
        if settling_steps is None:
            settling_steps = int(0.1 * steps)
        if zero_weights:
            self.zero_weights() 
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

