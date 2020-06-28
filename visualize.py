from __future__ import division
import numpy as np
from scipy.integrate import ode
from scipy import zeros_like
from scipy.stats import linregress
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import pdb

class visualRC( object ):

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

        # prep for plot
        ims = []
#        im = plt.imshow(self.x.reshape(8,8), animated=True)
#        ims.append([im])
        for ii in range(steps):
            tmp = U[ii,:].reshape(-1,1)
            self.update(tmp)
#            pdb.set_trace()
            im = plt.imshow(self.x.reshape(8,8), animated=True)
            ims.append([im])
            X.append(np.vstack((np.ones((1,1)), tmp, self.x)))
        X = np.concatenate(X, axis=1)
        I = np.identity(X.shape[0])
        self.Wout = np.dot(np.dot(Y, X.T), np.linalg.inv(np.dot(X, X.T) +
            gamma**2 * I))

        return ims
 
def main(noise=False, lag=10, fore=2.):

    fg = plt.figure()

    # set up Kuramoto systems

    N = 6
    K = 0.01

    omega= np.random.rand(N) * 2. * np.pi

    def model(t, theta, arg1):
        K = arg1[0]
        omega = arg1[1]
        dtheta = zeros_like(theta)
        sum1 = [0.] * N
        for ii in range(N):
            for jj in range(N):
                sum1[ii] += np.sin(theta[jj] - theta[ii])
                
            dtheta[ii] = omega[ii] + (K / N) * sum1[ii]
        
        return dtheta

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

    c = 0.2

    if noise:
        x += c * np.random.random_sample(x.shape)

    # prepare training and test data
    x = x.T

    # data for predicting the future
    nn = x.shape[0]
    cut = int(0.8 * nn)
    stop = int(fore/dt)
    train_u = x[:cut, :][:-stop,:]
    train_y = x[:cut, :][stop:,:]
    test_u = x[cut:,:][:-stop,:]
    test_y = x[cut:,:][stop:,:]

    # setup an RC
    print("Setting up RC...")
    nn = 64
    sparsity = 0.5
    gamma = 0.01

    rc_predict = visualRC(2*N, nn, 2*N, sparsity=sparsity)
    ims = rc_predict.train(train_u, train_y, gamma=gamma)
    preds = rc_predict.predict(train_u)
    error = np.sqrt(np.mean((train_y - preds)**2))
    print("Error on training set: {}".format(error))
    preds = rc_predict.predict(test_u)
    error = np.sqrt(np.mean((test_y - preds)**2))
    print("Error on test set: {}".format(error))
    print("Building animation...")
    ani = animation.ArtistAnimation(fg, ims, interval=33, repeat_delay=500, blit=True)

    plt.show()


if __name__ == "__main__":
    main(noise=False, fore=10.)
