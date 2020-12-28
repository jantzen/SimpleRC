# file: scalefreeRC.py

from simpleRC import simpleRC
import numpy as np
import networkx as nx

""" Implemens a simple RC using a reservoir with scale-free connectivity.
"""

class scalefreeRC( simpleRC ):

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
        tmp = nx.adjacency_matrix(nx.scale_free_graph(nn),
                weight=None).todense()
        edge_matrix = np.where(tmp > 0, np.ones(tmp.shape), np.zeros(tmp.shape))
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
 

