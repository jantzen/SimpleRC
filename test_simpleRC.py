# file: test_simpleRC.py

import unittest
from simpleRC import *
import numpy as np
import pdb


class TestSimpleRC(unittest.TestCase):
    def setUp(self):
        self.rc = simpleRC(5, 20, 4, sparsity=1.0)

    def test_init(self):
        self.assertEqual(self.rc.Win.shape, (20, 6))
        self.assertEqual(self.rc.Wres.shape, (20, 20))
        self.assertEqual(self.rc.Wout.shape, (4, 26))
        self.assertTrue(np.all(self.rc.x==0.))
        self.assertTrue(np.all(self.rc.y==0))

    def test_update(self):
        pass

#    def test_train(self):
#        pass
#        # prepare some training data
#        t = np.linspace(0., 8 * np.pi, 100)
#        v = np.sin(t)
#        u = v[:-4].reshape(-1,1)
#        y = []
#        for ii in range(4, u.shape[0]):
#            tmp = v[ii:ii+4].reshape(1,-1)
#            y.append(tmp)
#        y = np.concatenate(y, axis=0)
#        self.rc.train(u, y)


    def test_predict(self):
        # prepare some training data
        t = np.linspace(0., 8 * np.pi, 1000)
        v = np.sin(t)
#        u = v[:-4].reshape(-1,1)
        U = []
        y = []
        for ii in range(v.shape[0] - 9):
            tmp_U = v[ii:ii+5].reshape(1, -1)
            tmp_y = v[ii+5:ii+9].reshape(1,-1)
            U.append(tmp_U)
            y.append(tmp_y)
        U = np.concatenate(U, axis=0)
        y = np.concatenate(y, axis=0)
        # train
        self.rc.train(U, y, gamma=0.1)
        # predict
        preds = self.rc.predict(U)
        # compute mean error
        error = np.sqrt(np.mean(np.sum((y - preds)**2, axis=1)))
        print(error)
        t2 = np.linspace(np.pi / 4., 0.8 * np.pi + np.pi / 4., 100)
        v2 = np.sin(t2)
        U2 = []
        y2 = []
        for ii in range(v2.shape[0] - 9):
            tmp_U = v2[ii:ii+5].reshape(1, -1)
            tmp_y = v2[ii+5:ii+9].reshape(1,-1)
            U2.append(tmp_U)
            y2.append(tmp_y)
        U2 = np.concatenate(U2, axis=0)
        y2 = np.concatenate(y2, axis=0)
        # predict
        preds = self.rc.predict(U2)
        # compute mean error
        error = np.sqrt(np.mean(np.sum((y2 - preds)**2, axis=1)))
        print(error)


if __name__ == '__main__':
    unittest.main()
