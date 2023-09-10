# file: test_simpleRC.py

import unittest
from simpleRC import simpleRC 
from stochasticRC import stochasticRC 
import numpy as np
import torch
import pdb


class TestSimpleRC(unittest.TestCase):
    def setUp(self):
        # create the RC
        self.nu = 5
        self.nn = 100
        self.no = 5
        self.rc = simpleRC(5, 100, 5, sparsity=1.0)
        # generate some test data
        t = np.linspace(0., 8 * np.pi, 1001).reshape(-1,1)
        v = np.concatenate([np.sin(t), np.sin(2*t), np.sin(3*t), np.sin(4*t),
                            np.sin(5*t)], axis=1)
        self.U = v[:-1,:]
        self.y = v[1:,:]

    def test_init(self):
        self.assertEqual(self.rc.Win.shape, (self.nn, self.nu + 1))
        self.assertEqual(self.rc.Wres.shape, (self.nn, self.nn))
        self.assertEqual(self.rc.Wout.shape, (self.no, self.nu + self.nn + 1))
        self.assertTrue(np.all(self.rc.x==0.))
        self.assertTrue(np.all(self.rc.y==0))

    def test_update(self):
        for ii in range(self.U.shape[0]):
            self.rc.update(self.U[ii,:].reshape(-1,1))
        self.assertTrue(np.all(np.abs(self.rc.x) > 0.))
        
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
        # predict
        U_truncated, preds = self.rc.predict(self.U)
        self.assertTrue(np.all(U_truncated.shape == np.array([900, self.rc.nu])))
        self.assertTrue(np.all(preds.shape == np.array([900, self.rc.no])))
        U_truncated, preds = self.rc.predict(self.U, settling_steps=50)
        self.assertTrue(np.all(U_truncated.shape == np.array([950, self.rc.nu])))
        self.assertTrue(np.all(preds.shape == np.array([950, self.rc.no])))
        U_truncated, preds = self.rc.predict(self.U, settling_steps=0)
        self.assertTrue(np.all(U_truncated.shape == np.array([1000, self.rc.nu])))
        self.assertTrue(np.all(preds.shape == np.array([1000, self.rc.no])))
#        # compute mean error
#        error = np.sqrt(np.mean(np.sum((y - preds)**2, axis=1)))
#        print(error)
#        t2 = np.linspace(np.pi / 4., 0.8 * np.pi + np.pi / 4., 100)
#        v2 = np.sin(t2)
#        U2 = []
#        y2 = []
#        for ii in range(v2.shape[0] - 9):
#            tmp_U = v2[ii:ii+5].reshape(1, -1)
#            tmp_y = v2[ii+5:ii+9].reshape(1,-1)
#            U2.append(tmp_U)
#            y2.append(tmp_y)
#        U2 = np.concatenate(U2, axis=0)
#        y2 = np.concatenate(y2, axis=0)
#        # predict
#        preds = self.rc.predict(U2)
#        # compute mean error
#        error = np.sqrt(np.mean(np.sum((y2 - preds)**2, axis=1)))
#        print(error)

#    def test_lag_data(self):
#        from dataprepRC import lag_data
#        lag = 100
#        X = np.ones((5, 1000))
#        Xlag = lag_data(X, lag)
#        assert(Xlag.shape[1] == X.shape[1] - lag)
#        t = np.ones(1000)
#        Xlag, tlag = lag_data(X, lag, times=t)
#        assert(Xlag.shape[1] == X.shape[1] - lag)
#        assert(tlag.shape[0] == t.shape[0] - lag)

    def test_train(self):
        # compute the error of the untrained RC
        U_trunc, preds = self.rc.predict(self.U, settling_steps=100)
        error_untrained = np.sum((self.y[100:,:] - preds) ** 2)
        # train the RC
        self.rc.train(self.U, self.y)
        # compute error of trained RC
        U_trunc, preds = self.rc.predict(self.U, settling_steps=100)
        error_trained = np.sum((self.y[100:,:] - preds) ** 2)
        self.assertTrue(error_untrained > error_trained)

    def test_project(self):
        U_train = self.U[:900, :]
        y_train = self.y[:900, :]
        U_test = self.U[900:, :]
        y_test = self.y[900:, :]
        self.rc.train(U_train, y_train, settling_steps=10)
        pred = self.rc.project(U_test[0,:].reshape(-1,1), 100)
        error = np.sum((pred - y_test) ** 2)
        error_baseline = np.sum((0.5 - y_test) ** 2)
        self.assertTrue(error < error_baseline)


class TestSimpleRC_gpu(unittest.TestCase):
    def setUp(self):
        # generate some test data
        t = np.linspace(0., 8 * np.pi, 1001).reshape(-1,1)
        v = np.concatenate([np.sin(t), np.sin(2*t), np.sin(3*t), np.sin(4*t),
                            np.sin(5*t)], axis=1)
        self.U = v[:-1,:]
        self.y = v[1:,:]

        # make the RC
        self.nu = 5
        self.nn = 100
        self.no = 5
        self.rc = simpleRC(5, 100, 5, sparsity=1.0, gpu=True)

    def test_init(self):
        self.assertEqual(self.rc.Win.shape, (self.nn, self.nu + 1))
        self.assertEqual(self.rc.Wres.shape, (self.nn, self.nn))
        self.assertEqual(self.rc.Wout.shape, (self.no, self.nn + self.nu + 1))
        self.assertTrue(np.all(self.rc.x.cpu().numpy()==0.))
        self.assertTrue(np.all(self.rc.y.cpu().numpy()==0))

    def test_update(self):
        for ii in range(self.U.shape[0]):
            self.rc.update(self.U[ii,:].reshape(-1,1))
        self.assertTrue(np.all(np.abs(self.rc.x.cpu().numpy()) > 0.))

    def test_predict(self):    
        U_truncated, preds = self.rc.predict(self.U)
        self.assertTrue(np.all(U_truncated.shape == np.array([900, self.rc.nu])))
        self.assertTrue(np.all(preds.shape == np.array([900, self.rc.no])))
        U_truncated, preds = self.rc.predict(self.U, settling_steps=50)
        self.assertTrue(np.all(U_truncated.shape == np.array([950, self.rc.nu])))
        self.assertTrue(np.all(preds.shape == np.array([950, self.rc.no])))
        U_truncated, preds = self.rc.predict(self.U, settling_steps=0)
        self.assertTrue(np.all(U_truncated.shape == np.array([1000, self.rc.nu])))
        self.assertTrue(np.all(preds.shape == np.array([1000, self.rc.no])))

    def test_train(self):
        # compute the error of the untrained RC
        U_trunc, preds = self.rc.predict(self.U, settling_steps=100)
        error_untrained = np.sum((self.y[100:,:] - preds) ** 2)
        # train the RC
        self.rc.train(self.U, self.y)
        # compute error of trained RC
        U_trunc, preds = self.rc.predict(self.U, settling_steps=100)
        error_trained = np.sum((self.y[100:,:] - preds) ** 2)
        self.assertTrue(error_untrained > error_trained)

    def test_project(self):
        U_train = self.U[:900, :]
        y_train = self.y[:900, :]
        U_test = self.U[900:, :]
        y_test = self.y[900:, :]
        self.rc.train(U_train, y_train, settling_steps=10)
        pred = self.rc.project(U_test[0,:].reshape(-1,1), 100)
        error = np.sum((pred - y_test) ** 2)
        error_baseline = np.sum((0.5 - y_test) ** 2)
        self.assertTrue(error < error_baseline)

#
#    def test_predict(self):
#        # prepare some training data
#        t = np.linspace(0., 8 * np.pi, 1000)
#        v = np.sin(t)
#        U = []
#        y = []
#        for ii in range(v.shape[0] - 9):
#            tmp_U = v[ii:ii+5].reshape(1, -1)
#            tmp_y = v[ii+5:ii+9].reshape(1,-1)
#            U.append(tmp_U)
#            y.append(tmp_y)
#        U = np.concatenate(U, axis=0)
#        y = np.concatenate(y, axis=0)
#        # train
#        self.rc.train(U, y, gamma=0.1)
#        # predict
#        preds = self.rc.predict(U)
#        assert isinstance(preds, np.ndarray)
#        # compute mean error
#        error = np.sqrt(np.mean(np.sum((y - preds)**2, axis=1)))
#        print(error)
#        t2 = np.linspace(np.pi / 4., 0.8 * np.pi + np.pi / 4., 100)
#        v2 = np.sin(t2)
#        U2 = []
#        y2 = []
#        for ii in range(v2.shape[0] - 9):
#            tmp_U = v2[ii:ii+5].reshape(1, -1)
#            tmp_y = v2[ii+5:ii+9].reshape(1,-1)
#            U2.append(tmp_U)
#            y2.append(tmp_y)
#        U2 = np.concatenate(U2, axis=0)
#        y2 = np.concatenate(y2, axis=0)
#        # predict
#        preds = self.rc.predict(U2)
#        # compute mean error
#        error = np.sqrt(np.mean(np.sum((y2 - preds)**2, axis=1)))
#        print(error)


if __name__ == '__main__':
    unittest.main()
