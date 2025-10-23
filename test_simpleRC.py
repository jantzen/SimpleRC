# file: test_simpleRC.py

import unittest
from simpleRC import simpleRC 
import numpy as np
import torch
import pdb
from pathlib import Path

def load_xor_dataset(npz_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the XOR dataset saved by the generator script.

    Parameters
    ----------
    npz_path : str or pathlib.Path
        Path to the .npz file (e.g. "../data/xor_dataset.npz").

    Returns
    -------
    X : ndarray, shape (n_samples, 2)
        Feature matrix.
    y : ndarray, shape (n_samples,)
        Binary labels (0 or 1).
    """
    # `np.load` returns a NpzFile object that behaves like a dict.
    with np.load(npz_path) as data:
        # Keys are whatever the saving script used.
        # In the generator we saved `X=` and `y=` so the keys are "X" and "y".
        X = data["X"]   # shape (n_samples, 2)
        y = data["y"]   # shape (n_samples,)

    return X, y


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

    def test_train_classify(self):
        # create a new RC with 2D inputs and a single output for classification
        self.nu = 2
        self.nn = 200
        self.no = 1
        self.rc = simpleRC(self.nu, self.nn, self.no, sparsity=0.2)
        settling_steps = 5
 
        # import the XOR data
        X, y = load_xor_dataset('./data/xor_dataset.npz')
        self.rc.classifier_train(X, y, settling_steps=settling_steps)
        X, preds = self.rc.classify(X, settling_steps=settling_steps)
        errors = y - preds 
        mse = np.sqrt(np.mean(errors ** 2))
        print("Error of classifer on XOR: {}".format(mse))

    def test_classify(self):
        pass



class TestSimpleRC_gpu(unittest.TestCase):
    def setUp(self):
        # check for gpu; set flag
        if not torch.cuda.is_available():
            print("No CUDA‑compatible GPU detected (or drivers missing).")
            self._gpu = False
        else:
            self._gpu = True

        if self._gpu:
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
        if not self._gpu:
            return

        self.assertEqual(self.rc.Win.shape, (self.nn, self.nu + 1))
        self.assertEqual(self.rc.Wres.shape, (self.nn, self.nn))
        self.assertEqual(self.rc.Wout.shape, (self.no, self.nn + self.nu + 1))
        self.assertTrue(np.all(self.rc.x.cpu().numpy()==0.))
        self.assertTrue(np.all(self.rc.y.cpu().numpy()==0))

    def test_update(self):
        if not self._gpu:
            return

        for ii in range(self.U.shape[0]):
            self.rc.update(self.U[ii,:].reshape(-1,1))
        self.assertTrue(np.all(np.abs(self.rc.x.cpu().numpy()) > 0.))

    def test_predict(self):    
        if not self._gpu:
            return

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
        if not self._gpu:
            return

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
        if not self._gpu:
            return

        U_train = self.U[:900, :]
        y_train = self.y[:900, :]
        U_test = self.U[900:, :]
        y_test = self.y[900:, :]
        self.rc.train(U_train, y_train, settling_steps=10)
        pred = self.rc.project(U_test[0,:].reshape(-1,1), 100)
        error = np.sum((pred - y_test) ** 2)
        error_baseline = np.sum((0.5 - y_test) ** 2)
        self.assertTrue(error < error_baseline)


if __name__ == '__main__':
    unittest.main()
