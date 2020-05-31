# file: weather.py

import pandas as pd
import numpy as np
from simpleRC import *
import matplotlib.pyplot as plt
import pdb

def open_data(filename):
    # import csv
    df = pd.read_csv(filename)

    # choose desired columns and convert to numeric
    T = pd.to_numeric(df['HourlyWetBulbTemperature'], errors='coerce')
    Precip = pd.to_numeric(df['HourlyPrecipitation'], errors='coerce')
    RH = pd.to_numeric(df['HourlyRelativeHumidity'], errors='coerce')
    P = pd.to_numeric(df['HourlyStationPressure'], errors='coerce')

    # replace NaN values with appropriate substitutes:
    # 0 for precipitation
    # mean of nearest numeric values for everything else
    T = T.fillna(method='ffill').fillna(method='bfill')
    Precip = Precip.fillna(0.)
    RH = RH.fillna(method='ffill').fillna(method='bfill')
    P = P.fillna(method='ffill').fillna(method='bfill')
    variables = [T, Precip, RH, P]

    # verify everything is now numeric
    for var in variables:
        if not np.all(np.isfinite(var)):
            raise ValueError(
            "There are unhandled non-numeric values in the data."
            )

    # stack into numpy data
    tmp = []
    for var in variables:
        tmp.append(var.to_numpy().reshape(-1,1))
    data = np.concatenate(tmp, axis=1)

    return data


def prep_training_and_test(data):
    """ Train the RC to use 120-hours worth of data to predict the hourly
    precipitation 72 hours in advance.
    """
    U = []
    y = []
    steps = int(data.shape[0] / 120) - 1
    for ii in range(steps):
        tmp_U = data[ii * 120 : (ii + 1) * 120, :]
        tmp_y = data[(ii + 1) * 120 + 72, 1].reshape(1,-1)
        U.append(tmp_U.flatten().reshape(1, -1))
        y.append(tmp_y.reshape(1, -1))
    U = np.concatenate(U, axis=0)
    y = np.concatenate(y, axis=0)
    print(U.shape, y.shape)

    # split into training (80%) and testing (20%)
    split = int(0.8 * U.shape[0])
    U_train = U[:split, :]
    y_train = y[:split, :]
    U_test = U[split:, :]
    y_test = y[split:, :]

    return U_train, y_train, U_test, y_test


def main(filename='./data/2166184.csv'):
    print("Opening data file...")
    data = open_data(filename)
    print("Building RC...")
    rc = simpleRC(480, 1000, 1)
    print("Constructing training and testing datasets...")
    U_train, y_train, U_test, y_test = prep_training_and_test(data)
    print(U_train.shape, y_train.shape, U_test.shape, y_test.shape)
    print("Training the RC...")
    rc.train(U_train, y_train)
    print("Testing the trained RC...")
    preds = rc.predict(U_test)
    error = np.sqrt(np.mean((y_test - preds)**2))
    print("Error on test set: {}".format(error))
    t = np.arange(y_test.shape[0])
    plt.plot(t, y_test, 'bo', t, preds, 'ro')
    plt.show()

if __name__ == '__main__':
    main()
