# file: weather.py

import pandas as pd
import numpy as np
from simpleRC import *
import matplotlib.pyplot as plt
import copy
import pdb

def open_data(filename):
    # import csv
    df = pd.read_csv(filename)

    VA = df[:105621]
    AZ = df[105621:]

    # assemble Virginia data
    # choose desired columns and convert to numeric
    T_VA = pd.to_numeric(VA['HourlyWetBulbTemperature'], errors='coerce')
    Precip_VA = pd.to_numeric(VA['HourlyPrecipitation'], errors='coerce')
    RH_VA = pd.to_numeric(VA['HourlyRelativeHumidity'], errors='coerce')
    P_VA = pd.to_numeric(VA['HourlyStationPressure'], errors='coerce')

    # replace NaN values with appropriate substitutes:
    # 0 for precipitation
    # mean of nearest numeric values for everything else
    T_VA = T_VA.fillna(method='ffill').fillna(method='bfill')
    Precip_VA = Precip_VA.fillna(0.)
    RH_VA = RH_VA.fillna(method='ffill').fillna(method='bfill')
    P_VA = P_VA.fillna(method='ffill').fillna(method='bfill')
    variables_VA = [T_VA, Precip_VA, RH_VA, P_VA]

    # verify everything is now numeric
    for var in variables_VA:
        if not np.all(np.isfinite(var)):
            raise ValueError(
            "There are unhandled non-numeric values in the data."
            )

    # stack into numpy data
    tmp = []
    for var in variables_VA:
        tmp.append(var.to_numpy().reshape(-1,1))
    data_VA = np.concatenate(tmp, axis=1)

    # assemble Arizona data
    # choose desired columns and convert to numeric
    T_AZ = pd.to_numeric(AZ['HourlyWetBulbTemperature'], errors='coerce')
    Precip_AZ = pd.to_numeric(AZ['HourlyPrecipitation'], errors='coerce')
    RH_AZ = pd.to_numeric(AZ['HourlyRelativeHumidity'], errors='coerce')
    P_AZ = pd.to_numeric(AZ['HourlyStationPressure'], errors='coerce')

    # replace NaN values with appropriate substitutes:
    # 0 for precipitation
    # mean of nearest numeric values for everything else
    T_AZ = T_AZ.fillna(method='ffill').fillna(method='bfill')
    Precip_AZ = Precip_AZ.fillna(0.)
    RH_AZ = RH_AZ.fillna(method='ffill').fillna(method='bfill')
    P_AZ = P_AZ.fillna(method='ffill').fillna(method='bfill')
    variables_AZ = [T_AZ, Precip_AZ, RH_AZ, P_AZ]

    # verify everything is now numeric
    for var in variables_AZ:
        if not np.all(np.isfinite(var)):
            raise ValueError(
            "There are unhandled non-numeric values in the data."
            )

    # stack into numpy data
    tmp = []
    for var in variables_AZ:
        tmp.append(var.to_numpy().reshape(-1,1))
    data_AZ = np.concatenate(tmp, axis=1)

    return data_VA, data_AZ


def prep_training_and_test(data, station):
    """ Train the RC to use 200 samples of data to predict the hourly
    temperature 72 hours in advance. There are 72 samples a day for the
    Blacksburg, VA data and 30 samples a day for the Phoenix, AZ data.
    """
    print("Data shape: {}".format(data.shape))
    U = []
    y = []
    if station == "va":
        k = int(72 / 24 * 72)
    elif station == "az":
        k = int(30 / 24 * 72)
    steps = data.shape[0] - 200 - k 
    for ii in range(steps):
#        tmp_U = data[ii * 200 : (ii + 1) * 200, :]
#        tmp_y = data[(ii + 1) * 200 + k, 0].reshape(1,-1)
        tmp_U = data[ii : ii + 200, :]
        tmp_y = data[ii + 200 + k, 0].reshape(1,-1)
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


def benchmark_predictions(U, station):
    """ Estimates the temperature 72 hours in advance by averaging the preceding
    72 hours.
    """
    if station == "va":
        k = int(72 / 24 * 72)
    elif station == "az":
        k = int(30 / 24 * 72)
    steps = U.shape[0]
    preds = []
    for ii in range(steps):
        tmp = U[ii,:].reshape(200, 4)
        preds.append(np.mean(tmp[:,0]).reshape(1,-1))
    preds = np.concatenate(preds, axis=0)
        
    return preds
def main(filename='./data/2166184.csv'):
    print("Opening data files...")
    data_VA, data_AZ = open_data(filename)
    print("Building RC...")
    rc = simpleRC(800, 800, 1)
    print("Constructing training and testing datasets for VA...")
    U_train, y_train, U_test, y_test = prep_training_and_test(data_VA, 'va')
    print(U_train.shape, y_train.shape, U_test.shape, y_test.shape)
    print("Training the RC for VA...")
    rc.train(U_train, y_train, gamma=0.5)
    print("Testing the trained RC for VA...")
    preds = rc.predict(U_train)
    error = np.sqrt(np.mean((y_train - preds)**2))
    print("Error on training set: {}".format(error))
    preds = rc.predict(U_test)
    error = np.sqrt(np.mean((y_test - preds)**2))
    print("Error on test set: {}".format(error))
    t = np.arange(y_test.shape[0])
    print("Getting benchmark predictions for VA...")
    preds = benchmark_predictions(U_test, 'va')
    error = np.sqrt(np.mean((y_test - preds)**2))
    print("Benchmark RMS for VA: {}".format(error))
    plt.plot(t, y_test, 'bo', t, preds, 'ro')

    print("Copying and initializing RC for use with AZ data...")
    rc2 = simpleRC(800, 800, 1)
    rc2.Win = copy.deepcopy(rc.Win)
    rc2.Wres = copy.deepcopy(rc.Wres)
    print("Constructing training and testing datasets for AZ...")
    U_train, y_train, U_test, y_test = prep_training_and_test(data_AZ, 'az')
    print(U_train.shape, y_train.shape, U_test.shape, y_test.shape)
    print("Training the RC for AZ...")
    rc2.train(U_train, y_train, gamma=0.5)
    print("Testing the trained RC for AZ...")
    preds = rc2.predict(U_train)
    error = np.sqrt(np.mean((y_train - preds)**2))
    print("Error on training set: {}".format(error))
    preds = rc2.predict(U_test)
    error = np.sqrt(np.mean((y_test - preds)**2))
    print("Error on test set: {}".format(error))
    t = np.arange(y_test.shape[0])
    print("Getting benchmark predictions for AZ...")
    preds = benchmark_predictions(U_test, 'az')
    error = np.sqrt(np.mean((y_test - preds)**2))
    print("Benchmark RMS for AZ: {}".format(error))
    plt.figure()
    plt.plot(t, y_test, 'bo', t, preds, 'ro')
    plt.show()

if __name__ == '__main__':
    main()
