# file: weather.py

import pandas as pd
import numpy as np
from simpleRC import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import copy
import pdb

def open_data(filename1, filename2):
    # import csv
    df1 = pd.read_csv(filename1)
    df2 = pd.read_csv(filename2)

    # keep only the FM-15 reports
    df1 = df1.loc[df1['REPORT_TYPE'] == 'FM-15']
    df2 = df2.loc[df2['REPORT_TYPE'] == 'FM-15']
    
    VA = df1.loc[df1['STATION'] == 72411353881]
    VA = VA.iloc[2:,:]
    AZ = df1.loc[df1['STATION'] == 72278023183]
    KY = df2

    # assemble Virginia data
    # choose desired columns and convert to numeric
    T_VA = pd.to_numeric(VA['HourlyWetBulbTemperature'], errors='coerce')
    Precip_VA = pd.to_numeric(VA['HourlyPrecipitation'], errors='coerce')
    RH_VA = pd.to_numeric(VA['HourlyRelativeHumidity'], errors='coerce')
    P_VA = pd.to_numeric(VA['HourlyStationPressure'], errors='coerce')
    WS_VA = pd.to_numeric(VA['HourlyWindSpeed'], errors='coerce')

    # replace NaN values with appropriate substitutes:
    # 0 for precipitation
    # mean of nearest numeric values for everything else
    T_VA = T_VA.fillna(method='ffill').fillna(method='bfill')
    Precip_VA = Precip_VA.fillna(0.)
    RH_VA = RH_VA.fillna(method='ffill').fillna(method='bfill')
    P_VA = P_VA.fillna(method='ffill').fillna(method='bfill')
    WS_VA = WS_VA.fillna(method='ffill').fillna(method='bfill')
    variables_VA = [T_VA, Precip_VA, RH_VA, P_VA, WS_VA]

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
    WS_AZ = pd.to_numeric(AZ['HourlyWindSpeed'], errors='coerce')

    # replace NaN values with appropriate substitutes:
    # 0 for precipitation
    # mean of nearest numeric values for everything else
    T_AZ = T_AZ.fillna(method='ffill').fillna(method='bfill')
    Precip_AZ = Precip_AZ.fillna(0.)
    RH_AZ = RH_AZ.fillna(method='ffill').fillna(method='bfill')
    P_AZ = P_AZ.fillna(method='ffill').fillna(method='bfill')
    WS_AZ = WS_AZ.fillna(method='ffill').fillna(method='bfill')
    variables_AZ = [T_AZ, Precip_AZ, RH_AZ, P_AZ, WS_AZ]

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


    # assemble Kentucky data
    # choose desired columns and convert to numeric
    T_KY = pd.to_numeric(KY['HourlyWetBulbTemperature'], errors='coerce')
    Precip_KY = pd.to_numeric(KY['HourlyPrecipitation'], errors='coerce')
    RH_KY = pd.to_numeric(KY['HourlyRelativeHumidity'], errors='coerce')
    P_KY = pd.to_numeric(KY['HourlyStationPressure'], errors='coerce')
    WS_KY = pd.to_numeric(KY['HourlyWindSpeed'], errors='coerce')

    # replace NaN values with appropriate substitutes:
    # 0 for precipitation
    # mean of nearest numeric values for everything else
    T_KY = T_KY.fillna(method='ffill').fillna(method='bfill')
    Precip_KY = Precip_KY.fillna(0.)
    RH_KY = RH_KY.fillna(method='ffill').fillna(method='bfill')
    P_KY = P_KY.fillna(method='ffill').fillna(method='bfill')
    WS_KY = WS_KY.fillna(method='ffill').fillna(method='bfill')
    variables_KY = [T_KY, Precip_KY, RH_KY, P_KY, WS_KY]

    # verify everything is now numeric
    for var in variables_KY:
        if not np.all(np.isfinite(var)):
            raise ValueError(
            "There are unhandled non-numeric values in the data."
            )

    # stack into numpy data
    tmp = []
    for var in variables_KY:
        tmp.append(var.to_numpy().reshape(-1,1))
    data_KY = np.concatenate(tmp, axis=1)

    # combine KY and VA data
    tmp = data_VA[::3]
    data_VA_KY = np.concatenate([tmp, data_KY[:tmp.shape[0],:]], axis=1)

    return data_VA, data_AZ, data_KY, data_VA_KY


def prep_training_and_test(data, station, num_samples):
    """ Train the RC to use num_samples of data to predict the hourly
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
    elif station == "va_ky":
        k = int(24/24 * 72)
    steps = data.shape[0] - num_samples - k 
    for ii in range(steps):
#        tmp_U = data[ii * 200 : (ii + 1) * 200, :]
#        tmp_y = data[(ii + 1) * 200 + k, 0].reshape(1,-1)
        tmp_U = data[ii : ii + num_samples, :]
        tmp_y = data[ii + num_samples + k, 0].reshape(1,-1)
        U.append(tmp_U.flatten().reshape(1, -1))
        y.append(tmp_y.reshape(1, -1))
    U = np.concatenate(U, axis=0)
    y = np.concatenate(y, axis=0)
    print(U.shape, y.shape)

    # split into training (70%) and testing (30%)
    split = int(0.7 * U.shape[0])
    U_train = U[:split, :]
    y_train = y[:split, :]
    U_test = U[split:, :]
    y_test = y[split:, :]

    return U_train, y_train, U_test, y_test


def benchmark_predictions(U, station, num_samples):
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
        tmp = U[ii,:].reshape(num_samples, 5)
        preds.append(np.mean(tmp[:k,0]).reshape(1,-1))
    preds = np.concatenate(preds, axis=0)
        
    return preds

def scale(X):
    # scales the input array X (assumed to have samples in rows) to have 0
    # mean and stdev=1
    mu = np.mean(X, axis=0)
#    stdev = np.std((X - mu), axis=0)
    stdev = np.abs(np.max(X - mu) - np.min(X - mu))
    X_scaled = (X - mu) / stdev
    return X_scaled, mu, stdev

def unscale(X, mu, stdev):
    # applies inverse transformation of scale
    return X * stdev + mu


def main(filename1='./data/2166184.csv', filename2='./data/2173692.csv'):

    num_samples = 100
#    nn = 500
#    sparsity = 0.01
    nn = 49
    sparsity = 0.9
    gamma = 0.01

    # open file for saving output
    f = open('weather_output', 'w')
    print("Opening data files...")
    data_VA, data_AZ, data_KY, data_VA_KY = open_data(filename1, filename2)
    print("Normalizing data...")
    data_VA, mu_VA, stdev_VA = scale(data_VA)
    data_AZ, mu_AZ, stdev_AZ = scale(data_AZ)
    data_KY, mu_KY, stdev_KY = scale(data_KY)
    data_VA_KY, mu_VA_KY, stdev_VA_KY = scale(data_VA_KY)
    print("Building RC...")
    rc = simpleRC(num_samples * 5, nn, 1, sparsity=sparsity)
    print("Revervoir size: {}".format(rc.Wres.shape))
    f.write(("Revervoir size: {}\n".format(rc.Wres.shape)))

    print("Constructing training and testing datasets for VA...")
    f.write("Constructing training and testing datasets for VA...\n")
    U_train, y_train, U_test, y_test = prep_training_and_test(data_VA, 'va',
            num_samples)
    print(U_train.shape, y_train.shape, U_test.shape, y_test.shape)

    # test untrained accuracy
    preds = rc.predict(U_train)
    error = np.sqrt(np.mean((y_train - preds)**2))
    print("Sanity check: untrained prediction accuracy = {}".format(error))
    f.write("Sanity check: untrained prediction accuracy = {}\n".format(error))

    print("Training the RC for VA...")
    f.write("Training the RC for VA...\n")
#    rc.visual_train(U_train, y_train, gamma=gamma, filename="weather_viz.mp4")
    rc.train(U_train, y_train, gamma=gamma)
    print("Testing the trained RC for VA...")
    f.write("Testing the trained RC for VA...\n")
    preds = rc.predict(U_train)
    error = np.sqrt(np.mean((y_train - preds)**2))
    print("Error on training set: {}".format(error))
    f.write("Error on training set: {}\n".format(error))
    preds = rc.predict(U_test)
    error = np.sqrt(np.mean((y_test - preds)**2))
    print("Error on test set: {}".format(error))
    f.write("Error on test set: {}\n".format(error))
    t = np.arange(y_test.shape[0])
    print("Saving the linear output layer for VA...")
    np.savetxt('Wout_VA', rc.Wout)
    print("Saving the reservoir weights for VA...")
    np.savetxt('W_VA', rc.Wres)
    print("Getting benchmark predictions for VA...")
    f.write("Getting benchmark predictions for VA...\n")
    preds = benchmark_predictions(U_test, 'va', num_samples)
    error = np.sqrt(np.mean((y_test - preds)**2))
    print("Benchmark RMS for VA: {}".format(error))
    f.write("Benchmark RMS for VA: {}\n".format(error))
    plt.plot(t, y_test, 'bo', t, preds, 'ro')
    plt.title("Blacksburg")
    np.savetxt('va_y_test', y_test)
    np.savetxt('va_preds', preds)

    print("Training a new RC for VA using KY data supplement...")
    f.write("Training a new RC for VA using KY data supplement...\n")
    U_train, y_train, U_test, y_test = prep_training_and_test(data_VA_KY,
            'va_ky', num_samples)
    rc_supp = simpleRC(num_samples * 5 * 2, nn, 1, sparsity=sparsity)
    rc_supp.train(U_train, y_train, gamma=gamma)
    print("Testing the trained RC for VA...")
    f.write("Testing the trained RC for VA...\n")
    preds = rc_supp.predict(U_train)
    error = np.sqrt(np.mean((y_train - preds)**2))
    print("Error on training set: {}".format(error))
    f.write("Error on training set: {}\n".format(error))
    preds = rc_supp.predict(U_test)
    error = np.sqrt(np.mean((y_test - preds)**2))
    print("Error on test set: {}".format(error))
    f.write("Error on test set: {}\n".format(error))
    t = np.arange(y_test.shape[0])
    plt.figure()
    plt.plot(t, y_test, 'bo', t, preds, 'ro')
    plt.title("Blacksburg (supplemented with Lexington data)")

    print("Copying and initializing original RC for use with AZ data...")
    f.write("Copying and initializing original RC for use with AZ data...\n")
    rc2 = simpleRC(num_samples * 5, nn, 1, sparsity=sparsity)
    rc2.Win = copy.deepcopy(rc.Win)
    rc2.Wres = copy.deepcopy(rc.Wres)
    print("Constructing training and testing datasets for AZ...")
    f.write("Constructing training and testing datasets for AZ...\n")
    U_train, y_train, U_test, y_test = prep_training_and_test(data_AZ, 'az',
            num_samples)
    print(U_train.shape, y_train.shape, U_test.shape, y_test.shape)
    print("Training the RC for AZ...")
    f.write("Training the RC for AZ...\n")
    rc2.train(U_train, y_train, gamma=gamma)
    print("Testing the trained RC for AZ...")
    f.write("Testing the trained RC for AZ...\n")
    preds = rc2.predict(U_train)
    error = np.sqrt(np.mean((y_train - preds)**2))
    print("Error on training set: {}".format(error))
    f.write("Error on training set: {}\n".format(error))
    preds = rc2.predict(U_test)
    error = np.sqrt(np.mean((y_test - preds)**2))
    print("Error on test set: {}".format(error))
    f.write("Error on test set: {}\n".format(error))
    t = np.arange(y_test.shape[0])
    print("Getting benchmark predictions for AZ...")
    f.write("Getting benchmark predictions for AZ...\n")
    preds = benchmark_predictions(U_test, 'az', num_samples)
    error = np.sqrt(np.mean((y_test - preds)**2))
    print("Benchmark RMS for AZ: {}".format(error))
    f.write("Benchmark RMS for AZ: {}\n".format(error))
    np.savetxt('az_y_test', y_test)
    np.savetxt('az_preds', preds)
    plt.figure()
    plt.plot(t, y_test, 'bo', t, preds, 'ro')
    plt.title("Phoenix")
    plt.show()
    f.close()

if __name__ == '__main__':
    main()
