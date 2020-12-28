# file: weather_graphs.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

steps = int(np.loadtxt('steps'))

# Virginia
t = np.loadtxt('va_t2')
y_units = np.loadtxt('va_test2')
preds_units = np.loadtxt('va_preds2')

fig, axes = plt.subplots(5, 1, sharex=True)
plt.suptitle("Weather in Blacksburg, Virginia")
labels = ['Temperature [deg F]', 'Precipitation [in]', 
        'Relative Humidity [%]', 'Pressure [in Hg]', 'Wind Speed [mph]']
for ii, ax in enumerate(axes):
    ax.plot(t[:steps], y_units[:steps,ii], 'bo', t[:steps], preds_units[:steps,ii], 'r-')
    ax.set_ylabel(labels[ii])
    ax.legend(('actual','predicted'))
axes[4].set_xlabel("Days")


# Utah
t = np.loadtxt('ut_t2')
y_units = np.loadtxt('ut_test2')
preds_units = np.loadtxt('ut_preds2')

fig, axes = plt.subplots(5, 1, sharex=True)
plt.suptitle("Weather in Salt Lake City, Utah")
labels = ['Temperature [deg F]', 'Precipitation [in]', 
        'Relative Humidity [%]', 'Pressure [in Hg]', 'Wind Speed [mph]']
for ii, ax in enumerate(axes):
    ax.plot(t[:steps], y_units[:steps,ii], 'bo', t[:steps], preds_units[:steps,ii], 'r-')
    ax.set_ylabel(labels[ii])
    ax.legend(('actual','predicted'))
axes[4].set_xlabel("Days")


plt.show()
