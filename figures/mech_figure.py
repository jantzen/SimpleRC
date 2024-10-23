# file: mech_figure.py

import matplotlib.pyplot as plt
import numpy as np

# import the data
preds_lin_osc = np.load('../data/mech_preds_double_linear.npy')
preds_pend = np.load('../data/mech_preds_double_pend.npy')


fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(6,5))
fig.tight_layout(rect=[0.05, 0.05, 0.95, 0.95], h_pad=4.)
axs[0].plot(preds_lin_osc[:,0], preds_lin_osc[:,1], 'k-')
axs[0].plot(preds_lin_osc[:,0], preds_lin_osc[:,5], 'k+', markersize=5)
axs[0].plot(preds_lin_osc[:,0], preds_lin_osc[:,2], 'k-')
axs[0].plot(preds_lin_osc[:,0], preds_lin_osc[:,6], 'k+', markersize=5)
axs0t = axs[0].twinx()
axs0t.plot(preds_lin_osc[:,0], preds_lin_osc[:,3], '--', color='tab:gray')
axs0t.plot(preds_lin_osc[:,0], preds_lin_osc[:,7], '.', color='tab:gray', markersize=5)
axs0t.plot(preds_lin_osc[:,0], preds_lin_osc[:,4], '--', color='tab:gray')
axs0t.plot(preds_lin_osc[:,0], preds_lin_osc[:,8], '.', color='tab:gray', markersize=5)
axs[0].set_title('(a)', weight='bold', fontsize=12, loc='center')
axs[0].set_xlabel('t', fontsize=10)
axs[0].set_ylabel(r'$x_i(t)$', fontsize=10)
axs0t.set_ylabel(r'$v_i(t)$', fontsize=10, color='tab:gray')
axs0t.tick_params(axis='y', colors='tab:gray')

axs[1].plot(preds_pend[:,0], preds_pend[:,1], 'k-')
axs[1].plot(preds_pend[:,0], preds_pend[:,5], 'k+', markersize=5)
axs[1].plot(preds_pend[:,0], preds_pend[:,2], 'k-')
axs[1].plot(preds_pend[:,0], preds_pend[:,6], 'k+', markersize=5)
axs1t = axs[1].twinx()
axs1t.plot(preds_pend[:,0], preds_pend[:,3], '--', color='tab:gray')
axs1t.plot(preds_pend[:,0], preds_pend[:,7], '.', color='tab:gray', markersize=5)
axs1t.plot(preds_pend[:,0], preds_pend[:,4], '--', color='tab:gray')
axs1t.plot(preds_pend[:,0], preds_pend[:,8], '.', color='tab:gray', markersize=5)
axs[1].set_title('(b)', weight='bold', fontsize=12, loc='center')
axs[1].set_xlabel('t', fontsize=10)
axs[1].set_ylabel(r'$x_i(t)$', fontsize=10)
axs1t.set_ylabel(r'$v_t(t)$', fontsize=10, color='tab:gray')
axs1t.tick_params(axis='y', colors='tab:gray')
plt.savefig('mech_figure.pdf', dpi=600)

