# file: lorenz_figure.py

import matplotlib.pyplot as plt
import numpy as np
import pdb

# import the data
traj = np.load('../data/lorenz_data.npy')
preds = np.load('../data/lorenz_predictions.npy')

# set up axes
#fig = plt.figure(figsize=(5, 5))
#fig.tight_layout(rect=[0.05, 0.05, 0.95, 0.95], h_pad=4., w_pad=2.7)
#ax1 = fig.add_subplot(221, projection='3d')
#ax1.plot(traj[1,::10], traj[2,::10], traj[3,::10], 'k.', markersize=1)
#ax2 = fig.add_subplot(312)
#ax2.plot(traj[0,:], traj[1,:])
#ax3 = fig.add_subplot(313)
#ax3.plot(preds[:,0], preds[:,1], 'k-')
#ax3.plot(preds[::10,0], preds[::10,2], 'k.', markersize=5)
fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(6,3))
fig.tight_layout(rect=[0.05, 0.05, 0.95, 0.95], h_pad=2., w_pad=4.)
gs = axs[0,0].get_gridspec()
axs[0,0].remove()
axs[1,0].remove()
axbig = fig.add_subplot(gs[0:, 0], projection='3d')
axbig.plot(traj[1,::30], traj[2,::30], traj[3,::30], 'k.', markersize=1)
axbig.set_xlabel('x', fontsize=8)
axbig.set_ylabel('y', fontsize=8)
axbig.set_zlabel('z', fontsize=8)
axbig.tick_params(axis='x', labelsize=6)
axbig.tick_params(axis='y', labelsize=6)
axbig.tick_params(axis='z', labelsize=6)
axbig.set_title('(a)', weight='bold', fontsize=10, loc='center')
axs[0,1].plot(traj[0,:], traj[1,:], 'k-', linewidth=1)
axs[0,1].set_xlabel('t', fontsize=8)
axs[0,1].set_ylabel('x(t)', fontsize=8)
axs[0,1].tick_params(axis='x', labelsize=6)
axs[0,1].tick_params(axis='y', labelsize=6)
axs[0,1].set_title('(b)', weight='bold', fontsize=10, loc='center')
axs[1,1].plot(preds[:,0], preds[:,1], 'k-')
axs[1,1].plot(preds[::30,0], preds[::30,2], '+', color='tab:gray', markersize=3)
axs[1,1].set_xlabel('t', fontsize=8)
axs[1,1].set_ylabel('x(t)', fontsize=8)
axs[1,1].tick_params(axis='x', labelsize=6)
axs[1,1].tick_params(axis='y', labelsize=6)
axs[1,1].set_title('(c)', weight='bold', fontsize=10, loc='center')
plt.savefig('lorenz_figure.pdf', dpi=600)
