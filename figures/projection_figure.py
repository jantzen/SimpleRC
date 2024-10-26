# file: projection_figure.py

import matplotlib.pyplot as plt
import numpy as np

# import the data
data = np.load('../data/projection_demo.npy')

fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(6,3))
fig.tight_layout(rect=[0.05, 0.05, 0.95, 0.95], w_pad=4.)
gs = axs[0].get_gridspec()
axs[0].remove()
ax3d = fig.add_subplot(gs[0], projection='3d')
ax3d.plot(data[10:,0], data[10:,1], data[10:,2], 'k.')
ax3d.set_xlabel(r'$x_1$', fontsize=8)
ax3d.set_ylabel(r'$x_2$', fontsize=8)
ax3d.set_zlabel(r'$x_3$', fontsize=8)
ax3d.tick_params(axis='x', labelsize=4)
ax3d.tick_params(axis='y', labelsize=4)
ax3d.tick_params(axis='z', labelsize=4)
ax3d.xaxis.labelpad = -10
ax3d.yaxis.labelpad = -10
ax3d.zaxis.labelpad = -10
ax3d.set_xticklabels([])
ax3d.set_yticklabels([])
ax3d.set_zticklabels([])
ax3d.set_title('(a)', weight='bold', fontsize=10, loc='center')
axs[1].plot(data[10:,3], data[10:,4], 'k.')
axs[1].set_aspect(1)
axs[1].set_xlabel(r'$y_1$', fontsize=8)
axs[1].set_ylabel(r'$y_2$', fontsize=8)
axs[1].tick_params(axis='x', labelsize=4)
axs[1].tick_params(axis='y', labelsize=4)
axs[1].xaxis.labelpad = 2
axs[1].yaxis.labelpad = 2
axs[1].set_xlim([-0.48, -0.12])
axs[1].set_xticklabels([])
axs[1].set_yticklabels([])
axs[1].set_title('(b)', weight='bold', fontsize=10, loc='center')
plt.savefig('projection_figure.pdf', dpi=600)

