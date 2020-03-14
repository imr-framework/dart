"""
===================
Image Slices Viewer
===================

Scroll through 2D image slices of a 3D array.
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


class IndexTracker(object):
    def __init__(self, ax, vols, vmax, disp_axis, cmap='gray'):
        self.ax = ax
        self.disp_axis = disp_axis

        self.X = vols
        if self.disp_axis == 0:
            self.slices, rows, cols = vols[0].shape
        elif self.disp_axis == 1:
            rows, self.slices, cols = vols[0].shape
        else:
            rows, cols, self.slices = vols[0].shape
        self.ind = self.slices // 2

        self.im = []
        for i in range(len(self.ax)):
            self.ax[i].axis('off')
            if self.disp_axis == 0:
                slice = self.X[i][self.ind, :, :]
            elif self.disp_axis == 1:
                slice = self.X[i][:, self.ind, :]
            else:
                slice = self.X[i][:, :, self.ind]
            self.im.append(self.ax[i].imshow(slice, cmap=cmap, vmax=vmax))
        self.update()

    def onscroll(self, event):
        # print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        for i in range(len(self.im)):
            if self.disp_axis == 0:
                slice = self.X[i][self.ind, :, :]
            elif self.disp_axis == 1:
                slice = self.X[i][:, self.ind, :]
            else:
                slice = self.X[i][:, :, self.ind]
            self.im[i].set_data(slice)
            self.ax[i].set_title(f'slice {self.ind}/{self.slices}')
            self.im[i].axes.figure.canvas.draw()


def multiplot3d(*vols, vmax=None, cmap='gray', disp_axis=2):
    """
    Plot one or two volumes.
    """
    if len(vols) == 1:
        fig, ax = plt.subplots(1, 1)
        ax = np.asarray([ax])
    elif len(vols) == 2:
        fig, ax = plt.subplots(1, 2)
    elif len(vols) == 3:
        fig, ax = plt.subplots(1, 3)
    elif len(vols) == 4:
        fig, ax = plt.subplots(2, 2)
    else:
        raise ValueError('multiplot3d currently does not support plotting more than 4 volumes simultaneously.')

    ax = ax.flatten()
    fig.suptitle('Use scroll wheel to navigate images')
    tracker = IndexTracker(ax, vols, vmax, disp_axis=disp_axis, cmap=cmap)
    for i in range(len(tracker.im)):
        divider = make_axes_locatable(tracker.ax[i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(tracker.im[i], ax=tracker.ax[i], cax=cax)

    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show()
