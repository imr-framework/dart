"""
===================
Image Slices Viewer
===================

Scroll through 2D image slices of a 3D array.
"""

import matplotlib.pyplot as plt

ALPHA_MASK = 0.25


class IndexTracker(object):
    def __init__(self, ax, vol, mask, vmax, cmap='gray'):
        self.ax = ax

        self.X = vol
        self.Y = mask
        rows, cols, self.slices = vol.shape
        self.ind = self.slices // 2

        self.im = self.ax.imshow(self.X[:, :, self.ind], cmap=cmap, vmax=vmax)
        self.im2 = self.ax.imshow(self.Y[:, :, self.ind], cmap='hot', alpha=ALPHA_MASK)
        self.ax.axis('off')
        self.update()

    def onscroll(self, event):
        # print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        self.im2.set_data(self.Y[:, :, self.ind])
        self.ax.set_title(f'slice {self.ind}/{self.slices}')
        self.im.axes.figure.canvas.draw()
        self.im2.axes.figure.canvas.draw()


def maskplot3d(vol, mask, vmax=None, cmap='gray'):
    """
    Plot one volume and overlay a mask.
    """
    fig, ax = plt.subplots(1, 1)
    tracker = IndexTracker(ax, vol, mask, vmax, cmap)
    fig.colorbar(tracker.im)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show()
