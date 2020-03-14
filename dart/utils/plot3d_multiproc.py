"""
===================
Image Slices Viewer
===================

Scroll through 2D image slices of a 3D array.
"""

import matplotlib.pyplot as plt

MASK_ALPHA_VAL = 0.2


class IndexTracker(object):
    """
    Matplotlib volume viewer - implements a mouse wheel scroll listener. The listener updates the plot.
    """

    def __init__(self, ax, X_q, Y_q):
        """
        ax : matplotlib.axes.Axes
            Axes instance
        X_q : multiprocessing.Queue
            Queue to receive brain data from parent process.
        Y_q : multiprocessing.Queue
            Queue to receive tumor mask data from parent process.
        """
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.X = X_q.get()
        self.Y_q = Y_q
        self.Y = None
        rows, cols, self.slices = self.X.shape
        self.ind = self.slices // 2

        self.im = ax.imshow(self.X[:, :, self.ind], cmap='gray')
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
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()
        if not self.Y_q.empty():  # If Queue has data, get it
            self.Y = self.Y_q.get()
        if self.Y is not None:  # If we have already retrieved data from Queue, plot it
            if not hasattr(self, 'im2'):  # If we have never plotted the tumor mask, create an AxesImage instance now
                self.im2 = self.ax.imshow(self.Y[:, :, self.ind], alpha=MASK_ALPHA_VAL)
            else:  # We have previously plotted tumor mask, so reuse AxesImage instance
                self.im2.set_data(self.Y[:, :, self.ind])


def multiproc_plot3d(vol_q, mask_q):
    """
    Multiprocessing plotting for DART talk.
    """
    fig, ax = plt.subplots(1, 1)
    tracker = IndexTracker(ax, X_q=vol_q, Y_q=mask_q)
    fig.colorbar(tracker.im)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show()
