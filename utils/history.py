import matplotlib.pyplot as plt
from IPython.display import clear_output
import numpy as np


class History (object):
    """
    class to hold `loss` and `accuracy` over `epoch`
    """

    def __init__(self, name=None):
        self.name = name
        self.epoch = []
        self.loss = []
        self.acc = []
        self.axes = []
        self.recent = None

    def add(self, logs, epoch):
        self.recent = logs
        self.epoch.append(epoch)
        self.loss.append(logs['loss'])
        self.acc.append(logs['acc'])

    def set_axes(self, axes=None):
        if axes:
            self.axes = axes
        # new figure and axis
        else:
            self.axes = []
            plt.figure()
            self.axes.append(plt.subplot(2, 1, 1))
            self.axes.append(plt.subplot(2, 1, 2))

    def _get_tick(self):
        tick_max = np.max(self.epoch)
        ticks_int = np.arange(0, tick_max, np.ceil(tick_max / 5))
        if max(ticks_int) != tick_max:
            ticks_int = np.append(ticks_int, tick_max)
        return ticks_int

    def plot(self, axes=None, show=True):
        """
        plot loss acc in subplots
        :param axes: # axes usually returned by subplot if provided
        :param show:
        :return:
        """
        # if provided, set, else create
        self.set_axes(axes=axes)

        ticks = self._get_tick()
        self.axes[0].plot(self.epoch, self.loss)
        self.axes[0].legend([self.name + "/loss"])
        self.axes[0].set_xticks(ticks)
        self.axes[0].set_xticklabels([str(e) for e in ticks])

        self.axes[1].plot(self.epoch, self.acc)
        self.axes[1].legend([self.name + "/acc"])
        self.axes[1].set_xticks(ticks)
        self.axes[1].set_xticklabels([str(e) for e in ticks])

        plt.show() if show else None

    def clc_plot(self, axes=None, show=True):
        """
        clear output before plot, use in jupyter notebook to dynamically plot
        :param axes:
        :param show:
        :return:
        """
        clear_output(wait=True)
        self.plot(axes=axes, show=show)

    def clear(self):
        clear_output(wait=True)

