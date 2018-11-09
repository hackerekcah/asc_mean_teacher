import matplotlib.pyplot as plt


class History (object):
    """
    class to hold `loss` and `accuracy` over `epoch`
    """

    def __init__(self, name=None):
        self.name = name
        self.epoch = []
        self.loss = []
        self.acc = []

    def add(self, logs, epoch):
        self.epoch.append(epoch)
        self.loss.append(logs['loss'])
        self.acc.append(logs['acc'])

    def plot(self, axes=None, show=False):
        # axes usually returned by subplot
        if axes:
            axes[0].plot(self.epoch, self.loss)
            axes[0].legend(self.name + "/loss")

            axes[1].plot(self.epoch, self.acc)
            axes[1].legend(self.name + "/acc")
        else:
            plt.figure()
            plt.subplot(2, 1, 1)
            plt.plot(self.epoch, self.loss)
            plt.legend([self.name + "/loss"])
            plt.xticks(self.epoch, [str(e) for e in self.epoch])

            plt.subplot(2, 1, 2)
            plt.plot(self.epoch, self.acc)
            plt.legend([self.name + "/acc"])
            plt.xticks(self.epoch, [str(e) for e in self.epoch])

        plt.show() if show else None



