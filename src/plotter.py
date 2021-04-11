from typing import List

import matplotlib.pyplot as plt


class Plotter:
    def __init__(self):
        pass

    def plot_epoch_losses(self, epochs: List[int], loss_train: List[float], loss_test: List[float]):
        plt.plot(epochs, loss_train)
        plt.plot(epochs, loss_test)
        plt.show()

    def plot_filters_losses(self, n_filters: List[int], loss_train: List[float], loss_test: List[float]):
        plt.plot(n_filters, loss_train)
        plt.plot(n_filters, loss_test)
        plt.show()
