import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import torch
from cryptodata import CryptoDataset

class ModelMetrics:
    def __init__(self, dataset: CryptoDataset) -> None:
        self.dataset = dataset

        self.max_subplots_per_row = 3
        self.num_rows = (self.dataset.features_length[1] + self.max_subplots_per_row - 1) // self.max_subplots_per_row
        self.num_cols = min(self.dataset.features_length[1], 3)
        self.lines = {}

        plt.ion()
        self.fig, self.axes = plt.subplots(self.num_rows, self.num_cols, figsize=(12, 8), constrained_layout=True)
        self.axes = np.atleast_1d(self.axes)

        for i in range(self.num_rows * self.num_cols):
            if i < self.dataset.features_length[1]:
                self.axes.flat[i].set_visible(True)
            else:
                self.axes.flat[i].set_visible(False)

    def plot(self, output, target):
        output = output.reshape(-1, output.shape[2])
        target = target.reshape(-1, target.shape[2])

        for i, feature in enumerate(self.dataset.features[1]):
            ax = self.axes.flat[i]

            target_x_data = range(output.shape[0])
            target_y_data = target[:, i]

            output_x_data = range(output.shape[0])
            output_y_data = output[:, i]

            if feature in self.lines:
                output_line, target_line = self.lines[feature]

                target_line.set_data(target_x_data, target_y_data)
                output_line.set_data(output_x_data, output_y_data)
            else:
                target_line, = ax.plot(target[:, i], label='Actual', color='orange')
                output_line, = ax.plot(output[:, i], label='Predicted', color='blue')
                
                self.lines[feature] = (output_line, target_line)

                ax.set_title(feature)
                ax.legend()

            ax.set_xlim(0, max(target_x_data))

            ax.relim()
            ax.autoscale_view(True, True, True)
            
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def close(self):
        plt.ioff()