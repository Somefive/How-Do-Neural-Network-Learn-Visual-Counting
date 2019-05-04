import torch
import numpy as np
import matplotlib.pyplot as plt
from mnist_dataset import MNISTDataset, F_MNISTDataset
from model import MNISTBaseLineModel
import os

class HeatMapVisualizer:

    def __init__(self, model, model_path, dataset, classes, fig_save_path=None, cmap='hot', interpolation='nearest', visual_sample_only=False):
        self.classes = classes
        self.model = model
        self.model.load_model(model_path)
        self.dataset = dataset
        self.index = 0
        self.plot_params = {'cmap': cmap, 'interpolation': interpolation}
        self.fig_save_path = fig_save_path
        self.visual_sample_only = visual_sample_only

    def plot(self, fig_index=0):
        X, y = self.dataset[self.index]
        X = torch.as_tensor([X])
        pred, hidden = self.model(X)
        pred, hidden = pred.detach().numpy()[0], hidden.detach().numpy()[0]
        classes_size = len(self.classes) if not self.visual_sample_only else 0
        width = np.ceil(np.sqrt(classes_size+1))
        height = np.ceil((classes_size+1) / width)
        fig = plt.figure(figsize=(width * 3, height * 3), dpi=100)
        plt.subplot(height, width, 1)
        plt.imshow(X.detach().numpy()[0], **self.plot_params)
        plt.title('Sample')
        for i in range(classes_size):
            plt.subplot(height, width, i+2)
            plt.imshow(hidden[i], **self.plot_params)
            plt.title('cls:%d pred/gt:%.2f/%.d' % (self.classes[i], pred[i], y[i]))
        if self.fig_save_path:
            if fig_index > 0:
                fig_save_path = self.fig_save_path[:self.fig_save_path.index('.')] + str(fig_index) + self.fig_save_path[self.fig_save_path.index('.'):]
            else:
                fig_save_path = self.fig_save_path
            plt.savefig(fig_save_path)
        else:
            plt.show()
        self.index = (self.index + 1) % len(self.dataset)


if __name__ == '__main__':
    from argsparser import args
    model = MNISTBaseLineModel(size=args.grid_size * 28, cls=len(args.classes), filter_size=args.filter_size).double()
    if args.fashion:
        dataset = F_MNISTDataset(args.figs_count+1, **args.dataset_params)
    else:
        dataset = MNISTDataset(args.figs_count+1, **args.dataset_params)
    visualizer = HeatMapVisualizer(
        model=model,
        model_path=args.load_model_path,
        dataset=dataset,
        fig_save_path=args.fig_save_path,
        classes=args.classes,
        visual_sample_only=args.visual_sample_only
    )
    if args.figs_count > 0:
        for fig_index in range(args.figs_count):
            visualizer.plot(fig_index+1)
    else:
        visualizer.plot(0)