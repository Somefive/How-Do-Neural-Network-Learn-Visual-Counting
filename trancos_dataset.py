from utils import load_image
from torch.utils import data
import numpy as np

class TRANCOSDataset(data.Dataset):

    def __init__(self, src):
        self.data = []
        with open('Datasets/TRANCOS_v3/image_sets/%s.txt' % src) as f:
            for line in f:
                filename = line[:line.find('.')]
                X = load_image('Datasets/TRANCOS_v3/images/%s.jpg' % filename) / 255
                y = load_image('Datasets/TRANCOS_v3/images/%sdots.png' % filename).sum() / 255
                self.data.append((X, np.array([y]), 0))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
