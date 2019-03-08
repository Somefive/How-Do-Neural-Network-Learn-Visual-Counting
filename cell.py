from utils import load_image
from torch.utils import data

class VGGDataset(data.Dataset):

    def __init__(self, ids):
        self.ids = []
    
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        idx = self.ids[index]
        X = load_image('Datasets/VGG_Cell_Dataset/%03dcell.png') / 255
        Y = load_image('Datasets/VGG_Cell_Dataset/%03ddots.png') / 255
        return X, y
