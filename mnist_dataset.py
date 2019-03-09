from utils import load_image
from torch.utils import data
from data_producer import MNISTDataProducer

class MNISTDataset(data.Dataset):

    def __init__(self, size, grid_size=3, confusion=True, target=8):
        self.size = size
        g = MNISTDataProducer()
        self.data = [g.generate(grid_size=grid_size, confusion=confusion, target=target) for i in range(size)]
    
    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.data[index]
