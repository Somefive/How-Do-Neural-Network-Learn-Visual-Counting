from utils import load_image
from torch.utils import data
from data_producer import MNISTDataProducer

class MNISTDataset(data.Dataset):

    def __init__(self, size):
        self.size = size
        g = MNISTDataProducer()
        self.data = [g.generate() for i in range(size)]
    
    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.data[index]
