from utils import load_image
from torch.utils import data
from data_producer import DotDataProducer

class DotDataset(data.Dataset):

    def __init__(self, size, min_dots=20, max_dots=200, image_size=64):
        self.size = size
        g = DotDataProducer(min_dots=min_dots, max_dots=max_dots, size=image_size)
        self.data = [g.generate() for i in range(size)]
    
    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.data[index]
