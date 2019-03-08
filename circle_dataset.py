from utils import load_image
from torch.utils import data
from data_producer import CircleDataProducer

class CircleDataset(data.Dataset):

    def __init__(self, size, min_circles=1, max_circles=41, image_size=64, radius=3):
        self.size = size
        g = CircleDataProducer(min_circles=min_circles, max_circles=max_circles, size=image_size, radius=radius)
        self.data = [g.generate() for i in range(size)]
    
    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.data[index]
