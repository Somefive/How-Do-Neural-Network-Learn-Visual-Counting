from utils import load_image
from torch.utils import data
from data_producer import MNISTDataProducer

class MNISTDataset(data.Dataset):

    def __init__(self, size, grid_size=3, target=[6, 8], interference=False, random=False, maxnum_perclass=5, overlap_rate=0.5):
        self.size = size
        g = MNISTDataProducer()
        if random:
            self.data = [g.generate_random(grid_size=grid_size, target=target, interference=interference, maxnum_perclass=maxnum_perclass, overlap_rate=overlap_rate) for i in range(size)]
        else:
            self.data = [g.generate(grid_size=grid_size, target=target, interference=interference) for i in range(size)]

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.data[index]
