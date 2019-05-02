from utils import load_image
from torch.utils import data
from data_producer import MNISTDataProducer
from torchvision import datasets, transforms
import os

class MNISTDataset(data.Dataset):

    def __init__(self, size, grid_size=3, target=[6, 8], interference=False, random=False, maxnum_perclass=5, overlap_rate=0.5):
        self.size = size
        self.data_path = 'Datasets/mnist.pkl'
        g = MNISTDataProducer(self.data_path)
        if random:
            self.data = [g.generate_random(grid_size=grid_size, target=target, interference=interference, maxnum_perclass=maxnum_perclass, overlap_rate=overlap_rate) for i in range(size)]
        else:
            self.data = [g.generate(grid_size=grid_size, target=target, interference=interference) for i in range(size)]

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.data[index]

class F_MNISTDataset(data.Dataset):
    def __init__(self, size, grid_size=3, target=[6, 8], interference=False, random=False, maxnum_perclass=5, overlap_rate=0.5):
        self.size = size
        self.data_path = 'Datasets/f_mnist.pkl'
        if not os.path.exists(self.data_path):
            self._preprocess_fashion_mnist(self.data_path)
        g = MNISTDataProducer(self.data_path)
        if random:
            self.data = [g.generate_random(grid_size=grid_size, target=target, interference=interference, maxnum_perclass=maxnum_perclass, overlap_rate=overlap_rate) for i in range(size)]
        else:
            self.data = [g.generate(grid_size=grid_size, target=target, interference=interference) for i in range(size)]

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.data[index]

    def _preprocess_fashion_mnist(self, data_path):
        import numpy as np
        trainset = datasets.FashionMNIST('Datasets/F_MNIST_data/', download=True, train=True)
        testset = datasets.FashionMNIST('Datasets/F_MNIST_data/', download=True, train=False)

        training_images, training_labels, test_images, test_labels = [], [], [], []
        for i in range(len(trainset)):
            im = np.array(trainset[i][0])
            training_images.append(im)
            training_labels.append(trainset[i][1])
        
        for i in range(len(testset)):
            im = np.array(testset[i][0])
            test_images.append(im)
            test_labels.append(testset[i][1])

        ds = {
            "training_images" : training_images,
            "training_labels" : training_labels,
            "test_images" : test_images,
            "test_labels" : test_labels,

        }
        import pickle
        with open(data_path, 'wb') as f:
            pickle.dump(ds, f)
        
        print('preprocess done')

if __name__ == '__main__':
    dataset = F_MNISTDataset(3*28)
    