import numpy as np
import pickle


class DotDataProducer(object):

    def __init__(self, size=64, min_dots=20, max_dots=200):
        self.size = size
        self.min_dots, self.max_dots = min_dots, max_dots
        self.scale, self.offset = max_dots - min_dots, min_dots

    def generate(self):
        X = np.zeros((self.size, self.size))
        y = np.random.randint(self.min_dots, self.max_dots)
        positions = (np.random.rand(2, y) * self.size).astype(int)
        X[positions[0], positions[1]] = 1
        y = X.sum()
        return X, (y - self.offset) / self.scale


from skimage import draw
class CircleDataProducer(object):

    def __init__(self, size=64, min_circles=1, max_circles=41, radius=3):
        self.size = size
        self.min_circles, self.max_circles = min_circles, max_circles
        self.radius = radius
        self.offset, self.scale = 1, max_circles - 1

    def generate(self):
        radius = self.radius
        size = self.size
        X = np.zeros((self.size, self.size))
        y = np.random.randint(self.min_circles, self.max_circles)
        positions = (np.random.rand(y, 2) * self.size).astype(int)
        pos = []
        for i in range(y):
            _x, _y = positions[i]
            if _x < radius or _x > size - radius or _y < radius or _y > size - radius:
                continue
            flag = False
            for __x, __y in pos:
                if (__x - _x) ** 2 + (__y - _y) ** 2 < radius ** 2:
                    flag = True
                    break
            if flag:
                continue
            rr, cc = draw.circle(_x, _y, radius)
            X[rr, cc] = 1
            pos.append((_x, _y))
        y = len(pos)
        return X, (y - self.offset) / self.scale


from collections import defaultdict
import random
class MNISTDataProducer(object):

    def __init__(self):
        ds = pickle.load(open("Datasets/mnist.pkl",'rb'))
        training_images, training_labels, test_images, test_labels = ds["training_images"], ds["training_labels"], ds["test_images"], ds["test_labels"]
        self.images = defaultdict(list)
        for i in range(len(training_labels)):
            self.images[training_labels[i]].append(training_images[i].reshape(28,28))
        for i in range(len(test_labels)):
            self.images[test_labels[i]].append(test_images[i].reshape(28,28))

    def generate(self, grid_size=3, target=[6, 8], maxnum_perclass=5, interference=False):
        not_target = list(set(np.arange(10)) - set(target))
        X = np.zeros((grid_size * 28, grid_size * 28))
        # part = np.random.choice(5, len(target), replace=False)+1
        # max_num = min(max_num, grid_size * grid_size)
        if interference:
            part = np.random.choice(grid_size * grid_size + 1, len(target)+1)
        else:
            part = np.random.choice(grid_size * grid_size + 1, len(target))
        part = np.sort(part)

        y = np.zeros((len(target)))

        pos = list(range(grid_size * grid_size))
        random.shuffle(pos)

        last = 0
        for j in range(len(target)):
            yc = part[j] - last # Let's generate yc many target[j]
            y[j] = min(yc, maxnum_perclass)
            for i in range(last, min(yc, maxnum_perclass)+last):
                _x, _y = pos[i] // grid_size * 28, pos[i] % grid_size * 28
                X[_x:_x+28,_y:_y+28] = self.images[target[j]][np.random.randint(len(self.images[target[j]]))]
            last = part[j]

        if interference:
            for i in range(last, part[len(target)], 1):
                _x, _y = pos[i] // grid_size * 28, pos[i] % grid_size * 28
                number = not_target[np.random.randint(len(not_target))]
                X[_x:_x+28,_y:_y+28] = self.images[number][np.random.randint(len(self.images[number]))]

        # print(y)
        # import matplotlib.pyplot as plt
        # plt.imshow(X)
        # plt.show()

        return X, y

    def generate_random(self, grid_size=3, target=[6, 8], maxnum_perclass=5, interference=False, overlap_rate=0.5):
        not_target = list(set(np.arange(10)) - set(target))
        X = np.zeros((grid_size * 28, grid_size * 28))
        y = np.zeros((len(target)))
        flag = np.zeros((grid_size * 28, grid_size * 28))

        if interference:
            part = np.random.choice(grid_size * grid_size + 1, len(target)+1)
        else:
            part = np.random.choice(grid_size * grid_size + 1, len(target))
        part = np.sort(part)

        last = 0
        for i in range(len(target)):
            y[i] = min(part[i] - last, maxnum_perclass)
            for j in range(min(part[i] - last, maxnum_perclass)):
                while True:
                    top, left = np.random.randint(X.shape[0]-28), np.random.randint(X.shape[1]-28)
                    if np.sum(flag[top:top+28, left:left+28]) < overlap_rate * (28*28):
                        break
                flag[top:top+28, left:left+28] = 1
                X[top:top+28, left:left+28] += self.images[target[i]][np.random.randint(len(self.images[target[i]]))]
            last = part[i]
        
        full = False
        if interference:
            for i in range(last, part[len(target)], 1):
                number = not_target[np.random.randint(len(not_target))]

                cnt = 0
                while True:
                    top, left = np.random.randint(X.shape[0]-28), np.random.randint(X.shape[1]-28)
                    if np.sum(flag[top:top+28, left:left+28]) < overlap_rate * (28*28):
                        flag[top:top+28, left:left+28] = 1
                        X[top:top+28, left:left+28] += self.images[number][np.random.randint(len(self.images[number]))]
                        break
                    cnt +=1 
                    full = (cnt > 30)
                    if full:
                        break
                if full:
                    break


        X = np.minimum(X, 255)
        return X, y

if __name__ == '__main__':
    gen = MNISTDataProducer()
    X, y = gen.generate_random(grid_size=4, target=[1,2,3,4,5,6])
    # X, y = gen.generate(grid_size=4, target=[1,2,3,4,5,6])
    import matplotlib.pyplot as plt
    print(y)
    plt.imshow(X, cmap='gray')
    plt.show()
