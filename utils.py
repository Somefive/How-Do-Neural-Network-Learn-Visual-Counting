import numpy as np
from PIL import Image
import time
def load_image(filename):
    img = Image.open(filename)
    return np.array(img).transpose(2,0,1)

def time_for_file(localtime=True):
    ISOTIMEFORMAT='%d-%h-at-%H-%M-%S'
    f = time.localtime if localtime else time.gmtime
    return '{}'.format(time.strftime(ISOTIMEFORMAT, f(time.time())))

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

