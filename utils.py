import numpy as np
from PIL import Image
def load_image(filename):
    img = Image.open(filename)
    return np.array(img).transpose(2,0,1)
