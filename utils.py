import numpy as np
from PIL import Image
import time
from skimage import measure
import logging

def load_image(filename):
    img = Image.open(filename)
    return np.array(img).transpose(2,0,1)

def time_for_file(localtime=True):
    ISOTIMEFORMAT='%d-%h-at-%H-%M-%S'
    f = time.localtime if localtime else time.gmtime
    return '{}'.format(time.strftime(ISOTIMEFORMAT, f(time.time())))

def output_bbox(heatmap, threshold):
    h, w, c = heatmap.shape
    blobs = heatmap > threshold
    bbox = []
    for i in range(c):
        bbox_c = []
        blob_c = blobs[:,:,i]
        blobs_labels, num_labels = measure.label(blob_c, background=0, connectivity=2, return_num=True)

        for j in range(1, num_labels+1):
            object_j = (blobs_labels==j)
            row_index = np.where(np.max(object_j, axis=0))
            left = np.min(row_index)
            right = np.max(row_index)
            col_index = np.where(np.max(object_j, axis=1))
            top = np.min(col_index)
            bottom = np.max(col_index)
            bbox_c.append(np.array([left, right, top, bottom, np.mean(heatmap[:,:,i][object_j])]))

        bbox.append(np.array(bbox_c))
    return np.array(bbox)

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

def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)