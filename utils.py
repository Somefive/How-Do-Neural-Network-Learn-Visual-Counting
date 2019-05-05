import numpy as np
from PIL import Image
import time
from skimage import measure
import logging
import sklearn.metrics

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
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.WARNING)
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

def compute_map(gt, pred, average=None):
    """
    Compute the multi-label classification accuracy.
    Args:
        gt (np.ndarray): Shape Nx20, 0 or 1, 1 if the object i is present in that
            image.
        pred (np.ndarray): Shape Nx20, probability of that object in the image
            (output probablitiy).
        valid (np.ndarray): Shape Nx20, 0 if you want to ignore that class for that
            image. Some objects are labeled as ambiguous.
    Returns:
        AP (list): average precision for all classes
    """

    nclasses = gt.shape[1]
    AP = []
    gt_cls = gt.astype('float32')
    pred_cls = (pred > 0.5).astype('float32')       # As per PhilK. code:
    # https://github.com/philkr/voc-classification/blob/master/src/train_cls.py
    # pred_cls -= 1e-5 * gt_cls
    pred_cls -= 1e-5 * gt_cls
    ap = sklearn.metrics.average_precision_score(gt_cls, pred_cls, average=average)
    return np.mean(ap)


# def eval_map(model, dataset):
#     """
#     Evaluate the model with the given dataset
#     Args:
#          model (keras.Model): model to be evaluated
#          dataset (tf.data.Dataset): evaluation dataset
#     Returns:
#          AP (list): Average Precision for all classes
#          MAP (float): mean average precision
#     """

#     # TODO not clear about how to calculate if divided into batches, so here just calculate as a whole thanks to the small dataset

#     AP = [0.0] * 20
#     all_preds = np.zeros((0, 20))
#     all_labels = np.zeros((0, 20))
#     all_weights = np.zeros((0, 20))
#     for batch, (images, labels, weights) in enumerate(dataset):
#         pred = tf.math.sigmoid(model(images))
#         all_preds = np.concatenate([all_preds, np.array(pred)], axis=0)
#         all_labels = np.concatenate([all_labels, np.array(labels)], axis=0)
#         all_weights = np.concatenate([all_weights, np.array(weights)], axis=0)
    
#     AP = compute_ap(all_labels, all_preds, all_weights)
#     # mAP = np.nanmean(AP)
#     mAP = sum(AP)/len(AP)
#     return AP, mAP
