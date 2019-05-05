import numpy as np
import torch
import argparse
from mnist_dataset import MNISTDataset, F_MNISTDataset
from model import MNISTBaseLineModel
from utils import *
from torch.utils import data
from tqdm import tqdm


str2list = lambda s: s.split(',')
str2ints = lambda l: [int(i) for i in l.split(',')]
str2bool = lambda s: s == 'True' or s == 'true'

parser = argparse.ArgumentParser()
parser.add_argument('--load_model_path', type=str, default=None)
parser.add_argument('--validate', type=str2bool, default=False)
parser.add_argument('--grid_size', type=int, default=4)
parser.add_argument('--classes', type=str2ints, default=[2,3,4])
parser.add_argument('--val_set_size', type=int, default=100)
parser.add_argument('--interf', type=str2bool, default=False)
parser.add_argument('--fig_save_path', type=str, default=None)
parser.add_argument('--filter_size', type=int, default=64)

parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--shuffle', type=str2bool, default=False)
parser.add_argument('--num_workers', type=int, default=6)

parser.add_argument('--dataset_random', type=str2bool, default=True)
parser.add_argument('--dataset_maxnum_perclass', type=int, default=3)
parser.add_argument('--dataset_overlap_rate', type=float, default=0.3)
parser.add_argument('--device', type=str, default='auto')

args = parser.parse_args()

if args.device == 'auto':
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'


args.data_generator_params = {
    'batch_size': args.batch_size,
    'shuffle': args.shuffle,
    'num_workers': args.num_workers
}
args.dataset_params = {
    'grid_size': args.grid_size,
    'target': args.classes,
    'interference': args.interf,
    'random': args.dataset_random,
    'maxnum_perclass': args.dataset_maxnum_perclass,
    'overlap_rate': args.dataset_overlap_rate
}


def load_model(model_path):
    model = MNISTBaseLineModel(size=args.grid_size * 28, cls=len(args.classes), filter_size=args.filter_size).double()
    model.load_model(model_path)
    device = torch.device(args.device)
    model.to(device)
    return model

def heatmap2center(heatmap, threshold=0.005):
    heatmap = np.transpose(heatmap.detach().cpu().numpy(), [1,2,0])
    heatmap = heatmap / (56*56)
    # print(heatmap)
    h, w, c = heatmap.shape
    blobs = heatmap > threshold
    bbox = []
    # import pdb; pdb.set_trace()
    c_coord = np.stack([np.arange(w)]*h, axis=0)
    r_coord = np.stack([np.arange(h)]*w, axis=1)
    for i in range(c):
        bbox_c = []
        blob_c = blobs[:,:,i]
        blobs_labels, num_labels = measure.label(blob_c, background=0, connectivity=2, return_num=True)

        for j in range(1, num_labels+1):
            object_j = (blobs_labels==j)
            heatmap_j = object_j*heatmap[:,:,i]
            if np.sum(heatmap_j) < 0.1:
                continue
            heatmap_j = heatmap_j/np.sum(heatmap_j)
#             pdb.set_trace()
            center_c = np.sum(heatmap_j*c_coord)
            center_r = np.sum(heatmap_j*r_coord)
            bbox_c.append(np.array([center_r, center_c, i]))

        bbox.append(np.array(bbox_c))
    return np.array(bbox)


def get_heatmap(model, loader):
    iterator = tqdm(enumerate(loader))
    heatmaps = []
    labels = []
    gt_bboxes, gt_bbox_cls = [], []
    device = torch.device(args.device)
    for idx,(local_batch, local_labels, local_labels_cls, bboxes, bbox_cls) in iterator:
        local_batch, local_labels, local_labels_cls= \
            local_batch.to(device), local_labels.to(device), local_labels_cls.to(device),
        y_pred, heatmap = model(local_batch)
        heatmaps.append(heatmap)
        gt_bboxes.append(torch.Tensor(bboxes))
        gt_bbox_cls.append(torch.Tensor(bbox_cls))

    heatmaps = torch.cat(heatmaps, dim=0)
    # gt_bboxes = torch.stack(gt_bboxes, dim=0)
    # gt_bbox_cls = torch.stack(gt_bbox_cls, dim=0)
    return heatmaps, gt_bboxes, gt_bbox_cls


def center2bbox(centers):
    bboxes = []
    bboxes_cls = []
    for c in range(len(args.classes)):
        for i in range(centers[c].shape[0]):
            x1, y1 = centers[c][i][1]*2-14, centers[c][i][0]*2-14
            x2, y2 = x1 + 28, y1 + 28
            bboxes.append((y1, x1, y2, x2))
            bboxes_cls.append(c)
    bboxes = np.array(bboxes)
    bboxes_cls = np.array(bboxes_cls)
    return bboxes, bboxes_cls

if __name__=="__main__":
    # cls_model_path = ""
    cnt_model_path = "models/model-234r-64-5w"

    draw = True

    dataset = MNISTDataset(size=args.val_set_size, **args.dataset_params, det=True)
    loader = data.DataLoader(dataset, **args.data_generator_params)
    print('data done!')

    # cls_model = load_model(cls_model_path)
    cnt_model = load_model(cnt_model_path)

    # cls_heatmap, gt_bboxes, gt_bbox_cls = get_heatmap(cls_model, loader)
    cnt_heatmap, gt_bboxes, gt_bbox_cls = get_heatmap(cnt_model, loader)

    all_cls_bboxes, all_cnt_bboxes = [], []
    all_cls_labels, all_cnt_labels = [], []
    all_cls_scores, all_cnt_scores = [], []
    for idx in range(args.val_set_size):

        # cls_center = heatmap2center(cls_heatmap[idx], threshold=0.005)
        # cls_bboxes, cls_labels = center2bbox(cls_center)
        # cls_scores = np.ones(cls_bboxes.shape[0])
        # all_cls_bboxes.append(cls_bboxes)
        # all_cls_labels.append(cls_labels)
        # all_cls_scores.append(cls_scores)

        cnt_center = heatmap2center(cnt_heatmap[idx], threshold=0.005)
        # print(cnt_center)
        # print('!!')
        cnt_bboxes, cnt_labels = center2bbox(cnt_center)
        cnt_scores = np.ones(cnt_bboxes.shape[0]) / 2
        # all_cnt_bboxes.append(cnt_bboxes.detach().cpu().numpy())
        # all_cnt_labels.append(cnt_labels.detach().cpu().numpy())
        # all_cnt_scores.append(cnt_scores.detach().cpu().numpy())

        
        with open('mAP/input/ground-truth/image_%d.txt'%idx, 'w') as f:
            for i in range(gt_bboxes[idx].shape[0]):
                f.write("%d %d %d %d %d\n"% (int(gt_bbox_cls[idx][i]), int(gt_bboxes[idx][i][0]), int(gt_bboxes[idx][i][1]), int(gt_bboxes[idx][i][2]), int(gt_bboxes[idx][i][3])))

        with open('mAP/input/detection-results/image_%d.txt'%idx, 'w') as f:
            for i in range(cnt_bboxes.shape[0]):
                f.write("%d %f %d %d %d %d\n"% (int(cnt_labels[i])+2, np.random.rand(), int(cnt_bboxes[i][0]), int(cnt_bboxes[i][1]), int(cnt_bboxes[i][2]), int(cnt_bboxes[i][3])))
        


        # import pdb; pdb.set_trace()
        # if draw:
        # 	draw_bbox(cls_center, da)
        # 	draw_bbox(cnt_center)

    # mAP_cls = calculate_mAP(all_cls_bboxes, all_cls_labels, all_cls_scores, gt_bboxes, gt_bbox_cls)
    # mAP_cnt = calculate_mAP(all_cnt_bboxes, all_cnt_labels, all_cnt_scores, gt_bboxes, gt_bbox_cls, len(args.classes))

    # print('mAP cnt', mAP_cnt)



    