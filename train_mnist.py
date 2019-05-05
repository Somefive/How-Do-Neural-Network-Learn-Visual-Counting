import torch
import torch.nn as nn
from torch.utils import data
from mnist_dataset import MNISTDataset, F_MNISTDataset
from trancos_dataset import TRANCOSDataset
from model import MNISTBaseLineModel, TRANCOSBaseLineModel
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from tensorboardX import SummaryWriter
import os
import random
from utils import *
import logging

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from argsparser import args

rand_seed=0
np.random.seed(rand_seed)
random.seed(rand_seed)
torch.manual_seed(rand_seed)
torch.cuda.manual_seed_all(rand_seed)

log_dir = os.path.join('runs', time_for_file() + '_seed' + str(rand_seed) + '_mnist_' + args.task +  ("_" + args.cm if args.cm != "" else ""))
writer = SummaryWriter(log_dir)
set_logger(os.path.join(log_dir, 'train.log'))
args.save_model_path.append(os.path.join(log_dir, 'base-model'))

# CUDA for PyTorch
device = torch.device(args.device)

# Model
Criterions = {
    "count"    : nn.SmoothL1Loss(),
    "cls"      : nn.MultiLabelSoftMarginLoss(),
}


criterion = Criterions[args.task]
model     = MNISTBaseLineModel(size=args.grid_size * 28, cls=len(args.classes), filter_size=args.filter_size).double()

from torch.optim.lr_scheduler import StepLR
optimizer = args.optim(model.parameters(), lr=args.lr)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

logging.critical(model)
logging.critical(args)

# Generators
if args.fashion:
    training_set = F_MNISTDataset(size=args.train_set_size, **args.dataset_params)
    validation_set = F_MNISTDataset(size=args.val_set_size, **args.dataset_params)
elif args.trancos:
    training_set = TRANCOSDataset('trainval')
    validation_set = TRANCOSDataset('test')
    model = TRANCOSBaseLineModel(filter_size=args.filter_size).double()
else:
    training_set = MNISTDataset(size=args.train_set_size, **args.dataset_params)
    validation_set = MNISTDataset(size=args.val_set_size, **args.dataset_params)

training_generator = data.DataLoader(training_set, **args.data_generator_params)
validation_generator = data.DataLoader(validation_set, **args.data_generator_params)

model.load_model(args.load_model_path)
model.to(device)

print('Dataloader initiated.')

def run(train_mode=True, epoch=0):
    phase = 'train' if train_mode else 'test'
    if train_mode:
        optimizer.zero_grad()
        scheduler.step()

    if args.task == "count":
        mse = AverageMeter()
        mde = AverageMeter()
        dos = AverageMeter()

    cnt = 0
    iterator = tqdm(enumerate(training_generator if train_mode else validation_generator))
    num_batches = len(training_generator)
    for idx, (local_batch, local_labels, local_labels_cls) in iterator:
        current_step = epoch * num_batches + idx

        local_batch, local_labels, local_labels_cls = local_batch.to(device), local_labels.to(device), local_labels_cls.to(device),
        y_pred, _ = model(local_batch)

        if args.task == "count":
            y_true = local_labels

        else:
            y_true = local_labels_cls


        loss = criterion(y_pred, y_true)
        if train_mode:
            loss.backward()
            optimizer.step()

        cnt += local_labels.size(0)

        if args.task == "count":
            mse.update(loss.item(), local_labels.size(0))
            diff = torch.abs(y_pred - y_true)
            sum = y_pred + y_true
            mde.update(torch.mean(torch.sum(diff, 1)).item(),local_labels.size(0))
            dos.update(torch.mean(torch.div(diff, sum+1e-8)).item(), local_labels.size(0))
            # if len(args.classes) == 1:
            #     log_string = '%s [%d,%d] (%.1f vs %.1f) loss: %.3e(%.3e) mse:%.3e(%.3e), mde:%.3e(%.3e), dos: %.3e(%.3e)' % (
            #         'Train' if train_mode else 'Val  ', epoch+1, cnt, y_true[0].item(), y_pred[0].item(), \
            #             mse.val, mse.avg, mde.val, mde.avg, dos.val, dos.avg)

            log_string = '%s [%d,%d] (%.1f vs %.1f) loss: %.3f mse:%.3e(%.3e), mde:%.3e(%.3e), dos: %.3e(%.3e)' % (
                'Train' if train_mode else 'Val  ', epoch+1, cnt, y_true[0][0].item(), y_pred[0][0].item(), \
                    loss.item(), mse.val, mse.avg, mde.val, mde.avg, dos.val, dos.avg)

        elif args.task == "cls":
            # mAP = compute_map(y_true.detach().cpu().numpy(), y_pred.detach().cpu().numpy())
            total_acc =  ((torch.sigmoid(y_pred) > 0.5).double() == y_true).sum().double()
            acc = total_acc * 1.0 / (y_true.shape[0] * y_true.shape[1])
            log_string = '%s [%d,%d] loss: %.3f acc: %.3f (%.1e / %.1e)' % (
                    'Train' if train_mode else 'Val  ', epoch+1, cnt, loss.item(), acc,
                    total_acc, y_true.shape[0] * y_true.shape[1] )

        iterator.set_description(log_string)
        logging.info(log_string)

        if idx % 30 == 0:
            writer.add_scalar(phase+'/loss', loss.item(), current_step)
            image = torch.stack((local_batch[0], local_batch[0], local_batch[0]), dim=0)
            writer.add_image('input', image, current_step)
            if args.task == "count":
                writer.add_scalar(phase+'/mse', mse.avg, current_step)
                writer.add_scalar(phase+'/mde', mde.avg, current_step)
                writer.add_scalar(phase+'/dos', dos.avg, current_step)
            else:
                writer.add_scalar(phase+'/acc', acc, current_step)

    if train_mode:
        for save_model_path in args.save_model_path:
            model.save_model(save_model_path, device=device)


# Loop over epochs
for epoch in range(args.max_epochs):
    run(True, epoch)
    run(False, epoch)


