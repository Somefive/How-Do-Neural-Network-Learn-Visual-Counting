import torch
from torch.utils import data
from trancos_dataset import TRANCOSDataset
from model import TRANCOSBaseLineModel, TRANCOSModel1
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from utils import AverageMeter
import argparse
import os
from tensorboardX import SummaryWriter
from utils import *
import random
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import logging
from argsparser import args

rand_seed=0
np.random.seed(rand_seed)
random.seed(rand_seed)
torch.manual_seed(rand_seed)
torch.cuda.manual_seed_all(rand_seed)

log_dir = os.path.join('runs', time_for_file() + '_seed' + str(rand_seed) + '_trancos' + ("_" + args.cm if args.cm != "" else ""))
writer = SummaryWriter(log_dir)
set_logger(os.path.join(log_dir, 'train.log'))
args.save_model_path = os.path.join(log_dir, 'base-model')


# CUDA for PyTorch
device = torch.device(args.device)

# Model
model = TRANCOSBaseLineModel().double().to(device)
criterion = args.loss()
from torch.optim.lr_scheduler import StepLR
optimizer = optim.SGD([
                {'params': model.feature.parameters(), 'lr':1e-3},
                {'params': model.fc.parameters(), 'lr': 1e-3}
            ], momentum=0.0)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
print(args.load_model_path)
model.load_model(args.load_model_path)
model.to(device)

logging.info(model)
logging.info(args)


# Generators
training_set = TRANCOSDataset('trainval')
training_generator = data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True, num_workers=20)

validation_set = TRANCOSDataset('test')
validation_generator = data.DataLoader(validation_set, batch_size=args.batch_size, shuffle=False, num_workers=10)

print('Dataloader initiated.')

def run(train_mode=True, epoch=0):
    phase = 'train' if train_mode else 'test'
    if train_mode:
        optimizer.zero_grad()
        scheduler.step()
    mse = AverageMeter()
    mde = AverageMeter()
    mde_ratio = AverageMeter()
    cnt = 0
    iterator = tqdm(enumerate(training_generator if train_mode else validation_generator))
    Xs, y_preds, y_trues = [], [], []
    num_batches = len(training_generator)
    for idx, (local_batch, local_labels, _) in iterator:
        local_batch, local_labels = local_batch.to(device), local_labels.to(device)
        current_step = epoch * num_batches + idx
        y_pred, _ = model(local_batch)
        y_true = local_labels
        loss = criterion(y_pred, y_true)
        if train_mode:
            loss.backward()
            optimizer.step()
        
        mse.update(loss.item(), local_labels.size(0))
        diff = torch.abs(y_pred - y_true)
        mde.update(torch.mean(diff).item(), local_labels.size(0))
        mde_ratio.update(torch.mean(torch.div(diff, y_true)).item(), local_labels.size(0))
        cnt += local_labels.size(0)

        iterator.set_description('%s [%d,%d] (%.1f vs %.1f) mse:%.3e(%.3e), mde:%.3e(%.3e), mde_ratio: %.3e(%.3e)' 
                    % ('Train' if train_mode else 'Val  ', epoch+1, cnt, y_true[0].item(), y_pred[0].item(), \
                        mse.val, mse.avg, mde.val, mde.avg, mde_ratio.val, mde_ratio.avg))
        if idx % 50 == 0:
            logging.info('%s [%d,%d] (%.1f vs %.1f) mse:%.3e(%.3e), mde:%.3e(%.3e), mde_ratio: %.3e(%.3e)' 
                    % ('Train' if train_mode else 'Val  ', epoch+1, cnt, y_true[0].item(), y_pred[0].item(), \
                        mse.val, mse.avg, mde.val, mde.avg, mde_ratio.val, mde_ratio.avg))
            
            writer.add_scalar(phase+'/mse', mse.avg, current_step)
            writer.add_scalar(phase+'/mde', mde.avg, current_step)
            writer.add_scalar(phase+'/mde_ratio', mde_ratio.avg, current_step)

    if train_mode:
        model.save_model(args.save_model_path, device=device)

# Loop over epochs
for epoch in range(args.max_epochs):
    run(True, epoch)
    run(False, epoch)

    
