import torch
from torch.utils import data
from mnist_dataset import MNISTDataset  
from model import MNISTBaseLineModel
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from tensorboardX import SummaryWriter
import argparse
import os
import random
from utils import *

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser()
parser.add_argument('--params', type=str)
parser.add_argument('--save', type=str, default='models/base-model')
parser.add_argument('--validate', type=bool, default=False)
parser.add_argument('--grid_size', type=int, default=4)
parser.add_argument('--cm', type=str, default="")
parser.add_argument('--target', type=str, default="6,8")
parser.add_argument('--max_num', type=int, default=5)
parser.add_argument('--interf', type=bool, default=False)

args = parser.parse_args()
print(args)

rand_seed=0
np.random.seed(rand_seed)
random.seed(rand_seed)
torch.manual_seed(rand_seed)
torch.cuda.manual_seed_all(rand_seed)

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
print('use_cuda: %s' % use_cuda)
device = torch.device("cuda" if use_cuda else "cpu")

target = [int(x) for x in args.target.split(',')]
print(target)

# Model
model = MNISTBaseLineModel(size=args.grid_size * 28, cls=len(target)).double().to(device)
criterion = torch.nn.SmoothL1Loss()
# criterion = torch.nn.MSELoss()
from torch.optim.lr_scheduler import StepLR
optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.0) # optim.Adam(model.parameters())
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
if args.params and os.path.exists(args.params):
  print('loading parameters from %s' % args.params)
  model.load_state_dict(torch.load(args.params))
  model.eval()
  print('parameter loaded')
print(model)

# Parameters
params = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 6}
max_epochs = 50

# config

# Generators
training_set = MNISTDataset(10000, grid_size=args.grid_size, max_num=args.max_num, target=target, interference=args.interf)
training_generator = data.DataLoader(training_set, **params)

validation_set = MNISTDataset(10, grid_size=args.grid_size, max_num=args.max_num, target=target, interference=args.interf)
validation_generator = data.DataLoader(validation_set, **params)

print('Dataloader initiated.')
writer = SummaryWriter('runs/'+ time_for_file() + '_seed' + str(rand_seed) + '_mnist' + ("_" + args.cm if args.cm != "" else ""))

def run(train_mode=True, epoch=0):
    phase = 'train' if train_mode else 'test'
    if train_mode:
        optimizer.zero_grad()
        scheduler.step()
    mse = AverageMeter()
    mde = AverageMeter()
    dos = AverageMeter()
    cnt = 0
    iterator = tqdm(enumerate(training_generator if train_mode else validation_generator))
    Xs, y_preds, y_trues = [], [], []
    num_batches = len(training_generator)
    for idx, (local_batch, local_labels) in iterator:
        current_step = epoch * num_batches + idx
        local_batch, local_labels = local_batch.to(device), local_labels.to(device)
        y_pred, _ = model(local_batch)
        y_true = local_labels

        loss = criterion(y_pred, y_true)
        if train_mode:
            loss.backward()
            optimizer.step()

        mse.update(loss.item(), local_labels.size(0))
        diff = torch.abs(y_pred - y_true)
        sum = y_pred + y_true
        mde.update(torch.mean(torch.sum(diff, 1)).item(),local_labels.size(0))
        dos.update(torch.mean(torch.div(diff, sum+1e-8)).item(), local_labels.size(0))
        cnt += local_labels.size(0)

        if len(target) == 1:
            iterator.set_description('%s [%d,%d] (%.1f vs %.1f) mse:%.3e(%.3e), mde:%.3e(%.3e), dos: %.3e(%.3e)' 
                % ('Train' if train_mode else 'Val  ', epoch+1, cnt, y_true[0].item(), y_pred[0].item(), \
                    mse.val, mse.avg, mde.val, mde.avg, dos.val, dos.avg))
        else:
            iterator.set_description('%s [%d,%d] (%.1f vs %.1f) mse:%.3e(%.3e), mde:%.3e(%.3e), dos: %.3e(%.3e)' 
                % ('Train' if train_mode else 'Val  ', epoch+1, cnt, y_true[0][0].item(), y_pred[0][0].item(), \
                    mse.val, mse.avg, mde.val, mde.avg, dos.val, dos.avg))
        if idx % 100 == 0:
            writer.add_scalar(phase+'/mse', mse.avg, current_step)
            writer.add_scalar(phase+'/mde', mde.avg, current_step)
            writer.add_scalar(phase+'/dos', dos.avg, current_step)

    if train_mode:
        torch.save(model.state_dict(), args.save)
    return Xs, y_preds, y_trues

# Loop over epochs
for epoch in range(max_epochs):
    run(True, epoch)
    run(False, epoch)

    
