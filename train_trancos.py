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

parser = argparse.ArgumentParser()
parser.add_argument('--params', type=str)
parser.add_argument('--save', type=str, default='models/base-model')
parser.add_argument('--validate', type=bool, default=False)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--cm', type=str, default="")
parser.add_argument('--lr', type=float, default=1e-3)

args = parser.parse_args()
print(args)

rand_seed=0
np.random.seed(rand_seed)
random.seed(rand_seed)
torch.manual_seed(rand_seed)
torch.cuda.manual_seed_all(rand_seed)

# dataset   = {'mnist-train'     : MNISTDataset,
#              'mnist-test'      : MNISTDataset,
#              'trancos-train'   : TRANCOSDataset,
#              'trancos-test'    : TRANCOSDataset,
#         #    'dot'   :DotDataset, 
#             }
# procedure = { 'mnist'    : MNISTBaseLineModel,
#               'trancos'  : TRANCOSModel,
#             }


# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
print('use_cuda: %s' % use_cuda)
device = torch.device("cuda:0" if use_cuda else "cpu")

# Model
# model = TRANCOSBaseLineModel().double().to(device)
model = TRANCOSModel1().double().to(device)


# model = MNISTBaseLineModel(size=args.grid_size * 28).double().to(device)
# criterion = torch.nn.MSELoss()
criterion = torch.nn.SmoothL1Loss()
from torch.optim.lr_scheduler import StepLR
# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.0) 
optimizer = optim.SGD([
                {'params': model.feature.parameters(), 'lr':1e-3},
                {'params': model.fc.parameters(), 'lr': 1e-3}
            ], momentum=0.0)
# optimizer = optim.Adam(model.parameters(),  lr=args.lr, momentum=0.0)
scheduler = StepLR(optimizer, step_size=15, gamma=0.5)
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
max_epochs = 90

# config

# Generators
training_set = TRANCOSDataset('trainval')
training_generator = data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True, num_workers=20)

validation_set = TRANCOSDataset('test')
validation_generator = data.DataLoader(validation_set, batch_size=args.batch_size, shuffle=False, num_workers=10)

print('Dataloader initiated.')
writer = SummaryWriter('runs/'+ time_for_file() + '_seed' + str(rand_seed) + '_trancos' + ("_" + args.cm if args.cm != "" else ""))

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
    for idx, (local_batch, local_labels) in iterator:
        local_batch, local_labels = local_batch.to(device), local_labels.to(device)
        current_step = epoch * num_batches + idx
        y_pred = model(local_batch)
        y_true = local_labels.unsqueeze(dim=1)
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
        # Xs.extend(local_batch)
        # y_preds.extend(y_pred)
        # y_trues.extend(y_true)
        writer.add_scalar(phase+'/mse', mse.avg, current_step)
        writer.add_scalar(phase+'/mde', mde.avg, current_step)
        writer.add_scalar(phase+'/mde_ratio', mde_ratio.avg, current_step)
    if train_mode:
        torch.save(model.state_dict(), args.save)
    # return Xs, y_preds, y_trues

# Loop over epochs
for epoch in range(max_epochs):
    run(True, epoch)
    run(False, epoch)

    
