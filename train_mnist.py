import torch
from torch.utils import data
from mnist_dataset import MNISTDataset
from model import MNISTBaseLineModel
import torch.optim as optim
from tqdm import tqdm
import numpy as np

import argparse
import os

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser()
parser.add_argument('--params', type=str)
parser.add_argument('--save', type=str, default='models/base-model')
parser.add_argument('--validate', type=bool, default=False)
args = parser.parse_args()

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
print('use_cuda: %s' % use_cuda)
device = torch.device("cuda:0" if use_cuda else "cpu")

# Model
model = MNISTBaseLineModel().double().to(device)
criterion = torch.nn.MSELoss()
from torch.optim.lr_scheduler import StepLR
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.0) # optim.Adam(model.parameters())
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
max_epochs = 0

# config

# Generators
training_set = MNISTDataset(10000)
training_generator = data.DataLoader(training_set, **params)

validation_set = MNISTDataset(100)
validation_generator = data.DataLoader(validation_set, **params)

print('Dataloader initiated.')

# Loop over epochs
for epoch in range(max_epochs):

    def run(train_mode=True):
        if train_mode:
            optimizer.zero_grad()
            scheduler.step()
        mse, cnt = 0, 0
        iterator = tqdm(enumerate(training_generator if train_mode else validation_generator))
        Xs, y_preds, y_trues = [], [], []
        for idx, (local_batch, local_labels) in iterator:
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            y_pred, _ = model(local_batch)
            y_true = local_labels.unsqueeze(dim=1)
            loss = criterion(y_pred, y_true)
            if train_mode:
                loss.backward()
                optimizer.step()
            mse += loss.item()
            cnt += local_labels.size(0)
            iterator.set_description('%s [%d,%d] mse:%.3e' % ('Train' if train_mode else 'Val', epoch+1, cnt, mse / cnt))
            Xs.extend(local_batch)
            y_preds.extend(y_pred)
            y_trues.extend(y_true)
        if train_mode:
            torch.save(model.state_dict(), args.save)
        return Xs, y_preds, y_trues

    run(True)
    run(False)

    
