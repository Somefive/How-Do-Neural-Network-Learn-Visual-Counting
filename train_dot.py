import torch
from torch.utils import data
from dot_dataset import DotDataset
from model import DotBaseLineModel
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

# Model
model = DotBaseLineModel().double()
criterion = torch.nn.MSELoss()
from torch.optim.lr_scheduler import StepLR
optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.0) # optim.Adam(model.parameters())
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
if args.params and os.path.exists(args.params):
  print('loading parameters from %s' % args.params)
  model.load_state_dict(torch.load(args.params))
  model.eval()
  print('parameter loaded')
print(model)

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
print('use_cuda: %s' % use_cuda)
device = torch.device("cuda:0" if use_cuda else "cpu")

# Parameters
params = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 6}
max_epochs = 50

# config
min_dots, max_dots = 20, 200

# Generators
training_set = DotDataset(10000, min_dots=min_dots, max_dots=max_dots, image_size=16)
training_generator = data.DataLoader(training_set, **params)

validation_set = DotDataset(100, min_dots=min_dots, max_dots=max_dots, image_size=16)
validation_generator = data.DataLoader(validation_set, **params)

print('Dataloader initiated.')

# model.fc.load_state_dict({'weight': torch.Tensor(np.ones((1, 16 * 16)) / 180).double(), 'bias': torch.Tensor([-20/180]).double()})

xs, yps, yts = [], [], []
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
            y_pred = model(local_batch)
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
        return Xs, y_preds, y_trues

    run(True)
    xs, yps, yts = run(False)

    