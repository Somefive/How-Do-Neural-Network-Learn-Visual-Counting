import argparse
import torch

str2ints = lambda l: [int(i) for i in l.split(',')]
str2bool = lambda s: s == 'True' or s == 'true'

def str2loss(s):
    if s == 'mse' or s == 'l2':
        return torch.nn.MSELoss
    elif s == 'mae' or s == 'l1':
        return torch.nn.L1Loss
    elif s == 'smoothl1':
        return torch.nn.SmoothL1Loss
    else:
        return torch.nn.MSELoss

def str2optim(s):
    if s == 'adam' or 'Adam':
        return torch.optim.Adam
    elif s == 'sgd' or 'SGD':
        return torch.optim.SGD
    else:
        return torch.optim.SGD


parser = argparse.ArgumentParser()
parser.add_argument('--load_model_path', type=str, default='models/base-model')
parser.add_argument('--save_model_path', type=str, default='models/base-model')
parser.add_argument('--validate', type=bool, default=False)
parser.add_argument('--grid_size', type=int, default=4)
parser.add_argument('--cm', type=str, default="")
parser.add_argument('--max_num', type=int, default=5)
parser.add_argument('--interf', type=bool, default=False)
parser.add_argument('--classes', type=str2ints, default=[6,8])

parser.add_argument('--train_set_size', type=int, default=10000)
parser.add_argument('--val_set_size', type=int, default=1000)

parser.add_argument('--fig_save_path', type=str, default=None)
parser.add_argument('--figs_count', type=int, default=0)

parser.add_argument('--max_epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--shuffle', type=str2bool, default=True)
parser.add_argument('--num_workers', type=int, default=6)

parser.add_argument('--loss', type=str2loss, default=torch.nn.SmoothL1Loss)
parser.add_argument('--optim', type=str2optim, default=torch.optim.SGD)
parser.add_argument('--lr', type=float, default=1e-4)

parser.add_argument('--device', type=str, default='auto')


args = parser.parse_args()

args.data_generator_params = {
    'batch_size': args.batch_size,
    'shuffle': args.shuffle,
    'num_workers': args.num_workers
}
if args.device == 'auto':
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(args)