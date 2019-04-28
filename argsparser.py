import argparse

str2ints = lambda l: [int(i) for i in l.split(',')]

parser = argparse.ArgumentParser()
parser.add_argument('--params', type=str, default='models/base-model')
parser.add_argument('--save', type=str, default='models/base-model')
parser.add_argument('--validate', type=bool, default=False)
parser.add_argument('--grid_size', type=int, default=4)
parser.add_argument('--cm', type=str, default="")
parser.add_argument('--target', type=str, default="6,8")
parser.add_argument('--max_num', type=int, default=5)
parser.add_argument('--interf', type=bool, default=False)
parser.add_argument('--classes', type=str2ints, default=[1,2,3])

args = parser.parse_args()
print(args)