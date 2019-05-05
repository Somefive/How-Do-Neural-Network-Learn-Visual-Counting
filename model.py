import torch.nn as nn
import torch.nn.functional as F
import torch
import os

class AutoLoadSaveModel(nn.Module):

    def load_model(self, load_model_path):
        print(load_model_path)
        print(os.path.exists(load_model_path), load_model_path)
        if load_model_path and os.path.exists(load_model_path):
            self.load_state_dict(torch.load(load_model_path))
            self.eval()
            print('model parameter loaded from %s' % load_model_path)

    def save_model(self, save_model_path, device='cpu'):
        self.to('cpu')
        torch.save(self.state_dict(), save_model_path)
        self.to(device)

class MNISTBaseLineModel(AutoLoadSaveModel):
    def __init__(self, size=84, cls=3, filter_size=16):
        super(MNISTBaseLineModel, self).__init__()

        self.cls = cls

        self.conv1 = nn.Conv2d(1, filter_size, 5, padding=2) # >> 16, 84, 84
        self.conv2 = nn.Conv2d(filter_size, filter_size, 5, padding=2) # >> 16, 84, 84
        self.conv3 = nn.Conv2d(filter_size, filter_size, 3, padding=1, stride=2) # >> 16, 42, 42
        self.conv4 = nn.Conv2d(filter_size, filter_size, 5, padding=2) # >> 16, 42, 42
        self.conv5 = nn.Conv2d(filter_size, cls, 5, padding=2) # >> cls, 42, 42
        self.pool = nn.AvgPool2d(size // 2, size // 2) # >> cls, 1, 1

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1) # << 1, 64, 64
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        h = F.relu(self.conv5(x))
        o = self.pool(h).view(-1, self.cls)
        return o, h

class CircleBaseLineModel(nn.Module):
    def __init__(self):
        super(CircleBaseLineModel, self).__init__()
        # self.conv = nn.Conv2d(1, 4, 3, padding=1) # >> 1, 64, 64
        # self.pool = nn.MaxPool2d(2, 2) # << 4, 32, 32
        self.fc = nn.Linear(4 * 32 * 32, 1)

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1) # << 1, 64, 64
        # h = self.pool(F.relu(self.conv(x)))
        o = self.fc(x.view(-1, 4 * 32 * 32))
        # o = torch.sigmoid(o)
        return o

class DotBaseLineModel(nn.Module):
    def __init__(self):
        super(DotBaseLineModel, self).__init__()
        # self.conv = nn.Conv2d(1, 4, 3, padding=1) # >> 1, 64, 64
        # self.pool = nn.MaxPool2d(2, 2) # << 4, 32, 32
        self.fc = nn.Linear(16 * 16, 1)

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1) # << 1, 64, 64
        # h = self.pool(F.relu(self.conv(x)))
        o = self.fc(x.view(-1, 16 * 16))
        # o = torch.sigmoid(o)
        return o

class TRANCOSBaseLineModel(AutoLoadSaveModel):
    def __init__(self, filter_size=16):
        super(TRANCOSBaseLineModel, self).__init__()
        self.pool0 = nn.AvgPool2d(4, 4)
        self.conv1 = nn.Conv2d(3, filter_size, 5, padding=2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(filter_size, filter_size, 5, padding=2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(filter_size, filter_size, 5, padding=2)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(filter_size, 1, 5, padding=2)
        #self.fc = nn.Linear(160*120*3,1)

    def forward(self, x):
        b = x.size(0)
        x = self.pool0(x)
        x = self.pool1(F.relu(self.conv1(x)))
        #x = self.pool2(F.relu(self.conv2(x)))
        #x = self.pool3(F.relu(self.conv3(x)))
        #x = self.conv4(x)
        #x = self.fc(self.pool0(x).view(b, -1))
        #return x, x
        return torch.mean(x.view(b, -1), dim=1, keepdim=True), x


class TRANCOSModel1(AutoLoadSaveModel):
    def __init__(self, bn=False, filter_size=16):
        super(TRANCOSModel1, self).__init__()
        # 3 x 480 x 640
        self.feature = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=2, padding=3),  # 32 x 240 x 320
            nn.ReLU(),
            nn.MaxPool2d(4, 4),
            nn.Conv2d(32, 64, 3, padding=1), # << 64 x 60 x 80
            nn.MaxPool2d(2, 2), # << 64 x 30 x 40
            nn.Conv2d(64, 64, 3, padding=1), # << 64 x 30 x 40
            nn.MaxPool2d(2, 2), # << 64 x 15 x 20
            nn.Conv2d(64, 1, 3, padding=1), # << 1, 15, 20
        )
        # self.conv1 = nn.Conv2d(3, 32, 7, stride=2, padding=3)  # 32 x 240 x 320
        # self.pool1 = nn.MaxPool2d(4, 4) # << 32 x 60 x 80
        # self.conv2 = nn.Conv2d(32, 64, 3, padding=1) # << 64 x 60 x 80
        # self.pool2 = nn.MaxPool2d(2, 2) # << 64 x 30 x 40
        # self.conv3 = nn.Conv2d(64, 64, 3, padding=1) # << 64 x 30 x 40
        # self.pool3 = nn.MaxPool2d(2, 2) # << 64 x 15 x 20
        # self.conv4 = nn.Conv2d(64, 1, 3, padding=1) # << 1, 15, 20

        self.fc = nn.Sequential(
            nn.Linear(15 * 20, 100),
            nn.Linear(100, 1),
        )
        # self.fc1 = nn.Linear(15 * 20, 100)
        # self.fc2 = nn.Linear(100, 1)

    def forward(self, x):
        # x = self.pool1(F.relu(self.conv1(x)))
        # x = self.pool2(F.relu(self.conv2(x)))
        # x = self.pool3(F.relu(self.conv3(x)))
        # x = self.conv4(x).view(-1, 15 * 20)
        x = self.feature(x).view(-1, 15*20)
        # x = self.fc1(x)
        # x = self.fc2(x)
        y = self.fc(x)
        return y, x
