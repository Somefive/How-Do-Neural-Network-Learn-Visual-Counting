import torch.nn as nn
import torch.nn.functional as F
import torch

class MNISTBaseLineModel(nn.Module):
    def __init__(self, size=84, cls=3):
        super(MNISTBaseLineModel, self).__init__()

        self.cls = cls

        self.conv1 = nn.Conv2d(1, 16, 5, padding=2) # >> 16, 84, 84
        self.conv2 = nn.Conv2d(16, 16, 5, padding=2) # >> 16, 84, 84
        self.conv3 = nn.Conv2d(16, 16, 3, padding=1, stride=2) # >> 16, 42, 42
        self.conv4 = nn.Conv2d(16, 16, 5, padding=2) # >> 16, 42, 42
        self.conv5 = nn.Conv2d(16, cls, 5, padding=2) # >> cls, 42, 42
        self.pool = nn.AvgPool2d(size // 2, size // 2) # << 1, 1, 1
        # self.fc = nn.Linear(4 * 32 * 32, 1)

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1) # << 1, 64, 64
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        h = F.relu(self.conv5(x))
        o = self.pool(h).view(-1, self.cls)
        # o = torch.sigmoid(o)
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

class TRANCOSBaseLineModel(nn.Module):
    def __init__(self):
        super(TRANCOSBaseLineModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5, padding=2)
        self.pool1 = nn.MaxPool2d(8, 8) # << 16, 60, 80
        self.conv2 = nn.Conv2d(16, 64, 5, padding=2)
        self.pool2 = nn.MaxPool2d(4, 4) # << 64, 15, 20
        self.conv3 = nn.Conv2d(64, 1, 5, padding=2) # << 1, 15, 20
        self.fc = nn.Linear(15 * 20, 1)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.fc(self.conv3(x).view(-1, 15 * 20))
        x = torch.sigmoid(x) * 15 * 20
        return x
