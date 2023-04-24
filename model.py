import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
import pandas as pd
class Tnet(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.k=k
        self.conv1 = nn.Conv1d(k,64,1)
        self.conv2 = nn.Conv1d(64,128,1)
        self.conv3 = nn.Conv1d(128,1024,1)
        self.fc1 = nn.Linear(1024,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,k*k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, input):
        bs = input.size(0)
        xb = F.relu(self.bn1(self.conv1(input)))
        xb = F.relu(self.bn2(self.conv2(xb)))
        xb = F.relu(self.bn3(self.conv3(xb)))
        pool = nn.MaxPool1d(xb.size(-1))(xb)
        flat = nn.Flatten(1)(pool)
        xb = F.relu(self.bn4(self.fc1(flat)))
        xb = F.relu(self.bn5(self.fc2(xb)))

        init = torch.eye(self.k, requires_grad=True).repeat(bs,1,1)
        if xb.is_cuda:
            init=init.cuda()
        matrix = self.fc3(xb).view(-1,self.k,self.k) + init
        return matrix

class PointNetBackbone(nn.Module):
    def __init__(self, num_points=1024, num_feats=1024):

        super(PointNetBackbone, self).__init__()

        self.num_points = num_points
        self.num_feats = num_feats

        # Spatial Transformer Networks (T-nets)
        self.tnet1 = Tnet(3)
        self.tnet2 = Tnet(64)

        # shared MLP 1
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1)

        # shared MLP 2
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv5 = nn.Conv1d(128, self.num_feats, kernel_size=1)

        # batch norms for both shared MLPs
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(self.num_feats)

        # max pool to get the global features
        self.max_pool = nn.MaxPool1d(kernel_size=num_points, return_indices=True)


    def forward(self, x):

        bs = x.shape[0]

        # pass through first Tnet to get transform matrix
        A_input = self.tnet1(x)

        # perform first transformation across each point in the batch
        x = torch.bmm(x.transpose(2, 1), A_input).transpose(2, 1)

        # pass through first shared MLP
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))

        # get feature transform
        A_feat = self.tnet2(x)

        # perform second transformation across each (64 dim) feature in the batch
        x = torch.bmm(x.transpose(2, 1), A_feat).transpose(2, 1)

        # pass through second MLP
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.bn4(F.relu(self.conv4(x)))
        x = self.bn5(F.relu(self.conv5(x)))

        # get feature vector and critical indexes
        features, critical_indexes = self.max_pool(x)
        features = features.view(bs, -1)
        critical_indexes = critical_indexes.view(bs, -1)

        return features, critical_indexes, A_feat
class PointNet(nn.Module):

    def __init__(self, classes = 6):
        super().__init__()
        self.transform = PointNetBackbone()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, classes)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, input):
        x, crit_idxs, A_feat = self.transform(input)
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x, crit_idxs, A_feat


class Model():
    def __init__(self):
        self.__predictioner = PointNet()
        self.__predictioner.load_state_dict(torch.load('models/modelv1'))
        self.__encoder = self.__load_encoder()
        self.__num_points = 1024
    def __load_encoder(self):
        with open('models/encoder.pkl', 'rb') as f:
            return pickle.load(f)

    def __data_transform(self, points):
        points = self.__downsample(points)
        points = (points - np.mean(points, axis=0)) / np.std(points, axis=0)
        points = torch.from_numpy(points).type(torch.float32)
        points = points.transpose(-2,1)
        points = points.view([1, 3, 1024])
        return points

    def __downsample(self, points):
        if len(points) > self.__num_points:
            choice = np.random.choice(len(points), self.__num_points, replace=False)
        else:
            choice = np.random.choice(len(points), self.__num_points, replace=True)
        points = points[choice, :]

        return points
    def prediction(self, points):
        with torch.no_grad():
            self.__predictioner.eval()
            points = self.__data_transform(points)
            preds, _, A = self.__predictioner(points)
            probabilities = pd.DataFrame(index = self.__encoder.classes_, columns = ['Probs'], data = torch.softmax(preds, dim=1)[0])
        return probabilities

#%%
