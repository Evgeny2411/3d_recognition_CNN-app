import pickle
from argparse import Namespace
from pathlib import Path
from config import config
from recognizer3d import data, utils

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


args = Namespace(**utils.load_dict(filepath=Path(config.CONFIG_DIR, "args.json")))
class Tnet(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.max_pool = nn.MaxPool1d(kernel_size=1)
        self.flatten = nn.Flatten(start_dim=1)

    def forward(self, input: torch.Tensor):
        bs = input.size(0)
        output = F.relu(self.bn1(self.conv1(input)))
        output = F.relu(self.bn2(self.conv2(output)))
        output = F.relu(self.bn3(self.conv3(output)))
        pool = self.max_pool(output)
        flat = self.flatten(pool)
        output = F.relu(self.bn4(self.fc1(flat)))
        output = F.relu(self.bn5(self.fc2(output)))

        init = torch.eye(self.k, requires_grad=True).repeat(bs, 1, 1)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        init = init.to(device)
        matrix = self.fc3(output).view(-1, self.k, self.k) + init
        return matrix


class PointNetBackbone(nn.Module):
    def __init__(self, num_points: int = args.num_points, num_feats: int = args.num_points):

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

    def forward(self, x: torch.Tensor):

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
    def __init__(self, classes: int = args.num_classes):
        super().__init__()
        self.transform = PointNetBackbone()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, classes)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, input: torch.Tensor):
        x, crit_idxs, A_feat = self.transform(input)
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x, crit_idxs, A_feat


class Model:
    def __init__(self, args_fp):
        self.__model = PointNet()
        self.__model.load_state_dict(torch.load(Path(config.MODELS_DIR, "modelv1")))
        self.__encoder = self.__load_encoder()
        self.__num_points = args.num_points

    def __load_encoder(self):
        with open(Path(config.MODELS_DIR, "encoder.pkl"), "rb") as f:
            return pickle.load(f)

    def create_prediction_df(selfs, probabilities):
        return pd.DataFrame(
            index=self.__encoder.classes_,
            columns=["Probs"],
            data=torch.softmax(preds, dim=1)[0],
        )

    def predict_probs(self, points: int):
        with torch.no_grad():
            self.__model.eval()
            points = data.points_transform(points, self.__num_points)
            preds, _, _ = self.__model(points)
            probabilities = create_prediction_df(preds)
        return probabilities

    def make_prediction(self, points: int) -> pd.DataFrame:
        prediction_df = self.predict_probs(points=points)
        prediction = prediction_df.sort_values(by="Probs", ascending=False)
        return prediction