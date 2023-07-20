from typing import Union
import os
from pathlib import Path
import pickle
import warnings
from argparse import Namespace

import numpy as np
import pandas as pd
import torch
import trimesh
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset

from config import config
from recognizer3d import utils


class PointCloudDataset(Dataset):
    """Improved version of torch-like dataset
    for cloud of points task"""

    def __init__(self, df: pd.DataFrame, num_points: int) -> None:
        self.df = df
        self.num_points = num_points

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor):
        file_path = self.df["filename"][idx]
        label = self.df["label"][idx]

        try:
            mesh = trimesh.load(file_path)
            points = mesh.vertices
        except Exception as e:
            print(f'Error loading file {file_path}: {e}')
            points = np.zeros((self.num_points, 3))
            label = -1

        points, label = self.downsample(points, label)
        points = torch.from_numpy(points).type(torch.float32)
        label = torch.tensor(label).type(torch.LongTensor)

        return points, label

    def downsample(
        self, points: torch.Tensor, targets: torch.Tensor
    ) -> (torch.Tensor, torch.Tensor):
        if len(points) > self.num_points:
            choice = torch.randperm(len(points))[:self.num_points]
        else:
            choice = torch.randint(len(points), (self.num_points,), dtype=torch.long)
        points = points[choice, :]
        targets = targets[choice]

        return points, targets

    def normalize(self, points: torch.Tensor) -> torch.Tensor:
        return (points - torch.mean(points, dim=0)) / torch.std(points, dim=0)


def collect_df(dir_path: str, split_name: str) -> pd.DataFrame:
    """Build dataframe with src to cloud of points,
    and it's label based on split you choose"""
    df = pd.DataFrame(columns=['filename', 'label'])

    for root, dirs, files in os.walk(dir_path):
        if split_name in dirs:
            split_path = os.path.join(root, split_name)
            file_data = [
                {"filename": str(file_path), "label": file_path.stem}
                for file_path in Path(split_path).glob("*.obj")
            ]
            df = df.append(file_data, ignore_index=True)

    return df


def load_data() -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    load data for each split
    """
    train = collect_df(config.DATA_DIR, "train")
    test = collect_df(config.DATA_DIR, "test")
    valid = collect_df(config.DATA_DIR, "valid")

    return train, test, valid


def save_encoder(encoder: LabelEncoder, dir: str, encoder_name: str) -> Union[None, str]:
    """Save the encoder in pickle format and return the file path"""
    dir_path = Path(dir)
    if not dir_path.exists():
        dir_path.mkdir(parents=True)
    file_path = dir_path / encoder_name
    with open(file_path, "wb") as f:
        pickle.dump(encoder, f)
    return str(file_path)


def create_datasets(data: dict, num_points: int) -> PointCloudDataset:
    return PointCloudDataset(data, num_points)

def create_dataloaders(dataset: PointCloudDataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
def load_process(args: dict, encoder: LabelEncoder) -> (Dataset, Dataset, Dataset):
    """
    transform labels to nums and fits data into trainable torch objects
    :return: DataLoaders for each split and fitted encoder
    """

    train, test, valid = load_data()
    # encode target
    encoder.fit(train["label"])

    train["label"] = encoder.transform(train["label"])
    test["label"] = encoder.transform(test["label"])
    valid["label"] = encoder.transform(valid["label"])

    # to torch-like Dataset
    train = create_datasets(train, args.num_points)
    test = create_datasets(test, args.num_points)
    valid = create_datasets(valid, args.num_points)

    return train, test, valid, encoder

def get_preprocessed(encoder: LabelEncoder, args: dict) -> (DataLoader, DataLoader, DataLoader, LabelEncoder):

    train_dataset, test_dataset, valid_dataset, encoder = load_process(args, encoder)

    # to torch-trainable Dataloader
    train_loader = create_dataloaders(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    test_loader = create_dataloaders(test_dataset, batch_size=args.train_batch_size, shuffle=False)
    valid_loader = create_dataloaders(valid_dataset, batch_size=args.valid_batch_size, shuffle=False)

    return train_loader, test_loader, valid_loader, encoder



def downsample(points: np.ndarray, num_pointns: int) -> np.ndarray:
    """
    Sample points to correct amount for model from whole cloud of points.
    
    Args:
        points (np.ndarray): The cloud of points to sample from.
        num_points (int): The number of points to sample.
    
    Returns:
        np.ndarray: The downsampled points.
    """
    if len(points) > num_pointns:
        choice = np.random.permutation(len(points))[:num_pointns]
    else:
        choice = np.random.choice(len(points), num_pointns, replace=True)
    return points[choice, :]


def points_transform(points: np.ndarray, num_points: int) -> torch.Tensor:
    """
    Downsample and normalize points for prediction.
    
    Args:
        points (np.ndarray): Array of points to be transformed.
        num_points (int): Number of points to downsample to.
    
    Returns:
        torch.Tensor: Transformed points as a tensor.
    """
    points = downsample(points, num_points)
    points = (points - np.mean(points, axis=0)) / np.std(points, axis=0)
    points = torch.from_numpy(points).type(torch.float32)
    points = points.transpose(-2, 1)
    points = points.view([1, 3, num_points])
    return points