from argparse import Namespace
from pathlib import Path
from model import PointNet
from config import config
from recognizer3d import data, predict, utils

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder

class PointNetLoss(nn.Module):
    def __init__(
        self,
        alpha: float = None,
        gamma: float = 0.0,
        reg_weight: float = 0.0,
        size_average: bool = True,
    ):
        super(PointNetLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reg_weight = reg_weight
        self.size_average = size_average

        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, (list, np.ndarray)):
            self.alpha = torch.Tensor(alpha)

        self.cross_entropy_loss = nn.CrossEntropyLoss(weight=self.alpha)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, A=None):

        bs = predictions.size(0)

        ce_loss = self.cross_entropy_loss(predictions, targets)

        pn = F.softmax(predictions, dim=0)
        pn = pn.gather(1, targets.view(-1, 1)).view(-1)

        # get regularization
        if self.reg_weight:
            I = torch.eye(64, device=A.device).expand(A.shape[0], -1, -1)
            if A.is_cuda:
                I = I.cuda()
            reg = torch.linalg.norm(I - torch.bmm(A, A.transpose(2, 1)))
            reg = self.reg_weight * reg / bs
        else:
            reg = 0

        loss = (1 - pn) ** self.gamma * ce_loss
        if self.size_average:
            return loss.mean() + reg
        else:
            return loss.sum() + reg


def train_test(
    args_fp: str,
    classifier: nn.Module,
    dataloader: DataLoader,
    num_batch: int,
    epoch: int,
    split: str = "train",
) -> (float, float):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = Namespace(**utils.load_dict(filepath=args_fp))

    alpha = np.ones(6)
    gamma = 2

    classifier = classifier.to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=0.0001
    )
    criterion = PointNetLoss(alpha=alpha, gamma=gamma, reg_weight=args.reg_weight).to(device)

    _loss = []
    _accuracy = []

    total_test_targets = []
    total_test_preds = []
    for i, (points, targets) in enumerate(dataloader, 0):

        points = points.transpose(2, 1).to(device)
        targets = targets.squeeze().to(device)

        optimizer.zero_grad()
        preds, _, A = classifier(points)
        loss = criterion(preds, targets, A)

        if split == "train":
            loss.backward()
            optimizer.step()
            scheduler.step()

        pred_choice = predict.custom_predictions(preds, 0.6, 0)
        correct = pred_choice.eq(targets.data).cpu().sum()
        accuracy = correct.item() / float(targets.size(0))

        _loss.append(loss.item())
        _accuracy.append(accuracy)

        if split == "test":
            total_test_targets += targets.reshape(-1).cpu().numpy().tolist()
            total_test_preds += pred_choice.reshape(-1).cpu().numpy().tolist()

        if i % 10 == 0:
            print(
                f"\t [{args.epohcs}: {i}/{num_batch}] "
                + f"{split} loss: {loss.item():.4f} "
                + f"accuracy: {accuracy:.4f}"
            )

    epoch_loss = np.mean(_loss)
    epoch_accuracy = np.mean(_accuracy)

    print(
        f"Epoch: {epoch} - {split} Loss: {epoch_loss:.4f} "
        + f"- {split} Accuracy: {epoch_accuracy:.4f} "
        + f"- {split}"
    )

    if split == "test":
        return epoch_loss, epoch_accuracy, total_test_targets, total_test_preds
    else:
        return epoch_loss, epoch_accuracy

train_dataset, test_dataset, valdation_dataset, encoder = load_process({num_points: 1024}, LabelEncoder())

num_train_batch = int(np.ceil(len(train_dataset) / args.train_batch_size))
num_test_batch = int(np.ceil(len(test_dataset) / args.train_batch_size))
def train_save_model(
    args: str,
    model: torch.nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_train_batch: int,
    num_test_batch: int
) -> None:
    for epoch in range(1, args.num_epochs):

        model = model.train()

        _train_metrics = train_test(
            args, model, train_loader, num_train_batch, epoch, split="train"
        )
        with torch.no_grad():

            model = model.eval()

            _test_metrics = train_test(
                args, model, test_loader, num_test_batch, epoch, split="test"
            )

    return model

def save_model(model: torch.nn.Module) -> None:
    # save model
    torch.save(model.state_dict(), Path(config.MODELS_DIR, "modelv1"))


def start_training():

    args = Namespace(**utils.load_dict(filepath=Path(config.CONFIG_DIR, "args.json")))

    train_loader, test_loader, valid_loader, encoder = data.get_preprocessed(LabelEncoder, args)
    save_encoder(encoder, config.MODELS_DIR, "encoder")
    return train_save_model(args, PointNet(), train_loader, test_loader)
