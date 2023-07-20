import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from argparse import Namespace
from pathlib import Path
from config import config
from recognizer3d import predict, train, utils, data


train_dataset, test_dataset, valdation_dataset, encoder = load_process({num_points: 1024}, LabelEncoder())

num_train_batch = int(np.ceil(len(train_dataset) / args.train_batch_size))
num_test_batch = int(np.ceil(len(test_dataset) / args.train_batch_size))
def train_metrics(
    args: dict,
    model: torch.nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_train_batch: int,
    num_test_batch: int
) -> (list, list):

    args = Namespace(**utils.load_dict(filepath=Path(config.CONFIG_DIR, "args.json")))

    train_metrics = []
    test_metrics = []

    for epoch in range(1, args.num_epochs):

        model = model.train()

        _train_metrics = train.train_test(
            args, model, train_loader, num_train_batch, epoch, split="train"
        )
        train_metrics.append(_train_metrics)

        with torch.no_grad():

            model = model.eval()

            _test_metrics = train.train_test(
                args, model, test_loader, num_test_batch, epoch, split="test"
            )
            test_metrics.append(_test_metrics)
    return train_metrics, test_metrics


def evaluate(model: torch.nn.Module, loader: DataLoader) -> (list, list):
    predictions = []
    labels = []

    for i, (points, targets) in enumerate(loader, 0):
        with torch.no_grad():
            model.eval().to("cpu")

            points = points.transpose(2, 1)
            targets = targets.squeeze()

            preds, _, _ = model(points)
            pred_choice = predict.custom_predictions(preds, 0.6, 0)

            predictions.extend(pred_choice)
            labels.extend(targets.data)

    return predictions, labels


def get_classification_report(
    model: torch.nn.Module, loader: DataLoader, encoder: LabelEncoder
) -> dict:
    '''
    This function returns a classification report for a given model and data loader.

    Args:
        model (torch.nn.Module): The trained model.
        loader (torch.utils.data.DataLoader): The data loader.
        encoder (LabelEncoder): The label encoder.

    Returns:
        dict: The classification report.
    '''
    predictions, labels = evaluate(model, loader)
    return classification_report(
        utils.get_labels(encoder, labels), utils.get_labels(encoder, predictions)
    )


def get_confusion_matrix(
    model: torch.nn.Module, loader: torch.utils.data.DataLoader, encoder: LabelEncoder
) -> dict:
    """Calculate the confusion matrix, accuracy, precision, recall, and F1-score for a given model and data loader.

   Args:
       model (torch.nn.Module): The trained model.
       loader (torch.utils.data.DataLoader): The data loader.
       encoder (LabelEncoder): The label encoder.

   Returns:
       dict: A dictionary containing the confusion matrix, accuracy, precision, recall, and F1-score.

   Raises:
       None
   """
    logging.info('Starting get_confusion_matrix function')
    predictions, labels = evaluate(model, loader)
    logging.info('Evaluating predictions and labels')
    cm = confusion_matrix(utils.get_labels(encoder, labels), utils.get_labels(encoder, predictions))
    acc = accuracy_score(labels, predictions)
    prec = precision_score(labels, predictions, average='weighted')
    rec = recall_score(labels, predictions, average='weighted')
    f1 = f1_score(labels, predictions, average='weighted')
    logging.info('Returning confusion matrix and metrics')
    return {
        'confusion_matrix': pd.DataFrame(columns=encoder.classes_, index=encoder.classes_, data=cm),
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1
    }