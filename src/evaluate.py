import torch
import numpy as np
from torch.utils.data import DataLoader
from .utils import compute_all_metrics, print_metrics_report
from .dataset import AudioDataset


@torch.no_grad()
def evaluate_model(model, data_loader, device):
    model.eval()
    all_labels = []
    all_scores = []
    all_preds = []

    for batch in data_loader:
        if isinstance(batch, (list, tuple)):
            features, labels = batch
            features = features.to(device)
            labels = labels.to(device)
        else:
            features = batch.to(device)
            labels = torch.zeros(1, dtype=torch.long)

        outputs = model(features)
        scores = torch.softmax(outputs, dim=-1)[:, 1]

        all_labels.append(labels.cpu().numpy())
        all_scores.append(scores.cpu().numpy())
        all_preds.append(outputs.argmax(dim=-1).cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_scores = np.concatenate(all_scores)
    y_pred = np.concatenate(all_preds)

    return compute_all_metrics(y_true, y_pred, y_scores)


@torch.no_grad()
def get_predictions(model, data_loader, device):
    model.eval()
    all_labels = []
    all_scores = []
    all_preds = []

    for batch in data_loader:
        if isinstance(batch, (list, tuple)):
            features, labels = batch
            features = features.to(device)
            labels = labels.to(device)
        else:
            features = batch.to(device)
            labels = torch.zeros(1, dtype=torch.long)

        outputs = model(features)
        scores = torch.softmax(outputs, dim=-1)[:, 1]

        all_labels.append(labels.cpu().numpy())
        all_scores.append(scores.cpu().numpy())
        all_preds.append(outputs.argmax(dim=-1).cpu().numpy())

    return (
        np.concatenate(all_labels),
        np.concatenate(all_scores),
        np.concatenate(all_preds)
    )
