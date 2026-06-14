import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import pickle
import json

from .dataset import AudioDataset
from .evaluate import evaluate_model


def train_model(
    model, train_loader, val_loader, device,
    epochs=50, lr=3e-4, weight_decay=1e-4,
    patience=10, checkpoint_dir='trained_models',
    model_name='model', class_weights=None
):
    os.makedirs(checkpoint_dir, exist_ok=True)

    if class_weights is not None:
        cw = class_weights.to(device)
    elif hasattr(train_loader.dataset, 'class_weights'):
        cw = train_loader.dataset.class_weights.to(device)
    elif hasattr(train_loader.dataset, 'dataset') and hasattr(train_loader.dataset.dataset, 'class_weights'):
        cw = train_loader.dataset.dataset.class_weights.to(device)
    else:
        cw = None
    criterion = nn.CrossEntropyLoss(weight=cw)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_eer = float('inf')
    best_epoch = 0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_eer': [], 'val_acc': []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for features, labels in pbar:
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)

        metrics = evaluate_model(model, val_loader, device)
        history['val_eer'].append(metrics['eer'])
        history['val_acc'].append(metrics['accuracy'])

        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, "
              f"Val EER={metrics['eer']:.2f}%, Val Acc={metrics['accuracy']:.2f}%")

        scheduler.step(avg_val_loss)

        if metrics['eer'] < best_eer:
            best_eer = metrics['eer']
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'{model_name}_best.pth'))
            print(f"  -> New best model saved (EER={best_eer:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"\nTraining complete. Best EER: {best_eer:.2f}% at epoch {best_epoch}")

    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, f'{model_name}_best.pth')))
    return model, history, best_eer


def save_scaler(scaler, path):
    with open(path, 'wb') as f:
        pickle.dump(scaler, f)


def load_scaler(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_model_config(config, path):
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)


def load_model_config(path):
    with open(path, 'r') as f:
        return json.load(f)
