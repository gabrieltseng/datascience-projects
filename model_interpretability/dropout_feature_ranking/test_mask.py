from get_mask import get_mask, train_val_test_split, preprocess_data as org_preprocessor
from data import PhysioNetDataset
from models import PhysioNet

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from tqdm import tqdm
import itertools
from pathlib import Path
import os
import pandas as pd
import numpy as np
import pickle
import argparse
from sklearn.metrics import roc_auc_score


def train(model, model_path, train_dataset, val_dataset, optimizer, patience, num_epochs=None,
          batch_size=64):
    """ Training loop
        """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    patience_counter = 0
    best_loss = np.inf

    if num_epochs:
        epoch_counter = range(1, num_epochs + 1)
    else:
        epoch_counter = itertools.count(start=1)

    for epoch in epoch_counter:
        model.train()
        train_loss = 0
        batch = tqdm(train_loader, total=len(train_loader),
                     desc='Epoch {:03d}'.format(epoch))

        for inputs, targets in batch:
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = F.binary_cross_entropy(outputs, targets.unsqueeze(1))
            loss.backward()
            train_loss += loss.data.item()
            optimizer.step()
        model.eval()
        val_loss = 0
        predictions_list, targets_list = [], []
        for inputs, targets in val_loader:
            outputs = model(inputs)
            val_loss += F.binary_cross_entropy(outputs, targets.unsqueeze(1)).item()
            predictions_list.extend(outputs.data.tolist())
            targets_list.extend(targets.data.tolist())

        val_auc = roc_auc_score(targets_list, predictions_list)
        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        print('loss: {:.6g}, val_loss: {:.6g}, val_auc: {:.6g}'.format(train_loss, val_loss, val_auc))
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            print('Saving new best model')
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
        if patience_counter == patience:
            print('Early stopping - best_loss: {:.6g}'.format(best_loss))
            break

    # Load the weights of the best model
    model.load_state_dict(torch.load(model_path))
    return model


def predict(model, predict_dataset):
    """Predict loop (just calculates the prediction loss)
    """
    predict_loader = DataLoader(predict_dataset, batch_size=64)
    loss = 0

    model.eval()
    output_list, target_list = [], []
    for inputs, targets in predict_loader:
        output = model(inputs)
        loss += F.binary_cross_entropy(output, targets.unsqueeze(1)).data.item()
        output_list.extend(output.tolist())
        target_list.extend(targets.tolist())
    loss /= len(predict_loader)
    pred_roc = roc_auc_score(target_list, output_list)
    return loss, pred_roc


def preprocess_data(data_path, features, masking_features):
    # delete all files
    k = len(features)
    for file in ['physio_input.npy', 'physio_outcomes.npy',
                 'physio_normalizing_dict.pkl']:
        try: os.remove(data_path/file)
        except FileNotFoundError: continue

    if not data_path.exists():
        data_path.mkdir()

    # get the original inputs
    org_data_path = data_path.parents[0]
    # check the input data exists; if it doesn't, generate it
    for file in ['physio_input.npy', 'physio_outcomes.npy',
                 'physio_normalizing_dict.pkl']:
        if not (org_data_path/file).exists():
            print(f'Missing {org_data_path/file}! Preprocessing')
            org_preprocessor(org_data_path, masking_features)

    org_input = np.load(org_data_path/'physio_input.npy')
    org_outcomes = np.load(org_data_path/'physio_outcomes.npy')
    with open(org_data_path/'physio_normalizing_dict.pkl', 'rb') as f:
        org_dict = pickle.load(f)

    # we can directly save the outputs
    np.save(data_path/'physio_outcomes.npy', org_outcomes)

    # now, we want to select the indices of the features we want
    relevant_indices = [org_dict[feat]['idx'] for feat in features]
    if masking_features:
        masking_indices = [idx + len(org_dict) for idx in relevant_indices]
        relevant_indices.extend(masking_indices)
    new_dict = {feat: {'idx': idx} for idx, feat in enumerate(features)}

    new_input = org_input[:, :, relevant_indices]
    np.save(data_path/'physio_input.npy', new_input)
    with open(data_path/'physio_normalizing_dict.pkl', 'wb') as f:
        pickle.dump(new_dict, f)


def get_features(importance_dict_path, k):
    importance_dict = pd.read_csv(importance_dict_path)
    importance_dict = importance_dict.sort_values('vals')
    return importance_dict.features.values[:k]


def test_mask(data_path=Path('data'), k=20, masking_features=False):
    mask_folder = 'with_masking' if masking_features else 'without_masking'

    if not data_path.exists():
        data_path.mkdir()

    # check the importance dict exists
    if not (data_path/mask_folder/'importance_dict.csv').exists():
        print("Mask missing! Running get_mask")
        get_mask(data_path, masking_features)

    topk_folder = f'top_{k}'
    # check the input data exists; if it doesn't, generate it
    for file in ['physio_input.npy', 'physio_outcomes.npy',
                 'physio_normalizing_dict.pkl']:
        if not (data_path/mask_folder/topk_folder/file).exists():
            print(f'Missing {data_path/mask_folder/topk_folder/file}! Preprocessing')
            features = get_features(data_path/mask_folder/'importance_dict.csv', k)
            preprocess_data(data_path/mask_folder/topk_folder, features, masking_features)

    X = np.load(data_path/mask_folder/topk_folder/'physio_input.npy')
    Y = np.load(data_path/mask_folder/topk_folder/'physio_outcomes.npy')

    train_set, val_set, test_set = train_val_test_split(X, Y)
    train_dataset = TensorDataset(*train_set)
    val_dataset = TensorDataset(*val_set)
    test_dataset = TensorDataset(*test_set)

    # define the model
    model = PhysioNet(input_size=2 * k if masking_features else k)

    model_optimizer = torch.optim.Adam(model.parameters())

    model = train(model, data_path/mask_folder/topk_folder/'model.pickle', train_dataset, val_dataset,
                  model_optimizer, patience=2)

    loss, pred_roc = predict(model, test_dataset)
    print('Prediction loss: {:.6g}, AUC ROC: {:.6g}'.format(loss, pred_roc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-path', default=None)
    parser.add_argument('--k', default=20)
    parser.add_argument('--masking-features', action='store_true')
    args = parser.parse_args()
    if args.data_path:
        test_mask(Path(args.data_path), int(args.k), args.masking_features)
    else:
        test_mask(Path('data'), int(args.k), args.masking_features)
