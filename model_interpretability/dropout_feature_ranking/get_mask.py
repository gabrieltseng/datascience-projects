import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

import numpy as np
import pandas as pd
from collections import defaultdict
import argparse
import os
import itertools
import pickle
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from concrete_dropout import ConcreteDropout, ConcreteRegularizer, Annealer
from models import PhysioNet
from data import PhysioNetDataset


def train(model, model_path, concrete_dropout, concrete_dropout_path,
          dropout_regularizer, train_dataset, val_dataset, model_optimizer,
          dropout_optimizer, patience, batch_size=64, num_epochs=None, annealing=True):
    """ Training loop
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    patience_counter = 0
    best_loss = np.inf

    if annealing:
        annealer = Annealer(dropout_optimizer)

    if num_epochs:
        epoch_counter = range(1, num_epochs + 1)
    else:
        epoch_counter = itertools.count(start=1)

    for epoch in epoch_counter:
        model.train()
        concrete_dropout.train()
        train_loss = 0
        reg_loss = 0
        batch = tqdm(train_loader, total=len(train_loader),
                     desc='Epoch {:03d}'.format(epoch))

        for inputs, targets in batch:
            model_optimizer.zero_grad()
            dropout_optimizer.zero_grad()
            dropped_inputs, mask = concrete_dropout(inputs)
            outputs = model(dropped_inputs)

            loss = F.binary_cross_entropy(outputs, targets.unsqueeze(1))
            reg = dropout_regularizer(mask)
            total_loss = loss + reg
            total_loss.backward()
            train_loss += loss.data.item()
            reg_loss += reg.data.item()
            model_optimizer.step()
            dropout_optimizer.step()
            if annealing:
                annealer.step()

        model.eval()
        concrete_dropout.eval()
        val_loss = 0
        predictions_list, targets_list = [], []
        for inputs, targets in val_loader:
            outputs = model(concrete_dropout(inputs))
            val_loss += F.binary_cross_entropy(outputs, targets.unsqueeze(1)).item()
            predictions_list.extend(outputs.data.tolist())
            targets_list.extend(targets.data.tolist())

        val_auc = roc_auc_score(targets_list, predictions_list)
        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        print('loss: {:.6g}, reg: {:.6g}, val_loss: {:.6g}, val_auc: {:.6g}'.format(train_loss, reg_loss,
                                                                                    val_loss, val_auc))
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            print('Saving new best model')
            torch.save(model.state_dict(), model_path)
            torch.save(concrete_dropout.state_dict(), concrete_dropout_path)
        else:
            patience_counter += 1
        if patience_counter == patience:
            print('Early stopping - best_loss: {:.6g}'.format(best_loss))
            break

    # Load the weights of the best model
    model.load_state_dict(torch.load(model_path))
    concrete_dropout.load_state_dict(torch.load(concrete_dropout_path))
    return model, concrete_dropout


def predict(model, concrete_dropout, predict_dataset):
    """Predict loop (just calculates the prediction loss)
    """
    predict_loader = DataLoader(predict_dataset, batch_size=64)
    loss = 0

    model.eval()
    concrete_dropout.eval()
    output_list, target_list = [], []
    for inputs, targets in predict_loader:
        output = model(concrete_dropout(inputs))
        loss += F.binary_cross_entropy(output, targets.unsqueeze(1)).data.item()
        output_list.extend(output.tolist())
        target_list.extend(targets.tolist())
    loss /= len(predict_loader)
    pred_roc = roc_auc_score(target_list, output_list)
    return loss, pred_roc


def feature_ranking(mask, normalizing_dict, binary=True):
    """if binary, expect 2*len(normalizing_dict) features.
    In this case, feature importance will be the sum of the binary
    and non-binary feature
    """
    importance_dict = defaultdict(list)
    for feature, vals in normalizing_dict.items():
        idx = vals['idx']
        feature_val = mask[idx]
        if binary:
            feature_val += mask[idx + len(normalizing_dict)]
        importance_dict['features'].append(feature)
        importance_dict['vals'].append(feature_val)
    return pd.DataFrame(importance_dict)


def preprocess_data(data_path, masking_features):
    # delete all files
    for file in ['raw_physio_input.npy', 'raw_physio_outcomes.npy',
                 'raw_physio_normalizing_dict.pkl']:
        try: os.remove(data_path/file)
        except FileNotFoundError: continue

    # make a folder if it doesn't already exist
    if not data_path.exists():
        data_path.mkdir()

    dataset = PhysioNetDataset(binary_features=masking_features)
    input_array, outcomes = dataset.preprocess_all()
    # input_array = outcomes = torch.tensor([1, 2, 3])
    np.save(data_path/'physio_input.npy', input_array.numpy())
    np.save(data_path/'physio_outcomes.npy', outcomes.numpy())
    normalizing_dict = dataset.get_normalizing_dict()
    # normalizing_dict = {1: 2}
    with open(data_path/'physio_normalizing_dict.pkl', 'wb') as f:
        pickle.dump(normalizing_dict, f)


def train_val_test_split(X, Y, val_size=0.1, test_size=0.1, return_tensors=True):
    """Split the input X and Y arrays into train, val and test sets.
    Stratify according to the Y labels.
    """
    change_to_tensor = False
    if return_tensors and 'numpy' in str(type(X)):
        change_to_tensor = True
    train_size = 1 - (val_size + test_size)
    pos_idx, neg_idx = np.where(Y == 1)[0], np.where(Y == 0)[0]

    pos_mask = np.random.uniform(size=len(pos_idx))
    neg_mask = np.random.uniform(size=len(neg_idx))

    train_idx = np.concatenate((pos_idx[pos_mask < train_size], neg_idx[neg_mask < train_size]))
    val_idx = np.concatenate((pos_idx[(pos_mask > train_size) & (pos_mask < 1 - test_size)],
                              neg_idx[(neg_mask > train_size) & (neg_mask < 1 - test_size)]))
    test_idx = np.concatenate((pos_idx[pos_mask > 1 - test_size], neg_idx[neg_mask > 1 - test_size]))

    if change_to_tensor:
        return (torch.FloatTensor(X[train_idx]), torch.FloatTensor(Y[train_idx])), \
               (torch.FloatTensor(X[val_idx]), torch.FloatTensor(Y[val_idx])), \
               (torch.FloatTensor(X[test_idx]), torch.FloatTensor(Y[test_idx]))
    else:
        return (X[train_idx], Y[train_idx]), (X[val_idx], Y[val_idx]), (X[test_idx], Y[test_idx])


def get_mask(data_path=Path('data'), masking_features=False):

    if not data_path.exists():
        data_path.mkdir()

    folder = 'with_masking' if masking_features else 'without_masking'
    # check the input data exists; if it doesn't, generate it
    for file in ['physio_input.npy', 'physio_outcomes.npy',
                 'physio_normalizing_dict.pkl']:
        if not (data_path/folder/file).exists():
            print(f'Missing {data_path/folder/file}! Preprocessing')
            preprocess_data(data_path/folder, masking_features)

    X = np.load(data_path/folder/'physio_input.npy')
    Y = np.load(data_path/folder/'physio_outcomes.npy')
    with open(data_path/folder/'physio_normalizing_dict.pkl', 'rb') as f:
        normalizing_dict = pickle.load(f)

    train_set, val_set, test_set = train_val_test_split(X, Y)
    train_dataset = TensorDataset(*train_set)
    val_dataset = TensorDataset(*val_set)
    test_dataset = TensorDataset(*test_set)

    # define the model
    model = PhysioNet(input_size=len(normalizing_dict) * 2 if masking_features else len(normalizing_dict))

    # take [2:] of the shape to only add a mask across the input features, not
    # across the time (so the mask will have shape (74,), not (48, 74))
    concrete_dropout = ConcreteDropout(input_shape=train_set[0].shape[2:])
    regularizer = ConcreteRegularizer(lam=0.001)

    model_optimizer = torch.optim.Adam(model.parameters())
    dropout_optimizer = torch.optim.Adam(concrete_dropout.parameters())

    model, dropout = train(model, data_path/folder/'dropout_model.pickle', concrete_dropout,
                           data_path/folder/'concrete_dropout.pickle', regularizer, train_dataset, val_dataset,
                           model_optimizer, dropout_optimizer, patience=2)

    loss, pred_roc = predict(model, dropout, test_dataset)
    print('Prediction loss: {:.6g}, AUC ROC: {:.6g}'.format(loss, pred_roc))

    mask = dropout.parameter_mask.data.numpy()
    importance_dict = pd.DataFrame(feature_ranking(mask, normalizing_dict, masking_features))
    importance_dict.to_csv(data_path/folder/'importance_dict.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-path', default=None)
    parser.add_argument('--masking-features', action='store_true')
    args = parser.parse_args()
    if args.data_path:
        get_mask(Path(args.data_path), args.masking_features)
    else:
        get_mask(Path('data'), args.masking_features)
