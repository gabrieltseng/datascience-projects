import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score


def train_classifier_epoch(model, optimizer, train_dataloader, val_dataloader):

    t_losses, t_auc_scores = [], []
    v_losses, v_auc_scores = [], []
    model.train()
    for x, y in tqdm(train_dataloader):
        optimizer.zero_grad()
        preds = model(x)

        loss = F.binary_cross_entropy(preds.squeeze(1), y)
        loss.backward()
        optimizer.step()
        t_losses.append(loss.item())
        t_auc_scores.append(roc_auc_score(y.cpu().detach().numpy(),
                                          preds.squeeze(1).cpu().detach().numpy()))

    with torch.no_grad():
        model.eval()
        for val_x, val_y in tqdm(val_dataloader):
            val_preds = model(val_x)
            val_loss = F.binary_cross_entropy(val_preds.squeeze(1), val_y)
            v_losses.append(val_loss.item())
            v_auc_scores.append(roc_auc_score(val_y.cpu().detach().numpy(),
                                val_preds.squeeze(1).cpu().detach().numpy()))

    print(f'Train loss: {np.mean(t_losses)}, Train AUC ROC: {np.mean(t_auc_scores)}, '
          f'Val loss: {np.mean(v_losses)}, Val AUC ROC: {np.mean(v_auc_scores)}')
    return (t_losses, t_auc_scores), (v_losses, v_auc_scores)
