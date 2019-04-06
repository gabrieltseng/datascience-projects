import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score


def train_classifier(model, train_dataloader, val_dataloader,
                     warmup=2, patience=5, max_epochs=100):

    best_state_dict = model.state_dict()
    best_val_auc_roc = 0.5
    patience_counter = 0
    for i in range(max_epochs):
        if i <= warmup:
            # we start by finetuning the model
            optimizer = torch.optim.Adam([pam for name, pam in
                                          model.named_parameters() if 'classifier' in name])
        else:
            # then, we train the whole thing
            optimizer = torch.optim.Adam(model.parameters())

        train_data, val_data = _train_classifier_epoch(model, optimizer, train_dataloader,
                                                       val_dataloader)
        if np.mean(val_data[1]) > best_val_auc_roc:
            best_val_auc_roc = np.mean(val_data[1])
            patience_counter = 0
            best_state_dict = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter == patience:
                print(f"Early stopping!")
                model.load_state_dict(best_state_dict)


def _train_classifier_epoch(model, optimizer, train_dataloader, val_dataloader):

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


def _train_segmenter_epoch(model, optimizer, train_dataloader, val_dataloader):

    t_losses, v_losses = [], []
    model.train()
    for x, y in tqdm(train_dataloader):
        optimizer.zero_grad()
        preds = model(x)

        loss = F.binary_cross_entropy(preds, y)
        loss.backward()
        optimizer.step()

        t_losses.append(loss.item())

    with torch.no_grad():
        model.eval()
        for val_x, val_y in tqdm(val_dataloader):
            val_preds = model(val_x)
            val_loss = F.binary_cross_entropy(val_preds, val_y.unsqueeze(1))
            v_losses.append(val_loss.item())
    print(f'Train loss: {np.mean(t_losses)}, Val loss: {np.mean(v_losses)}')

    return t_losses, v_losses
