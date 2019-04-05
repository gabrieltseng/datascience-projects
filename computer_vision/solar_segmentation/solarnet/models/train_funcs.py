import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score


def train_classifier_epoch(model, optimizer, dataloader):

    losses, auc_scores = [], []
    for x, y in tqdm(dataloader):
        optimizer.zero_grad()
        preds = model(x)

        loss = F.binary_cross_entropy(preds.squeeze(1), y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        auc_scores.append(roc_auc_score(y.cpu().detach().numpy(),
                                        preds.squeeze(1).cpu().detach().numpy()))

    print(f'Epoch loss: {np.mean(losses)}, Epoch AUC ROC: {np.mean(auc_scores)}')
    return losses, auc_scores
