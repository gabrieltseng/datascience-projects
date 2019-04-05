import torch.nn.functional as F
from tqdm import tqdm
import numpy as np


def train_classifier_epoch(model, optimizer, dataloader):

    losses = []
    for x, y in tqdm(dataloader):
        optimizer.zero_grad()
        preds = model(x)

        loss = F.binary_cross_entropy(preds.squeeze(1), y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print(f"Batch loss: {loss.item()}")

    print(f'Epoch loss: {np.mean(losses)}')
    return losses
