from tqdm import tqdm
import torch
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt


def find_learning_rate(model, crit, dataloader, optimizer, min_lr=1e-5, max_lr=1, ax=None):
    """
    All outputs of the model will be passed to the loss function.
    """

    info = defaultdict(list)

    for batch_number, batch in (enumerate(tqdm(dataloader))):

        addition = (batch_number / len(dataloader)) * (max_lr - min_lr)
        learning_rate = min_lr + addition
        # update the optimizer learning rates
        for g in optimizer.param_groups:
            g['lr'] = learning_rate
        info['lr'].append(learning_rate)
        # Zero the gradients of my optimizer - 'reset' it
        optimizer.zero_grad()

        x, y = batch
        pred = model(x)

        loss = crit(pred, y)

        info['loss'].append(loss.item())
        loss.backward()
        optimizer.step()
    show_plot = False
    if ax is None:
        show_plot = True
        fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(info['lr'], info['loss'])
    ax.set_xlabel('Learning rate')
    ax.set_label('Loss')

    if show_plot:
        plt.show()

    return info


def train(model, crit, optimizer, num_epochs, train_dataloader, val_dataloader=None,
          lr_scheduler=None, forcing_scheduler=None):
    overall_train = defaultdict(list)
    overall_val = defaultdict(list)

    for epoch in range(num_epochs):
        # set the model to train
        model.train()
        # keep track of training scores, so they can be displayed later
        train_scores = defaultdict(list)

        for batch_number, batch in (enumerate(tqdm(train_dataloader))):
            # Zero the gradients of my optimizer - 'reset' it
            optimizer.zero_grad()

            x, y = batch

            # pass the y as well, for teacher forcing
            pred = model(x, y)

            loss = crit(pred, y)
            train_scores['loss'].append(loss.item())
            loss.backward()
            optimizer.step()

            if lr_scheduler:
                lr_scheduler.step()
            if forcing_scheduler:
                forcing_scheduler.step()

        train_output_strings = []
        for key, val in train_scores.items():
            mean_value = np.array(val).mean()
            train_output_strings.append('{}: {}'.format(key, round(mean_value, 5)))
            overall_train[key].append(mean_value)

        if val_dataloader:
            model.eval()
            val_scores = defaultdict(list)
            with torch.no_grad():
                for batch_number, batch in enumerate(val_dataloader):
                    x, y = batch
                    pred = model(x)
                    val_loss = crit(pred, y)

                    val_scores['loss'].append(val_loss)

            val_output_strings = []
            for key, val in val_scores.items():
                mean_value = np.array(val).mean()
                val_output_strings.append('{}: {}'.format(key, round(mean_value, 5)))
                overall_val[key].append(mean_value)
        print('TRAINING: {}'.format(*train_output_strings))
        if val_dataloader:
            print('VALIDATION: {}'.format(*val_output_strings))
    if val_dataloader:
        return overall_train, overall_val
    else:
        return overall_train
