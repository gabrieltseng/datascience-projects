import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from pathlib import Path
import numpy as np
from collections import defaultdict
from tqdm import tqdm


class ModelBase:
    """
    Base class for all models
    """
    def __init__(self, savedir):
        self.savedir = savedir
        self.model = None

    def run(self, path_to_histogram=Path('data/img_output/histogram_all_full.npz'),
              pred_years=None, num_runs=2, train_steps=5, batch_size=32,
              starter_learning_rate=1e-3, return_last_dense=True):
        """
        Train the models. Note that multiple models are trained: as per the paper, a model
        is trained for each year, with all preceding years used as training values. In addition,
        for each year, 2 models are trained to account for random initialization.

        Parameters
        ----------
        path_to_histogram: pathlib Path, default=Path('data/img_output/histogram_all_full.npz')
            The location of the training data
        pred_years: list or None, default=None
            Which years to build models for. If None, the default values from the paper (range(2009, 2016))
            are used.
        num_runs: int, default=2
            The number of runs to do per year. Default taken from the paper
        train_steps: int, default=25000
            The number of steps for which to train the model. Default taken from the paper.
        batch_size: int, default=32
            Batch size when training. Default taken from the paper
        starter_learning_rate: float, default=1e-3
            Starter learning rate. Note that the learning rate is divided by 10 after 2000 and 4000 training
            steps. Default taken from the paper
        return_last_dense: boolean, default=True
            Whether or not to save the second-to-last vector produced by the network. This is necessary to train
            a Gaussian process.
        """

        with np.load(path_to_histogram) as hist:
            images = hist['output_image']
            locations = hist['output_locations']
            yields = hist['output_yield']
            years = hist['output_year']
            indices = hist['output_index']

        if pred_years is None:
            pred_years = range(2009, 2016)
        for pred_year in pred_years:
            for run_number in range(1, num_runs + 1):
                print(f'Training to predict on {pred_year}, Run number {run_number}')
                self._run_1_year(images, yields, years, locations, indices, pred_year,
                                 run_number, train_steps, batch_size, starter_learning_rate,
                                 return_last_dense)
                print('-----------')

        # TODO: delete broken images (?)

    def _run_1_year(self, images, yields, years, locations, indices, predict_year, run_number,
                      train_steps, batch_size, starter_learning_rate, return_last_dense):
        """
        Train one model on one year of data, and then save the model predictions.
        To be called by run().
        """
        train_indices = np.nonzero(years < predict_year)[0]
        val_indices = np.nonzero(years == predict_year)[0]

        train_images, val_images = self._normalize(images[train_indices], images[val_indices])

        print(f'Train set size: {train_indices.shape[0]}, Val set size: {val_indices.shape[0]}')

        train_images = torch.tensor(train_images).float()
        train_yields = torch.tensor(yields[train_indices]).float().unsqueeze(1)
        train_locations = torch.tensor(locations[train_indices])
        train_indices = torch.tensor(indices[train_indices])

        val_images = torch.tensor(val_images).float()
        val_yields = torch.tensor(yields[val_indices]).float().unsqueeze(1)
        val_locations = torch.tensor(locations[val_indices])
        val_indices = torch.tensor(indices[val_indices])

        # reinitialize the weights, since self.model may be trained multiple
        # times in one call to run()
        self.model.initialize_weights()

        train_scores, val_scores = self._train(train_images, train_yields, val_images,
                                               val_yields, train_steps, batch_size,
                                               starter_learning_rate)
        results = self._predict(train_images, train_yields, train_locations, train_indices,
                                val_images, val_yields, val_locations, val_indices,
                                return_last_dense, batch_size)

        model_information = {
            'state_dict': self.model.state_dict(),
            'val_loss': val_scores['loss'],
            'train_loss': train_scores['loss'],
        }
        for key in results:
            model_information[key] = results[key]
        filename = f'{predict_year}_{run_number}.pth.tar'
        torch.save(model_information, self.savedir / filename)

    def _train(self, train_images, train_yields, val_images, val_yields, train_steps,
               batch_size, starter_learning_rate):

        train_images, train_yields
        train_dataset = TensorDataset(train_images, train_yields)
        val_dataset = TensorDataset(val_images, val_yields)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

        criterion = nn.MSELoss()  # TODO: L1 loss as well?
        optimizer = torch.optim.Adam([pam for pam in self.model.parameters()],
                                      lr=starter_learning_rate)

        num_epochs = int(train_steps / train_images.shape[0])
        print(f'Training for {num_epochs} epochs')
        step_number = 0

        train_scores = defaultdict(list)
        val_scores = defaultdict(list)

        for epoch in range(num_epochs):
            self.model.train()

            # running train and val scores are only for printing out
            # information
            running_train_scores = defaultdict(list)

            for train_x, train_y in tqdm(train_dataloader):
                optimizer.zero_grad()
                pred_y = self.model(train_x)
                loss = criterion(pred_y, train_y)
                loss.backward()
                optimizer.step()

                running_train_scores['loss'].append(loss.item())
                train_scores['loss'].append(loss.item())

                step_number += 1
                if (step_number == 2000) or (step_number == 4000):
                    for param_group in optimizer.param_groups:
                        param_group['lr'] /= 10

            train_output_strings = []
            for key, val in running_train_scores.items():
                train_output_strings.append('{}: {}'.format(key, round(np.array(val).mean(), 5)))

            running_val_scores = defaultdict(list)
            self.model.eval()
            with torch.no_grad():
                for val_x, val_y, in tqdm(val_dataloader):
                    val_pred_y = self.model(val_x)
                    val_loss = criterion(val_pred_y, val_y)
                    running_val_scores['loss'].append(val_loss.item())

                    val_scores['loss'].append(val_loss.item())

            val_output_strings = []
            for key, val in running_val_scores.items():
                val_output_strings.append('{}: {}'.format(key, round(np.array(val).mean(), 5)))
            print('TRAINING: {}'.format(*train_output_strings))
            print('VALIDATION: {}'.format(*val_output_strings))

            return train_scores, val_scores

    def _predict(self, train_images, train_yields, train_locations, train_indices,
                 val_images, val_yields, val_locations, val_indices, return_last_dense, batch_size):
        """
        Predict on the training and validation data. Optionally, return the last
        feature vector of the model.
        """
        train_dataset = TensorDataset(train_images, train_yields, train_locations, train_indices)
        val_dataset = TensorDataset(val_images, val_yields, val_locations, val_indices)

        train_dataloader = DataLoader(train_dataset, batch_size=1)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

        results = defaultdict(list)

        self.model.eval()
        with torch.no_grad():
            for train_im, train_yield, train_loc, train_idx in tqdm(train_dataloader):
                model_output = self.model(train_im, return_last_dense=return_last_dense)
                if return_last_dense:
                    pred, feat = model_output
                    results['train_feat'].append(feat.numpy())
                else:
                    pred = model_output
                results['train_pred'].extend(pred.squeeze(1).tolist())
                results['train_real'].extend(train_yield.squeeze(1).tolist())
                results['train_loc'].extend(train_loc.numpy())
                results['train_indices'].extend(train_idx.numpy())
            for val_im, val_yield, val_loc, val_idx in tqdm(val_dataloader):
                model_output = self.model(val_im, return_last_dense=return_last_dense)
                if return_last_dense:
                    pred, feat = model_output
                    results['val_feat'].append(feat.numpy())
                else:
                    pred = model_output
                results['val_pred'].extend(pred.squeeze(1).tolist())
                results['val_real'].extend(val_yield.squeeze(1).tolist())
                results['val_loc'].extend(val_loc.numpy())
                results['val_indices'].extend(val_idx.numpy())

        for key in results:
            if key in ['train_pred', 'train_real', 'val_pred', 'val_real']:
                results[key] = np.array(results[key])
            else:
                results[key] = np.concatenate(results[key], axis=0)
        return results

    @staticmethod
    def _normalize(train_images, val_images):
        """
        Find the mean values of the bands in the train images. Use these values
        to normalize both the training and validation images.

        A little awkward, since transpositions are necessary to make array broadcasting work
        """
        mean = np.mean(train_images, axis=(0, 2, 3))

        train_images = (train_images.transpose(0, 2, 3, 1) - mean).transpose(0, 3, 1, 2)
        val_images = (val_images.transpose(0, 2, 3, 1) - mean).transpose(0, 3, 1, 2)

        return train_images, val_images
