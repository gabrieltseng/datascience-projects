from pathlib import Path
import torch

from collections import defaultdict
import pandas as pd
import numpy as np

from concurrent.futures import ProcessPoolExecutor
from itertools import repeat, islice


class PhysioNetDataset(object):

    def __init__(self, data_dir=Path('PhysioNet/set-a'), outcomes_dir=Path('PhysioNet/Outcomes-a.txt'),
                 mask=None, device=torch.device("cpu"), normalizing_dict=None,
                 processes=6, parallelism=4, chunks=1):
        self.device = device
        self.data_dir = data_dir
        self.outcomes = self._load_outcomes(outcomes_dir)

        # load up all the filenames (patient ids), and sort them
        files = np.array(sorted([x for x in data_dir.iterdir()]))
        if mask is not None:
            self.files = files[mask]
        else:
            self.files = files

        if normalizing_dict is None:
            # finally, we need to map all the variables to an index
            parameters = self.get_values()

            normalizing_dict = {}
            for idx, dict_vals in enumerate(parameters.items()):
                key, vals = dict_vals
                normalizing_dict[key] = {
                    'mean': np.mean(vals),
                    'std': np.std(vals),
                    'idx': idx
                }
        self.normalizing_dict = normalizing_dict

        # for multiprocessing
        self.processes = processes
        self.parallelism = parallelism
        self.chunks = chunks

    @staticmethod
    def chunk(it, size):
        """
        An iterator which returns chunks of items (i.e. size items per call, instead of 1).
        Setting size=1 returns a normal iterator.
        """
        it = iter(it)
        return iter(lambda: tuple(islice(it, size)), ())

    def _load_outcomes(self, outcomes_dir):
        outcomes_csv = pd.read_csv(outcomes_dir)
        outcomes = {}
        for i in outcomes_csv.itertuples():
            # In-hospital_death turned to its index
            outcomes[i.RecordID] = i._6
        return outcomes

    def get_normalizing_dict(self):
        return self.normalizing_dict

    def get_values(self):
        """Iterates through all the files, and accumulates
        all the variables, and their values, to calculate their mean and
        standard deviations
        """
        ignore = ['RecordID', 'Age', 'Gender', 'Height', 'ICUType']
        parameters = defaultdict(list)
        for file in self.files:
            file_csv = pd.read_csv(file)
            for row in file_csv.itertuples():
                if row.Parameter not in ignore:
                    parameters[row.Parameter].append(row.Value)
        return parameters

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return process_record(self.files[idx], self.outcomes,
                              self.normalizing_dict, self.device.type)[0]

    def preprocess_all(self):

        files_iter = self.chunk(self.files, size=self.chunks)
        length = int(len(self.files) / self.chunks)

        # iterators for all the arguments
        outcomes_iter = repeat(self.outcomes)
        normalizing_dict_iter = repeat(self.normalizing_dict)
        device_iter = repeat(self.device.type)

        input_arrays = []
        outcomes = []
        with ProcessPoolExecutor() as executor:
            chunksize = int(max(length / (self.processes * self.parallelism), 1))
            i = 0
            for result in executor.map(process_record, files_iter, outcomes_iter,
                                       normalizing_dict_iter, device_iter,
                                       chunksize=chunksize):
                for input_array, outcome in result:
                    input_arrays.append(input_array)
                    outcomes.append(outcome)
                    i += 1
                    if i % 100 == 0:
                        print('Processed {} records'.format(i))
        return torch.stack(input_arrays, 0), torch.stack(outcomes, 0)


def process_record(filepath, outcomes, normalizing_dict, device):
    device = torch.device(device)  # the device object itself can't be pickled
    if type(filepath) is tuple:
        # happens when an iterator is used
        filepath = filepath[0]
    file_csv = pd.read_csv(filepath)
    outcome = outcomes[file_csv[file_csv.Parameter == 'RecordID'].Value.iloc[0]]

    # split the time into hours
    file_csv['hour'] = file_csv['Time'].apply(lambda x: int(x.split(':')[0]))

    output_array = torch.zeros((48, 37), device=device)
    for i in range(48):
        hourly_data = file_csv[file_csv.hour == i]
        for param, vals in normalizing_dict.items():
            hourly_param = hourly_data[hourly_data.Parameter == param]
            if len(hourly_param) > 0:
                normalized_hourly_average = (hourly_param.Value.mean() - vals['mean']) / \
                                            (vals['std'] if vals['std'] != 0 else 1)
                output_array[i, vals['idx']] = normalized_hourly_average
    # map() flattens the results; a 2d list prevents this
    return [[output_array, torch.tensor(outcome, device=device)]]
