from pathlib import Path
import torch

from collections import defaultdict
import pandas as pd
import numpy as np

from tqdm import tqdm


class PhysioNetDataset(object):

    def __init__(self, data_dir=Path('PhysioNet/set-a'), outcomes_dir=Path('PhysioNet/Outcomes-a.txt'),
                 mask=None, device=torch.device("cpu"),
                 normalizing_dict=None):
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

    @staticmethod
    def time_to_hour(time, idx=0):
        return int(time.split(':')[idx])

    def __getitem__(self, idx):
        file_csv = pd.read_csv(self.files[idx])
        outcome = self.outcomes[file_csv[file_csv.Parameter == 'RecordID'].Value.iloc[0]]

        # split the time into hours
        file_csv['hour'] = file_csv['Time'].apply(self.time_to_hour)

        output_array = torch.zeros((48, 37), device=self.device)
        for i in range(48):
            hourly_data = file_csv[file_csv.hour == i]
            for param, vals in self.normalizing_dict.items():
                hourly_param = hourly_data[hourly_data.Parameter == param]
                if len(hourly_param) > 0:
                    normalized_hourly_average = (hourly_param.Value.mean() - vals['mean']) / \
                                                (vals['std'] if vals['std'] != 0 else 1)
                    output_array[i, vals['idx']] = normalized_hourly_average
        return output_array, torch.tensor(outcome, device=self.device)

    def preprocess_all(self):
        """
        The processing time for each invidividual example is significant. Therefore,
        all examples can be returned as a single tensor, for easy loading.
        """
        input_arrays = []
        outcomes = []
        for i in tqdm(range(len(self.files))):
            input_array, outcome = self.__getitem__(i)
            input_arrays.append(input_array)
            outcomes.append(outcome)

        return torch.stack(input_arrays, 0), torch.stack(outcomes, 0)
