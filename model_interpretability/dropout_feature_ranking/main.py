import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from test_mask import test_topk


def run_experiment(data_path, masking_features):
    """
    Compares concrete dropout to randomly picking k features.
    """
    random_roc = []
    dropout_roc = []
    k_values = [5, 10, 15, 20, 25, 30, 37]
    for k in k_values:
        random_roc.append(test_topk(data_path, k, masking_features, random_k=True))
        dropout_roc.append(test_topk(data_path, k, masking_features, random_k=False))

    # save the arrays
    mask_folder = 'with_masking' if masking_features else 'without_masking'
    np.save(data_path/mask_folder/'random_roc.npy', random_roc)
    np.save(data_path/mask_folder/'dropout_roc.npy', dropout_roc)

    # plot the roc curve, and save that too
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(k_values, random_roc, label='Random')
    ax.plot(k_values, dropout_roc, label='Dropout FR')
    ax.set_xlabel('Number of features')
    ax.set_ylabel('Test AUROC')
    ax.legend()
    plt.savefig(data_path/mask_folder/'Result.png', bbox_inches='tight', transparent=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-path', default=None)
    parser.add_argument('--masking-features', action='store_true')
    args = parser.parse_args()
    if args.data_path:
        run_experiment(Path(args.data_path), args.masking_features)
    else:
        run_experiment(Path('data'), args.masking_features)
