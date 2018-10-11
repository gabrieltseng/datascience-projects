from test_mask import test_mask
import argparse
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-path', default=None)
    parser.add_argument('--k', help="Number of features to test", default=20)
    parser.add_argument('--masking-features', action='store_true')
    args = parser.parse_args()
    if args.data_path:
        test_mask(Path(args.data_path), int(args.k), args.masking_features)
    else:
        test_mask(Path('data'), int(args.k), args.masking_features)
