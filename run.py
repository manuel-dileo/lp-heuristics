import argparse
from heuristic import RA,PA,AA,CN
from evaluation import evaluate_heuristic

import scipy.sparse as ssp
import torch
from torch_geometric.data import Data

from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
from tgb.linkproppred.evaluate import Evaluator
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset

parser = argparse.ArgumentParser(description="Example argparse with two options")
    
# Adding the "heuristic" option
parser.add_argument('--heuristic', choices=['PA', 'CN', 'RA', 'AA'],
                    help='Choose a heuristic: PA, CN, RA, AA')

# Adding the "dataset" option
parser.add_argument('--dataset', choices=['tgbl-wiki', 'tgbl-review', 'tgbl-coin', 'tgbl-comment'],
                    help='Choose a dataset: tgbl-wiki, tgbl-review, tgbl-coin, tgbl-comment')

# Adding the boolean option "--use_recent_edges"
parser.add_argument('--use_recent_edges', action='store_true',
                    help='Flag to indicate whether to use recent edges only')


if __name__=='__main__':
    args = parser.parse_args()
    
    # Accessing the selected options
    heuristic = args.heuristic
    name = args.dataset
    
    dataset = PyGLinkPropPredDataset(name=name, root="datasets")
    dataset.load_test_ns()
    dataset.load_val_ns()
    
    evaluator = Evaluator(name=name)
    
    mrr_test = evaluate_heuristic_tgb(evaluator, RA, dataset, dataset.test_mask, split_mode='test', recent_edges=args.use_recent_edges)
    
    print(f'{heuristic} on {name} MRR Test: {mrr_test}')