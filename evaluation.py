from heuristic import CN, AA, RA, PA
from torch_geometric.data import Data
import torch
import numpy as np
import scipy.sparse as ssp
from tqdm import tqdm

def evaluate_heuristic_tgb(evaluator, heuristic, dataset, mask, split_mode='test', recent_edges=False):
    
    num_nodes = dataset.get_TemporalData().num_nodes
    if not recent_edges:
        data = Data(edge_index=torch.stack((dataset.src[dataset.train_mask], dataset.dst[dataset.train_mask])), num_nodes = num_nodes)
    else:
        data = Data(edge_index=torch.stack((dataset.src[dataset.val_mask], dataset.dst[dataset.val_mask])), num_nodes = num_nodes)
    edge_weight = torch.ones(data.edge_index.size(1), dtype=int)
    A = ssp.csr_matrix((edge_weight, (data.edge_index[0], data.edge_index[1])), shape=(num_nodes, num_nodes))
    
    neg_edges = dataset.negative_sampler.query_batch(dataset.src[mask],
                                     dataset.dst[mask],
                                     dataset.ts[mask], 
                                     split_mode=split_mode)
    
    metric = dataset.eval_metric
    
    pos_edge = torch.stack((dataset.src[mask], dataset.dst[mask]))
    perf_list = []
    
    A_ = None
    if heuristic.__name__ == 'AA':
        multiplier = 1 / np.log(A.sum(axis=0))
        multiplier[np.isinf(multiplier)] = 0
        A_ = A.multiply(multiplier).tocsr()
    elif heuristic.__name__ == 'RA':
        multiplier = 1 / (A.sum(axis=0))
        multiplier[np.isinf(multiplier)] = 0
        A_ = A.multiply(multiplier).tocsr()
    elif heuristic.__name__ == 'PA':
        degree_col = A.sum(axis=0)
        degree_row = A.sum(axis=1)
        A_ = A.multiply(degree_col).multiply(degree_row).tocsr()
    
    for idx in tqdm(range(len(neg_edges))):
        neg_batch = neg_edges[idx]
        pos_src = pos_edge[0, idx]
        pos_dst = pos_edge[1, idx]
        query_src = torch.Tensor([int(pos_src) for _ in range(len(neg_batch) + 1)])
        query_dst = torch.Tensor([int(pos_dst)]+ neg_batch)

        edge_index = torch.stack((query_src, query_dst))
        y_pred, _ = heuristic(A, edge_index, A_)
        # compute MRR
        input_dict = {
            "y_pred_pos": np.array([y_pred[0]]),
            "y_pred_neg": np.array(y_pred[1:]),
            "eval_metric": [metric]
        }
        perf_list.append(evaluator.eval(input_dict)[metric])
    perf_metrics = float(np.mean(perf_list))
    return perf_metrics