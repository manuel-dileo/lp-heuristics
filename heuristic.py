import torch
from torch_geometric.data import DataLoader
from torch_geometric.utils import degree
import numpy as np
from tqdm import tqdm

"""
Heuristic scores for link prediction.
Args:
   A: sparse matrix, adjacency matrix of the training graph
   edge_index: torch_geometric edge_index, representing the edge list for which compute the scores
   A_: sparse matrix, utility matrix for computing heuristic scores.
"""

def CN(A, edge_index, A_=None, batch_size=100000):
    # The Common Neighbor heuristic score.
    link_loader = DataLoader(range(edge_index.size(1)), batch_size)
    scores = []
    for ind in link_loader:
        src, dst = edge_index[0, ind], edge_index[1, ind]
        cur_scores = np.array(np.sum(A[src].multiply(A[dst]), 1)).flatten()
        scores.append(cur_scores)
    return np.concatenate(scores, 0), edge_index


def AA(A, edge_index, A_ = None, batch_size=100000):
    # The Adamic-Adar heuristic score.
    if A_ is None:
        multiplier = 1 / np.log(A.sum(axis=0))
        multiplier[np.isinf(multiplier)] = 0
        A_ = A.multiply(multiplier).tocsr()
    link_loader = DataLoader(range(edge_index.size(1)), batch_size)
    scores = []
    for ind in link_loader:
        src, dst = edge_index[0, ind], edge_index[1, ind]
        cur_scores = np.array(np.sum(A[src].multiply(A_[dst]), 1)).flatten()
        scores.append(cur_scores)
    scores = np.concatenate(scores, 0)
    return scores, edge_index

def RA(A, edge_index, A_=None, batch_size=100000):
    # The Resource-Allocation heuristic score.
    if A_ is None:
        multiplier = 1 / (A.sum(axis=0))
        multiplier[np.isinf(multiplier)] = 0
        A_ = A.multiply(multiplier).tocsr()
    link_loader = DataLoader(range(edge_index.size(1)), batch_size)
    scores = []
    for ind in link_loader:
        src, dst = edge_index[0, ind], edge_index[1, ind]
        cur_scores = np.array(np.sum(A[src].multiply(A_[dst]), 1)).flatten()
        scores.append(cur_scores)
    scores = np.concatenate(scores, 0)
    return scores, edge_index

def PA(A, edge_index, A_=None, batch_size=100000):
    # The Preferential-Attachment heuristic score.
    if A_ is None:
        degree_col = A.sum(axis=0)
        degree_row = A.sum(axis=1)
        A_ = A.multiply(degree_col).multiply(degree_row).tocsr()
    link_loader = DataLoader(range(edge_index.size(1)), batch_size)
    scores = []
    for ind in link_loader:
        src, dst = edge_index[0, ind], edge_index[1, ind]
        cur_scores = A_[src.long(), :][:, dst.long()].diagonal()
        scores.append(cur_scores)
    scores = np.concatenate(scores, 0)
    return scores, edge_index