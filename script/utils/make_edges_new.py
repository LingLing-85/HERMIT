import torch
import numpy as np
import scipy.sparse as sp
import pickle
#import torch_geometric.utils.to_dense_adj as to_dense_adj       # this line is for the version in requirement.txt
from torch_geometric.utils import to_dense_adj    # this line is for new version of torch_geometric
from torch_geometric.utils.undirected import to_undirected
from torch_geometric.utils.convert import to_scipy_sparse_matrix
#from torch_geometric.utils.negative_sampling import negative_sampling, structured_negative_sampling     # this line is for the version in requirement.txt
from torch_geometric.utils import negative_sampling, structured_negative_sampling     # this line is for new version of torch_geometric
from torch_geometric.utils import remove_self_loops
from torch_geometric.utils import softmax, k_hop_subgraph


def tuple_to_array(lot):
    out = np.array(list(lot[0]))
    for i in range(1, len(lot)):
        out = np.vstack((out, np.array(list(lot[i]))))
    return out


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def to_one_directed_edge(undirected_edge):
    return torch.from_numpy(sparse_to_tuple(sp.triu(to_scipy_sparse_matrix(undirected_edge)))[0]).transpose(1, 0)


def get_edges(edge_index_list, edge_weight_list=None):
    directed_edge_list = []
    directed_weight_list = []
    
    for i in range(0, len(edge_index_list)):
        data = torch.from_numpy(np.array(edge_index_list[i]))
        # Free memory of the source element immediately
        edge_index_list[i] = None
        
        # Ensure shape is [2, N]
        if data.shape[0] != 2 and data.shape[1] == 2:
            data = data.transpose(1, 0)
            
        if edge_weight_list is not None:
            weights = edge_weight_list[i]
            # Free memory of the source weight immediately
            edge_weight_list[i] = None
            
            # Remove self-loops from both edges and weights
            edge_index, weights = remove_self_loops(data, weights)
            directed_weight_list.append(weights)
        else:
            edge_index, _ = remove_self_loops(data)  # remove self-loop
            
        # Keep edges directed to preserve routing directionality
        directed_edge_list.append(edge_index)
        
    if edge_weight_list is not None:
        return directed_edge_list, directed_weight_list
    return directed_edge_list


def get_prediction_edges(directed_edge_index_list, num_nodes):
    pos_edges_list = []
    neg_edges_list = []
    for directed_edge in directed_edge_index_list:
        # Edges are already directed, no need to convert
        pos_edges = directed_edge

        pos_edges_list.append(pos_edges)
        neg_edges = negative_sampling(directed_edge, num_nodes=num_nodes, num_neg_samples=pos_edges.size(1))
        neg_edges_list.append(neg_edges)
    return pos_edges_list, neg_edges_list


def get_new_prediction_edges(directed_edge_index_list, num_nodes):
    pos_edges_list = [torch.zeros((2, 100))]  # ignore the first pos edges
    neg_edges_list = [torch.zeros((2, 100))]  # ignore the first neg edges

    for i in range(1, len(directed_edge_index_list)):
        # Edges are already directed, no need to convert
        current_edges = directed_edge_index_list[i].long()
        last_edges = directed_edge_index_list[i - 1].long()
        
        # Check for index out of bounds
        if current_edges.max() >= num_nodes:
            print(f"❌ ERROR: current_edges max {current_edges.max()} >= num_nodes {num_nodes}")
            
        edges_perm = current_edges[0] * num_nodes + current_edges[1]  # hash current edges
        last_edges_perm = last_edges[0] * num_nodes + last_edges[1]  # hash last edges

        perm = np.setdiff1d(edges_perm, np.intersect1d(edges_perm, last_edges_perm))  # new edges: edge-edge^last_edge
        edges_pos = np.vstack(np.divmod(perm, num_nodes)).transpose().astype(np.int64)  # convert perm to indices
        edges_pos = torch.from_numpy(edges_pos).transpose(1, 0)

        pos_edges_list.append(edges_pos)
        # Use directed edges for negative sampling
        neg_edges_list.append(negative_sampling(edges_pos, num_nodes=num_nodes, num_neg_samples=edges_pos.size(1)))

    return pos_edges_list, neg_edges_list
