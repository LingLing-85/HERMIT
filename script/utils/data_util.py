import os
import numpy as np
import torch
from torch_geometric.utils import train_test_split_edges
from torch_geometric.data import Data
import pickle
from script.utils.make_edges_orign import mask_edges_det, mask_edges_prd, mask_edges_prd_new_by_marlin
from script.utils.make_edges_new import get_edges, get_prediction_edges, get_new_prediction_edges


def mkdirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path


def prepare_dir(output_folder):
    mkdirs(output_folder)
    log_folder = mkdirs(output_folder)
    return log_folder


def load_vgrnn_dataset(dataset):
    assert dataset in ['enron10', 'dblp']  # using vgrnn dataset
    print('>> loading on vgrnn dataset')
    with open('../data/input/raw/{}/adj_time_list.pickle'.format(dataset), 'rb') as handle:
        adj_time_list = pickle.load(handle, encoding='iso-8859-1')
    print('>> generating edges,negative edges and new edges, wait for a while ...')
    data = {}
    edges, biedges = mask_edges_det(adj_time_list)  # list
    pedges, nedges = mask_edges_prd(adj_time_list)  # list
    new_pedges, new_nedges = mask_edges_prd_new_by_marlin(adj_time_list)  # list
    print('>> processing finished!')
    assert len(edges) == len(biedges) == len(pedges) == len(nedges) == len(new_nedges) == len(new_pedges)
    edge_index_list, pedges_list, nedges_list, new_nedges_list, new_pedges_list = [], [], [], [], []
    for t in range(len(biedges)):
        edge_index_list.append(torch.tensor(np.transpose(biedges[t]), dtype=torch.long))
        pedges_list.append(torch.tensor(np.transpose(pedges[t]), dtype=torch.long))
        nedges_list.append(torch.tensor(np.transpose(nedges[t]), dtype=torch.long))
        new_pedges_list.append(torch.tensor(np.transpose(new_pedges[t]), dtype=torch.long))
        new_nedges_list.append(torch.tensor(np.transpose(new_nedges[t]), dtype=torch.long))

    data['edge_index_list'] = edge_index_list
    data['pedges'], data['nedges'] = pedges_list, nedges_list
    data['new_pedges'], data['new_nedges'] = new_pedges_list, new_nedges_list  # list
    data['num_nodes'] = int(np.max(np.vstack(edges))) + 1

    data['time_length'] = len(edge_index_list)
    data['weights'] = None
    print('>> data: {}'.format(dataset))
    print('>> total length:{}'.format(len(edge_index_list)))
    print('>> number nodes: {}'.format(data['num_nodes']))
    return data


    print('>> loading on new dataset')
    data = {}
    
    # Get the project root directory (two levels up from utils/data_util.py)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    
    # Construct absolute paths
    rawfile = os.path.join(project_root, 'data/input/processed/{}/{}.pt'.format(dataset, dataset))
    edge_index_list = torch.load(rawfile)  # format: list:[[[1,2],[2,3],[3,4]]]
    
    # Load edge weights if available
    weight_file = os.path.join(project_root, 'data/input/processed/{}/{}_weights.pt'.format(dataset, dataset))
    if os.path.exists(weight_file):
        edge_weight_list = torch.load(weight_file)
        print('>> loaded edge features from {}'.format(weight_file))
    else:
        edge_weight_list = None
        print('>> no edge features found, using uniform features')
    
    if edge_weight_list is not None:
        directed_edges, directed_weights = get_edges(edge_index_list, edge_weight_list)
        data['weights'] = directed_weights
        del edge_weight_list  # Free memory
    else:
        directed_edges = get_edges(edge_index_list)
        data['weights'] = None
    
    time_length = len(edge_index_list)
    del edge_index_list  # Free memory
    import gc
    gc.collect()  # Force garbage collection
        
    num_nodes = int(np.max(np.hstack(directed_edges))) + 1
    pedges, nedges = get_prediction_edges(directed_edges, num_nodes)  # list
    new_pedges, new_nedges = get_new_prediction_edges(directed_edges, num_nodes)

    data['edge_index_list'] = directed_edges
    data['pedges'], data['nedges'] = pedges, nedges
    data['new_pedges'], data['new_nedges'] = new_pedges, new_nedges  # list
    data['num_nodes'] = num_nodes
    data['time_length'] = time_length
    # Load RTT stats for caida
    if dataset == 'caida':
        stats_file = os.path.join(project_root, 'data/input/processed/{}/rtt_stats.pt'.format(dataset))
        if os.path.exists(stats_file):
            stats = torch.load(stats_file)
            data['rtt_avg_min'] = stats['avg_min']
            data['rtt_avg_max'] = stats['avg_max']
        else:
            data['rtt_avg_min'] = 0.0
            data['rtt_avg_max'] = 1.0
    print('>> data: {}'.format(dataset))
    print('>> total length: {}'.format(time_length))
    print('>> number nodes: {}'.format(data['num_nodes']))
    return data


def load_vgrnn_dataset_det(dataset):
    assert dataset in ['enron10', 'dblp']  # using vgrnn dataset
    print('>> loading on vgrnn dataset')
    with open('../data/input/raw/{}/adj_time_list.pickle'.format(dataset), 'rb') as handle:
        adj_time_list = pickle.load(handle, encoding='iso-8859-1')
    print('>> generating edges, negative edges and new edges, wait for a while ...')
    data = {}
    edges, biedges = mask_edges_det(adj_time_list)  # list
    pedges, nedges = mask_edges_prd(adj_time_list)  # list
    new_pedges, new_nedges = mask_edges_prd_new_by_marlin(adj_time_list)  # list
    print('>> processing finished!')
    assert len(edges) == len(biedges) == len(pedges) == len(nedges) == len(new_nedges) == len(new_pedges)
    edge_index_list, pedges_list, nedges_list, new_nedges_list, new_pedges_list = [], [], [], [], []
    for t in range(len(biedges)):
        edge_index_list.append(torch.tensor(np.transpose(biedges[t]), dtype=torch.long))
        pedges_list.append(torch.tensor(np.transpose(pedges[t]), dtype=torch.long))
        nedges_list.append(torch.tensor(np.transpose(nedges[t]), dtype=torch.long))
        new_pedges_list.append(torch.tensor(np.transpose(new_pedges[t]), dtype=torch.long))
        new_nedges_list.append(torch.tensor(np.transpose(new_nedges[t]), dtype=torch.long))

    data['edge_index_list'] = edge_index_list
    data['pedges'], data['nedges'] = pedges_list, nedges_list
    data['new_pedges'], data['new_nedges'] = new_pedges_list, new_nedges_list  # list
    data['num_nodes'] = int(np.max(np.vstack(edges))) + 1

    data['time_length'] = len(edge_index_list)
    data['weights'] = None
    print('>> data: {}'.format(dataset))
    print('>> total length:{}'.format(len(edge_index_list)))
    print('>> number nodes: {}'.format(data['num_nodes']))
    return data


def load_new_dataset_det(dataset):
    print('>> loading on new dataset')
    data = {}
    rawfile = '../data/input/processed/{}/{}.pt'.format(dataset, dataset)
    edge_index_list = torch.load(rawfile)  # format: list:[[[1,2],[2,3],[3,4]]]
    directed_edges = get_edges(edge_index_list)
    num_nodes = int(np.max(np.hstack(directed_edges))) + 1

    gdata_list = []
    for edge_index in directed_edges:
        gdata = Data(x=None, edge_index=edge_index, num_nodes=num_nodes)
        gdata_list.append(train_test_split_edges(gdata, 0.1, 0.4))

    data['gdata'] = gdata_list
    data['num_nodes'] = num_nodes
    data['time_length'] = len(edge_index_list)
    data['weights'] = None
    print('>> data: {}'.format(dataset))
    print('>> total length: {}'.format(len(edge_index_list)))
    print('>> number nodes: {}'.format(data['num_nodes']))
    return data


def loader(dataset='enron10'):
    # if cached, load directly
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    data_root = os.path.join(project_root, 'data/input/cached/{}/'.format(dataset))
    filepath = mkdirs(data_root) + '{}.data'.format(dataset)
    if os.path.isfile(filepath):
        print('loading {} directly'.format(dataset))
        return torch.load(filepath)
    # if not cached, to process and cached
    print('>>data is not exits, processing ...')
    if dataset in ['enron10', 'dblp']:
        data = load_vgrnn_dataset(dataset)
    if dataset in ['as733', 'fbw', 'HepPh30', 'disease', 'caida']:
        data = load_new_dataset(dataset)
    torch.save(data, filepath)
    print('saved!')
    return data
