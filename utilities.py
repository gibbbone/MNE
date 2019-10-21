import logging
import sys
import os
from copy import deepcopy
from collections import defaultdict
import itertools
import numpy as np
import networkx as nx
from math import log as math_log
from numpy import exp, log, dot, zeros, outer, random, dtype, float32 as REAL,\
    double, uint32, seterr, array, uint8, vstack, fromstring, sqrt, newaxis,\
    ndarray, empty, sum as np_sum, prod, ones, ascontiguousarray, vstack, logaddexp
from sklearn.metrics import roc_auc_score
import argparse

def get_parser():
    # Parses the node2vec arguments.
    parser = argparse.ArgumentParser(description="Run node2vec.")

    parser.add_argument('--input', nargs='?', default='graph/karate.edgelist',
                        help='Input graph path')

    parser.add_argument('--output', nargs='?', default='emb/karate.emb',
                        help='Embeddings path')

    parser.add_argument('--dimensions', type=int, default=200,
                        help='Number of dimensions. Default is 100.')

    parser.add_argument('--walk-length', type=int, default=10,
                        help='Length of walk per source. Default is 80.')

    parser.add_argument('--num-walks', type=int, default=20,
                        help='Number of walks per source. Default is 10.')

    parser.add_argument('--window-size', type=int, default=10,
                        help='Context size for optimization. Default is 10.')

    parser.add_argument('--iter', type=int, default=10,
                        help='Number of epochs in SGD')

    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')

    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default=1,
                        help='Inout hyperparameter. Default is 1.')

    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=False)

    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.set_defaults(directed=False)

    parser.add_argument('--file', help='Input graph filepath')
    
    parser.add_argument('--folds', type=int, default=5,
                        help='Number of repetitions for k-fold validation')

    return parser

def load_network_data(f_name):
    '''
    This function is used to load multiplex data. The accepted format is:
    
    layer  | node_i | node_j | (weighted)
    1  12 17 1
    2  12 15 1
    2  15 3 1

    The (main) output is a dict where layer are keys and values are list of 
    edges (tuples of nodes)
    '''
    print('We are loading data from:', f_name)
    edge_data_by_type = dict()
    all_edges = list()
    all_nodes = list()
    with open(f_name, 'r') as f:
        for line in f:
            # remove bottom line and separate at space
            words = line[:-1].split(' ')
            # separate by layer            
            if words[0] not in edge_data_by_type:
                edge_data_by_type[words[0]] = list()
            edge_data_by_type[words[0]].append((words[1], words[2]))
            # store everything
            all_edges.append((words[1], words[2]))
            all_nodes.append(words[1])
            all_nodes.append(words[2])
    all_nodes = list(set(all_nodes))
    # create common layer.
    all_edges = list(set(all_edges))
    edge_data_by_type['Base'] = all_edges
    print('Finish loading data')
    return edge_data_by_type, all_edges, all_nodes

def divide_data(input_list, group_number):
    '''
    Randomly divide data into few parts for the purpose of cross-validation.
    It accepts a dict of layer-edges elements and split each of the edges in 5 groups
    '''
    
    # get length of each group
    group_size = len(input_list) / float(group_number)
    # shuffle the data
    random.shuffle(input_list)
    # divide in groups
    groups = [
        input_list[int(round(group_size * i)): int(round(group_size * (i + 1)))] 
        for i in range(group_number)]
    return groups

def get_training_eval_data(edge_data_by_type_by_group, group, number_of_groups=5):
    training_data_by_type = dict()
    evaluation_data_by_type = dict()    
    for edge_type, edge_data in edge_data_by_type_by_group.items():
        eval_data = []
        train_data = []
        for j in range(number_of_groups):
            # the number of the fold fixes the evaluation data
            if j == group:
                for tmp_edge in edge_data[j]:
                    eval_data.append((tmp_edge[0], tmp_edge[1]))
            else:
                for tmp_edge in edge_data[j]:
                    train_data.append((tmp_edge[0], tmp_edge[1]))
                    
        training_data_by_type[edge_type] = train_data
        evaluation_data_by_type[edge_type] = eval_data
    return training_data_by_type, evaluation_data_by_type

def select_true_edges(train_edges, eval_edges, training_nodes):
    tmp_training_nodes = set(itertools.chain.from_iterable(train_edges))
    selected_true_edges = list()
    for edge in eval_edges:
        if edge[0] in tmp_training_nodes and edge[1] in tmp_training_nodes:
            if edge[0] == edge[1]:
                continue
            selected_true_edges.append(edge)
    return selected_true_edges

def randomly_choose_false_edges(nodes, true_edges):
    tmp_list = list()
    all_edges = list()
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            all_edges.append((i, j))
    random.shuffle(all_edges)
    for edge in all_edges:
        if edge[0] == edge[1]:
            continue
        if (nodes[edge[0]], nodes[edge[1]]) not in true_edges and (nodes[edge[1]], nodes[edge[0]]) not in true_edges:
            tmp_list.append((nodes[edge[0]], nodes[edge[1]]))
    return tmp_list

def get_G_from_edges(edges):
    edge_dict = dict()
    for edge in edges:
        edge_key = str(edge[0]) + '_' + str(edge[1])
        if edge_key not in edge_dict:
            edge_dict[edge_key] = 1
        else:
            edge_dict[edge_key] += 1
    tmp_G = nx.DiGraph()
    for edge_key in edge_dict:
        weight = edge_dict[edge_key]
        tmp_G.add_edge(edge_key.split('_')[0], edge_key.split('_')[1])
        tmp_G[edge_key.split('_')[0]][edge_key.split('_')[1]]['weight'] = weight
    return tmp_G

def get_dict_neighbourhood_score(local_model, node1, node2):
    try:
        vector1 = local_model[node1]
        vector2 = local_model[node2]
        return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    except:
        return 2+random.random()

def get_dict_AUC(model, true_edges, false_edges):
    true_list = list()
    prediction_list = list()
    for edge in true_edges:
        tmp_score = get_dict_neighbourhood_score(model, str(edge[0]), str(edge[1]))
        true_list.append(1)
        # prediction_list.append(tmp_score)
        # for the unseen pair, we randomly give a prediction
        if tmp_score > 2:
            if tmp_score > 2.5:
                prediction_list.append(1)
            else:
                prediction_list.append(-1)
        else:
            prediction_list.append(tmp_score)
    for edge in false_edges:
        tmp_score = get_dict_neighbourhood_score(model, str(edge[0]), str(edge[1]))
        true_list.append(0)
        # prediction_list.append(tmp_score)
        # for the unseen pair, we randomly give a prediction
        if tmp_score > 2:
            if tmp_score > 2.5:
                prediction_list.append(1)
            else:
                prediction_list.append(-1)
        else:
            prediction_list.append(tmp_score)
    y_true = np.array(true_list)
    y_scores = np.array(prediction_list)
    return roc_auc_score(y_true, y_scores)

def get_neighbourhood_score(local_model, node1, node2):
    try:
        vector1 = local_model.wv.syn0[local_model.wv.index2word.index(node1)]
        vector2 = local_model.wv.syn0[local_model.wv.index2word.index(node2)]
        return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    except:
        return 2+random.random()

def get_AUC(model, true_edges, false_edges):
    true_list = list()
    prediction_list = list()
    for edge in true_edges:
        tmp_score = get_neighbourhood_score(model, str(edge[0]), str(edge[1]))
        true_list.append(1)
        # prediction_list.append(tmp_score)
        # for the unseen pair, we randomly give a prediction
        if tmp_score > 2:
            if tmp_score > 2.5:
                prediction_list.append(1)
            else:
                prediction_list.append(-1)
        else:
            prediction_list.append(tmp_score)

    for edge in false_edges:
        tmp_score = get_neighbourhood_score(model, str(edge[0]), str(edge[1]))
        true_list.append(0)
        # prediction_list.append(tmp_score)
        # for the unseen pair, we randomly give a prediction
        if tmp_score > 2:
            if tmp_score > 2.5:
                prediction_list.append(1)
            else:
                prediction_list.append(-1)
        else:
            prediction_list.append(tmp_score)
    y_true = np.array(true_list)
    y_scores = np.array(prediction_list)
    return roc_auc_score(y_true, y_scores)

def get_common_neighbor_score(networks, target_A, target_B):
    common_neighbor_counter = 0
    tmp_network = networks
    A_neighbors = list()
    B_neighbors = list()
    for edge in tmp_network:
        if edge[0] == target_A:
            A_neighbors.append(edge[1])
        if edge[1] == target_A:
            A_neighbors.append(edge[0])
        if edge[0] == target_B:
            B_neighbors.append(edge[1])
        if edge[1] == target_B:
            B_neighbors.append(edge[0])
    for neighbor in A_neighbors:
        if neighbor in B_neighbors:
            common_neighbor_counter += 1
    return common_neighbor_counter

def get_Jaccard_score(networks, target_A, target_B):
    tmp_network = networks
    A_neighbors = list()
    B_neighbors = list()
    for edge in tmp_network:
        if edge[0] == target_A:
            A_neighbors.append(edge[1])
        if edge[1] == target_A:
            A_neighbors.append(edge[0])
        if edge[0] == target_B:
            B_neighbors.append(edge[1])
        if edge[1] == target_B:
            B_neighbors.append(edge[0])
    common_neighbor_counter = 0
    for neighbor in A_neighbors:
        if neighbor in B_neighbors:
            common_neighbor_counter += 1
    if len(A_neighbors) == 0 and len(B_neighbors) == 0:
        Jaccard_score = 1
    else:
        Jaccard_score = common_neighbor_counter/(len(A_neighbors) + len(B_neighbors) - common_neighbor_counter)
    return Jaccard_score

def get_AA_score(networks, target_A, target_B, frequency_dict):
    AA_score = 0
    A_neighbors = list()
    B_neighbors = list()
    for edge in networks:
        if edge[0] == target_A:
            A_neighbors.append(edge[1])
        if edge[1] == target_A:
            A_neighbors.append(edge[0])
        if edge[0] == target_B:
            B_neighbors.append(edge[1])
        if edge[1] == target_B:
            B_neighbors.append(edge[0])
    for neighbor in A_neighbors:
        if neighbor in B_neighbors:
            if frequency_dict[neighbor] > 1:
                AA_score += 1/(math_log(frequency_dict[neighbor]))
    return AA_score

def get_frequency_dict(networks):
    counting_dict = dict()
    for edge in networks:
        if edge[0] not in counting_dict:
            counting_dict[edge[0]] = 0
        if edge[1] not in counting_dict:
            counting_dict[edge[1]] = 0
        counting_dict[edge[0]] += 1
        counting_dict[edge[1]] += 1
    return counting_dict