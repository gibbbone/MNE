import logging
import sys
import os
from copy import deepcopy
from collections import defaultdict
import itertools
import numpy as np
import networkx as nx
import math
from numpy import exp, log, dot, zeros, outer, random, dtype, float32 as REAL,\
    double, uint32, seterr, array, uint8, vstack, fromstring, sqrt, newaxis,\
    ndarray, empty, sum as np_sum, prod, ones, ascontiguousarray, vstack, logaddexp
from sklearn.metrics import roc_auc_score
from gensim.models import Word2Vec

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

def load_network_data(f_name):
    # This function is used to load multiplex data
    print('We are loading data from:', f_name)
    edge_data_by_type = dict()
    all_edges = list()
    all_nodes = list()
    with open(f_name, 'r') as f:
        for line in f:
            words = line[:-1].split(' ')
            if words[0] not in edge_data_by_type:
                edge_data_by_type[words[0]] = list()
            edge_data_by_type[words[0]].append((words[1], words[2]))
            all_edges.append((words[1], words[2]))
            all_nodes.append(words[1])
            all_nodes.append(words[2])
    all_nodes = list(set(all_nodes))
    # create common layer.
    all_edges = list(set(all_edges))
    edge_data_by_type['Base'] = all_edges
    print('Finish loading data')
    return edge_data_by_type, all_edges, all_nodes

def train_deepwalk_embedding(walks, iteration=None):
    if iteration is None:
        iteration = 100
    model = Word2Vec(walks, size=200, window=5, min_count=0, sg=1, workers=4, iter=iteration)
    return model

# randomly divide data into few parts for the purpose of cross-validation
def divide_data(input_list, group_number):
    local_division = len(input_list) / float(group_number)
    random.shuffle(input_list)
    return [input_list[int(round(local_division * i)): int(round(local_division * (i + 1)))] for i in
            range(group_number)]

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
                AA_score += 1/(math.log(frequency_dict[neighbor]))
    return AA_score
