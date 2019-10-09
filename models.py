import logging
import sys
import os
from copy import deepcopy
from collections import defaultdict
import itertools
import numpy as np
import networkx as nx
import subprocess 

def read_LINE_vectors(file_name):
    tmp_embedding = dict()
    file = open(file_name, 'r')
    for line in file.readlines()[1:]:
        numbers = line[:-2].split(' ')
        tmp_vector = list()
        for n in numbers[1:]:
            tmp_vector.append(float(n))
            tmp_embedding[numbers[0]] = np.asarray(tmp_vector)
    file.close()
    return tmp_embedding


def train_LINE_model(edges, epoch_num=1, dimension=100, negative=5):
    preparation_command = 'LD_LIBRARY_PATH=/usr/local/lib\nexport LD_LIBRARY_PATH'
    file_name = 'LINE_tmp_edges.txt'
    file = open(file_name, 'w')
    for edge in edges:
        file.write(edge[0] + ' ' + edge[1] + ' 1\n')
    file.close()
    command1 = 'C++/LINE/linux/line -train LINE_tmp_edges.txt -output LINE_tmp_embedding1.txt -order 1 base-negative ' + str(
        negative) + ' -dimension ' + str(dimension / 2)
    command2 = 'C++/LINE/linux/line -train LINE_tmp_edges.txt -output LINE_tmp_embedding2.txt -order 2 -negative ' + str(
        negative) + ' -dimension ' + str(dimension / 2)
    subprocess.call(preparation_command + '\n' + command1 + '\n' + command2, shell=True)
    print('finish training')
    first_order_embedding = read_LINE_vectors('LINE_tmp_embedding1.txt')
    second_order_embedding = read_LINE_vectors('LINE_tmp_embedding2.txt')
    final_embedding = dict()
    for node in first_order_embedding:
        final_embedding[node] = np.append(first_order_embedding[node], second_order_embedding[node])
    return final_embedding


def Evaluate_basic_methods(input_network):
    print('Start to analyze the base methods')
    training_network = input_network['training']
    test_network = input_network['test_true']
    false_network = input_network['test_false']
    all_network = list()
    all_test_network = list()
    all_false_network = list()
    all_nodes = list()
    for edge_type in training_network:
        for edge in training_network[edge_type]:
            all_network.append(edge)
            if edge[0] not in all_nodes:
                all_nodes.append(edge[0])
            if edge[1] not in all_nodes:
                all_nodes.append(edge[1])
        for edge in test_network[edge_type]:
            all_test_network.append(edge)
        for edge in false_network[edge_type]:
            all_false_network.append(edge)
    print('We are analyzing the common neighbor method')
    all_network = set(all_network)
    true_list = list()
    prediction_list = list()
    for edge in all_test_network:
        true_list.append(1)
        prediction_list.append(get_common_neighbor_score(all_network, edge[0], edge[1]))
    for edge in all_false_network:
        true_list.append(0)
        prediction_list.append(get_common_neighbor_score(all_network, edge[0], edge[1]))
    y_true = np.array(true_list)
    y_scores = np.array(prediction_list)
    common_neighbor_performance = roc_auc_score(y_true, y_scores)
    print('Performance of common neighbor:', common_neighbor_performance)
    print('We are analyzing the Jaccard method')
    true_list = list()
    prediction_list = list()
    for edge in all_test_network:
        true_list.append(1)
        prediction_list.append(get_Jaccard_score(all_network, edge[0], edge[1]))
    for edge in all_false_network:
        true_list.append(0)
        prediction_list.append(get_Jaccard_score(all_network, edge[0], edge[1]))
    y_true = np.array(true_list)
    y_scores = np.array(prediction_list)
    Jaccard_performance = roc_auc_score(y_true, y_scores)
    print('Performance of Jaccard:', Jaccard_performance)
    print('We are analyzing the AA method')
    true_list = list()
    prediction_list = list()
    frequency_dict = get_frequency_dict(all_network)
    for edge in all_test_network:
        true_list.append(1)
        prediction_list.append(get_AA_score(all_network, edge[0], edge[1], frequency_dict))
    for edge in all_false_network:
        true_list.append(0)
        prediction_list.append(get_AA_score(all_network, edge[0], edge[1], frequency_dict))
    y_true = np.array(true_list)
    y_scores = np.array(prediction_list)
    AA_performance = roc_auc_score(y_true, y_scores)
    print('Performance of AA:', AA_performance)
    return common_neighbor_performance, Jaccard_performance, AA_performance


def merge_PMNE_models(input_all_models, all_nodes):
    final_model = dict()
    for tmp_model in input_all_models:
        for node in all_nodes:
            if node in final_model:
                if node in tmp_model.wv.index2word:
                    final_model[node] = np.concatenate((final_model[node], tmp_model.wv.syn0[tmp_model.wv.index2word.index(node)]), axis=0)
                else:
                    final_model[node] = np.concatenate((final_model[node], np.zeros([args.dimensions])), axis=0)
            else:
                if node in tmp_model.wv.index2word:
                    final_model[node] = tmp_model.wv.syn0[tmp_model.wv.index2word.index(node)]
                else:
                    final_model[node] = np.zeros([args.dimensions])
    return final_model


def Evaluate_PMNE_methods(input_network):
    # we need to write codes to implement the co-analysis method of PMNE
    print('Start to analyze the PMNE method')
    training_network = input_network['training']
    test_network = input_network['test_true']
    false_network = input_network['test_false']
    all_network = list()
    all_test_network = list()
    all_false_network = list()
    all_nodes = list()
    for edge_type in training_network:
        for edge in training_network[edge_type]:
            all_network.append(edge)
            if edge[0] not in all_nodes:
                all_nodes.append(edge[0])
            if edge[1] not in all_nodes:
                all_nodes.append(edge[1])
        for edge in test_network[edge_type]:
            all_test_network.append(edge)
        for edge in false_network[edge_type]:
            all_false_network.append(edge)
    # print('We are working on method one')
    all_network = set(all_network)
    G = Random_walk.RWGraph(get_G_from_edges(all_network), args.directed, args.p, args.q)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(args.num_walks, args.walk_length)
    model_one = train_deepwalk_embedding(walks)
    method_one_performance = get_AUC(model_one, all_test_network, all_false_network)
    print('Performance of PMNE method one:', method_one_performance)
    # print('We are working on method two')
    all_models = list()
    for edge_type in training_network:
        tmp_edges = training_network[edge_type]
        tmp_G = Random_walk.RWGraph(get_G_from_edges(tmp_edges), args.directed, args.p, args.q)
        tmp_G.preprocess_transition_probs()
        walks = tmp_G.simulate_walks(args.num_walks, args.walk_length)
        tmp_model = train_deepwalk_embedding(walks)
        all_models.append(tmp_model)
    model_two = merge_PMNE_models(all_models, all_nodes)
    method_two_performance = get_dict_AUC(model_two, all_test_network, all_false_network)
    print('Performance of PMNE method two:', method_two_performance)
    # print('We are working on method three')
    tmp_graphs = list()
    for edge_type in training_network:
        tmp_G = get_G_from_edges(training_network[edge_type])
        tmp_graphs.append(tmp_G)
    MK_G = Node2Vec_LayerSelect.Graph(tmp_graphs, args.p, args.q, 0.5)
    MK_G.preprocess_transition_probs()
    MK_walks = MK_G.simulate_walks(args.num_walks, args.walk_length)
    model_three = train_deepwalk_embedding(MK_walks)
    method_three_performance = get_AUC(model_three, all_test_network, all_false_network)
    print('Performance of PMNE method three:', method_three_performance)
    return method_one_performance, method_two_performance, method_three_performance

