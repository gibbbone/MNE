import logging
import sys
import os
from copy import deepcopy
from collections import defaultdict
import itertools
import numpy as np
import networkx as nx
import subprocess 
import random_walk 
import node2vec_layer_select
from utilities import * 
from MNE import *
from gensim.models import Word2Vec

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
    def get_command_string(input_string, output_string, order, negative, dimension):
        base_string = 'C++/LINE/linux/line -train {} -output {} -order {} -negative {} -dimension {}' 
        output_string_i = output_string.format(order)
        command_string = base_string.format(
            input_string,
            output_string_i,
            order, 
            negative, 
            int(dimension/2))             
        return command_string
    
    file_name = 'LINE_tmp_edges.txt'
    with open(file_name, 'w') as edge_file:
        for edge in edges:
            edge_file.write(edge[0] + ' ' + edge[1] + ' 1\n')
    
    command1 = get_command_string(
        'LINE_tmp_edges.txt', 'LINE_tmp_embedding{}.txt', 
        1, negative, dimension)
    command2 = get_command_string(
        'LINE_tmp_edges.txt', 'LINE_tmp_embedding{}.txt', 
        2, negative, dimension)
    
    #preparation_command = 'LD_LIBRARY_PATH=/usr/local/lib\nexport LD_LIBRARY_PATH'    
    preparation_command = 'LD_LIBRARY_PATH=$HOME/gls/lib\nexport LD_LIBRARY_PATH'
    subprocess.call(preparation_command, shell=True)
    subprocess.call(command1, shell=True)
    subprocess.call(command2, shell=True)    
    print('finish training')
    first_order_embedding = read_LINE_vectors('LINE_tmp_embedding1.txt')
    second_order_embedding = read_LINE_vectors('LINE_tmp_embedding2.txt')
    final_embedding = dict()
    for node in first_order_embedding:
        final_embedding[node] = np.append(first_order_embedding[node], second_order_embedding[node])
    return final_embedding

def evaluate_basic_methods(input_network):
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


def merge_PMNE_models(input_all_models, all_nodes, args):
    final_model = dict()
    for tmp_model in input_all_models:
        for node in all_nodes:
            if node in final_model:
                if node in tmp_model.wv.index2word:
                    final_model[node] = np.concatenate(
                        (final_model[node],
                         tmp_model.wv.syn0[tmp_model.wv.index2word.index(node)]), 
                        axis=0)
                else:
                    final_model[node] = np.concatenate(
                        (final_model[node], 
                         np.zeros([args.dimensions])), 
                        axis=0)
            else:
                if node in tmp_model.wv.index2word:
                    final_model[node] = tmp_model.wv.syn0[tmp_model.wv.index2word.index(node)]
                else:
                    final_model[node] = np.zeros([args.dimensions])
    return final_model


def evaluate_PMNE_methods(input_network, args):
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
    G = random_walk.RWGraph(get_G_from_edges(all_network), args.directed, args.p, args.q)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(args.num_walks, args.walk_length)
    model_one = train_deepwalk_embedding(walks)
    method_one_performance = get_AUC(model_one, all_test_network, all_false_network)
    print('Performance of PMNE method one:', method_one_performance)
    # print('We are working on method two')
    all_models = list()
    for edge_type in training_network:
        tmp_edges = training_network[edge_type]
        tmp_G = random_walk.RWGraph(get_G_from_edges(tmp_edges), args.directed, args.p, args.q)
        tmp_G.preprocess_transition_probs()
        walks = tmp_G.simulate_walks(args.num_walks, args.walk_length)
        tmp_model = train_deepwalk_embedding(walks)
        all_models.append(tmp_model)
    model_two = merge_PMNE_models(all_models, all_nodes, args)
    method_two_performance = get_dict_AUC(model_two, all_test_network, all_false_network)
    print('Performance of PMNE method two:', method_two_performance)
    # print('We are working on method three')
    tmp_graphs = list()
    for edge_type in training_network:
        tmp_G = get_G_from_edges(training_network[edge_type])
        tmp_graphs.append(tmp_G)
    MK_G = node2vec_layer_select.Graph(tmp_graphs, args.p, args.q, 0.5)
    MK_G.preprocess_transition_probs()
    MK_walks = MK_G.simulate_walks(args.num_walks, args.walk_length)
    model_three = train_deepwalk_embedding(MK_walks)
    method_three_performance = get_AUC(model_three, all_test_network, all_false_network)
    print('Performance of PMNE method three:', method_three_performance)
    return method_one_performance, method_two_performance, method_three_performance

def train_embedding(current_embedding, walks, layer_id, iter=10, info_size=10, base_weight=1):
    training_data = list()
    for walk in walks:
        tmp_walk = list()
        for node in walk:
            tmp_walk.append(str(node))
        training_data.append(tmp_walk)
    base_embedding = dict()
    if current_embedding is not None:
        for pos in range(len(current_embedding['index2word'])):
            base_embedding[current_embedding['index2word'][pos]] = current_embedding['base'][pos]
        if layer_id in current_embedding['tran']:
            current_tran = current_embedding['tran'][layer_id]
            current_additional_embedding = dict()
            for pos in range(len(current_embedding['index2word'])):
                current_additional_embedding[current_embedding['index2word'][pos]] = current_embedding['addition'][layer_id][pos]
            initial_embedding = {'base': base_embedding, 'tran': current_tran, 'addition': current_additional_embedding}
        else:
            initial_embedding = {'base': base_embedding, 'tran': None, 'addition': None}
    else:
        initial_embedding = None
    new_model = MNE(training_data, size=200, window=5, min_count=0, sg=1, workers=4, iter=iter, small_size=info_size, initial_embedding=initial_embedding, base_weight=base_weight)
    # new_model = merge_model(tmp_model, new_model, w=learning_rate)

    return new_model.in_base, new_model.in_tran, new_model.in_local, new_model.wv.index2word


def train_model(network_data):
    base_network = network_data['Base']
    base_G = random_walk.RWGraph(get_G_from_edges(base_network), 'directed', 1, 1)
    print('finish building the graph')
    base_G.preprocess_transition_probs()
    base_walks = base_G.simulate_walks(20, 10)
    base_embedding, _, _, index2word = train_embedding(None, base_walks, 'Base', 100, 10, 1)
    final_model = dict()
    final_model['base'] = base_embedding
    final_model['tran'] = dict()
    final_model['addition'] = dict()
    final_model['index2word'] = index2word
    # you can repeat this process for multiple times
    for layer_id in network_data:
        if layer_id == 'Base':
            continue
        print('We are training model for layer:', layer_id)
        if layer_id not in final_model['addition']:
            final_model['addition'][layer_id] = zeros((len(final_model['index2word']), 10), dtype=REAL)
        tmp_data = network_data[layer_id]
        # start to do the random walk on a layer
        layer_G = random_walk.RWGraph(get_G_from_edges(tmp_data), 'directed', 1, 1)
        layer_G.preprocess_transition_probs()
        layer_walks = layer_G.simulate_walks(20, 10)
        tmp_base, tmp_tran, tmp_local, tmp_index2word = train_embedding(final_model, layer_walks, layer_id, 20, 10, 0)
        base_embedding_dict = dict()
        local_embedding_dict = dict()
        for pos in range(len(tmp_index2word)):
            base_embedding_dict[tmp_index2word[pos]] = tmp_base[pos]
            local_embedding_dict[tmp_index2word[pos]] = tmp_local[pos]
        final_model['tran'][layer_id] = tmp_tran
        for tmp_word in tmp_index2word:
            final_model['addition'][layer_id][final_model['index2word'].index(tmp_word)] = local_embedding_dict[tmp_word]
    return final_model


def save_model(final_model, save_folder_name):
    with open(save_folder_name+'/'+'index2word.json', 'w') as f:
        json.dump(final_model['index2word'], f)
    np.save(save_folder_name+'/base.npy', final_model['base'])
    for layer_id in final_model['addition']:
        np.save(save_folder_name+'/tran_'+str(layer_id)+'.npy', final_model['tran'][layer_id])
        np.save(save_folder_name+'/addition_'+str(layer_id)+'.npy', final_model['addition'][layer_id])

def load_model(data_folder_name):
    file_names = os.listdir(data_folder_name)
    layer_ids = list()
    for name in file_names:
        if name[:4] == 'tran':
            tmp_id_name = name[5:-4]
            if tmp_id_name not in layer_ids:
                layer_ids.append(tmp_id_name)
    final_model = dict()
    final_model['base'] = np.load(data_folder_name+'/base.npy')
    final_model['tran'] = dict()
    final_model['addition'] = dict()
    with open(data_folder_name+'/'+'index2word.json', 'r') as f:
        final_model['index2word'] = json.load(f)
    for layer_id in layer_ids:
        final_model['tran'][layer_id] = np.load(data_folder_name+'/tran_'+str(layer_id)+'.npy')
        final_model['addition'][layer_id] = np.load(data_folder_name+'/addition_'+str(layer_id)+'.npy')
    return final_model

def get_local_MNE_model(MNE_model, edge_type):
    # MNE local model
    local_model = dict()
    for pos in range(len(MNE_model['index2word'])):
        # 0.5 is the weight parameter mentioned in the paper, which is used to show 
        # how important each relation type is and can be tuned based on the network.
        local_model[MNE_model['index2word'][pos]] = MNE_model['base'][pos] + 0.5*np.dot(
            MNE_model['addition'][edge_type][pos], MNE_model['tran'][edge_type])
    return local_model            
            
def get_node2vec_model(train_edges, args):
    # node2vec model
    G = get_G_from_edges(train_edges)
    node2vec_G = random_walk.RWGraph(G, args.directed, 2, 0.5)
    node2vec_G.preprocess_transition_probs()
    node2vec_walks = node2vec_G.simulate_walks(20, 10)
    node2vec_model = train_deepwalk_embedding(node2vec_walks)
    return node2vec_model
   
def get_deepwalk_model(train_edges, args):
    # Deepwalk model
    G = get_G_from_edges(train_edges)
    deepwalk_G = random_walk.RWGraph(G, args.directed, 1, 1)
    deepwalk_G.preprocess_transition_probs()
    deepwalk_walks = deepwalk_G.simulate_walks(args.num_walks, 10)
    deepwalk_model = train_deepwalk_embedding(deepwalk_walks)
    return deepwalk_model

def train_deepwalk_embedding(walks, iteration=None):
    if iteration is None:
        iteration = 100
    model = Word2Vec(walks, size=200, window=5, min_count=0, sg=1, workers=4, iter=iteration)
    return model