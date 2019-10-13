# This python file is used to reproduce our link prediction experiment
# Author: Hongming ZHANG, HKUST KnowComp Group

from utilities import *
from models import *
from MNE import *

if __name__ == "__main__":
    args = parse_args()
    # logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    file_name = args.file #sys.argv[1]
    # file_name = 'data/Vickers-Chan-7thGraders_multiplex.edges'
    edge_data_by_type, _, all_nodes = load_network_data(file_name)
    # model = train_model(edge_data_by_type)

    # In our experiment, we use 5-fold cross-validation, but you can change that
    number_of_groups = 5
    edge_data_by_type_by_group = dict()
    for edge_type in edge_data_by_type:
        all_data = edge_data_by_type[edge_type]
        separated_data = divide_data(all_data, number_of_groups)
        edge_data_by_type_by_group[edge_type] = separated_data

    overall_MNE_performance = list()
    overall_node2Vec_performance = list()
    overall_LINE_performance = list()
    overall_Deepwalk_performance = list()
    overall_common_neighbor_performance = list()
    overall_Jaccard_performance = list()
    overall_AA_performance = list()
    overall_PMNE_1_performance = list()
    overall_PMNE_2_performance = list()
    overall_PMNE_3_performance = list()

    for i in range(number_of_groups):
        training_data_by_type = dict()
        evaluation_data_by_type = dict()
        for edge_type in edge_data_by_type_by_group:
            training_data_by_type[edge_type] = list()
            evaluation_data_by_type[edge_type] = list()
            for j in range(number_of_groups):
                if j == i:
                    for tmp_edge in edge_data_by_type_by_group[edge_type][j]:
                        evaluation_data_by_type[edge_type].append((tmp_edge[0], tmp_edge[1]))
                else:
                    for tmp_edge in edge_data_by_type_by_group[edge_type][j]:
                        training_data_by_type[edge_type].append((tmp_edge[0], tmp_edge[1]))

        base_edges = list()
        training_nodes = list()
        for edge_type in training_data_by_type:
            for edge in training_data_by_type[edge_type]:
                base_edges.append(edge)
                training_nodes.append(edge[0])
                training_nodes.append(edge[1])
        training_nodes = list(set(training_nodes))
        training_data_by_type['Base'] = base_edges

        MNE_model = train_model(training_data_by_type)

        tmp_MNE_performance = 0
        tmp_node2Vec_performance = 0
        tmp_LINE_performance = 0
        tmp_Deepwalk_performance = 0
        merged_networks = dict()
        merged_networks['training'] = dict()
        merged_networks['test_true'] = dict()
        merged_networks['test_false'] = dict()
        for edge_type in training_data_by_type:
            # Get data
            if edge_type == 'Base':
                continue
            print('We are working on edge:', edge_type)
            selected_true_edges = list()
            tmp_training_nodes = list()
            for edge in training_data_by_type[edge_type]:
                tmp_training_nodes.append(edge[0])
                tmp_training_nodes.append(edge[1])
            tmp_training_nodes = set(tmp_training_nodes)
            for edge in evaluation_data_by_type[edge_type]:
                if edge[0] in tmp_training_nodes and edge[1] in tmp_training_nodes:
                    if edge[0] == edge[1]:
                        continue
                    selected_true_edges.append(edge)
            if len(selected_true_edges) == 0:
                continue
            selected_false_edges = randomly_choose_false_edges(training_nodes, edge_data_by_type[edge_type])
            print('number of info network edges:', len(training_data_by_type[edge_type]))
            print('number of evaluation edges:', len(selected_true_edges))
            merged_networks['training'][edge_type] = set(training_data_by_type[edge_type])
            merged_networks['test_true'][edge_type] = selected_true_edges
            merged_networks['test_false'][edge_type] = selected_false_edges
            # MNE model
            local_model = dict()
            for pos in range(len(MNE_model['index2word'])):
                # 0.5 is the weight parameter mentioned in the paper, which is used to show how important each relation type is and can be tuned based on the network.
                local_model[MNE_model['index2word'][pos]] = MNE_model['base'][pos] + 0.5*np.dot(MNE_model['addition'][edge_type][pos], MNE_model['tran'][edge_type])
            tmp_MNE_score = get_dict_AUC(local_model, selected_true_edges, selected_false_edges)
            # tmp_MNE_score = get_AUC(MNE_model['addition'][edge_type], selected_true_edges, selected_false_edges)
            print('MNE score:', tmp_MNE_score)
            # node2vec model
            node2vec_G = Random_walk.RWGraph(get_G_from_edges(training_data_by_type[edge_type]), args.directed, 2, 0.5)
            node2vec_G.preprocess_transition_probs()
            node2vec_walks = node2vec_G.simulate_walks(20, 10)
            node2vec_model = train_deepwalk_embedding(node2vec_walks)
            tmp_node2vec_score = get_AUC(node2vec_model, selected_true_edges, selected_false_edges)
            print('node2vec score:', tmp_node2vec_score)
            # Deepwalk model
            Deepwalk_G = Random_walk.RWGraph(get_G_from_edges(training_data_by_type[edge_type]), args.directed, 1, 1)
            Deepwalk_G.preprocess_transition_probs()
            Deepwalk_walks = Deepwalk_G.simulate_walks(args.num_walks, 10)
            Deepwalk_model = train_deepwalk_embedding(Deepwalk_walks)
            tmp_Deepwalk_score = get_AUC(Deepwalk_model, selected_true_edges, selected_false_edges)
            print('Deepwalk score:', tmp_Deepwalk_score)
            # Line model
            LINE_model = train_LINE_model(training_data_by_type[edge_type])
            tmp_LINE_score = get_dict_AUC(LINE_model, selected_true_edges, selected_false_edges)
            print('LINE score:', tmp_LINE_score)
            # Update performances
            tmp_MNE_performance += tmp_MNE_score
            tmp_node2Vec_performance += tmp_node2vec_score
            tmp_LINE_performance += tmp_LINE_score
            tmp_Deepwalk_performance += tmp_Deepwalk_score

        # Print intermediate outputs
        perf = [
            tmp_MNE_performance, 
            tmp_node2Vec_performance, 
            tmp_LINE_performance, 
            tmp_Deepwalk_performance,            
        ]
        perf_list = [
            overall_MNE_performance, 
            overall_node2Vec_performance, 
            overall_LINE_performance, 
            overall_Deepwalk_performance,            
        ]
        perf_string = ['MNE','node2vec','LINE','Deepwalk']

        for p,pl,ps in zip(perf, perf_list, perf_string):
            pp = p / (len(training_data_by_type)-1)
            print('{} performance: {:10}'.format(ps, pp))
            pl.append(pp)            
        
        # Common methods
        common_neighbor_performance, Jaccard_performance, AA_performance = Evaluate_basic_methods(merged_networks)        
        overall_common_neighbor_performance.append(common_neighbor_performance)
        overall_Jaccard_performance.append(Jaccard_performance)
        overall_AA_performance.append(AA_performance)
        # PMNE model
        performance_1, performance_2, performance_3 = Evaluate_PMNE_methods(merged_networks, args)
        overall_PMNE_1_performance.append(performance_1)
        overall_PMNE_2_performance.append(performance_2)
        overall_PMNE_3_performance.append(performance_3)

    # Print final output
    performances = [    
        overall_MNE_performance,
        overall_node2Vec_performance,
        overall_LINE_performance,
        overall_Deepwalk_performance,
        overall_common_neighbor_performance,
        overall_Jaccard_performance,
        overall_AA_performance,
        overall_PMNE_1_performance,
        overall_PMNE_2_performance,
        overall_PMNE_3_performance,
    ]
    perf_str = [
        'MRNE',
        'node2Vec',
        'LINE',
        'Deepwalk',
        'Common neighbor',
        'Jaccard',
        'AA',
        'PMNE 1',
        'PMNE 2',
        'PMNE 3',       
    ]

    for perf,pstring in zip(performances, perf_str):
        perf = np.asarray(perf)
        print('Overall {} AUC: '.format(
            pstring), perf)
        print('Overall {} AUC mean: {:8}'.format(
            pstring, np.mean(perf)))
        print('Overall {} AUC std: {:8}'.format(
            pstring, np.std(perf)))
        print('')
    print('end')
