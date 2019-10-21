# This python file is used to reproduce our link prediction experiment
# Author: Hongming ZHANG, HKUST KnowComp Group
# file_name = 'data/Vickers-Chan-7thGraders_multiplex.edges'

from utilities import *
from models import *
from MNE import *

if __name__ == "__main__":    
    # logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    
    # In our experiment, we use 5-fold cross-validation
    # customizable providing another number via the -folds argument
    args = get_parser().parse_args()
    file_name = args.file 
    number_of_groups = args.folds

    # load data
    edge_data_by_type, _, all_nodes = load_network_data(file_name)    
    
    # create k-fold groups
    edge_data_by_type_by_group = {
        edge_type: divide_data(all_data, number_of_groups) 
        for edge_type, all_data in edge_data_by_type.items()} 

    # initialize output lists
    overall_MNE_performance = list()
    overall_node2Vec_performance = list()
    overall_LINE_performance = list()
    overall_deepwalk_performance = list()
    overall_common_neighbor_performance = list()
    overall_Jaccard_performance = list()
    overall_AA_performance = list()
    overall_PMNE_1_performance = list()
    overall_PMNE_2_performance = list()
    overall_PMNE_3_performance = list()

    # iterate k-fold
    for i in range(number_of_groups):
        # train-test split
        training_data_by_type, evaluation_data_by_type = get_training_eval_data(
            edge_data_by_type_by_group, i, number_of_groups)
        
        # collect edges and nodes from training set
        base_edges = list(
            itertools.chain.from_iterable(
                [l for l in training_data_by_type.values()]))
        training_data_by_type['Base'] = base_edges
        training_nodes = list(set(itertools.chain.from_iterable(base_edges)))        
        
        # train global MNE model with all the data in advance
        MNE_model = train_model(training_data_by_type)
        
        #train or load Ohment here 
        
        # prepare results collection
        tmp_MNE_performance = 0
        tmp_node2Vec_performance = 0
        tmp_LINE_performance = 0
        tmp_deepwalk_performance = 0
        merged_networks = {
            'training': {},
            'test_true': {},
            'test_false': {},
        }
        # iterate over layers 
        for edge_type, train_edges in training_data_by_type.items():            
            if edge_type == 'Base':
                continue
            print('We are working on edge:', edge_type)      
            
            # get true/false edges
            eval_edges = edge_data_by_type[edge_type]
            selected_true_edges = select_true_edges(
                train_edges, eval_edges, training_nodes)            
            if len(selected_true_edges) == 0:
                continue                
            selected_false_edges = randomly_choose_false_edges(
                training_nodes, eval_edges)
            
            # log and store details
            print('number of info network edges:', len(train_edges))
            print('number of evaluation edges:', len(selected_true_edges))
            merged_networks['training'][edge_type] = set(train_edges)
            merged_networks['test_true'][edge_type] = selected_true_edges
            merged_networks['test_false'][edge_type] = selected_false_edges
            
            # MNE model            
            local_model = get_local_MNE_model(MNE_model, edge_type)
            tmp_MNE_score = get_dict_AUC(
                local_model, 
                selected_true_edges, 
                selected_false_edges)            
            print('MNE score:', tmp_MNE_score)
            
            # node2vec model
            G = get_G_from_edges(train_edges)
            # node2vec_model = get_node2vec_model(G, args)
            node2vec_model = get_random_walk_model(G, args, 2, 0.5, 20)
            tmp_node2vec_score = get_AUC(
                node2vec_model, 
                selected_true_edges, 
                selected_false_edges)
            print('Node2vec score:', tmp_node2vec_score)
            
            # Deepwalk model
            # G = get_G_from_edges(train_edges)
            # deepwalk_model = get_deepwalk_model(G, args)
            deepwalk_model = get_random_walk_model(G, args, 1, 1, args.num_walks)
            tmp_deepwalk_score = get_AUC(
                deepwalk_model, 
                selected_true_edges, 
                selected_false_edges)
            print('Deepwalk score:', tmp_deepwalk_score)
            
            # Line model
            LINE_model = train_LINE_model(train_edges)
            tmp_LINE_score = get_dict_AUC(
                LINE_model, 
                selected_true_edges, 
                selected_false_edges)
            print('LINE score:', tmp_LINE_score)
            
            # Update performances
            tmp_MNE_performance += tmp_MNE_score
            tmp_node2Vec_performance += tmp_node2vec_score
            tmp_LINE_performance += tmp_LINE_score
            tmp_deepwalk_performance += tmp_deepwalk_score

        # Print intermediate outputs for each fold
        perf = [
            tmp_MNE_performance, 
            tmp_node2Vec_performance, 
            tmp_LINE_performance, 
            tmp_deepwalk_performance,            
        ]
        perf_list = [
            overall_MNE_performance, 
            overall_node2Vec_performance, 
            overall_LINE_performance, 
            overall_deepwalk_performance,            
        ]
        perf_string = ['MNE','node2vec','LINE','Deepwalk']

        for p,pl,ps in zip(perf, perf_list, perf_string):
            pp = p / (len(training_data_by_type)-1)
            print('{} performance: {:10}'.format(ps, pp))
            pl.append(pp)            
        
        # Common methods
        common_neighbor_performance, Jaccard_performance, AA_performance = evaluate_basic_methods(merged_networks)        
        overall_common_neighbor_performance.append(common_neighbor_performance)
        overall_Jaccard_performance.append(Jaccard_performance)
        overall_AA_performance.append(AA_performance)
        # PMNE model
        performance_1, performance_2, performance_3 = evaluate_PMNE_methods(merged_networks, args)
        overall_PMNE_1_performance.append(performance_1)
        overall_PMNE_2_performance.append(performance_2)
        overall_PMNE_3_performance.append(performance_3)

    # Print final output
    performances = [    
        overall_MNE_performance,
        overall_node2Vec_performance,
        overall_LINE_performance,
        overall_deepwalk_performance,
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
        'deepwalk',
        'common neighbor',
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
