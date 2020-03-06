import random
import numpy as np
import pandas as pd
from sklearn import preprocessing
from Sources.Preparation.Features.get_qualified_edges import get_all_GeneDisease_unweighted_disease2disease_qualified_edges
from Sources.Preparation.Features.get_qualified_edges import get_GPSim_disease2disease_qualified_edges
# from Sources.Preparation.Features.test import get_all_GeneDisease_unweighted_disease2disease_qualified_edges
# from Sources.Preparation.Features.test import get_GPSim_disease2disease_qualified_edges
from itertools import combinations


def onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def remove_edges_from_graph_for_link_prediction(data, graph, train_set_np, test_set_np):
# def remove_edges_from_graph_for_link_prediction(data, graph_with_no_added_qualified_edges, train_set_np, split):
    '''

    @param data:
    @param graph_with_no_added_qualified_edges:
    @param train_set_np:
    @param split: desc= test_split
    @return:
    '''
    graph_with_no_added_qualified_edges = graph.copy()
    # TODO here>>-1 what is splti used for; hwo can it make it to be compatible with cross valiation and train_test_splti
    train_set_edges = set(map(tuple, train_set_np[:, :2].tolist()))
    # assert len(train_set_edges) == int(len(data.gene_disease_edges) * (1-split)) * 2, "train_set_edges = int(len(original_gene_disease_edges * training_split)) * 2"
    # assert

    count_removed_edges = 0
    # print(len(graph_with_no_added_qualified_edges.edges))
    for edge in data.gene_disease_edges:  # np or pd ?
        if graph_with_no_added_qualified_edges.has_edge(*edge):
            edge_with_gene_as_str = tuple(str(i) for i in
                                          edge)  # I hould have use gene as str in the first place, I instead used int, so I am *patching* that mistake
            if edge_with_gene_as_str not in train_set_edges:
                count_removed_edges += 1
                graph_with_no_added_qualified_edges.remove_edge(*edge)
        else:
            print(edge)
            raise ValueError(
                "some edges from dataset.gene_disease-edges is not contained in starter_graph (aka. graph_with_no_added_qualified_edges)")

    # print(count_removed_edges)
    graph_with_no_added_qualified_edges_with_removed_test_edges = graph_with_no_added_qualified_edges


    return graph_with_no_added_qualified_edges_with_removed_test_edges


def select_emb_save_path(save_path_base = None, emb_type=None, add_qualified_edges=None, dataset=None,
                         use_weighted_edges=None, edges_number=None,
                         edges_percent=None,
                         added_edges_percent_of=None,
                         use_shared_gene_edges = None,
                         use_shared_phenotype_edges = None,
                         use_shared_gene_and_phenotype_edges= None,
                         use_shared_gene_but_not_phenotype_edges=None,
                         use_shared_phenotype_but_not_gene_edges=None,
                         use_gene_disease_graph=None,
                         use_phenotype_gene_disease_graph=None,
                         graph_edges_type = None,
                         task = None,
                         split=None,
                         k_fold =None,
                         k_fold_ind = None,
                         split_by_node=None
                         ):
    """

    @param add_qualified_edges: type = Boolean eg True or False:
    @param dataset: type = String eg. GPSim, GeneDisease
    @return: valid path to saved emb_file
    """

    # if added_edges_percent_of is not None:
    #     raise ValueError("not yet implemented ")
    assert save_path_base is not None, "save_path_base must be specified to avoid ambiguity"
    assert emb_type is not None, "emb_type mst be specified to avoide ambiguity "
    # assert add_qualified_edges is not None, "added_qualified_edges must be specified to avoid ambiguity"
    assert dataset is not None, "dataset must be specified to avoid ambiguity"
    assert use_weighted_edges is not None, "use_weighted_edges must be specified to avoid ambiguity "
    assert use_shared_gene_edges is not None, "use_shared_gene_edges must be specified to avoid ambiguity"
    assert use_shared_phenotype_edges is not None, "use_shared_phenotype_edges must be specified to avoid ambiguity"
    assert use_shared_gene_and_phenotype_edges is not None, "use_shared_gene_and_phenotype_edges must be specified to avoid ambiguity"
    assert use_shared_gene_but_not_phenotype_edges is not None, "use_shared_gene_but_not_phenotype_edges must be specified to avoid ambiguity"
    assert use_shared_phenotype_but_not_gene_edges is not None, "use_shared_phenotype_but_not_gene_edges must be specified to avoid ambiguity"
    assert use_gene_disease_graph is not None, "use_gene_disease_graph must be specified to avoid ambiguity"
    assert use_phenotype_gene_disease_graph is not None, "use_phenotype_gene_disease_graph must be specified to avoid ambiguity"
    assert graph_edges_type is not None, "graph_edges_type must be specified to avoid ambiguity"
    assert task is not None, "task must be specified to avoid ambiguity"
    # assert split is not None, "split must be specified to avoid ambiguity"

    # if task ==
    # if k_fold is not None:
    #     assert k_fold_ind is not None, "k_fold_ind must be specified to avoid ambiguity"

    if save_path_base == 'data':
        save_path_base = f"C:\\Users\\Anak\\PycharmProjects\\recreate_gene_disease\\Data\\processed\\"
    elif save_path_base == 'report_performance':
        save_path_base = f"C:\\Users\\Anak\\PycharmProjects\\recreate_gene_disease\\PerformanceResult\\"
        raise ValueError("looks of the saved pd are not readable, This option will be available when I make the saved file readable")

    if task == 'link_prediction':
        task_dir = 'LinkPrediction\\'
        assert split_by_node is not None, "split_by_node must be specified to avoid ambiguity"
        # TODO here>> how should I name folder for k-fold vs split
        # assert split is not None, "split must be specified to avoid ambiguity"
        if split is not None:
            split_dir = f'TrainingSplit=={1 - split}\\'
        elif k_fold is not None:
            assert k_fold_ind is not None, "k_fold_ind must be specified to avoid ambiguity"
            k_fold_dir = f'KFold={k_fold}\\{k_fold_ind}\\'
        else:
            raise ValueError("when task_predictiion, K_fold or split must be used as folder name ")
        if split_by_node:
            split_by_node_dir = 'SplitByNode\\'
        else:
            split_by_node_dir = 'SplitByEdge\\'
    elif task == 'node_classification':
        task_dir = 'NodeClassification\\'
    else:
        raise ValueError("for task, please choose either link_prediction or node_classification")

    if add_qualified_edges is not None:
        assert dataset != "no", "NoAddedEdges Folder can be accessed only when add_qualified_edges is False"
        if edges_percent is not None:
            # note that input of added_edges_percent_of must be valid folder naming (look at notion/principle/ for more information of folder naming)
            assert added_edges_percent_of in ['GeneDisease', 'GPSim','no'], ""
            number_of_added_edges = f"EdgesPercent\\{added_edges_percent_of}\\"
        elif edges_number is not None:
            number_of_added_edges = f"EdgesNumber\\"
        else:
            raise ValueError(
                " when add_qualified_edges is true, only edges_percent and edges_nuber id acceptable")

    if dataset == 'GeneDisease' or dataset == 'no':
        dataset_processed_dir = f"GeneDiseaseProject\\copd\\"
    elif dataset == "GPSim":
        dataset_processed_dir = f"GPSim\\"
    else:
        raise ValueError("please specified dataset that existed or available ")

    if use_gene_disease_graph:
        starter_graph_dir = "GeneDisease\\"
        starter_graph_with_edges_type_dir = starter_graph_dir
    elif use_phenotype_gene_disease_graph:
        starter_graph_dir = "PhenotypeGeneDisease\\"
        if graph_edges_type == 'phenotype_gene_disease':
            graph_edges_type_dir = 'PGD\\'
        elif graph_edges_type == 'phenotype_gene_disease_phenotype':
            graph_edges_type_dir = 'PGDP\\'
        else:
            raise ValueError('graph_edges_type only support for phenotype_gene_disease and phenotype_gene_disease_phenotype')
        starter_graph_with_edges_type_dir  = starter_graph_dir + graph_edges_type_dir
    else:
        raise ValueError("currently there are only 2 options for starter graph\n"
                         "1. use_gene_disease_graph\n"
                         "2. use_phenotype_gene_disease_graph\n")


    if emb_type == 'node2vec':
       emb_type_dir = "Node2Vec\\"
    else:
        raise ValueError("please specified em_type correctly")

    if use_weighted_edges:
        assert dataset != 'no', "not yet implmented: dataset == 'no' and use_weighted_edges is True implies adding weighted gene2disease edges which is not yet implmented "
        weighted_status_dir = "WeightedEdges\\"
    else:
        weighted_status_dir = "UnweightedEdges\\"

    if add_qualified_edges is not None:
        add_edges_status = "AddedEdges\\"

        # code below within add_qualified_edges is not None is folder for "use_initial_qualified_edges_option" condition

        use_initial_qualified_edges_option1 = [use_shared_gene_edges,
                                               use_shared_phenotype_edges]
        use_initial_qualified_edges_option2 = [
            use_shared_gene_and_phenotype_edges,
            use_shared_gene_but_not_phenotype_edges,
            use_shared_phenotype_but_not_gene_edges]
        assert sum(
            use_initial_qualified_edges_option2) <= 1, "no more than 1 of the following can be true at the same time:" \
                                                       "1. shared_gene_and_phenotype_edges OR" \
                                                       "2. use_shared_gene_but_not_phenotype_edges OR " \
                                                       "3. use_shared_phenotype_but_not_gene_edges "
        if sum(use_initial_qualified_edges_option2) > 0:

            assert sum(
                use_initial_qualified_edges_option1) == 0, "option1: use_shared_gene_edges or use_shared_phenotypes_edges are Ture or " \
                                                           "option2: one of the following is true " \
                                                           "         1. shared_gene_and_phenotype_edges OR" \
                                                           "         2. use_shared_gene_but_not_phenotype_edges OR " \
                                                           "         3. use_shared_phenotype_but_not_gene_edges "
            if use_shared_gene_and_phenotype_edges:
                shared_nodes_edges_dir = "SharedGeneAndPhenotypeEdges\\"

            elif use_shared_gene_but_not_phenotype_edges:
                shared_nodes_edges_dir = "SharedGeneNotPhenotypeEdges\\"

            elif use_shared_phenotype_but_not_gene_edges:
                shared_nodes_edges_dir = "SharedPhenotypeNotGeneEdges\\"

            else:
                raise ValueError('For option1, There are 3 option: '
                                 'shared_gene_and_phenotype_edges'
                                 'shared_gene_but_not_phenotype_edges'
                                 'shared_phenotype_but_not_gene_edges')
        else:
            assert sum(
                use_initial_qualified_edges_option1) > 0, "option1: use_shared_gene_edges or use_shared_phenotypes_edges are Ture or " \
                                                          "option2: one of the following is true " \
                                                          "         1. shared_gene_and_phenotype_edges OR" \
                                                          "         2. use_shared_gene_but_not_phenotype_edges OR " \
                                                          "         3. use_shared_phenotype_but_not_gene_edges "

            ## assign folder name of the given option for "use_initial_qualified_edges_option"
            assert dataset == "GeneDisease", "currenlty, only implmented shared_nodes for dataset == GeneDisease"

            if use_shared_phenotype_edges and use_shared_gene_edges:
                shared_nodes_edges_dir = "SharedGeneOrPhenotypeEdges\\"

            elif use_shared_phenotype_edges:
                shared_nodes_edges_dir = "SharedPhenotypeEdges\\"

            elif use_shared_gene_edges :
                shared_nodes_edges_dir = 'SharedGeneEdges\\'

            else:
                raise ValueError('For option1, There are 3 shared noded implmented: shared_gene, shared_phenotype, and shared_gene_and_phenotype')

        # if use_shared_gene_edges:
        #     assert dataset == "GeneDisease", "currenlty, only implmented shared_nodes for dataset == GeneDisease"
        #     shared_nodes_edges_dir = 'SharedGeneEdges\\'
        # elif use_shared_phenotype_edges:
        #     assert dataset == "GeneDisease", "currenlty, only implmented shared_nodes for dataset == GeneDisease"
        #     shared_nodes_edges_dir = "SharedPhenotypeEdges\\"
        # elif use_shared_phenotype_edges and use_shared_gene_edges:
        #     shared_nodes_edges_dir = "SharedGeneAndPhenotypeEdges\\"
        #     # raise ValueError("not yet implemented: first I need to resolve validateion process of shared nodes condition")
        #     # shared_nodes_edges_dir = "SharedGenePhenotype\\"
        # else:
        #     raise ValueError('There are 3 shared noded implmented: shared_gene, shared_phenotype, and shared_gene_and_phenotype')

    else:
        add_edges_status = 'NoAddedEdges\\'

        if use_shared_gene_edges:
            raise ValueError("argument combination is not correct; when NoAddedEdges, no shared_gene should be specified ")
        elif use_shared_phenotype_edges:
            raise ValueError("argument combination is not correct; when NoAddedEdges, no shared_gene should be specified ")
        elif not use_shared_phenotype_edges and not use_shared_gene_edges:
            shared_nodes_edges_dir = None
        else:
            raise ValueError("argument combination is not correct; when NoAddedEdges, no shared_gene should be specified ")

    if shared_nodes_edges_dir is None:
        assert dataset == 'no', "shared_nodes_edges_dir can only be None when NoAddedEdges "
        embedding_model_file_path = save_path_base + task_dir + dataset_processed_dir + starter_graph_with_edges_type_dir + emb_type_dir + weighted_status_dir +   add_edges_status
        # if task == 'link_prediction':
        if task == 'link_prediction':
            embedding_model_file_path += split_by_node_dir
            if split is not None:
                embedding_model_file_path += split_dir
            elif k_fold is not None:
                embedding_model_file_path += k_fold_dir
            else:
                raise ValueError(" ")
    else:
        embedding_model_file_path = save_path_base + task_dir +dataset_processed_dir + starter_graph_with_edges_type_dir + emb_type_dir + weighted_status_dir +  add_edges_status + shared_nodes_edges_dir +number_of_added_edges

    return embedding_model_file_path


def split_train_test(data, split,task):
    if task == 'node_classification':
        raise ValueError('no longer use; split_train_test migrate to dataset.split_train_test')

        # TODO move this whole function of node_classification into split_train_Test
        # # split into train_set and test_set
        # convert_disease2class_id = np.vectorize(
        #     lambda x: data.disease2class_id_dict[x])
        # disease_class = convert_disease2class_id(data.diseases_np)
        #
        # train_set, test_set = data.split_train_test(data.diseases_np,
        #                                             disease_class, split,
        #                                             stratify=disease_class)
        # (x_train, y_train), (x_test, y_test) = train_set, test_set
        # return train_set, test_set

        # return data.split_train_test(split, stratify=True)

    elif task == 'link_prediction':
        raise ValueError('no longer use; split_train_test migrate to dataset.split_train_test')

        # # TODO move split_train_test like prediction to dataset.split_train_testkk
        # # note: I use combination to produce non_directional edges
        # all_possible_gene_disease_edges = list(combinations(np.concatenate(( data.diseases_np, data.genes_np )),2))
        # all_possible_gene_disease_edges = np.unique(np.array(all_possible_gene_disease_edges))
        #
        # all_possible_gene_disease_edges_with_label_dict = {}
        # for edge in all_possible_gene_disease_edges:
        #     if edge in data.edges_np.tolist():
        #         all_possible_gene_disease_edges_with_label_dict.setdefault('existed_edges', []).append(edge)
        #     else:
        #         all_possible_gene_disease_edges_with_label_dict.setdefault('non_existed_edges', []).append(edge)
        #
        # pos_edges = all_possible_gene_disease_edges_with_label_dict['existed_edges']
        # neg_edges = all_possible_gene_disease_edges_with_label_dict['non_existed_edges']
        #
        # # spliiting pos
        # pos_x_train, pos_x_test, pos_y_train, pos_y_test = data.split_train_test(pos_edges, np.ones_like(np.array(pos_edges)), split)
        #
        # # splitting neg
        # # TODO for neg, just select neg of size pos_x_train + pos_x_test ==> pass it to data.split
        # train_test_size =  pos_x_train.shape[0] + pos_x_test.shape[0]
        # selected_neg_edges = np.random.choice(neg_edges, train_test_size,replace=False)
        # neg_x_train, neg_x_test, neg_y_train, neg_y_test = data.split_train_test(neg_edges, np.zeros_like(np.array(neg_edges)), split)
        #
        #
        # # TODO x and y have to be shuffled the same way
        # def concat_and_shuffle(pos_x,pos_y, neg_x,neg_y):
        #     x = np.concatenate([pos_x,neg_x])
        #     y = np.concatenate([pos_y,neg_y])
        #     x_y = [(i,j) for i,j in zip(x,y)]
        #
        #     random.shuffle(x_y)
        #     shuffled_x = np.array(x_y)[:, 0]
        #     shuffled_y =np.array(x_y)[:, 1]
        #     return shuffled_x, shuffled_y
        #
        #
        # shuffled_x_train, shuffled_y_train = concat_and_shuffle(pos_x_train, pos_y_train,  neg_x_train, neg_y_train)
        # shuffled_x_test, shuffled_y_test = concat_and_shuffle(pos_x_test, pos_y_test,  neg_x_test, neg_y_test)
        #
        # train_set, test_set =  (shuffled_x_train, shuffled_y_train), (shuffled_x_test, shuffled_y_test)
        #
        # return train_set, test_set



def apply_normalization(data_with_features, use_saved_emb_file,
                        normalized_weighted_edges):
    """

    @param data_with_features: type = pd.DataFrame
    @param normalized_weighted_edges:
    @return:
    """

    if normalized_weighted_edges:
        assert (
            not use_saved_emb_file), "Cannot normalized embedding data: used_saved_emb_file  = True means data is embededing which implied that it is already normalied."
    # else:
    #     assert ( use_saved_emb_file) , "Cannot normalized embedding data: used_saved_emb_file  = True means data is embededing which implied that it is already normalied."

    if normalized_weighted_edges:
        # x = df.values  # returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(data_with_features)
        normalized_data_with_features = pd.DataFrame(x_scaled,
                                                     columns=data_with_features.columns,
                                                     index=data_with_features.index)

        return normalized_data_with_features
        # return  x_scaled
    else:
        return data_with_features  # not apply normalization


def multiply_constant_multiplier_to_weighted_disease2disease_edges(
        constant_multiplier=10,
        weighted_disease2disease_edges=None):
    """

    @param constant_multiplier: type = float/int
    @param weighted_disease2disease_edges: type = np; shape = (# disease2disease edges, 3) where the last coloumns = weight
    @return:
    """

    assert weighted_disease2disease_edges is not None, "weighted_disease2disease_edges must be specified to avoid ambiguity"

    # multiply weighted diseease2disease edges with constant multiplier
    multiplied_weighted_disease2disease_edges = constant_multiplier * weighted_disease2disease_edges[
                                                                      :,
                                                                      2].astype(
        float)

    # clips max value to 1
    ## note: min should not be specified in clippping to detect weighted edges value error at validation step
    multiplied_weighted_disease2disease_edges = multiplied_weighted_disease2disease_edges.clip(
        max=1)  # output is expected to be converted to str

    # validate that min value = 0
    assert multiplied_weighted_disease2disease_edges.astype(
        float).min() > 0, "no edges should indicate the least weighted value of 0 (no relationship) "

    weighted_disease2disease_edges[:,
    2] = multiplied_weighted_disease2disease_edges

    return weighted_disease2disease_edges


def get_number_of_added_edges(data,all_qualified_edges_df , edges_percent,
                              edges_number, added_edges_percent_of):
    if edges_percent is not None:
        if added_edges_percent_of == 'GeneDisease':
            # note: GeneDisease originally use shared gene
            all_qualified_GeneDisease_disease2disease_edges_df = get_all_GeneDisease_unweighted_disease2disease_qualified_edges(data,
                                                                                                                                use_shared_phenotype_edges=False,
                                                                                                                                use_shared_gene_edges=True,
                                                                                                                                use_shared_gene_and_phenotype_edges = False,
                                                                                                                                use_shared_gene_but_not_phenotype_edges = False,
                                                                                                                                use_shared_phenotype_but_not_gene_edges = False)


            # qualified_edges_df = get_all_GeneDisease_qualified_edges(data,
            #                                     data.G,
            #                                     data.diseases_np)

            number_of_qualified_edges = all_qualified_GeneDisease_disease2disease_edges_df.shape[0]
            number_of_added_edges = int(edges_percent * number_of_qualified_edges)
            return number_of_added_edges

        elif added_edges_percent_of == 'GPSim':
            all_qualified_GPSim_weighted_disease2disease_edges_pd = get_GPSim_disease2disease_qualified_edges(data, False) # what is the expected output of this
            number_of_qualfied_edges = all_qualified_GPSim_weighted_disease2disease_edges_pd.shape[0]
            number_of_added_edges = int(edges_percent * number_of_qualfied_edges)
            return number_of_added_edges

        elif added_edges_percent_of == 'no':

            # all_qualified_GPSim_weighted_disease2disease_edges_pd = get_GPSim_disease2disease_qualified_edges(data, False) # what is the expected output of this

            number_of_qualfied_edges = all_qualified_edges_df.shape[0]
            number_of_added_edges = int(edges_percent * number_of_qualfied_edges)
            return number_of_added_edges

        else:
            raise ValueError("added_percent_edges_of only have 3 options currently including GeneDisease, GPSim, and no")

    if edges_number is not None:
        return edges_number


def get_saved_file_name_for_emb(add_qualified_edges, edges_percent,
                                edges_number, dim, walk_len, num_walks,
                                window):
    """This function is create in case that file name getting increasingly more complicated as development process continues"""

    # assert (edges_number is not None) or (edges_percent is not None), ""
    if add_qualified_edges is not None:
        if edges_number is not None:
            return f'{add_qualified_edges}={edges_number}_dim{dim}_walk_len{walk_len}_num_walks{num_walks}_window{window}.txt'
        elif edges_percent is not None:
            return f'{add_qualified_edges}={edges_percent}_dim{dim}_walk_len{walk_len}_num_walks{num_walks}_window{window}.txt'
        else:
            raise ValueError(
                "only edges_number and edges_pecent is acceptable as subparser for add qulified_edges")
    else:
        return f'dim{dim}_walk_len{walk_len}_num_walks{num_walks}_window{window}.txt'

# def get_all_GPsim_weighted_disease2disease_qualified_edges():
#     # assert isinstance( with_weight , bool), "weith_weight is expected to be type bool"
#     GPSim_cui_qualified_edges_files_path = r'C:\Users\Anak\PycharmProjects\recreate_gene_disease\Data\raw\GPSim\Edges\cui_edges_weight.csv'
#     GPSim_cui_qualified_edges_pd = pd.read_csv(GPSim_cui_qualified_edges_files_path, sep = ',')
#     return GPSim_cui_qualified_edges_pd
#     # if with_weight:
#     #    pass


# def get_all_GeneDisease_unweighted_disease2disease_qualified_edges(data):
#
#     nodes = data.diseases_np
#
#     all_disease2disease_edges = list(combinations(nodes, 2))
#
#     all_disease2disease_qualified_edges = np.array(
#         [edge for edge in all_disease2disease_edges if
#          len(list(nx.common_neighbors(data.G, edge[0], edge[1]))) > 0])
#
#     G = nx.Graph()
#     G.add_edges_from(all_disease2disease_qualified_edges, weight=1)
#     all_qualified_disease2disease_edges_df = nx.to_pandas_edgelist(G)
#
#     return all_qualified_disease2disease_edges_df

# def get_all_disease2disease_GeneDisease_qualified_edges(G, nodes):
# 
#     all_disease2disease_edges = list(combinations(nodes, 2))
#     disease2disease_qualified_edges = np.array(
#         [edge for edge in all_disease2disease_edges if
#          len(list(nx.common_neighbors(G, edge[0], edge[1]))) > 0])
# 
#     return disease2disease_qualified_edges
# 
# def get_all_GeneDisease_qualified_edges(data, G, nodes):
# 
#     all_disease2disease_qualified_edges = get_all_disease2disease_GeneDisease_qualified_edges(G,nodes)
# 
#     G = nx.Graph()
#     G.add_edges_from(all_disease2disease_qualified_edges, weight=1)
#     all_qualified_edges_df = nx.to_pandas_edgelist(G)
# 
#     # # TODO below is the code that produces the the following
#     # # : processed\GeneDiseaseProject\copd\Node2Vec\UnweightedEdges\AddedEdges\EdgesNumber
#     # # : processed\GeneDiseaseProject\copd\Node2Vec\UnweightedEdges\NoAddedEdges\EdgesNumber
#     # all_disease2disease_edges = list(combinations(nodes, 2))
#     # all_disease2disease_qualified_edges = np.array(
#     #     [edge for edge in all_disease2disease_edges if
#     #      len(list(nx.common_neighbors(G, edge[0], edge[1]))) > 0])
#     # G = nx.Graph()
#     # G.add_edges_from(all_disease2disease_edges, weight=1) # I added "all_disease2disease_edges" instead of "all_disease2disease_qualified_edges"
#     # all_qualified_edges_df = nx.to_pandas_edgelist(G)
# 
#     assert not (data.is_graph_edges_weighted(G,
#                                              use_outside_graph=True)), "use_weighted_edges is True, but all edges in graph has weight == 1"
#     return all_qualified_edges_df
