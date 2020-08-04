import os
from os import path

import numpy as np
import pandas as pd
from node2vec import Node2Vec

from Sources.Preparation.Data import GeneDiseaseGeometricDataset
# from Sources.Preparation.Data.conversion import create_grpah_from_emb_file
from Sources.Preparation.Features.split_data import get_k_fold_data
from Sources.Preprocessing.apply_preprocessing import \
    get_saved_file_name_for_emb
from Sources.Preprocessing.apply_preprocessing import \
    remove_edges_from_graph_for_link_prediction
# from Sources.Preparation.Features.build_features import \
#     get_data_without_using_emb_as_feat
# from Sources.Preprocessing.preprocessing import add_disease2disease_edges
from Sources.Preprocessing.apply_preprocessing import select_emb_save_path
from arg_parser import apply_parser_constraint
from arg_parser import args
from arg_parser import run_args_conditions
# from Models.train_model import run_train_model
from global_param import GENEDISEASE_ROOT


def run_node2vec_emb(data,
                     G,
                     embedding_model_file_path,
                     enforce_end2end,
                     add_qualified_edges,
                     use_weighted_edges,
                     edges_percent,
                     edges_number,
                     dim,
                     walk_len,
                     num_walks,
                     window,
                     added_edges_percent_of,
                     emb_type,
                     save_path):
    # dim = 1
    # walk_len = 1
    # num_walks =1
    # window =1

    node2vec = Node2Vec(G, weight_key='weight', dimensions=dim,
                        walk_length=walk_len, num_walks=num_walks, workers=4)
    model = node2vec.fit(window=window, min_count=1, batch_words=4)

    # save model to
    model.wv.save_word2vec_format(save_path)


def run_specified_emb(data=None, G=None, embedding_model_file_path=None,
                      enforce_end2end=None, add_qualified_edges=None,
                      use_weighted_edges=None,
                      edges_percent=None,
                      edges_number=None,
                      dim=64, walk_len=30, num_walks=200,
                      window=10,
                      added_edges_percent_of=None,
                      emb_type=None):
    # def run_node2vec_emb(data=None, G=None, embedding_model_file_path=None,
    #                      enforce_end2end=None, add_qualified_edges=None,
    #                      use_weighted_edges=None,
    #                      edges_percent=None,
    #                      edges_number=None,
    #                      dim=64, walk_len=30, num_walks=200,
    #                      window=10,
    #                      added_edges_percent_of=None):
    """

    @param data:
    @param G:
    @param embedding_model_file_path:
    @param add_qualified_edges: type = str ;  it specified name of edges adding strategy being used
    @param use_weighted_edges:
    @param dim:
    @param walk_len:
    @param num_walks:
    @param window:
    @return:
    """

    # # catching not yet implemented error
    # if added_edges_percent_of is not None:
    #     raise ValueError("not yet implemented")

    assert data is not None, "dataset Class must be explitcitly specified to avoid ambiguity"
    assert G is not None, "Graph of type nx.Graph() must be explicitly specified to avoide ambiguity"
    assert embedding_model_file_path is not None, "please specifiied embedding_model_file_path  to save emb_file "
    # assert add_qualified_edges is not None, "add_qualified_edges must be specified to avoid ambiguity"
    assert use_weighted_edges is not None, "use_weighted_edges must be explicitly specified to avoide ambiguity"
    assert emb_type is not None, "emb_type must be specified to avoid ambiguity"

    file_name = get_saved_file_name_for_emb(add_qualified_edges, edges_percent,
                                            edges_number, dim, walk_len,
                                            num_walks, window)

    # file_name = f'{add_qualified_edges}={}_dim{dim}_walk_len{walk_len}_num_walks{num_walks}_window{window}.txt'
    save_path = embedding_model_file_path + file_name

    print("save emb_file to " + save_path)

    # check that no files with the same name existed within these folder
    if path.exists(save_path):
        if enforce_end2end:
            print(f'emb_file is found at {save_path}')
            return
        else:
            raise ValueError(
                "emb_file already exist, Please check if you argument is correct")

    # assert not path.exists(
    #     save_path), "emb_file already exist, Please check if you argument is correct"

    # create dir if not alreayd existed
    if not os.path.exists(embedding_model_file_path):
        os.makedirs(embedding_model_file_path)

    if use_weighted_edges:
        assert data.is_graph_edges_weighted(G,
                                            use_outside_graph=True), "use_weighted_edges is True, but graph contains no weighted edges (defined as edges with weight != 1)"
    else:
        assert not data.is_graph_edges_weighted(G,
                                                use_outside_graph=True), "use_weighted_edges is Flase, but graph contains weighted edges (defined as edges with weight != 1)"

    import networkx as nx

    edge_weight_dict = {}
    for edge_tuple, weight in nx.get_edge_attributes(G, 'weight').items():
        G[edge_tuple[0]][edge_tuple[1]]['weight'] = float(weight)

        # edge_weight_dict[edge_tuple] = float(weight)
        # assert isinstance(edge_weight_dict[edge_tuple], float), f'edge_weight_dict[edge_tuple] must have type = float '

    # nx.set_node_attributes(G, edge_weight_dict, 'weight')
    if emb_type == 'node2vec':
        # print(f'save_path of node2vec = {save_path}')
        run_node2vec_emb(data,
                         G,
                         embedding_model_file_path,
                         enforce_end2end,
                         add_qualified_edges,
                         use_weighted_edges,
                         edges_percent,
                         edges_number,
                         dim,
                         walk_len,
                         num_walks,
                         window,
                         added_edges_percent_of,
                         emb_type,
                         save_path)
    elif emb_type == 'gcn':
        assert False, 'time to create gcn I assume :) '
        pass
    else:
        raise NotImplementedError()


# def run_node2vec(data, graph_with_added_qualified_edges, data_with_features,
#                  enforce_end2end=None, task=None):

def run_emb(data, graph_with_added_qualified_edges, data_with_features,
            use_saved_emb_file=None,
            add_qualified_edges=None,
            dataset=None, use_weighted_edges=None,
            normalized_weighted_edges=None,
            edges_percent=None,
            edges_number=None,
            added_edges_percent_of=None,
            use_shared_gene_edges=None,
            use_shared_phenotype_edges=None,
            use_shared_gene_and_phenotype_edges=None,
            use_shared_gene_but_not_phenotype_edges=None,
            use_shared_phenotype_but_not_gene_edges=None,
            use_shared_gene_or_phenotype_edges=None,
            use_gene_disease_graph=None,
            use_phenotype_gene_disease_graph=None,
            graph_edges_type=None,
            task=None,
            enforce_end2end=None,
            cross_validation=None,
            k_fold=None,
            split=None,
            get_data_with_emb_as_feat=None,
            split_by_node=None,
            emb_type=None
            ):
    # def run_node2vec(data, graph_with_added_qualified_edges, data_with_features,
    #                  use_saved_emb_file=None,
    #                  add_qualified_edges=None,
    #                  dataset=None, use_weighted_edges=None,
    #                  normalized_weighted_edges=None,
    #                  edges_percent=None,
    #                  edges_number=None,
    #                  added_edges_percent_of=None,
    #                  use_shared_gene_edges=None,
    #                  use_shared_phenotype_edges=None,
    #                  use_shared_gene_and_phenotype_edges=None,
    #                  use_shared_gene_but_not_phenotype_edges=None,
    #                  use_shared_phenotype_but_not_gene_edges=None,
    #                  use_gene_disease_graph=None,
    #                  use_phenotype_gene_disease_graph=None,
    #                  graph_edges_type=None,
    #                  task=None,
    #                  enforce_end2end=None,
    #                  cross_validation=None,
    #                  k_fold=None,
    #                  split=None,
    #                  get_data_with_emb_as_feat=None,
    #                  split_by_node = None
    #                  ):

    # assert data is not None, 'data must be explicitly stated to avoid ambiguity'
    assert use_saved_emb_file is not None, 'use_saved_emb_file must be explicitly stated to avoid ambiguity'
    assert dataset is not None, 'dataset must be explicitly stated to avoid ambiguity'
    assert use_weighted_edges is not None, 'use_weighted_edges must be explicitly stated to avoid ambiguity'
    assert normalized_weighted_edges is not None, "normalized_weighted_edges must be specified to avoid ambiguity"
    assert use_shared_gene_edges is not None, "use_shared_gene_edges must be specified to avoid ambiguity"
    assert use_shared_phenotype_edges is not None, "use_shared_phenotype_edges must be specified to avoid ambiguity"
    assert use_shared_gene_and_phenotype_edges is not None, "use_shared_gene_and_phenotype_edges must be specified to avoid ambiguity"
    assert use_shared_gene_but_not_phenotype_edges is not None, "use_shared_gene_but_not_phenotype_edges must be specified to avoid ambiguity"
    assert use_shared_phenotype_but_not_gene_edges is not None, "use_shared_phenotype_but_not_gene_edges must be specified to avoid ambiguity"
    assert use_gene_disease_graph is not None, "use_gene_disease_graph must be specified to avoid ambiguity"
    assert use_phenotype_gene_disease_graph is not None, " use_phenotype_gene_disease_graph must be specified to avoid ambiguity"
    # assert graph_edges_type is not None, "graph_edges_type must be specified to avoid ambiguity"
    assert task is not None, "task must be specified to avoid ambiguity"
    assert enforce_end2end is not None, "enforce_end2end must be specified to avoid ambiguity"
    assert cross_validation is not None, "cross_validation must be specified to avoid ambiguity"
    assert enforce_end2end is not None, "enforce_end2end must be specified to avoid ambiguity"
    assert task is not None, "task must be specified to avoid ambiguity"
    assert get_data_with_emb_as_feat is not None, "get_data_with_added_feat must be specified to avoid ambiguity"
    assert split_by_node is not None, "split_by_node must be specified to avoid ambiguity"
    assert emb_type is not None, "emb_type must be specified to avoid ambiguity"

    # =====================
    # == run embedding (it should save result in appropriate folder within Data
    # =====================
    if task == 'link_prediction':

        # assert not data.is_disease2disease_edges_added_to_graph(outside_graph=graph_with_added_qualified_edges, use_outside_graph=True), 'link_prediction does not support grpaph with disease2diseaes edes'

        graph_with_no_added_qualified_edges = graph_with_added_qualified_edges

        # # validate that graph of link_prediction has no disease2disease edges
        # assert not data.is_disease2disease_edges_added_to_graph(
        #     outside_graph=graph_with_no_added_qualified_edges,
        #     use_outside_graph=True), "currently, link_prediction only support PGDP graph with no disease2disease edges added "

        if split is not None:

            embedding_model_file_path = select_emb_save_path(
                save_path_base='data',
                # emb_type='node2vec',
                emb_type=emb_type,
                add_qualified_edges=add_qualified_edges,
                dataset=dataset,
                use_weighted_edges=use_weighted_edges,
                edges_percent=edges_percent,
                edges_number=edges_number,
                added_edges_percent_of=added_edges_percent_of,
                use_shared_phenotype_edges=use_shared_phenotype_edges,
                use_shared_gene_edges=use_shared_gene_edges,
                use_shared_gene_and_phenotype_edges=use_shared_gene_and_phenotype_edges,
                use_shared_gene_but_not_phenotype_edges=use_shared_gene_but_not_phenotype_edges,
                use_shared_phenotype_but_not_gene_edges=use_shared_phenotype_but_not_gene_edges,
                use_shared_gene_or_phenotype_edges=use_shared_gene_or_phenotype_edges,
                use_gene_disease_graph=use_gene_disease_graph,
                use_phenotype_gene_disease_graph=use_phenotype_gene_disease_graph,
                graph_edges_type=graph_edges_type,
                task=task,
                split=split,
                k_fold=k_fold,
                split_by_node=split_by_node
            )

            # TODO make this function compatible with k_fold => what do I need
            splitted_edges_dir = f'C:\\Users\\Anak\\PycharmProjects\\recreate_gene_disease\\Data\\processed\\LinkPrediction\\GeneDiseaseProject\\copd\\PhenotypeGeneDisease\\PGDP\\Node2Vec\\UnweightedEdges\\NoAddedEdges\\'
            if split_by_node:
                splitted_edges_dir += f'SplitByNode\\TrainingSplit=={1 - split}\\'
            else:
                splitted_edges_dir += f'SplitByEdge\\TrainingSplit=={1 - split}\\'
            train_set_np, test_set_np = data.split_train_test(split,
                                                              stratify=True,
                                                              task=task,
                                                              splitted_edges_dir=splitted_edges_dir,
                                                              split_by_node=split_by_node)
            # reset_train_test_split=True)

            # TODO paragraph below has been moved to run_node2vec_emb_with_removed_edges_from_graph_for_link_prediction
            # graph_with_no_added_qualified_edges_with_removed_test_edges = remove_edges_from_graph_for_link_prediction(
            #     data, graph_with_no_added_qualified_edges, train_set_np, split)

            import networkx as nx
            pos_train_bool_np = [(float(i) == 1) for i in train_set_np[:, -1]]
            pos_train_set_np = train_set_np[pos_train_bool_np]

            graph_with_no_added_qualified_edges_with_removed_test_edges = nx.Graph()
            graph_with_no_added_qualified_edges_with_removed_test_edges.add_edges_from(
                pos_train_set_np[:, :2], weight=1)
            # graph_with_no_added_qualified_edges_with_removed_test_edges = nx.from_edgelist(train_set_np[:, :2])
            # graph_with_no_added_qualified_edges_with_removed_test_edges = remove_edges_from_graph_for_link_prediction(
            #     data, graph_with_no_added_qualified_edges, train_set_np, test_set_np)

            run_specified_emb(data=data,
                              G=graph_with_no_added_qualified_edges_with_removed_test_edges,
                              embedding_model_file_path=embedding_model_file_path,
                              # enforce_end2end=enforce_end2end,
                              edges_percent=edges_percent,
                              edges_number=edges_number,
                              use_weighted_edges=use_weighted_edges,
                              add_qualified_edges=add_qualified_edges,
                              added_edges_percent_of=added_edges_percent_of,
                              enforce_end2end=enforce_end2end,
                              emb_type=emb_type
                              )

            # node_with_feature = get_data_with_emb_as_feat(data, graph_with_added_qualified_edges, data_with_features,
            #                                                 use_saved_emb_file=use_saved_emb_file,
            #                                                 add_qualified_edges=add_qualified_edges,
            #                                                 dataset=dataset, use_weighted_edges=use_weighted_edges,
            #                                                 normalized_weighted_edges=normalized_weighted_edges,
            #                                                 edges_percent=edges_percent,
            #                                                 edges_number=edges_number,
            #                                                 added_edges_percent_of=added_edges_percent_of,
            #                                                 use_shared_gene_edges=use_shared_gene_edges,
            #                                                 use_shared_phenotype_edges=use_shared_phenotype_edges,
            #                                                 use_shared_gene_and_phenotype_edges=use_shared_gene_and_phenotype_edges,
            #                                                 use_shared_gene_but_not_phenotype_edges=use_shared_gene_but_not_phenotype_edges,
            #                                                 use_shared_phenotype_but_not_gene_edges=use_shared_phenotype_but_not_gene_edges,
            #                                                 use_shared_gene_or_phenotype_edges=use_shared_gene_or_phenotype_edges,
            #                                                 use_gene_disease_graph=use_gene_disease_graph,
            #                                                 use_phenotype_gene_disease_graph=use_phenotype_gene_disease_graph,
            #                                                 graph_edges_type=graph_edges_type,
            #                                                 task=task,
            #                                                 enforce_end2end=enforce_end2end,
            #                                                 cross_validation=cross_validation,
            #                                                 k_fold=k_fold,
            #                                                 split=split,
            #                                                 get_data_with_emb_as_feat=get_data_with_emb_as_feat,
            #                                                 split_by_node = split_by_node,
            #                                                 emb_type=emb_type)

            node_with_feature = get_data_with_emb_as_feat(data,
                                                          # use_saved_emb_file,
                                                          add_qualified_edges,
                                                          dataset,
                                                          use_weighted_edges,
                                                          normalized_weighted_edges,
                                                          edges_percent,
                                                          edges_number,
                                                          added_edges_percent_of,
                                                          use_shared_gene_edges,
                                                          use_shared_phenotype_edges,
                                                          use_shared_gene_and_phenotype_edges,
                                                          use_shared_gene_but_not_phenotype_edges,
                                                          use_shared_phenotype_but_not_gene_edges,
                                                          use_shared_gene_or_phenotype_edges,
                                                          use_gene_disease_graph,
                                                          use_phenotype_gene_disease_graph,
                                                          graph_edges_type,
                                                          task,
                                                          split,
                                                          k_fold,
                                                          split_by_node
                                                          # need it to choose file name
                                                          )

            # TODO return edges_with_features
            return node_with_feature

            # TODO paragraph below has been moved to removed_edges_from_graph_for_link_prediction
            # train_set_edges = set(map(tuple, train_set_np[:, :2].tolist()))
            # assert len(train_set_edges) == int(len(data.gene_disease_edges) * (1-split)) * 2, "train_set_edges = int(len(original_gene_disease_edges * training_split)) * 2"
            #
            # count_removed_edges = 0
            # for edge in data.gene_disease_edges:  # np or pd ?
            #     if graph_with_no_added_qualified_edges.has_edge(*edge):
            #         edge_with_gene_as_str = tuple(str(i) for i in
            #                                       edge)  # I hould have use gene as str in the first place, I instead used int, so I am *patching* that mistake
            #         if edge_with_gene_as_str not in train_set_edges:
            #             count_removed_edges += 1
            #             graph_with_no_added_qualified_edges.remove_edge(*edge)
            #     else:
            #         print(edge)
            #         raise ValueError(
            #             "some edges from dataset.gene_disease-edges is not contained in starter_graph (aka. graph_with_no_added_qualified_edges)")

            # run_node2vec_emb(data=data, G=graph_with_no_added_qualified_edges,
            #                  embedding_model_file_path=embedding_model_file_path,
            #                  # enforce_end2end=enforce_end2end,
            #                  edges_percent=edges_percent,
            #                  edges_number=edges_number,
            #                  use_weighted_edges=use_weighted_edges,
            #                  add_qualified_edges=add_qualified_edges,
            #                  added_edges_percent_of=added_edges_percent_of,
            #                  enforce_end2end=enforce_end2end
            #                  )

        elif k_fold is not None:
            get_k_fold_data(run_emb, data, graph_with_no_added_qualified_edges,
                            use_saved_emb_file=use_saved_emb_file,
                            add_qualified_edges=add_qualified_edges,
                            dataset=dataset,
                            use_weighted_edges=use_weighted_edges,
                            normalized_weighted_edges=normalized_weighted_edges,
                            edges_percent=edges_percent,
                            edges_number=edges_number,
                            added_edges_percent_of=added_edges_percent_of,
                            use_shared_gene_edges=use_shared_gene_edges,
                            use_shared_phenotype_edges=use_shared_phenotype_edges,
                            use_shared_gene_and_phenotype_edges=use_shared_gene_and_phenotype_edges,
                            use_shared_gene_but_not_phenotype_edges=use_shared_gene_but_not_phenotype_edges,
                            use_shared_phenotype_but_not_gene_edges=use_shared_phenotype_but_not_gene_edges,
                            use_shared_gene_or_phenotype_edges=use_shared_gene_or_phenotype_edges,
                            use_gene_disease_graph=use_gene_disease_graph,
                            use_phenotype_gene_disease_graph=use_phenotype_gene_disease_graph,
                            graph_edges_type=graph_edges_type,
                            task=task,
                            enforce_end2end=enforce_end2end,
                            cross_validation=cross_validation,
                            k_fold=k_fold,
                            split=split,
                            get_data_with_emb_as_feat=get_data_with_emb_as_feat,
                            split_by_node=split_by_node)
        else:
            raise ValueError(' ')

    elif task == 'node_classification':

        # Note: Testing get_k_fold_data function
        # assert not data.is_disease2disease_edges_added_to_graph(outside_graph=graph_with_added_qualified_edges, use_outside_graph=True), 'link_prediction does not support grpaph with disease2diseaes edes'
        #
        # graph_with_no_added_qualified_edges = graph_with_added_qualified_edges
        #
        # get_k_fold_data(run_node2vec_emb, data, graph_with_no_added_qualified_edges,
        #          use_saved_emb_file=use_saved_emb_file,
        #          add_qualified_edges=add_qualified_edges,
        #          dataset=dataset, use_weighted_edges=use_weighted_edges,
        #          normalized_weighted_edges=normalized_weighted_edges,
        #          edges_percent=edges_percent,
        #          edges_number=edges_number,
        #          added_edges_percent_of=added_edges_percent_of,
        #          use_shared_gene_edges=use_shared_gene_edges,
        #          use_shared_phenotype_edges=use_shared_phenotype_edges,
        #          use_shared_gene_and_phenotype_edges=use_shared_gene_and_phenotype_edges,
        #          use_shared_gene_but_not_phenotype_edges=use_shared_gene_but_not_phenotype_edges,
        #          use_shared_phenotype_but_not_gene_edges=use_shared_phenotype_but_not_gene_edges,
        #          use_gene_disease_graph=use_gene_disease_graph,
        #          use_phenotype_gene_disease_graph=use_phenotype_gene_disease_graph,
        #          graph_edges_type=graph_edges_type,
        #          task=task,
        #          enforce_end2end=enforce_end2end,
        #          cross_validation=cross_validation,
        #          k_fold=k_fold,
        #          split=split,
        #          get_data_with_emb_as_feat=get_data_with_emb_as_feat,
        #          split_by_node = split_by_node)

        # Note: the paragaph below will be remove if get_k_fold_data work. it is currently being tested
        # Bug: Figure out why node_classification does not have emb folder that specified split type (by node vs by edges)
        embedding_model_file_path = select_emb_save_path(save_path_base='data',
                                                         # emb_type='node2vec',
                                                         # emb_type=emb_type,
                                                         add_qualified_edges=add_qualified_edges,
                                                         dataset=dataset,
                                                         use_weighted_edges=use_weighted_edges,
                                                         edges_percent=edges_percent,
                                                         edges_number=edges_number,
                                                         added_edges_percent_of=added_edges_percent_of,
                                                         use_shared_phenotype_edges=use_shared_phenotype_edges,
                                                         use_shared_gene_edges=use_shared_gene_edges,
                                                         use_shared_gene_and_phenotype_edges=use_shared_gene_and_phenotype_edges,
                                                         use_shared_gene_but_not_phenotype_edges=use_shared_gene_but_not_phenotype_edges,
                                                         use_shared_phenotype_but_not_gene_edges=use_shared_phenotype_but_not_gene_edges,
                                                         use_shared_gene_or_phenotype_edges=use_shared_gene_or_phenotype_edges,
                                                         use_gene_disease_graph=use_gene_disease_graph,
                                                         use_phenotype_gene_disease_graph=use_phenotype_gene_disease_graph,
                                                         graph_edges_type=graph_edges_type,
                                                         task=task,
                                                         split=split,
                                                         k_fold=k_fold
                                                         )

        if emb_type == 'node2vec':
            # TODO here>>-4 add link_prediction and node_classification to node_classification
            run_specified_emb(data=data, G=graph_with_added_qualified_edges,
                              embedding_model_file_path=embedding_model_file_path,
                              # enforce_end2end=enforce_end2end,
                              edges_percent=edges_percent,
                              edges_number=edges_number,
                              use_weighted_edges=use_weighted_edges,
                              add_qualified_edges=add_qualified_edges,
                              added_edges_percent_of=added_edges_percent_of,
                              enforce_end2end=enforce_end2end
                              )

            node_with_feature = get_data_with_emb_as_feat(data,
                                                          graph_with_added_qualified_edges,
                                                          data_with_features,
                                                          use_saved_emb_file=use_saved_emb_file,
                                                          add_qualified_edges=add_qualified_edges,
                                                          dataset=dataset,
                                                          use_weighted_edges=use_weighted_edges,
                                                          normalized_weighted_edges=normalized_weighted_edges,
                                                          edges_percent=edges_percent,
                                                          edges_number=edges_number,
                                                          added_edges_percent_of=added_edges_percent_of,
                                                          use_shared_gene_edges=use_shared_gene_edges,
                                                          use_shared_phenotype_edges=use_shared_phenotype_edges,
                                                          use_shared_gene_and_phenotype_edges=use_shared_gene_and_phenotype_edges,
                                                          use_shared_gene_but_not_phenotype_edges=use_shared_gene_but_not_phenotype_edges,
                                                          use_shared_phenotype_but_not_gene_edges=use_shared_phenotype_but_not_gene_edges,
                                                          use_shared_gene_or_phenotype_edges=use_shared_gene_or_phenotype_edges,
                                                          use_gene_disease_graph=use_gene_disease_graph,
                                                          use_phenotype_gene_disease_graph=use_phenotype_gene_disease_graph,
                                                          graph_edges_type=graph_edges_type,
                                                          task=task,
                                                          # enforce_end2end=enforce_end2end,
                                                          # cross_validation=cross_validation,
                                                          k_fold=k_fold,
                                                          split=split,
                                                          # get_data_with_emb_as_feat=get_data_with_emb_as_feat,
                                                          split_by_node=split_by_node)

            data_with_features = get_data_with_emb_as_feat(data,
                                                           # use_saved_emb_file,
                                                           add_qualified_edges,
                                                           dataset,
                                                           use_weighted_edges,
                                                           normalized_weighted_edges,
                                                           edges_percent,
                                                           edges_number,
                                                           added_edges_percent_of,
                                                           use_shared_gene_edges,
                                                           use_shared_phenotype_edges,
                                                           use_shared_gene_and_phenotype_edges,
                                                           use_shared_gene_but_not_phenotype_edges,
                                                           use_shared_phenotype_but_not_gene_edges,
                                                           use_gene_disease_graph,
                                                           use_phenotype_gene_disease_graph,
                                                           graph_edges_type,
                                                           task,
                                                           split,
                                                           k_fold,
                                                           split_by_node=split_by_node
                                                           # need it to choose file name
                                                           )

            return data_with_features
        elif emb_type == 'gcn':
            raise NotImplementedError('')
        else:
            raise NotImplementedError()

    else:
        raise ValueError(
            "task only accept link_prediction or node_classification as its value")


if __name__ == '__main__':
    """
    example of running node2vec_emb.py
       --use_phenotype_gene_disease_graph --graph_edges_type phenotype_gene_disease_phenotype --add_qualified_edges top_k --dataset GeneDisease --edges_percent 0.05 --added_edges_percent_of GeneDisease --use_shared_phenotype_edges 
       --run_multiple_args_conditions node2vec
    """

    from Sources.Preparation.Features.build_features import \
        get_data_without_using_emb_as_feat

    if args.run_multiple_args_conditions is not None:
        assert args.run_multiple_args_conditions == 'node2vec', "you can only run args.run_multiple_args_conditions == 'train_model' in train_model.py "
        run_args_conditions(run_emb, apply_parser_constraint,
                            args.run_multiple_args_conditions)
    else:
        apply_parser_constraint()
        # =====================
        # == Datsets
        # =====================
        ## GeneDisease
        # GeneDisease_root = r'c:\users\anak\pycharmprojects\recreate_gene_disease\data'  # indicate where file should be stored
        # data = GeneDiseaseGeometricDataset(GeneDisease_root)

        data = GeneDiseaseGeometricDataset(GENEDISEASE_ROOT)

        # # split into train_set and test_set
        # convert_disease2class_id = np.vectorize(lambda x: data.disease2class_id_dict[x])
        # disease_class = convert_disease2class_id(data.diseases_np)
        # train_set, test_set = data.split_train_test(data.diseases_np,
        #                                             disease_class, 0.4)
        #
        # (x_train, y_train), (x_test, y_test) = train_set, test_set

        # =====================
        # == Preprocessing
        # =====================

        graph_with_added_qualified_edges, data_with_features = get_data_without_using_emb_as_feat(
            data=data,
            add_qualified_edges=args.add_qualified_edges,
            dataset=args.dataset,
            use_weighted_edges=args.use_weighted_edges,
            normalized_weighted_edges=args.normalized_weighted_edges,
            # return_graph_and_data_with_features = True,
            # use_saved_emb_file=args.use_saved_emb_file,
            edges_number=args.edges_number,
            edges_percent=args.edges_percent,
            added_edges_percent_of=args.added_edges_percent_of,
            use_shared_phenotype_edges=args.use_shared_phenotype_edges,
            use_shared_gene_edges=args.use_shared_gene_edges,
            use_shared_gene_and_phenotype_edges=args.use_shared_gene_and_phenotype_edges,
            use_shared_gene_but_not_phenotype_edges=args.use_shared_gene_but_not_phenotype_edges,
            use_shared_phenotype_but_not_gene_edges=args.use_shared_phenotype_but_not_gene_edges,
            use_gene_disease_graph=args.use_gene_disease_graph,
            use_phenotype_gene_disease_graph=args.use_phenotype_gene_disease_graph,
            use_shared_gene_or_phenotype_edges=args.use_shared_gene_or_phenotype_edges,
            graph_edges_type=args.graph_edges_type,
        )

        # run_node2vec(data, graph_with_added_qualified_edges,
        #              data_with_features,
        #              enforce_end2end=args.enforce_end2end,
        #              task=args.task)

        run_emb(data, graph_with_added_qualified_edges,
                data_with_features,
                use_saved_emb_file=args.use_saved_emb_file,
                add_qualified_edges=args.add_qualified_edges,
                dataset=args.dataset,
                use_weighted_edges=args.use_weighted_edges,
                normalized_weighted_edges=args.normalized_weighted_edges,
                edges_percent=args.edges_percent,
                edges_number=args.edges_number,
                added_edges_percent_of=args.added_edges_percent_of,
                use_shared_gene_edges=args.use_shared_gene_edges,
                use_shared_phenotype_edges=args.use_shared_phenotype_edges,
                use_shared_gene_and_phenotype_edges=args.use_shared_gene_and_phenotype_edges,
                use_shared_gene_but_not_phenotype_edges=args.use_shared_gene_but_not_phenotype_edges,
                use_shared_phenotype_but_not_gene_edges=args.use_shared_phenotype_but_not_gene_edges,
                use_shared_gene_or_phenotype_edges=args.use_shared_gene_or_phenotype_edges,
                use_gene_disease_graph=args.use_gene_disease_graph,
                use_phenotype_gene_disease_graph=args.use_phenotype_gene_disease_graph,
                graph_edges_type=args.graph_edges_type,
                task=args.task,
                enforce_end2end=args.enforce_end2end,
                cross_validation=args.cross_validation,
                k_fold=args.k_fold,
                split=args.split,
                split_by_node=args.split_by_node)

    # # =====================
    # # == Datsets
    # # =====================
    # ## GeneDisease
    # GeneDisease_root = r'c:\users\anak\pycharmprojects\recreate_gene_disease\data'  # indicate where file should be stored
    # data = GeneDiseaseGeometricDataset(GeneDisease_root)
    #
    # # # split into train_set and test_set
    # # convert_disease2class_id = np.vectorize(lambda x: data.disease2class_id_dict[x])
    # # disease_class = convert_disease2class_id(data.diseases_np)
    # # train_set, test_set = data.split_train_test(data.diseases_np,
    # #                                             disease_class, 0.4)
    # #
    # # (x_train, y_train), (x_test, y_test) = train_set, test_set
    #
    # # =====================
    # # == Preprocessing
    # # =====================
    # graph_with_added_qualified_edges, data_with_features = get_data_without_using_emb_as_feat(
    #     data=data,
    #     add_qualified_edges=args.add_qualified_edges,
    #     dataset=args.dataset,
    #     use_weighted_edges=args.use_weighted_edges,
    #     normalized_weighted_edges=args.normalized_weighted_edges,
    #     return_graph_and_data_with_features = True,
    #     use_saved_emb_file=args.use_saved_emb_file,
    #     edges_number = args.edges_number,
    #     edges_percent= args.edges_percent,
    #     added_edges_percent_of= args.added_edges_percent_of,
    #     use_shared_phenotype_edges=args.use_shared_phenotype_edges,
    #     use_shared_gene_edges = args.use_shared_gene_edges,
    #     use_shared_gene_and_phenotype_edges=args.use_shared_gene_and_phenotype_edges,
    #     use_shared_gene_but_not_phenotype_edges=args.use_shared_gene_but_not_phenotype_edges,
    #     use_shared_phenotype_but_not_gene_edges=args.use_shared_phenotype_but_not_gene_edges)
    #
    # emb_type = "node2vec"
    # embedding_model_file_path = select_emb_save_path(emb_type=emb_type,
    #                                                  add_qualified_edges=args.add_qualified_edges,
    #                                                  dataset=args.dataset,
    #                                                  use_weighted_edges=args.use_weighted_edges,
    #                                                  edges_percent = args.edges_percent,
    #                                                  edges_number = args.edges_number,
    #                                                  added_edges_percent_of = args.added_edges_percent_of,
    #                                                  use_shared_phenotype_edges=args.use_shared_phenotype_edges,
    #                                                  use_shared_gene_edges=args.use_shared_gene_edges,
    #                                                  use_shared_gene_and_phenotype_edges=args.use_shared_gene_and_phenotype_edges,
    #                                                  use_shared_gene_but_not_phenotype_edges=args.use_shared_gene_but_not_phenotype_edges,
    #                                                  use_shared_phenotype_but_not_gene_edges=args.use_shared_phenotype_but_not_gene_edges)
    #
    # # =====================
    # # == run embedding (it should save result in appropriate folder within Data
    # # =====================
    # run_node2vec_emb(data=data, G=graph_with_added_qualified_edges,
    #              embedding_model_file_path=embedding_model_file_path,
    #              edges_percent= args.edges_percent,
    #              edges_number = args.edges_number,
    #              use_weighted_edges=args.use_weighted_edges,
    #              add_qualified_edges = args.add_qualified_edges,
    #              added_edges_percent_of=args.added_edges_percent_of
    #              )
