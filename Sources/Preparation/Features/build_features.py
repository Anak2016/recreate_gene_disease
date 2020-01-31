import networkx as nx
import numpy as np
import pandas as pd

from Sources.Preparation.Features.get_qualified_edges import \
    get_GPSim_disease2disease_qualified_edges
from Sources.Preparation.Features.get_qualified_edges import \
    get_GeneDisease_disease2disease_qualified_edges
from Sources.Preparation.Features.select_strategy import \
    apply_edges_adding_strategies
from Sources.Preprocessing.preprocessing import apply_normalization
from Sources.Preprocessing.preprocessing import get_number_of_added_edges
from Sources.Preprocessing.preprocessing import \
    multiply_constant_multiplier_to_weighted_disease2disease_edges


def get_instances_with_features(graph_with_added_edges=None,
                                path_to_saved_emb_file=None,
                                use_saved_emb_file=False,
                                normalized_weighted_edges=None):
    """

    @param graph_with_added_edges: nx.Graph with added edges
    @param use_saved_emb_file:
    @return: adj_df: type = pandas.DataFrame; shape = (# of instance, # of features)

    """
    assert use_saved_emb_file is not None, "use_saved_emb_file must be specified to avoid ambiguity"
    assert normalized_weighted_edges is not None, "normalized_weighted_edges must be specified to avoid ambiguity"

    if use_saved_emb_file:
        assert path_to_saved_emb_file is not None, "if use_saved_emb_file is not None, please specified path to emb_file"

        # emb_df = pd.read_csv(path_to_saved_emb_file, sep=' ', skiprows=0)
        df = pd.read_csv(path_to_saved_emb_file, sep=' ', skiprows=0, header=1)
        emb_df = pd.DataFrame(
            np.insert(df.values, 0, values=list(df.columns), axis=0))
        print(emb_df.head())
        emb_df.set_index(0, inplace=True)
        del emb_df.index.name
        emb_df.columns = list(range(emb_df.shape[1]))

    else:
        assert graph_with_added_edges is not None, "if use_saved_emb_file is False, please pass in nx.Graph object with added egdes"
        emb_df = nx.to_pandas_adjacency(graph_with_added_edges)

    # choose to activate apply normalization or not
    normalized_emb_df = apply_normalization(emb_df,
                                            use_saved_emb_file,
                                            normalized_weighted_edges)
    return normalized_emb_df


def get_data_without_using_emb_as_feat(data=None,
                                       add_qualified_edges=None,
                                       dataset=None,
                                       use_weighted_edges=None,
                                       normalized_weighted_edges=None,
                                       return_graph_and_data_with_features=None,
                                       use_saved_emb_file=None,
                                       edges_percent=None,
                                       edges_number=None,
                                       added_edges_percent_of=None):
    """

    @param data: type = nx.Graph()
    @param add_qualified_edges: type = Boolean;
    @param dataset: type = String eg. GPSim, GeneDisease; specified dataset to be used by its named
    @return:
    """

    assert data is not None, 'data must be explicitly stated to avoid ambiguity'
    # assert add_qualified_edges is not None, 'add_qualified_edges must be explicitly stated to avoid ambiguity'
    assert dataset is not None, 'dataset must be explicitly stated to avoid ambiguity'
    assert use_weighted_edges is not None, 'use_weighted_eFalsedges must be explicitly stated to avoid ambiguity'
    assert normalized_weighted_edges is not None, 'use_weighted_eFalsedges must be explicitly stated to avoid ambiguity'
    assert return_graph_and_data_with_features is not None, 'use_weighted_eFalsedges must be explicitly stated to avoid ambiguity'
    assert use_saved_emb_file is not None, "use_saved_emb_file must be speherecified to avoid ambiguity"
    assert (edges_percent is not None) or (
                edges_number is not None), "either edges_percent or edges_number must be specified to avoid ambiguity"
    # assert edges_percent is not None, "edges_percent must be specified to avoid ambiguity"
    # assert edges_number is not None, "edges_number must be specified to avoid ambiguity"

    if add_qualified_edges is not None:
        ## select disease2disease edges
        ### for all edges if it has common neighbor select them
        if dataset == "GeneDisease":

            all_qualified_disease2disease_edges_df = get_GeneDisease_disease2disease_qualified_edges(
                data,
                use_weighted_edges)  # expect to return df

        elif dataset == 'GPSim':
            all_qualified_disease2disease_edges_df = get_GPSim_disease2disease_qualified_edges(
                data,
                use_weighted_edges)  #
        # elif dataset == 'no':

        else:
            raise ValueError(
                'please specify existed/available dataset eg GeneDisease or GPSim')

        ## applly strategies to add edges from qualified_edges
        # TODO implementing applying_edges_adding_strategies
        number_of_added_edges = get_number_of_added_edges(
            data,
            add_qualified_edges, edges_percent, edges_number,
            added_edges_percent_of)

        disease2disease_edges_to_be_added = apply_edges_adding_strategies(
            add_qualified_edges,
            data.G,
            all_qualified_disease2disease_edges_df.to_numpy(),
            number_of_added_edges)


        disease2disease_edges_to_be_added = multiply_constant_multiplier_to_weighted_disease2disease_edges(
            constant_multiplier=1,
            weighted_disease2disease_edges=disease2disease_edges_to_be_added)

        data.add_weighted_qualified_edges_to_graph(
            disease2disease_edges_to_be_added)
        graph_with_added_disease2disease = data.G.copy()

        assert data.is_disease2disease_edges_added_to_graph(
            graph_with_added_disease2disease,
            use_outside_graph=True), "added_qualified_edges is True, but no disease2disease are added "

        data_with_features = get_instances_with_features(
            graph_with_added_disease2disease,
            use_saved_emb_file=use_saved_emb_file,
            normalized_weighted_edges=normalized_weighted_edges)

        # returning graph with added disease2disease edges
        if return_graph_and_data_with_features:
            return graph_with_added_disease2disease, data_with_features


    else:
        assert dataset == 'no', 'no need to specified dataset if no qualified edges will be added (this prevent unexpected beahabior that could be caused by unintentionally provided dataset as argument)'

        data_with_features = get_instances_with_features(data.G,
                                                         use_saved_emb_file=use_saved_emb_file,
                                                         normalized_weighted_edges=normalized_weighted_edges)

        # return graph with no added edges
        if return_graph_and_data_with_features:
            return data.G, data_with_features

    return data_with_features


def get_data_with_emb_as_feat(data,
                              use_saved_emb_file,
                              add_qualified_edges,
                              dataset,
                              use_weighted_edges,
                              normalized_weighted_edges,
                              edges_percent,
                              edges_number,
                              added_edges_percent_of):
    # catching not yet implmented conditions
    if added_edges_percent_of is not None:
        raise ValueError("not yet implemented")

    # TODO implement added_edges_percent_of when edgs_percent is not None
    # TODO implement edges_percent and edges_nubmer when add_qualified_edges is true.
    # TODO add weight of gene2disease edges
    # TODO add normalized weight edges (min-max normalization)
    if use_weighted_edges:
        if add_qualified_edges is not None:
            if dataset == 'GeneDisease':
                raise ValueError(
                    "no yet implemented => haven't yet created a emb_file for --use_weight_edges --dataset GeneDisease --add_qualified_edges")

            elif dataset == 'GPSim':
                path_to_saved_emb_file = r"C:\Users\Anak\PycharmProjects\recreate_gene_disease\Data\processed\GPSim\Node2Vec\WeightedEdges\AddedEdges\EdgesNumber\top_k=24_dim64_walk_len30_num_walks200_window10.txt"
            else:
                raise ValueError("correct/availble dataaset must be specified")
        else:
            assert dataset == 'no', 'no need to specified dataset if no qualified edges will be added (this prevent unexpected beahabior that could be caused by unintentionally provided dataset as argument)'
            raise ValueError(
                "no yet implemented => currently there is no weighted_edges that run with node2vec.")

    else:
        if add_qualified_edges is not None:
            if dataset == 'GeneDisease':
                # path_to_saved_emb_file = r'C:\Users\Anak\PycharmProjects\recreate_gene_disease\Data\processed\GeneDiseaseProject\copd\Node2Vec\AddedEdges\dim64_walk_len30_num_walks200_window10.txt'
                path_to_saved_emb_file = r'C:\Users\Anak\PycharmProjects\recreate_gene_disease\Data\processed\GeneDiseaseProject\copd\Node2Vec\UnweightedEdges\AddedEdges\EdgesNumber\top_k=24_dim64_walk_len30_num_walks200_window10.txt'
            elif dataset == 'GPSim':
                path_to_saved_emb_file = r'C:\Users\Anak\PycharmProjects\recreate_gene_disease\Data\processed\GPSim\Node2Vec\UnweightedEdges\AddedEdges\EdgesNumber\top_k=24_dim64_walk_len30_num_walks200_window10.txt'
            else:
                raise ValueError("correct/availble dataaset must be specified")
        else:
            assert dataset == 'no', 'no need to specified dataset if no qualified edges will be added (this prevent unexpected beahabior that could be caused by unintentionally provided dataset as argument)'
            path_to_saved_emb_file = r'C:\Users\Anak\PycharmProjects\recreate_gene_disease\Data\processed\GeneDiseaseProject\copd\Node2Vec\UnweightedEdges\NoAddedEdges\dim64_walk_len30_num_walks200_window10.txt'

    data_with_features = get_instances_with_features(
        path_to_saved_emb_file=path_to_saved_emb_file,
        use_saved_emb_file=use_saved_emb_file,
        normalized_weighted_edges=normalized_weighted_edges)

    return data_with_features


def get_data_feat(data=None, use_saved_emb_file=None, add_qualified_edges=None,
                  dataset=None, use_weighted_edges=None,
                  normalized_weighted_edges=None,
                  edges_percent=None,
                  edges_number=None,
                  added_edges_percent_of=None):
    assert data is not None, 'data must be explicitly stated to avoid ambiguity'
    assert use_saved_emb_file is not None, 'use_saved_emb_file must be explicitly stated to avoid ambiguity'
    # assert add_qualified_edges is not None, 'add_qualified_edges must be explicitly stated to avoid ambiguity'
    assert dataset is not None, 'dataset must be explicitly stated to avoid ambiguity'
    assert use_weighted_edges is not None, 'use_weighted_edges must be explicitly stated to avoid ambiguity'
    assert normalized_weighted_edges is not None, "normalized_weighted_edges must be specified to avoid ambiguity"
    assert (edges_percent is not None) or (
                edges_number is not None), "either edges_percent or edges_number must be specified to avoid ambiguity"
    # assert edges_number is not None, "edges_number must be specified to avoid ambiguity"

    data_with_features = None
    if use_saved_emb_file:
        # add features to data (read from emb_File)
        # note: step to create emb_file is as followed
        #     1. run Embedding.node2vec_emb.py (it will save to file location for you)
        #     2. apply get_instances_with_features with use_saved_emb_file = Ture (use_saved_emb_file is expected as command argument)
        # TODO all of the argument variation output same report_performancea
        # TODO add agument option for use_weighted_edges
        # TODO edges_number of edges_adding strategy is not yet implementing
        data_with_features = get_data_with_emb_as_feat(data,
                                                       use_saved_emb_file,
                                                       add_qualified_edges,
                                                       dataset,
                                                       use_weighted_edges,
                                                       normalized_weighted_edges,
                                                       edges_percent,
                                                       edges_number,
                                                       added_edges_percent_of)

    else:
        # add feature to Data (no embedding)
        # note: node embedding is only expected to be read from emb_file
        # TODO all of the argument variation output same report_performancea
        data_with_features = get_data_without_using_emb_as_feat(data=data,
                                                                add_qualified_edges=add_qualified_edges,
                                                                dataset=dataset,
                                                                use_weighted_edges=use_weighted_edges,
                                                                return_graph_and_data_with_features=False,
                                                                use_saved_emb_file=use_saved_emb_file,
                                                                normalized_weighted_edges=normalized_weighted_edges,
                                                                edges_percent=edges_percent,
                                                                edges_number=edges_number,
                                                                added_edges_percent_of=added_edges_percent_of)
    assert data_with_features is not None, ""

    # # get edges that were added
    # x = data_with_features.to_numpy()[data_with_features.to_numpy().nonzero()]
    # print(x[x != 0])

    return data_with_features