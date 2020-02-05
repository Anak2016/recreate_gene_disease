from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd

# from Sources.Preparation.Features.test import get_all_GeneDisease_unweighted_disease2disease_qualified_edges
# from Sources.Preparation.Features. import get_all_GeneDisease_unweighted_disease2disease_qualified_edges
# from Sources.Preprocessing.convertion import Converter
from Sources.Preparation.Data.conversion import Converter
from Sources.Preparation.Data.make_dataset import \
    get_diseases_that_are_overlapped_between_GPSim_and_GeneDisease_graph

def get_disease2disease_qualified_edges(data, dataset, use_weighted_edges,use_shared_gene_edges ,use_shared_phenotype_edges ):

    ### for all edges if it has common neighbor select them
    if dataset == "GeneDisease":
        # TODO here>>
        all_qualified_disease2disease_edges_df = get_GeneDisease_disease2disease_qualified_edges(
            data,
            use_weighted_edges,
            use_shared_gene_edges,
            use_shared_phenotype_edges)  # expect to return df

    elif dataset == 'GPSim':

        # TODO after get_GEneDisease_disease2disease_qualified_edges are validated for code correctness; implement use_shared_genes and use_shared_disease_edges
        all_qualified_disease2disease_edges_df = get_GPSim_disease2disease_qualified_edges(
            data,
            use_weighted_edges)  #
    else:
        raise ValueError(
            'please specify existed/available dataset eg GeneDisease or GPSim')

    return all_qualified_disease2disease_edges_df

def get_GPSim_disease2disease_qualified_edges(data, use_weight_edges):
    """
    note: self loop is not a qualified edges ( so it will be removed here)

    @return:
    """
    # TODO read_csv from qualfied_cui_disease2disease_edges_with_no_self_loop ( I have not create this yet)
    # # note: This should be created in GPSim (because it is EDA)

    # # TODO use cui_edges_weighted for use_weighted_edges = True nad False
    # # note: code that create cui_edges_weight.csv is in notebook/Dataset/GPSim.ipynb (which is where code support to be because it is a part of Exploratory Data Analysis aka eda)
    # file_name = r'C:\Users\Anak\PycharmProjects\recreate_gene_disease\Data\raw\GPSim\Edges\cui_edges_weight.csv'
    # non_zero_GPsim_disease_cui_disease_pair_with_no_self_loop_df = pd.read_csv(file_name, sep=",") # varaible name should reflect state of its content
    #
    # return non_zero_GPsim_disease_cui_disease_pair_with_no_self_loop_df

    # TODO code below can be found in GPSim.ipynb as stated in note of get_GPSim_disease2disease_qulified_edges()
    # TODO code below should be substituted with code above (reason: way less code + derived from saved file which provide predictable behavior)
    disease_pair_similarity = r'C:\Users\Anak\PycharmProjects\recreate_gene_disease\Data\raw\GPSim\copd_comorbidity_similarity.txt'
    disease_pair_similarity_pd = pd.read_csv(disease_pair_similarity, sep='\t',
                                             header=None)

    # replace na with zero and select only non zero value
    disease_pair_similarity_np = disease_pair_similarity_pd.fillna(
        0).to_numpy()
    non_zero_GPsim_disease_pairs = disease_pair_similarity_np.nonzero()

    non_zero_GPsim_disease_pair_coeff = disease_pair_similarity_np[
        non_zero_GPsim_disease_pairs]
    # non_zero_GPsim_disease_pair_coeff[np.logical_not(np.isnan(non_zero_GPsim_disease_pair_coeff))].shape

    # convert non_zero_GPSim_disease_pairs (index of matrix) to disease represenetation of its index (it have be to loaded from GPSim_overlaped_diseases because it preserved order)
    GPsim_overlapped_diseases_file_path = r"C:\Users\Anak\PycharmProjects\recreate_gene_disease\Data\raw\GPSim\GPsim_overlapped_diseases.txt"
    GPsim_overlapped_diseases = pd.read_csv(
        GPsim_overlapped_diseases_file_path, header=None)

    GPsim_overlapped_diseases_np = GPsim_overlapped_diseases.to_numpy()
    GPsim_overlapped_diseases_dict = {i: GPsim_overlapped_diseases_np[i][0] for
                                      i in
                                      range(GPsim_overlapped_diseases_np.shape[
                                                0])}

    # get doid_id
    disease_1 = np.array([list(map(lambda x: GPsim_overlapped_diseases_dict[x],
                                   non_zero_GPsim_disease_pairs[0]))])
    disease_2 = np.array([list(map(lambda x: GPsim_overlapped_diseases_dict[x],
                                   non_zero_GPsim_disease_pairs[1]))])

    non_zero_GPsim_disease_doid_disease_pair = np.concatenate(
        (disease_1, disease_2), axis=0)

    converter = Converter(data)
    original_disease_101 = np.array(list(converter.cui2class_id_dict.keys()))

    # Create cui2doid_dict and doid2cui_dict
    disease_mapping_df_for_orignal_disease_101 = converter.disease_mapping_df[
        converter.disease_mapping_df['diseaseId'].isin(original_disease_101)]
    doid2cui_dict = {i: j for i, j in
                     zip(disease_mapping_df_for_orignal_disease_101['doid'],
                         disease_mapping_df_for_orignal_disease_101[
                             'diseaseId'])}
    cui2doid_dict = {i: j for i, j in doid2cui_dict.items()}

    # For non_zero_GPsim_disease_doid_disease_pair, map doid to cui
    vfunc = np.vectorize(lambda x: doid2cui_dict[x])

    non_zero_GPsim_disease_cui_disease_pair = vfunc(
        non_zero_GPsim_disease_doid_disease_pair)

    # remove self loop (probly there is a better but I choose to use third party libary)
    if use_weight_edges:

        non_zero_GPsim_disease_pair_coeff = non_zero_GPsim_disease_pair_coeff[
                                            np.newaxis, :]
        G = nx.Graph()

        G.add_weighted_edges_from(zip(
            non_zero_GPsim_disease_cui_disease_pair[0, :],
            non_zero_GPsim_disease_cui_disease_pair[1, :],
            non_zero_GPsim_disease_pair_coeff[0, :].astype(float)))

        G.remove_edges_from(nx.selfloop_edges(G))

        assert data.is_graph_edges_weighted(G,
                                            use_outside_graph=True), "use_weighted_edges is True, but all edges in graph has weight == 1"

        non_zero_GPsim_disease_cui_disease_pair_with_no_self_loop_df = nx.to_pandas_edgelist(
            G)

    else:
        G = nx.Graph()

        # use weighted_edges = 1
        unweighted_edges_value = np.ones(
            non_zero_GPsim_disease_cui_disease_pair.shape[1])[np.newaxis,
                                 :]  # type = float64

        G.add_weighted_edges_from(zip(
            non_zero_GPsim_disease_cui_disease_pair[0, :],
            non_zero_GPsim_disease_cui_disease_pair[1, :],
            unweighted_edges_value[0, :].astype(float)))

        # G.add_edges_from(non_zero_GPsim_disease_cui_disease_pair.T)
        G.remove_edges_from(nx.selfloop_edges(G))

        assert not (data.is_graph_edges_weighted(G,
                                                 use_outside_graph=True)), "use_weighted_edges is True, but all edges in graph has weight == 1"

        non_zero_GPsim_disease_cui_disease_pair_with_no_self_loop_df = nx.to_pandas_edgelist(
            G)
    return non_zero_GPsim_disease_cui_disease_pair_with_no_self_loop_df



def get_GeneDisease_disease2disease_qualified_edges(data,
                                                    use_weighted_edges,
                                                    use_shared_gene_edges = None,
                                                    use_shared_phenotype_edges = None):
    """

    @param G: networkx.Graph()
    @param nodes: type = np; shape = (-1,1)
    @return: list of qualified edges; shape = (-1, 2)
    """
    assert use_shared_gene_edges is not None, "use_shared_gene_edges must be specified to avoid ambiguity"
    assert use_shared_phenotype_edges is not None, "use_shared_phenotype_edges must be specified to avoid ambiguity"

    if use_shared_gene_edges:
        assert not use_shared_phenotype_edges, "only use_shared_gene_edges or use_shared_phenotype_edges can be True"
    if use_shared_phenotype_edges:
        assert not use_shared_gene_edges, "only use_shared_gene_edges or use_shared_phenotype_edges can be True"

    # all_disease2disease_edges = list(combinations(nodes, 2))

    # TODO currently there is no different between use_weighted_edges = True/False because qualified edges of GeneDisease has no weight
    # note: potential weight of qualified edges are jaccard coefficient among other
    if use_weighted_edges:
        raise ValueError(
            'not yet implemented: currently i am not working any qualified edges of GeneDisease that has weight')

    else:

        all_qualified_disease2disease_edges = get_all_GeneDisease_unweighted_disease2disease_qualified_edges(data,
                                                                                                             use_shared_gene_edges=use_shared_gene_edges,
                                                                                                             use_shared_phenotype_edges=use_shared_phenotype_edges)

        # TODO code from function below below is placed within this function instead (reason: function names is does not match what it does tho it output expected result)
        # TODO below is the code that produces the the following
        # : processed\GeneDiseaseProject\copd\Node2Vec\UnweightedEdges\AddedEdges\EdgesNumber
        # : processed\GeneDiseaseProject\copd\Node2Vec\UnweightedEdges\NoAddedEdges\EdgesNumber
        # all_qualified_edges = get_all_GeneDisease_qualified_edges(data,
        #                                                       data.G,
        #                                                       data.diseases_np)

        # qualified_edges = np.array(
        #     [edge for edge in all_disease2disease_edges if
        #      len(list(nx.common_neighbors(G, edge[0], edge[1]))) > 0])
        # G = nx.Graph()
        # G.add_edges_from(all_disease2disease_edges, weight=1)
        # all_qualified_edges = nx.to_pandas_edgelist(G)
        #
        # assert not (data.is_graph_edges_weighted(G,
        #                                          use_outside_graph=True)), "use_weighted_edges is True, but all edges in graph has weight == 1"
    return all_qualified_disease2disease_edges


def get_all_GeneDisease_unweighted_disease2disease_qualified_edges(data,
                                                                   use_shared_gene_edges = None,
                                                                   use_shared_phenotype_edges = None):

    assert use_shared_gene_edges is not None, "use_shared_gene_edges must be specified to avoid ambiguity"
    assert use_shared_phenotype_edges is not None, "use_shared_phenotype_edges must be specified to avoid ambiguity"

    if use_shared_gene_edges and (not use_shared_phenotype_edges ):
        all_qualified_disease2disease_edges_pd = get_qualified_diseases2disease_edges_with_shared_genes(data, use_overlapped_disease = False)

    elif use_shared_phenotype_edges and ( not use_shared_gene_edges ):
        all_qualified_disease2disease_edges_pd = get_qualified_disease2disease_edges_with_shared_phenotype(data,  use_overlapped_disease = False)

    elif (not use_shared_gene_edges ) and ( not use_shared_phenotype_edges ):
        all_qualified_disease2disease_edges_pd = get_qualified_disease2disease_edges_with_shared_genes(data, use_overlapped_disease = False)

        # nodes = data.diseases_np
        # all_disease2disease_edges = list(combinations(nodes, 2))
        # all_qualified_disease2disease_edges = get_qualified_disease2disease_edges_with_shared_genes(
        #     data.G, all_disease2disease_edges)
        #
        # G = nx.Graph()
        # G.add_edges_from(all_qualified_disease2disease_edges, weight=1)
        # # remove selfloop
        # G.remove_edges_from(nx.selfloop_edges(G))
        # all_qualified_disease2disease_edges_pd = nx.to_pandas_edgelist(G)
        # raise ValueError(
        #     "not yet implmented: currently, the program only support edges between diseases with shared_genes or shared_phenotyp")

    else:
        raise ValueError(" ")

    return all_qualified_disease2disease_edges_pd

    # TODO paragraph of code below is modified to be used as shared_gene_edges option in code above
    # nodes = data.diseases_np
    #
    # all_disease2disease_edges = list(combinations(nodes, 2))
    #
    # all_disease2disease_qualified_edges = np.array(
    #     [edge for edge in all_disease2disease_edges if
    #      len(list(nx.common_neighbors(data.G, edge[0], edge[1]))) > 0])


    # # TODO code below is implemented in get_qualified_disease2disease_edges_with_shared_genes()
    # G = nx.Graph()
    # G.add_edges_from(all_disease2disease_qualified_edges, weight=1)
    # # remove selfloop
    # G.remove_edges_from(nx.selfloop_edges(G))
    # all_qualified_disease2disease_edges_df = nx.to_pandas_edgelist(G)

    # return all_qualified_disease2disease_edges_pd

def get_qualified_disease2disease_edges_with_shared_nodes(graph, node_pairs):
    """

    @param graph: type = nx.Graph();
    @param node_pairs: any
    @return:
    """

    qualified_edges = np.array(
        [edge for edge in node_pairs if
         len(list(nx.common_neighbors(graph, edge[0], edge[1]))) > 0])

    return qualified_edges

# def get_qualified_disease2disease_edges_with_shared_phenotypes(graph_with_phenotype2disease_edges,disease2disease_edges):
#
#     all_disease2disease_qualified_edges = get_qualified_disease2disease_edges_with_shared_nodes(graph_with_phenotype2disease_edges, disease2disease_edges)
#
#     return all_disease2disease_qualified_edges

# def get_qualified_disease2disease_edges_with_shared_genes(graph_with_gene2disease_edges,
#                                                           disease2disease_edges):
#     """
#
#     @param graph: type = nx.Graph;
#         desc:
#             > graph contains all diseases in disease2disease edges
#             > graph contains gene2disease edges
#     @param disease2disease_edges:
#     @return: all_disease2disease_qualified_edges: type = np
#     """
#     all_disease2disease_qualified_edges = get_qualified_disease2disease_edges_with_shared_nodes(graph_with_gene2disease_edges, disease2disease_edges)
#
#     return all_disease2disease_qualified_edges


def get_qualified_diseases2disease_edges_with_shared_genes(data, use_overlapped_disease=None):
    """
        shared_gene implies shared at least 1 gene
        overllaped impplies diseases overlapped between GPSim and GeneDisease dataset (file_name GPSim_overlapped_diseases.txt)
    """

    assert isinstance(use_overlapped_disease, bool), "isinstances must be of type boolean "

    qualified_diseases_np = get_qualified_diseases(data, use_overlapped_disease)

    # overlapped_diseases_np = get_diseases_that_are_overlapped_between_GPSim_and_GeneDisease_graph()
    all_overlapped_disease2disease_edges = list(
        combinations(qualified_diseases_np, 2))

    assert data.is_in_original_diseases(
        qualified_diseases_np), "some of the generated overlapped diseases are not in the original GEneDisease diseases nodes "

    # overlapped_disease_graph = data.original_GeneDisease_edges.copy()
    # overlapped_disease_graph.add_edges_from(
    #     all_overlapped_disease2disease_edges)

    # all_disease2disease_qualified_edges = get_qualified_disease2disease_edges_with_shared_genes(
    #     overlapped_disease_graph, all_overlapped_disease2disease_edges)

    all_disease2disease_qualified_edges = get_qualified_disease2disease_edges_with_shared_nodes(
        data.original_GeneDisease_edges.copy(), all_overlapped_disease2disease_edges)

    graph_with_no_self_loop_edges = get_graph_with_no_self_loop_edges(all_disease2disease_qualified_edges)
    all_qualified_disease2disease_edges_df = nx.to_pandas_edgelist(graph_with_no_self_loop_edges)

    # # remove selfloop
    # G = nx.Graph()
    # G.add_edges_from(all_disease2disease_qualified_edges, weight=1)
    # G.remove_edges_from(nx.selfloop_edges(G))
    # all_qualified_disease2disease_edges_df = nx.to_pandas_edgelist(G)

    return all_qualified_disease2disease_edges_df

def get_qualified_diseases(data, use_overlapped_disease):

    if use_overlapped_disease:
        qualified_diseases_np = get_diseases_that_are_overlapped_between_GPSim_and_GeneDisease_graph()
    else:
        qualified_diseases_np = data.diseases_np

    assert data.is_in_original_diseases(
        qualified_diseases_np), "some of the generated overlapped diseases are not in the original GEneDisease diseases nodes "

    return qualified_diseases_np

def get_qualified_disease2disease_edges_with_shared_phenotype(data, use_overlapped_disease = None):
    """
        shared_phenotype implies shared at least 1 phenotype
        overllaped impplies diseases overlapped between GPSim and GeneDisease dataset (file_name GPSim_overlapped_diseases.txt)
    """

    assert isinstance(use_overlapped_disease, bool), "isinstances must be of type boolean "

    qualified_diseases_np = get_qualified_diseases(data, use_overlapped_disease)


    # convert cui2doid using dict contains key of 101 original diseases 
    convertion = Converter(data)
    cui2doid_vectorized = np.vectorize(lambda x: convertion.original_cui2doid_dict[x])
    qualified_cui_diseases = qualified_diseases_np # renaming for self-documentation purposes
    qualified_doid_diseases = cui2doid_vectorized(qualified_cui_diseases)

    graph_with_phenotype2disease_edges = nx.Graph()
    phenotype2disease_edges_that_contain_qualified_nodes_pd = get_phenotype2disease_edges_that_contain_qualified_nodes(data,
                                                             qualified_doid_diseases)

    graph_with_phenotype2disease_edges.add_edges_from(phenotype2disease_edges_that_contain_qualified_nodes_pd.to_numpy())

    # get uniqued quaified_doid_diseases
    qualified_doid_diseases_in_phenotype2disease_edges =np.unique(phenotype2disease_edges_that_contain_qualified_nodes_pd['disease_id'].to_numpy())

    all_overlapped_disease2disease_edges_that_have_diseases_in_phenotype2disease_edges = list(
        combinations(qualified_doid_diseases_in_phenotype2disease_edges, 2))
    
    all_doid_disease2disease_qualified_edges = get_qualified_disease2disease_edges_with_shared_nodes(
        graph_with_phenotype2disease_edges, all_overlapped_disease2disease_edges_that_have_diseases_in_phenotype2disease_edges)

    doid2cui_vectorize = np.vectorize(lambda x: convertion.original_doid2cui_dict[x])
    all_cui_disease2disease_qualified_edges =  doid2cui_vectorize(all_doid_disease2disease_qualified_edges)

    graph_with_no_self_loop_edges = get_graph_with_no_self_loop_edges(all_cui_disease2disease_qualified_edges)
    all_qualified_disease2disease_edges_df = nx.to_pandas_edgelist(graph_with_no_self_loop_edges)

    return all_qualified_disease2disease_edges_df


def get_qualified_disease2disease_edges_with_shared_genes(data, use_overlapped_disease = None):
    """
        notice: function name does not have overlaped in in. This implies that disease2disease edges are created from original_set of GeneDisease diseases
    @return:
    """

    assert isinstance(use_overlapped_disease, bool), "isinstances must be of type boolean "

    qualified_diseases_np = get_qualified_diseases(data, use_overlapped_disease)

    all_disease2disease_edges = list(combinations(qualified_diseases_np, 2))
    all_qualified_disease2disease_edges = get_qualified_disease2disease_edges_with_shared_nodes(
        data.G, all_disease2disease_edges)

    graph_with_no_self_loop_edges = get_graph_with_no_self_loop_edges(all_qualified_disease2disease_edges)
    all_qualified_disease2disease_edges_pd = nx.to_pandas_edgelist(graph_with_no_self_loop_edges)

    return all_qualified_disease2disease_edges_pd


def get_graph_with_no_self_loop_edges(edges):

    # TODO please use add_weighted_edges_from instead of add_edges_from (reason: it is just more flexible overall)
    G = nx.Graph()
    G.add_edges_from(edges, weight=1)
    G.remove_edges_from(nx.selfloop_edges(G))

    return G

def get_phenotype2diseiase_edges_from_disease_hpo2_dataset():
    disease2phenotype_edges_file_path = r'C:\Users\Anak\PycharmProjects\recreate_gene_disease\Data\raw\GPSim\Edges\DiseasePhenotype\disease_hpo.csv'

    # disease_id, hpo_id, disease_name, hpo_name, is_do
    edges_disease2phenotypes_pd = pd.read_csv(disease2phenotype_edges_file_path)[['disease_id', 'hpo_id']]
    return edges_disease2phenotypes_pd

def get_phenotype2disease_edges_that_contain_qualified_nodes(data, qualified_nodes):
    """qualified_nodes are used to filter  edges whose either nodes are qualified_nodes"""

    qualified_doid_nodes = qualified_nodes # renaming for self-documentation purposes
    
    disease2phenotypes_edges_pd = get_phenotype2diseiase_edges_from_disease_hpo2_dataset()
    edges_disease2phenotypes_that_contain_qualified_nodes_pd = disease2phenotypes_edges_pd[disease2phenotypes_edges_pd['disease_id'].isin(qualified_doid_nodes)]

    return edges_disease2phenotypes_that_contain_qualified_nodes_pd


