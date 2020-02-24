from itertools import combinations
from os import path

import networkx as nx
import numpy as np
import pandas as pd

from Sources.Preparation.Data.conversion import Converter
from Sources.Preparation.Data.make_dataset import \
    get_diseases_that_are_overlapped_between_GPSim_and_GeneDisease_graph
from global_param import *

# =====================
# ==Conditional Code
# desc: main purpose is use in placed of if-else condition
# =====================

def get_disease2disease_qualified_edges(data, dataset, use_weighted_edges,
                                        use_shared_gene_edges,
                                        use_shared_phenotype_edges,
                                        use_shared_gene_and_phenotype_edges,
                                        use_shared_gene_but_not_phenotype_edges,
                                        use_shared_phenotype_but_not_gene_edges):  # expect to return df
    ### for all edges if it has common neighbor select them
    if dataset == "GeneDisease":
        all_qualified_disease2disease_edges_df = get_GeneDisease_disease2disease_qualified_edges(
            data,
            use_weighted_edges,
            use_shared_gene_edges,
            use_shared_phenotype_edges,
            use_shared_gene_and_phenotype_edges,
            use_shared_gene_but_not_phenotype_edges,
            use_shared_phenotype_but_not_gene_edges)  # expect to return df

    elif dataset == 'GPSim':

        # TODO I agree with professor zhu that there is no need to use 55 qualified (overllaped diseases) from GPSim because
        #       > we can directly used all 101 original diseases from GeneDisease
        all_qualified_disease2disease_edges_df = get_GPSim_disease2disease_qualified_edges(
            data,
            use_weighted_edges)  #
    else:
        raise ValueError(
            'please specify existed/available dataset eg GeneDisease or GPSim')

    return all_qualified_disease2disease_edges_df


def get_all_GeneDisease_unweighted_disease2disease_qualified_edges(data,
                                                                   use_shared_gene_edges=None,
                                                                   use_shared_phenotype_edges=None,
                                                                   use_shared_gene_and_phenotype_edges=None,
                                                                   use_shared_gene_but_not_phenotype_edges=None,
                                                                   use_shared_phenotype_but_not_gene_edges=None):
    """
        This function currently only implmented for UNWEIGHTED edges 
        @todo implement the function for weighted verion -> where should I change or adjust it?
    @param data: 
    @param use_shared_gene_edges: 
    @param use_shared_phenotype_edges: 
    @return: 
    """
    assert use_shared_gene_edges is not None, "use_shared_gene_edges must be specified to avoid ambiguity"
    assert use_shared_phenotype_edges is not None, "use_shared_phenotype_edges must be specified to avoid ambiguity"
    assert use_shared_gene_and_phenotype_edges is not None, "use_shared_gene_and_phenotype_edges must be specified to avoid ambiguity"
    assert use_shared_gene_but_not_phenotype_edges is not None, "use_shared_gene_but_not_phenotype_edges must be specified to avoid ambiguity"
    assert use_shared_phenotype_but_not_gene_edges is not None, "use_shared_phenotype_but_not_gene_edges must be specified to avoid ambiguity"

    if use_shared_gene_edges and (not use_shared_phenotype_edges):
        all_qualified_disease2disease_edges_pd = get_qualified_disease2disease_edges_with_shared_genes(
            data, use_overlapped_disease=False)

    elif use_shared_phenotype_edges and (not use_shared_gene_edges):
        all_qualified_disease2disease_edges_pd = get_qualified_disease2disease_edges_with_shared_phenotype(
            data, use_overlapped_disease=False)

    elif use_shared_gene_edges and use_shared_phenotype_edges:
        all_qualified_disease2disease_edges_pd = get_qualified_disease2disease_edges_with_shared_gene_or_phenotype(
            data, use_overlapped_disease=False)
    elif use_shared_gene_and_phenotype_edges:
        all_qualified_disease2disease_edges_pd = get_qualified_disease2disease_edges_with_shared_gene_and_phenotype(
            data, use_overlapped_disease=False)
    elif use_shared_gene_but_not_phenotype_edges:
        all_qualified_disease2disease_edges_pd = get_qualified_disease2disease_edges_with_shared_gene_but_not_phenotype_edges(
            data, use_overlapped_disease=False)
    elif use_shared_phenotype_but_not_gene_edges:
        all_qualified_disease2disease_edges_pd = get_qualified_disease2disease_edges_with_shared_phenotype_but_not_gene_edges(
            data, use_overlapped_disease=False)
    else:
        raise ValueError(" ")

    return all_qualified_disease2disease_edges_pd


# =====================
# ==Specialized Code
# desc: specialized code is code that are intended to do one specific job that is not expected (tho possible) to be used cross different function
#     : main purpose of specialized code is to enhance self-documented code
# =====================

def get_GeneDisease_disease2disease_qualified_edges(data,
                                                    use_weighted_edges,
                                                    use_shared_gene_edges,
                                                    use_shared_phenotype_edges,
                                                    use_shared_gene_and_phenotype_edges,
                                                    use_shared_gene_but_not_phenotype_edges,
                                                    use_shared_phenotype_but_not_gene_edges):
    """

    @param G: networkx.Graph()
    @param nodes: type = np; shape = (-1,1)
    @return: list of qualified edges; shape = (-1, 2)
    """

    # TODO currently there is no different between use_weighted_edges = True/False because qualified edges of GeneDisease has no weight
    # note: potential weight of qualified edges are jaccard coefficient among other
    if use_weighted_edges:
        raise ValueError(
            'not yet implemented: currently i am not working any qualified edges of GeneDisease that has weight')

    else:

        all_qualified_disease2disease_edges = get_all_GeneDisease_unweighted_disease2disease_qualified_edges(
            data,
            use_shared_gene_edges=use_shared_gene_edges,
            use_shared_phenotype_edges=use_shared_phenotype_edges,
            use_shared_gene_and_phenotype_edges=use_shared_gene_and_phenotype_edges,
            use_shared_gene_but_not_phenotype_edges=use_shared_gene_but_not_phenotype_edges,
            use_shared_phenotype_but_not_gene_edges=use_shared_phenotype_but_not_gene_edges)

    return all_qualified_disease2disease_edges


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

    # converter = Converter(data)
    # original_disease_101 = np.array(list(converter.cui2class_id_dict.keys()))
    #
    # # Create cui2doid_dict and doid2cui_dict
    # disease_mapping_df_for_orignal_disease_101 = converter.disease_mapping_df[
    #     converter.disease_mapping_df['diseaseId'].isin(original_disease_101)]
    # doid2cui_dict = {i: j for i, j in
    #                  zip(disease_mapping_df_for_orignal_disease_101['doid'],
    #                      disease_mapping_df_for_orignal_disease_101[
    #                          'diseaseId'])}
    # cui2doid_dict = {i: j for i, j in doid2cui_dict.items()}
    #
    # # For non_zero_GPsim_disease_doid_disease_pair, map doid to cui
    # vfunc = np.vectorize(lambda x: doid2cui_dict[x])
    # non_zero_GPsim_disease_cui_disease_pair = vfunc(
    #     non_zero_GPsim_disease_doid_disease_pair)

    converter = Converter(data)
    non_zero_GPsim_disease_cui_disease_pair = converter.original_doid2cui_mapping(
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


def get_qualified_disease2disease_edges_with_shared_genes(data,
                                                          use_overlapped_disease=None):
    """
        @todo implement the function for weighted verion -> where should I change or adjust it?
        shared_gene implies shared at least 1 gene
        overllaped impplies diseases overlapped between GPSim and GeneDisease dataset (file_name GPSim_overlapped_diseases.txt)
    """

    assert isinstance(use_overlapped_disease,
                      bool), "isinstances must be of type boolean "

    qualified_diseases_np = get_qualified_diseases(data,
                                                   use_overlapped_disease)

    # overlapped_diseases_np = get_diseases_that_are_overlapped_between_GPSim_and_GeneDisease_graph()
    all_qualified_disease2disease_edges = list(
        combinations(qualified_diseases_np, 2))

    assert data.is_in_original_diseases(
        qualified_diseases_np), "some of the generated diseases are not in the original GEneDisease diseases nodes "

    all_disease2disease_qualified_edges = get_qualified_disease2disease_edges_with_shared_nodes(
        data.original_GeneDisease_edges.copy(),
        all_qualified_disease2disease_edges)

    graph_with_no_self_loop_edges = get_graph_with_no_self_loop_edges(
        all_disease2disease_qualified_edges)
    all_qualified_disease2disease_edges_df = nx.to_pandas_edgelist(
        graph_with_no_self_loop_edges)

    return all_qualified_disease2disease_edges_df


def get_qualified_disease2disease_edges_with_shared_phenotype(data,
                                                              use_overlapped_disease=None):
    """
        shared_phenotype implies shared at least 1 phenotype
        overllaped impplies diseases overlapped between GPSim and GeneDisease dataset (file_name GPSim_overlapped_diseases.txt)
    """

    assert isinstance(use_overlapped_disease,
                      bool), "isinstances must be of type boolean "

    qualified_diseases_np = get_qualified_diseases(data,
                                                   use_overlapped_disease)

    # convert cui2doid using dict contains key of 101 original diseases
    convertion = Converter(data)

    qualified_cui_diseases = qualified_diseases_np  # renaming for self-documentation purposes
    qualified_doid_diseases = convertion.original_cui2doid_mapping(qualified_cui_diseases )

    # create graph_with_phenotype2disease_edges
    graph_with_phenotype2disease_edges = nx.Graph()

    # used saved file, if already existed if not create target file and saved it

    phenotype2disease_edges_that_contain_qualified_nodes_pd = get_phenotype2disease_edges_that_contain_qualified_nodes(
        data,
        qualified_doid_diseases,
        saved_file_path=PHENOTYPE2DISEASE_EDGES_THAT_CONTAIN_QUALIFIED_NODES_FILE_PATH
    )

    graph_with_phenotype2disease_edges.add_edges_from(
        phenotype2disease_edges_that_contain_qualified_nodes_pd.to_numpy())

    # get uniqued quaified_doid_diseases
    qualified_doid_diseases_in_phenotype2disease_edges = np.unique(
        phenotype2disease_edges_that_contain_qualified_nodes_pd[
            'disease_id'].to_numpy())

    # get all pairs of qualified nodes
    all_disease2disease_edges_that_have_diseases_in_phenotype2disease_edges = list(
        combinations(qualified_doid_diseases_in_phenotype2disease_edges, 2))

    # get qualified_edges with shared phenotype
    all_doid_disease2disease_qualified_edges = get_qualified_disease2disease_edges_with_shared_nodes(
        graph_with_phenotype2disease_edges,
        all_disease2disease_edges_that_have_diseases_in_phenotype2disease_edges)

    all_cui_disease2disease_qualified_edges = convertion.original_doid2cui_mapping(all_doid_disease2disease_qualified_edges)

    graph_with_no_self_loop_edges = get_graph_with_no_self_loop_edges(
        all_cui_disease2disease_qualified_edges)
    all_qualified_disease2disease_edges_df = nx.to_pandas_edgelist(
        graph_with_no_self_loop_edges)

    return all_qualified_disease2disease_edges_df


def get_qualified_disease2disease_edges_with_shared_gene_or_phenotype(data,
                                                                      use_overlapped_disease=None):
    # TODO validate that qualified_diseases_edges both shared_gene and shared_phenotype are unweighted
    # todo haven't yet implmented for weighted version
    all_qualified_disease2disease_edges_with_shared_gene_df = get_qualified_disease2disease_edges_with_shared_genes(
        data, use_overlapped_disease)
    all_qualified_disease2disease_edges_with_shared_phenotype_df = get_qualified_disease2disease_edges_with_shared_phenotype(
        data, use_overlapped_disease)

    # convert df to grpah to be validate whether both are weighted or unweighted
    ## create graph for shared_gene
    graph_with_shared_gene = nx.Graph()
    graph_with_shared_gene.add_weighted_edges_from(
        all_qualified_disease2disease_edges_with_shared_gene_df.to_numpy())
    ## create graph for shared_phenotype
    graph_with_shared_phenotype = nx.Graph()
    graph_with_shared_phenotype.add_weighted_edges_from(
        all_qualified_disease2disease_edges_with_shared_phenotype_df.to_numpy())

    # validate that both are qualified_disease2disease_edges are both unweighted or both weighted
    assert (not data.is_graph_edges_weighted(graph_with_shared_gene,
                                             use_outside_graph=True)) \
           and (not data.is_graph_edges_weighted(graph_with_shared_phenotype,
                                                 use_outside_graph=True)), "oboth shared_gene and shared_phenotype should be both unweighted "

    # drop dubplicate row
    concatenate_edges_bewteen_shared_gene_and_shared_phenotype_edges_df = pd.concat(
        [all_qualified_disease2disease_edges_with_shared_phenotype_df,
         all_qualified_disease2disease_edges_with_shared_gene_df],
        ignore_index=True)

    ## create undirected graph and output shared_gene OR shared_phenotype edges
    # all_qualified_disease2disease_edges_with_shared_gene_and_phenotype_df = concatenate_edges_bewteen_shared_gene_and_shared_phenotype_edges_df.drop_duplicates()
    undirected_graph = nx.Graph()
    undirected_graph.add_weighted_edges_from(
        concatenate_edges_bewteen_shared_gene_and_shared_phenotype_edges_df.to_numpy())
    all_qualified_disease2disease_edges_with_shared_gene_and_phenotype_df = pd.DataFrame(
        np.array(list(undirected_graph.edges.data('weight'))),
        columns=concatenate_edges_bewteen_shared_gene_and_shared_phenotype_edges_df.columns)

    # code below is self documentation that is used to validate varaibles content during DEBUGGING
    number_of_overlapped_edges_between_shared_gene_and_shared_phenotype_edges = \
        concatenate_edges_bewteen_shared_gene_and_shared_phenotype_edges_df.shape[
            0] - \
        all_qualified_disease2disease_edges_with_shared_gene_and_phenotype_df.shape[
            0]

    return all_qualified_disease2disease_edges_with_shared_gene_and_phenotype_df


def get_qualified_disease2disease_edges_with_shared_gene_and_phenotype(data,
                                                                       use_overlapped_disease=None):
    """
        qualified_edges (in this function) = disease2disease edges whose diseases nodes have have lest 1 shared_gene OR 1 shared_node
    @param data:
    @param use_overlapped_disease:
    @return:
    """

    # TODO validate that qualified_diseases_edges both shared_gene and shared_phenotype are unweighted
    # todo haven't yet implmented for weighted version
    all_qualified_disease2disease_edges_with_shared_gene_df = get_qualified_disease2disease_edges_with_shared_genes(
        data, use_overlapped_disease)
    all_qualified_disease2disease_edges_with_shared_phenotype_df = get_qualified_disease2disease_edges_with_shared_phenotype(
        data, use_overlapped_disease)

    # convert df to grpah to be validate whether both are weighted or unweighted
    ## create graph for shared_gene
    graph_with_shared_gene = nx.Graph()
    graph_with_shared_gene.add_weighted_edges_from(
        all_qualified_disease2disease_edges_with_shared_gene_df.to_numpy())
    ## create graph for shared_phenotype
    graph_with_shared_phenotype = nx.Graph()
    graph_with_shared_phenotype.add_weighted_edges_from(
        all_qualified_disease2disease_edges_with_shared_phenotype_df.to_numpy())

    # validate that both are qualified_disease2disease_edges are both unweighted or both weighted
    assert (not data.is_graph_edges_weighted(graph_with_shared_gene,
                                             use_outside_graph=True)) \
           and (not data.is_graph_edges_weighted(graph_with_shared_phenotype,
                                                 use_outside_graph=True)), "oboth shared_gene and shared_phenotype should be both unweighted "

    # TODO after weight version is implemented, I need to check if there is a chance that shared_gene and shared_phenotype can have diferent weighted_edges
    # drop dubplicate row
    ## create undirected graph and output shared_gene and shared_phenotype edges
    undirected_graph_with_shred_gene = nx.Graph()
    undirected_graph_with_shred_gene.add_weighted_edges_from(
        all_qualified_disease2disease_edges_with_shared_gene_df.to_numpy())

    ## get overlapped shared_gene and shared_phenotype
    all_qualified_disease2disease_edges_with_shared_gene_and_phenotype = []
    for edge_with_weight in all_qualified_disease2disease_edges_with_shared_phenotype_df.to_numpy():
        edge = edge_with_weight[:2]
        if undirected_graph_with_shred_gene.has_edge(*edge):
            all_qualified_disease2disease_edges_with_shared_gene_and_phenotype.append(
                list(edge_with_weight))

    all_qualified_disease2disease_edges_with_shared_gene_and_phenotyp_np = np.array(
        all_qualified_disease2disease_edges_with_shared_gene_and_phenotype)

    all_qualified_disease2disease_edges_with_shared_gene_and_phenotyp_df = pd.DataFrame(
        all_qualified_disease2disease_edges_with_shared_gene_and_phenotyp_np,
        columns=all_qualified_disease2disease_edges_with_shared_gene_df.columns)

    return all_qualified_disease2disease_edges_with_shared_gene_and_phenotyp_df


def get_qualified_disease2disease_edges_with_shared_gene_but_not_phenotype_edges(
        data,
        use_overlapped_disease=None):
    # TODO validate that qualified_diseases_edges both shared_gene and shared_phenotype are unweighted
    # todo haven't yet implmented for weighted version
    all_qualified_disease2disease_edges_with_shared_gene_df = get_qualified_disease2disease_edges_with_shared_genes(
        data, use_overlapped_disease)
    all_qualified_disease2disease_edges_with_shared_phenotype_df = get_qualified_disease2disease_edges_with_shared_phenotype(
        data, use_overlapped_disease)

    # convert df to grpah to be validate whether both are weighted or unweighted
    ## create graph for shared_gene
    graph_with_shared_gene = nx.Graph()
    graph_with_shared_gene.add_weighted_edges_from(
        all_qualified_disease2disease_edges_with_shared_gene_df.to_numpy())
    ## create graph for shared_phenotype
    graph_with_shared_phenotype = nx.Graph()
    graph_with_shared_phenotype.add_weighted_edges_from(
        all_qualified_disease2disease_edges_with_shared_phenotype_df.to_numpy())

    # validate that both are qualified_disease2disease_edges are both unweighted or both weighted
    assert (not data.is_graph_edges_weighted(graph_with_shared_gene,
                                             use_outside_graph=True)) \
           and (not data.is_graph_edges_weighted(graph_with_shared_phenotype,
                                                 use_outside_graph=True)), "oboth shared_gene and shared_phenotype should be both unweighted "

    # TODO after weight version is implemented, I need to check if there is a chance that shared_gene and shared_phenotype can have diferent weighted_edges
    # drop dubplicate row

    ## change variable name to reflect its types
    all_qualified_disease2disease_edges_with_shared_phenotype_np = all_qualified_disease2disease_edges_with_shared_phenotype_df.to_numpy()
    all_qualified_disease2disease_edges_with_shared_gene_np = all_qualified_disease2disease_edges_with_shared_gene_df.to_numpy()

    ## create undirected graph and output shared_gene but_not shared_phenotype edges
    undirected_graph_with_shared_phenotype = nx.Graph()
    undirected_graph_with_shared_phenotype.add_weighted_edges_from(
        all_qualified_disease2disease_edges_with_shared_phenotype_np)

    ## get overlapped shared_gene_but_not shared_phenotype
    # all_qualified_disease2disease_edges_with_shared_gene_but_not_phenotype = all_qualified_disease2disease_edges_with_shared_gene_np.tolist()
    all_qualified_disease2disease_edges_with_shared_gene_but_not_phenotype_graph = nx.Graph()
    all_qualified_disease2disease_edges_with_shared_gene_but_not_phenotype_graph.add_weighted_edges_from(
        all_qualified_disease2disease_edges_with_shared_gene_np)

    for edge_with_weight in all_qualified_disease2disease_edges_with_shared_gene_but_not_phenotype_graph.edges:
        edge = edge_with_weight[:2]
        if undirected_graph_with_shared_phenotype.has_edge(*edge):
            all_qualified_disease2disease_edges_with_shared_gene_but_not_phenotype_graph.remove_edge(
                *list(edge))

    all_qualified_disease2disease_edges_with_shared_gene_but_not_phenotype_np = np.array(
        list(
            all_qualified_disease2disease_edges_with_shared_gene_but_not_phenotype_graph.edges.data(
                'weight')))

    # all_qualified_disease2disease_edges_with_shared_gene_but_not_phenotype_np = np.array(
    #     all_qualified_disease2disease_edges_with_shared_gene_but_not_phenotype_graph)

    all_qualified_disease2disease_edges_with_shared_gene_but_not_phenotype_df = pd.DataFrame(
        all_qualified_disease2disease_edges_with_shared_gene_but_not_phenotype_np,
        columns=all_qualified_disease2disease_edges_with_shared_gene_df.columns)

    return all_qualified_disease2disease_edges_with_shared_gene_but_not_phenotype_df


def get_qualified_disease2disease_edges_with_shared_phenotype_but_not_gene_edges(
        data,
        use_overlapped_disease=None):
    # TODO validate that qualified_diseases_edges both shared_gene and shared_phenotype are unweighted
    # todo haven't yet implmented for weighted version
    all_qualified_disease2disease_edges_with_shared_gene_df = get_qualified_disease2disease_edges_with_shared_genes(
        data, use_overlapped_disease)
    all_qualified_disease2disease_edges_with_shared_phenotype_df = get_qualified_disease2disease_edges_with_shared_phenotype(
        data, use_overlapped_disease)

    # convert df to grpah to be validate whether both are weighted or unweighted
    ## create graph for shared_gene
    graph_with_shared_gene = nx.Graph()
    graph_with_shared_gene.add_weighted_edges_from(
        all_qualified_disease2disease_edges_with_shared_gene_df.to_numpy())
    ## create graph for shared_phenotype
    graph_with_shared_phenotype = nx.Graph()
    graph_with_shared_phenotype.add_weighted_edges_from(
        all_qualified_disease2disease_edges_with_shared_phenotype_df.to_numpy())

    # validate that both are qualified_disease2disease_edges are both unweighted or both weighted
    assert (not data.is_graph_edges_weighted(graph_with_shared_gene,
                                             use_outside_graph=True)) \
           and (not data.is_graph_edges_weighted(graph_with_shared_phenotype,
                                                 use_outside_graph=True)), "oboth shared_gene and shared_phenotype should be both unweighted "

    # TODO after weight version is implemented, I need to check if there is a chance that shared_gene and shared_phenotype can have diferent weighted_edges
    # drop dubplicate row

    ## change variable name to reflect its types
    all_qualified_disease2disease_edges_with_shared_phenotype_np = all_qualified_disease2disease_edges_with_shared_phenotype_df.to_numpy()
    all_qualified_disease2disease_edges_with_shared_gene_np = all_qualified_disease2disease_edges_with_shared_gene_df.to_numpy()

    ## create undirected graph and output shared_phenotype but_not shared_gene edges
    undirected_graph_with_shared_gene = nx.Graph()
    undirected_graph_with_shared_gene.add_weighted_edges_from(
        all_qualified_disease2disease_edges_with_shared_gene_np)

    ## get overlapped shared_phenotype_but_not shared_gene
    # all_qualified_disease2disease_edges_with_shared_phenotype_but_not_gene = all_qualified_disease2disease_edges_with_shared_phenotype_np.tolist()
    all_qualified_disease2disease_edges_with_shared_phenotype_but_not_gene_graph = nx.Graph()
    all_qualified_disease2disease_edges_with_shared_phenotype_but_not_gene_graph.add_weighted_edges_from(
        all_qualified_disease2disease_edges_with_shared_phenotype_np)

    for edge_with_weight in all_qualified_disease2disease_edges_with_shared_phenotype_but_not_gene_graph.edges:
        edge = edge_with_weight[:2]
        if undirected_graph_with_shared_gene.has_edge(*edge):
            all_qualified_disease2disease_edges_with_shared_phenotype_but_not_gene_graph.remove_edge(
                *list(edge))

    all_qualified_disease2disease_edges_with_shared_phenotype_but_not_gene_np = np.array(
        list(
            all_qualified_disease2disease_edges_with_shared_phenotype_but_not_gene_graph.edges.data(
                'weight')))
    # all_qualified_disease2disease_edges_with_shared_phenotype_but_not_gene_np = np.array(
    #     all_qualified_disease2disease_edges_with_shared_phenotype_but_not_gene_graph)

    all_qualified_disease2disease_edges_with_shared_phenotype_but_not_gene_df = pd.DataFrame(
        all_qualified_disease2disease_edges_with_shared_phenotype_but_not_gene_np,
        columns=all_qualified_disease2disease_edges_with_shared_gene_df.columns)

    return all_qualified_disease2disease_edges_with_shared_phenotype_but_not_gene_df


# =====================
# ==Utility Code
# desc: common/freqently used code that are used across other functions (preferably) within the same module/files
# =====================

# --------for shared_phenotype
def get_phenotype2disease_edges_that_contain_qualified_nodes(data,
                                                             qualified_nodes,
                                                             saved_file_path=None):
    """qualified_nodes are used to filter  edges whose either nodes are qualified_nodes"""
    assert saved_file_path is not None, "if save_to_file is true, saved_file_path must be specified"

    # check that no files with the same name existed within these folder
    if path.exists(saved_file_path):
        edges_disease2phenotypes_that_contain_qualified_nodes_pd = pd.read_csv(
            saved_file_path, sep=',')
    else:
        qualified_doid_nodes = qualified_nodes  # renaming for self-documentation purposes

        disease2phenotypes_edges_pd = get_phenotype2diseiase_edges_from_disease_hpo2_dataset()
        edges_disease2phenotypes_that_contain_qualified_nodes_pd = \
            disease2phenotypes_edges_pd[
                disease2phenotypes_edges_pd['disease_id'].isin(
                    qualified_doid_nodes)]
        edges_disease2phenotypes_that_contain_qualified_nodes_pd.to_csv(
            saved_file_path, index=False)

    return edges_disease2phenotypes_that_contain_qualified_nodes_pd


def get_phenotype2diseiase_edges_from_disease_hpo2_dataset():
    disease2phenotype_edges_file_path = r'C:\Users\Anak\PycharmProjects\recreate_gene_disease\Data\raw\GPSim\Edges\disease_hpo.csv'

    # disease_id, hpo_id, disease_name, hpo_name, is_do
    edges_disease2phenotypes_pd = \
        pd.read_csv(disease2phenotype_edges_file_path)[
            ['disease_id', 'hpo_id']]
    return edges_disease2phenotypes_pd


# --------for shared_gene

# --------not belong to any of the UtilityCode/SubCategory

def get_qualified_disease2disease_edges_with_shared_nodes(graph, node_pairs):
    """

    @param graph: type = nx.Graph(); desc: graph must contain all node_pairs + nodes that are not included in node_pairs
    @param node_pairs: any
    @return:
    """

    qualified_edges = np.array(
        [edge for edge in node_pairs if
         len(list(nx.common_neighbors(graph, edge[0], edge[1]))) > 0])

    return qualified_edges


def get_graph_with_no_self_loop_edges(edges):
    # TODO please use add_weighted_edges_from instead of add_edges_from (reason: it is just more flexible overall)
    G = nx.Graph()
    G.add_edges_from(edges, weight=1)
    G.remove_edges_from(nx.selfloop_edges(G))

    return G


def get_qualified_diseases(data, use_overlapped_disease):
    if use_overlapped_disease:
        qualified_diseases_np = get_diseases_that_are_overlapped_between_GPSim_and_GeneDisease_graph()
    else:
        qualified_diseases_np = data.diseases_np

    assert data.is_in_original_diseases(
        qualified_diseases_np), "some of the generated overlapped diseases are not in the original GEneDisease diseases nodes "

    return qualified_diseases_np
