import networkx as nx
import numpy as np
import pandas as pd

# from Sources.Preprocessing.preprocessing import Converter
from Sources.Preprocessing.convertion import Converter
from itertools import combinations
from Sources.Preprocessing.preprocessing import get_all_GeneDisease_unweighted_disease2disease_qualified_edges

def get_GPSim_disease2disease_qualified_edges(data, use_weight_edges):
    """
    note: self loop is not a qualified edges ( so it will be removed here)

    # TODO create "weighted_edges" argument. and pass qualified_edges with weighted here
    @return:
    """
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

    converter = Converter()
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

        # TODO error
        assert data.is_graph_edges_weighted(G,
                                            use_outside_graph=True), "use_weighted_edges is True, but all edges in graph has weight == 1"

        non_zero_GPsim_disease_cui_disease_pair_with_no_self_loop_df = nx.to_pandas_edgelist(
            G)

    else:
        G = nx.Graph()
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
                                                    use_weighted_edges):
    """

    @param G: networkx.Graph()
    @param nodes: type = np; shape = (-1,1)
    @return: list of qualified edges; shape = (-1, 2)
    """
    # all_disease2disease_edges = list(combinations(nodes, 2))

    # TODO currently there is no different between use_weighted_edges = True/False because qualified edges of GeneDisease has no weight
    # note: potential weight of qualified edges are jaccard coefficient among other
    if use_weighted_edges:
        raise ValueError(
            'not yet implemented: currently i am not working any qualified edges of GeneDisease that has weight')

    else:

        all_qualified_disease2disease_edges = get_all_GeneDisease_unweighted_disease2disease_qualified_edges(data)

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

