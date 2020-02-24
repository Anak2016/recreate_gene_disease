import numpy as np
import pandas as pd

from Sources.Preparation.Data.conversion import Converter
from Sources.Preparation.Data.make_dataset import GeneDiseaseGeometricDataset
from global_param import *


def get_graph(use_gene_disease_graph=None,
              use_phenotype_gene_disease_graph=None,
              graph_edges_type=None):
    """

    @param add_phenotype_node: type = boolean
    @return: starter graph: type=nx.Graph(), desc= graph that disease2disease edges will be added to
    """
    assert use_gene_disease_graph is not None, "use_gene_disease_graph must be specified to avoid ambiguity"
    assert use_phenotype_gene_disease_graph is not None, "use_phenotype_gene_disease_graph must be specified to avoid ambiguity"
    assert graph_edges_type is not None, "graph_edges_type must be specified to avoid ambiguity"

    if use_phenotype_gene_disease_graph:
        assert isinstance(use_phenotype_gene_disease_graph,
                          bool), "add_phenotype_nodes only accept boolean type"

    # get GeneDisease graph as a starter grap
    data = GeneDiseaseGeometricDataset(GENEDISEASE_ROOT)
    GeneDisease_graph = data.original_GeneDisease_edges

    if use_phenotype_gene_disease_graph:
        if graph_edges_type == 'phenotype_gene_disease':

            starter_graph = add_phenotype_gene_disease_edges_to_graph(data,
                                                                      GeneDisease_graph)

            # phenotype2disease_edges_that_contain_qualified_doid_nodes_pd = pd.read_csv(
            #     PHENOTYPE2DISEASE_EDGES_THAT_CONTAIN_QUALIFIED_NODES_FILE_PATH,
            #     sep=',')
            #
            # # doid_disease_incident_to_phenotype2disease_edges_np = np.unique(
            # #     phenotype2disease_edges_that_contain_qualified_doid_nodes_pd[
            # #         'disease_id'])
            #
            # converter = Converter(data)
            #
            # # TODO here>>
            #     # if graph_edges_type == ''
            # try:
            #     phenotype2disease_edges_that_contain_qualified_doid_nodes_pd['disease_id'] = converter.original_doid2cui_mapping(
            #         phenotype2disease_edges_that_contain_qualified_doid_nodes_pd['disease_id'])
            #     phenotype2disease_edges_that_contain_qualified_cui_nodes_pd = phenotype2disease_edges_that_contain_qualified_doid_nodes_pd
            #     original_graph_edges = phenotype2disease_edges_that_contain_qualified_cui_nodes_pd
            # except:
            #     raise ValueError(
            #         "There exists disease nodes incident to phenotype2disease edges that are not contained with in original disease of GeneDisease dataser")
            #
            # # add weight column to DataFrame
            # weight = pd.DataFrame(np.ones(
            #     phenotype2disease_edges_that_contain_qualified_cui_nodes_pd.shape[0]),
            #     columns=['weight'])
            # weighted_phenotype2disease_edges_that_contain_qualified_cui_nodes_np = pd.concat(
            #     [phenotype2disease_edges_that_contain_qualified_cui_nodes_pd, weight],
            #     axis=1).to_numpy()

            # # add phenotype edges to graph
            # GeneDisease_graph.add_weighted_edges_from(
            #     weighted_phenotype2disease_edges_that_contain_qualified_cui_nodes_np)
            # starter_graph = GeneDisease_graph

        elif graph_edges_type == 'phenotype_gene_disease_phenotype':
            starter_graph = add_phenotype_gene_disease_phenotype_edges_to_graph(
                data, GeneDisease_graph)

        else:
            raise ValueError("")


    elif use_gene_disease_graph:
        starter_graph = GeneDisease_graph
    else:
        raise ValueError(
            "currently only 2 options are implemented: use_gene_disease_graph and use_phenotype_gene_disease_graph")

    return starter_graph


def add_phenotype_gene_disease_phenotype_edges_to_graph(data, G):
    weighted_phenotype_gene_disease_phenotype_edges_np, phenotype2disease_pd, phenotype2gene_pd  = get_phenotype_gene_disease_phenotype_edges(data)
    uniq_nodes = np.unique(weighted_phenotype_gene_disease_phenotype_edges_np[:,
              :-1].flatten().astype(str))

    # TODO
    # add phenotype edges to graph
    G.add_weighted_edges_from(
        weighted_phenotype_gene_disease_phenotype_edges_np)
    graph_with_added_phenotype_gene_disease_phenotype_edges = G

    return graph_with_added_phenotype_gene_disease_phenotype_edges


def get_phenotype_gene_disease_phenotype_edges(data):
    phenotype_gene_disease_file_path = r'C:\Users\Anak\PycharmProjects\recreate_gene_disease\Data\raw\GPSim\Edges\phenotype_gene_disease.csv'
    phenotype_gene_disease_df = pd.read_csv(phenotype_gene_disease_file_path)
    phenotype_gene_disease_df.drop_duplicates(inplace=True)  # just to be sure

    phenotype2gene_pd = phenotype_gene_disease_df[
        ['hpo_id', 'gene_id']].dropna().drop_duplicates(inplace=False)

    # phenotype2gene_pd['gene_id'] = phenotype2gene_pd['gene_id'].to_numpy().astype(int)
    gene_id_pd = phenotype2gene_pd['gene_id'].astype(int)
    gene_id_np = gene_id_pd.to_numpy().astype(int)

    phenotype2gene_np = phenotype2gene_pd.to_numpy()
    phenotype2gene_np[:,1] = gene_id_np


    # qualified indicated that gene and disease are contained wihtin original graph.
    ## (in the otheword) no new gene/disease nodes are created because the goal here is to add hpo node. and Thast is it )
    # phenotype2qualified_gene_np = gene_id_pd[[ i for i in data.genes_np if i in gene_id_pd].any()]
    phenotype2qualified_gene_pd = phenotype2gene_pd[
        phenotype2gene_pd.iloc[:, -1].astype(int).isin(data.genes_np)]
    phenotype2qualified_gene_np = phenotype2qualified_gene_pd.to_numpy()

    phenotype2disease_pd = phenotype_gene_disease_df[
        ['hpo_id', 'disease_id']].dropna().drop_duplicates(inplace=False)
    # gene2disease_np = phenotype_gene_disease_df[['gene_id','disease_id']].drop_duplicates(inplace=False).to_numpy()

    converter = Converter(data)
    # TODO convert doid to cui
    phenotype2disease_pd['disease_id'] = converter.original_doid2cui_mapping(phenotype2disease_pd['disease_id'])
    phenotype2disease_np = phenotype2disease_pd.to_numpy()

    # phenotype_gene_disease_phenotype_edges_np = np.concatenate(
    #     [phenotype2gene_np, phenotype2disease_np], axis=0)
    phenotype_gene_disease_phenotype_edges_np = np.concatenate(
        [phenotype2qualified_gene_np, phenotype2disease_np], axis=0)

    weight = pd.DataFrame(np.ones(
        phenotype_gene_disease_phenotype_edges_np.shape[0]),
        columns=['weight'])

    weighted_phenotype_gene_disease_phenotype_edges_np = np.concatenate(
        [phenotype_gene_disease_phenotype_edges_np, weight], axis=1)

    return weighted_phenotype_gene_disease_phenotype_edges_np, phenotype2disease_pd, phenotype2qualified_gene_pd


def add_phenotype_gene_disease_edges_to_graph(data, G):
    """

    @param G: type=nx.Graph
    @return: graph_with_added_phenotype_gene_disease_edges: type=nx.Graph
    """
    weighted_phenotype2disease_edges_that_contain_qualified_cui_nodes_np = get_phenotype_gene_disease_edges(
        data)

    uniq_nodes = np.unique(weighted_phenotype2disease_edges_that_contain_qualified_cui_nodes_np[:,
                           :-1].flatten().astype(str))

    # add phenotype edges to graph
    G.add_weighted_edges_from(
        weighted_phenotype2disease_edges_that_contain_qualified_cui_nodes_np)
    graph_with_added_phenotype_gene_disease_edges = G

    return graph_with_added_phenotype_gene_disease_edges


def get_phenotype_gene_disease_edges(data):
    """

    @param data:
    @return: weighted_phenotype2disease_edges_that_contain_qualified_cui_nodes_np
    """
    # TODO this function can be greatly shorten by extract phenotype2disease frmo phenotype_gene_disease.csv (file is documented in notebook/eda/dataset/GPSim.ipynb)
    ## I have not validate the correctness of the new approach
    ## note: before extracting phenotype2disease from phenotype_gene_disease.csv (make sure that it produce the same set of edges)
    ##          > it is expected that result will not be the same this is because original_doid2cui_mapping is 1:1
    ##              (data is in fact many to many which should produce original_doid2cui_mapping with 1:many mapping)

    phenotype2disease_edges_that_contain_qualified_doid_nodes_pd = pd.read_csv(
        PHENOTYPE2DISEASE_EDGES_THAT_CONTAIN_QUALIFIED_NODES_FILE_PATH,
        sep=',')

    # doid_disease_incident_to_phenotype2disease_edges_np = np.unique(
    #     phenotype2disease_edges_that_contain_qualified_doid_nodes_pd[
    #         'disease_id'])

    converter = Converter(data)

    # TODO here>>
    # if graph_edges_type == ''
    try:
        phenotype2disease_edges_that_contain_qualified_doid_nodes_pd[
            'disease_id'] = converter.original_doid2cui_mapping(
            phenotype2disease_edges_that_contain_qualified_doid_nodes_pd[
                'disease_id'])
        phenotype2disease_edges_that_contain_qualified_cui_nodes_pd = phenotype2disease_edges_that_contain_qualified_doid_nodes_pd
        original_graph_edges = phenotype2disease_edges_that_contain_qualified_cui_nodes_pd
    except:
        raise ValueError(
            "There exists disease nodes incident to phenotype2disease edges that are not contained with in original disease of GeneDisease dataser")

    # add weight column to DataFrame
    weight = pd.DataFrame(np.ones(
        phenotype2disease_edges_that_contain_qualified_cui_nodes_pd.shape[0]),
        columns=['weight'])
    weighted_phenotype2disease_edges_that_contain_qualified_cui_nodes_np = pd.concat(
        [phenotype2disease_edges_that_contain_qualified_cui_nodes_pd, weight],
        axis=1).to_numpy()

    return weighted_phenotype2disease_edges_that_contain_qualified_cui_nodes_np


def add_nodes_to_graph():
    pass
