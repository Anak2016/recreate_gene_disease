import pandas as pd
import networkx as nx

from sklearn.model_selection import train_test_split
from torch_geometric.data import InMemoryDataset

from Sources.Preprocessing import *


class Dataset():
    def __init__(self):
        pass

    def split_train_test(self, x, y, split, stratify=None):
        """

        @param x: type = np; shape = (-1,1)
        @param y: type = np: shape = (-1,1)
        @param split:
        @return:
        """
        # use existing library of train_test_split
        # randomize int in rep_instances then split
        test_size = int(x.shape[0] * split)
        random_state = 42
        if stratify is None:
            x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                                test_size=test_size,
                                                                random_state=42)
        else:
            x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                                test_size=test_size,
                                                                random_state=42,
                                                                stratify=stratify)
        return (x_train, y_train), (x_test, y_test)

    def cv_split(self, x):
        return


class GeneDiseaseGeometricDataset(InMemoryDataset, Dataset):
    def __init__(self, root):

        super(GeneDiseaseGeometricDataset, self).__init__(root)

        # Data related variables
        self.data_df = pd.read_csv(self.raw_paths[0], sep=',')
        self.data_np = self.data_df[
            ['diseaseId', 'geneId']].to_numpy()  # (# instances, # Features)
        self.class_np = pd.read_csv(self.raw_paths[1], sep='\t',
                                    names=['diseaseId', 'class_id']).to_numpy()

        # graph related variables
        data_df_groupby_diseaseid = self.data_df['geneId'].groupby(
            self.data_df['diseaseId']).value_counts()
        self.gene_disease_edges = data_df_groupby_diseaseid.index.tolist()

        self.genes_np = self.data_df['geneId'].unique()
        self.diseases_np = self.data_df['diseaseId'].unique()

        # G = preprocessing.convert_numpy_to_networkx_graph(self.gene_disease_edges)
        G = nx.Graph()
        G.add_edges_from(self.gene_disease_edges, weight=1)

        self.G = G
        self.edges_np = np.array(list(G.edges))
        self.node_np = np.array(list(G.nodes))
        self.subgraphs_list = [G.subgraph(c) for c in
                               nx.connected_components(G)]
        self.largest_subgraphs = max(self.subgraphs_list, key=len)

        # dict coversion
        self.disease2class_id_dict = {val[0]: val[1] for val in self.class_np}

        # separate between disease and gene

    @property
    def raw_file_names(self):
        return [r'GeneDiseaseProject\COPD\Nodes\copd_label07_14_19_46.txt',
                r'GeneDiseaseProject\COPD\Nodes\copd_label_content07_14_19_46.txt',
                r'']

    @property
    def processed_file_names(self):
        return []

    def _download(self):
        """dataset to be download"""
        pass

    def _process(self):
        """use process when i need to process raw Data and save it some where"""
        pass

    def add_weighted_qualified_edges_to_graph(self, weighted_qualified_edges):
        """

        @param weighted_qualified_edges: type = np; shape=(-1,3) where last column = weight
        @return:
        """
        self.G.add_weighted_edges_from(weighted_qualified_edges)

    def is_disease2disease_edges_added_to_graph(self, outside_graph,
                                                use_outside_graph=False):
        """
        check self.G whether the graph has disease2disease edges
        @return: type = Boolean
        """
        assert use_outside_graph and (
                outside_graph is not None), " if use_outside_graph is true, outside_graph of type nx.Graph must be passed in"

        if use_outside_graph:
            graph = outside_graph
        else:
            graph = self.G.copy()

        is_node_disease_func = lambda x: x in self.diseases_np
        is_node_disease_vectorized = np.vectorize(is_node_disease_func)
        graph_edges_np = np.array(list(graph.edges))

        has_disease2disease_edges = is_node_disease_vectorized(graph_edges_np)
        has_disease2disease_edges = has_disease2disease_edges.all(axis=1).any()

        return has_disease2disease_edges


    def is_graph_edges_weighted(self, outside_graph, use_outside_graph=False):
        """
        check self.G whether or not edges of graph is weighted
        @return: type = Boolean
        """
        assert use_outside_graph and (
                outside_graph is not None), " if use_outside_graph is true, outside_graph of type nx.Graph must be passed in"

        if use_outside_graph:
            graph = outside_graph
        else:
            graph = self.G.copy()

        # is_type_float_vectorize = np.vectorize(lambda x: isinstance(x, float))
        # assert is_type_float_vectorize(np.array(list(graph.edges.data('weight')))[:, 2]), "For consistency, all value of weighted graph of graph must be of type float "

        is_weighted_one_func = lambda x: x != 1
        is_weighted_one_vectorize = np.vectorize(is_weighted_one_func)

        # TODO fix: AttributeError: 'tuple' object has no attribute 'edges'
        graph_edges_with_weight = np.array(list(graph.edges.data('weight')))
        # graph_edges_with_weight[:, 2] = graph_edges_with_weight[:,2].astype(float)
        weight_np = graph_edges_with_weight[:, 2].astype(float)

        has_weighted_edges = is_weighted_one_vectorize(weight_np).any()

        return has_weighted_edges


if __name__ == '__main__':
    # =====================
    # ==genedisease dataset
    # =====================

    # -- datasets
    gene_disease_root = r'c:\users\anak\pycharmprojects\recreate_gene_disease\data'  # indicate where file should be stored
    gene_disease_data = GeneDiseaseGeometricDataset(gene_disease_root)

    convert_node2class_id = np.vectorize(
        lambda x: gene_disease_data.node2class_id[x])
    gene_disease_class_id = convert_node2class_id(
        gene_disease_data.diseases_np)
    gene_disease_train_set, gene_disease_test_set = gene_disease_data.split_train_test(
        gene_disease_data.diseases_np, gene_disease_class_id, 0.6)
