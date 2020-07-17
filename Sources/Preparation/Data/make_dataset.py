import os
from itertools import product
from os import path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from torch_geometric.data import InMemoryDataset


# from Sources.Preprocessing.apply_preprocessing import select_emb_save_path


# from Sources.Preprocessing import *


class Dataset():
    def __init__(self):
        pass

    def split_edges_train_test(self, data_edges, ):
        test_size = int(data_edges, )
        pass

    # def split_train_test(self, x, y, split, stratify=None):
    #     """
    #
    #     @param x: type = np; shape = (-1,1)
    #     @param y: type = np: shape = (-1,1)
    #     @param split:
    #     @return:
    #     """
    #     # use existing library of train_test_split
    #     # randomize int in rep_instances then split
    #
    #     convert_disease2class_id = np.vectorize(
    #         lambda z: self.disease2class_id_dict[z])
    #     disease_class = convert_disease2class_id(self.diseases_np)
    #
    #     test_size = int(x.shape[0] * split)
    #     random_state = 42
    #     if stratify is None:
    #         x_train, x_test, y_train, y_test = train_test_split(x, y,
    #                                                             test_size=test_size,
    #                                                             random_state=42)
    #     else:
    #         x_train, x_test, y_train, y_teargst = train_test_split(x, y,
    #                                                             test_size=test_size,
    #                                                             random_state=42,
    #                                                             stratify=stratify)
    #     return (x_train, y_train), (x_test, y_test)

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
        self.diseases_np= self.data_df['diseaseId'].unique()
        self.disease_df = pd.DataFrame(self.diseases_np, index = np.array(list(range(self.diseases_np.shape[0]))))

        np.random.shuffle(self.diseases_np)

        # G = preprocessing.convert_numpy_to_networkx_graph(self.gene_disease_edges)
        G = nx.Graph()
        G.add_edges_from(self.gene_disease_edges, weight=1)

        self.original_GeneDisease_edges = G.copy()
        self.G = G
        self.edges_np = np.array(list(G.edges))
        self.node_np = np.array(list(G.nodes))
        self.subgraphs_list = [G.subgraph(c) for c in
                               nx.connected_components(G)]
        self.largest_subgraphs = max(self.subgraphs_list, key=len)

        # dict coversion
        self.disease2class_id_dict = {val[0]: val[1] for val in self.class_np}


        # train and test set splitting
        self.train_set = None
        self.test_set = None

        # cross validation split
        self.stratified_k_fold = []
        print()

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

    def get_existed_and_non_existed_gene_disease_edges(self):
        '''

        @return: all return variable are list (notice no _np as variable's suffix)
        '''

        all_possible_gene_disease_edges = list(
            product(self.diseases_np, self.genes_np))

        # all_possible_gene_disease_edges = np.unique(
        #     np.array(all_possible_gene_disease_edges))

        all_possible_gene_disease_edges_with_label_dict = {}

        # get gene_disease edges from getting incident of disease from data.original_edges
        # do all_possible_gene_disease_edges.differences(existed_gene_disease_edges)
        # existed_gene_disease_edges = [(i, j) for i in self.diseases_np for
        #                               j in
        #                               self.original_GeneDisease_edges.neighbors(
        #                                   i)]

        # existed_gene_disease_edges = self.gene_disease_edges
        existed_gene_disease_edges = list(self.largest_subgraphs.edges)

        existed_gene_disease_edges = list(set(existed_gene_disease_edges))
        existed_gene_disease_edges_label = list(
            np.ones((len(existed_gene_disease_edges), 1)))

        non_existed_gene_disease_edges = list(set(
            all_possible_gene_disease_edges).difference(
            set(existed_gene_disease_edges)))
        non_existed_gene_disease_edges_label = list(
            np.zeros((len(non_existed_gene_disease_edges), 1)))

        return existed_gene_disease_edges, existed_gene_disease_edges_label, non_existed_gene_disease_edges, non_existed_gene_disease_edges_label

    def run_cross_validation(self, x_np, y_np, k_fold, stratify):
        """

        @return:

        """
        if stratify:
            return list(StratifiedKFold(n_splits=k_fold).split(x_np, y_np))
        else:
            raise ValueError('not yet implemented')

    def run_split_train_test_df(self, x_df, y_df, split, stratify):
        """

        @param x_df:  type = np
        @param y_df:  type = np
        @param split:
        @param stratify:
        @return:
        """

        test_size = int(x_df.shape[0] * split)
        random_state = 42

        if not stratify:
            x_train, x_test, y_train, y_test = train_test_split(x_df, y_df,
                                                                test_size=test_size,
                                                                random_state=random_state)
        else:
            x_train, x_test, y_train, y_test = train_test_split(x_df, y_df,
                                                                test_size=test_size,
                                                                random_state=random_state,
                                                                stratify=y_df)

        return (x_train, y_train), (x_test, y_test)

    def run_split_train_test_np(self, x_np, y_np, split, stratify):
        """

        @param x_np:  type = np
        @param y_np:  type = np
        @param split:
        @param stratify:
        @return:
        """

        test_size = int(x_np.shape[0] * split)
        random_state = 42

        if not stratify:
            x_train, x_test, y_train, y_test = train_test_split(x_np, y_np,
                                                                test_size=test_size,
                                                                random_state=random_state)
        else:
            x_train, x_test, y_train, y_test = train_test_split(x_np, y_np,
                                                                test_size=test_size,
                                                                random_state=random_state,
                                                                stratify=y_np)

        return (x_train, y_train), (x_test, y_test)

    def run_split_cross_validation_for_node_classification(self, k_fold,
                                                           stratify):
        # TODO here>>-3 finish building run_split_cross_vlaition_for_node_classifition
        convert_disease2class_id = np.vectorize(
            lambda x: self.disease2class_id_dict[x])
        disease_class = convert_disease2class_id(self.diseases_np)

        selected_shuffled_x_with_label = np.concatenate(
            [self.disease_np, disease_class], axis=1)

        return self.create_stratified_k_fold(self.diseases_np, disease_class,
                                             k_fold,
                                             selected_shuffled_x_with_label,
                                             stratify)

    def create_split_by_nodes(self,
                              selected_edges_with_label_dict,
                              nodes_to_be_splitted,
                              train_set=None, test_set=None,
                              train_set_ind=None, test_set_ind=None,
                              ):

        if train_set_ind is not None and test_set_ind is not None:
            assert train_set is None and test_set is None, ''
            train_pos_edges = np.array([[i, j, '1.0'] for i in
                                        nodes_to_be_splitted[
                                            train_set_ind] for j in
                                        self.original_GeneDisease_edges.neighbors( # TODO here>>-11 check that original_GeneDisease_edges.neightbors(i) output expected result
                                            i)])
            test_pos_edges = np.array([[i, j, '1.0'] for i in
                                       nodes_to_be_splitted[
                                           test_set_ind] for j in
                                       self.original_GeneDisease_edges.neighbors(
                                           i)])
        else:
            raise ValueError('there is not yet a usecase that would feed train_set and test_Set and train_pos_edges and test_pos_edges respectively')
            # assert train_set is not None and test_set is not None, ''
            # train_pos_edges = train_set
            # test_pos_edges = test_set

        selected_non_existed_edges = \
            selected_edges_with_label_dict['non_existed_edges']

        selected_train_neg_edges_ind = np.random.choice(
            selected_non_existed_edges.shape[0],
            len(train_pos_edges), replace=False)
        # selected_train_neg_edges = \
        # selected_edges_with_label_dict['non_existed_edges'].iloc[
        #     selected_train_neg_edges_ind]

        # TODO here>>-11 check correctness of the paragraph of code below
        selected_train_neg_edges = selected_non_existed_edges.iloc[
            selected_train_neg_edges_ind]
        selected_non_existed_edges = selected_non_existed_edges.drop(
            selected_non_existed_edges.index[
                selected_train_neg_edges_ind])

        # test_pos_edges = [f'{i}_{j}' for i in nodes_to_be_splitted[test_set_ind] for j in self.original_GeneDisease_edges.neighbors(i)]

        selected_test_neg_edges_ind = np.random.choice(
            selected_non_existed_edges.shape[0],
            len(test_pos_edges),
            replace=False)
        # selected_test_neg_edges = \
        # selected_edges_with_label_dict['non_existed_edges'].iloc[
        #     selected_test_neg_edges_ind]
        selected_test_neg_edges = selected_non_existed_edges.iloc[
            selected_test_neg_edges_ind]

        # train_set = np.concatenate([train_pos_edges, selected_edges_with_label_dict['non_existed_edges'].iloc[train_neg_edges_ind]], axis=0)
        train_set = np.concatenate(
            [train_pos_edges, selected_train_neg_edges], axis=0)
        np.random.shuffle(train_set)
        shuffled_train_set = train_set

        test_set = np.concatenate(
            [test_pos_edges, selected_test_neg_edges], axis=0)
        # test_set = np.concatenate([test_pos_edges, selected_edges_with_label_dict['non_existed_edges'][test_neg_edges_ind]], axis=0)
        np.random.shuffle(test_set)
        shuffled_test_set = test_set

        # validate that test and test have no common edges.
        tmp = np.concatenate((train_set, test_set), axis=0).astype(
            str)
        uniq_tmp = np.unique(tmp, axis=0)

        assert tmp.shape == uniq_tmp.shape, 'DATA LEAKAGE! => train_set and test_set contain same edges'

        return shuffled_train_set, shuffled_test_set

    def rank_disease_by_degree_in_place(self, ascending_order=True):
        # TODO validate that self.G is orignal graph that new edges has been added to it.
        disease_degree_tuple = [(i,self.G.degree[i]) for i in self.diseases_np]
        ascendingly_sorted_tuple = np.array(
            sorted(disease_degree_tuple, key=lambda x: x[1])[::-1])
        if ascending_order:
            self.diseases_np = ascendingly_sorted_tuple[:,0]
        else:
            self.diseases_np = ascendingly_sorted_tuple[:, 0][::-1]

    def create_stratified_k_fold(self, data, x_np, y_np, k_fold,
                                 selected_x_with_label, stratify,
                                 split_by_node=None,
                                 nodes_to_be_splitted=None,
                                 splitted_edges_file_path=None
                                 ):

        assert selected_x_with_label.shape[0] == x_np.shape[0] == y_np.shape[
            0], "all three must have the number of row "
        assert split_by_node is not None, "split_by_node must be specified to avoid ambiguity"
        assert splitted_edges_file_path is not None, "splitted_edges_file_path must be specified to avoid ambiguity"

        splitted_edges_dict = {}

        if stratify:
            if not split_by_node:

                for ind, (train_set_ind, test_set_ind) in enumerate(self.run_cross_validation(
                        x_np,
                        y_np, k_fold,
                        stratify,
                )):

                    train_set = selected_x_with_label[
                        train_set_ind]
                    test_set = selected_x_with_label[
                        test_set_ind]

                    splitted_edges_dict[ind] = {'edges_train_set': train_set,
                                           'edges_test_set': test_set}

                    self.stratified_k_fold.append(
                        {'train_set': train_set, 'test_set': test_set})

                    # # check there is no spliting of graph and number of nodes is same
                    # if (nx.number_connected_components(G_temp) == 1) and (
                    #         len(G_temp.nodes) == initial_node_count):
                    #     omissible_links_index.append(i)
                    #     fb_df_temp = fb_df_temp.drop(index=i)

            else:

                selected_x_with_label_pd = pd.DataFrame(
                    selected_x_with_label)
                selected_edges_with_label_dict = {}
                for label, group in selected_x_with_label_pd.groupby(
                        selected_x_with_label_pd.columns[-1]):
                    label = float(label)
                    if label == 1:
                        selected_edges_with_label_dict['existed_edges'] = group
                    elif label == 0:
                        selected_edges_with_label_dict[
                            'non_existed_edges'] = group
                    else:
                        raise ValueError(
                            'label is expected to be 0 or 0 (existed_edges or non-existed_edges)')

                # TODO ascending_order = True, actually works like decending oder
                self.rank_disease_by_degree_in_place(ascending_order=False)

                for ind, (train_set_ind, test_set_ind) in enumerate(
                        self.run_cross_validation(
                            nodes_to_be_splitted,
                            np.ones(nodes_to_be_splitted.shape[0]),
                            # place holder with constanit that shape is euqla to data.disease_np.shape[0] and only contain 1 unique value
                            k_fold,
                            stratify)):

                    # TODO This is a quick fix. Utimately, I want nodes_to_be_splitted to be passed in as argument,
                    #  but for now, there is only disease that is used as nodes to be split so this fix should works
                    nodes_to_be_splitted = self.diseases_np

                    shuffled_train_set, shuffled_test_set = self.create_split_by_nodes(
                        selected_edges_with_label_dict,
                        nodes_to_be_splitted,
                        train_set_ind=train_set_ind, test_set_ind=test_set_ind)

                    splitted_edges_dict[ind] = {
                        'edges_train_set': shuffled_train_set,
                        'edges_test_set': shuffled_test_set}

                    self.stratified_k_fold.append(
                        {'train_set': shuffled_train_set,
                         'test_set': shuffled_test_set})

                    # print(shuffled_train_set.shape)
                    # print(shuffled_test_set.shape)
                    # # TODO refactor the paragraph of code below
                    # ## Goal = reuse it in split_Train_test_for_tlink_prediction/split_by_node
                    # ## What should be the name of the function?
                    # ##      > can I reuse create_train_test_set?
                    #
                    # # TODO how to get dataframe using train_pos_edges (using string? using
                    # # train_pos_edges = [f'{i}_{j}' for i in nodes_to_be_splitted[train_set_ind] for j in self.original_GeneDisease_edges.neighbors(i)]
                    # train_pos_edges = np.array([[i, j, '1.0'] for i in
                    #                             nodes_to_be_splitted[
                    #                                 train_set_ind] for j in
                    #                             self.original_GeneDisease_edges.neighbors(
                    #                                 i)])
                    # selected_non_existed_edges = \
                    #     selected_edges_with_label_dict['non_existed_edges']
                    #
                    # selected_train_neg_edges_ind = np.random.choice(
                    #     selected_non_existed_edges.shape[0],
                    #     len(train_pos_edges), replace=False)
                    # # selected_train_neg_edges = \
                    # # selected_edges_with_label_dict['non_existed_edges'].iloc[
                    # #     selected_train_neg_edges_ind]
                    # selected_train_neg_edges = selected_non_existed_edges.iloc[
                    #     selected_train_neg_edges_ind]
                    #
                    # selected_non_existed_edges = selected_non_existed_edges.drop(
                    #     selected_non_existed_edges.index[
                    #         selected_train_neg_edges_ind])
                    #
                    # # test_pos_edges = [f'{i}_{j}' for i in nodes_to_be_splitted[test_set_ind] for j in self.original_GeneDisease_edges.neighbors(i)]
                    # test_pos_edges = np.array([[i, j, '1.0'] for i in
                    #                            nodes_to_be_splitted[
                    #                                test_set_ind] for j in
                    #                            self.original_GeneDisease_edges.neighbors(
                    #                                i)])
                    #
                    # selected_test_neg_edges_ind = np.random.choice(
                    #     selected_non_existed_edges.shape[0],
                    #     len(test_pos_edges),
                    #     replace=False)
                    # # selected_test_neg_edges = \
                    # # selected_edges_with_label_dict['non_existed_edges'].iloc[
                    # #     selected_test_neg_edges_ind]
                    # selected_test_neg_edges = selected_non_existed_edges.iloc[
                    #     selected_test_neg_edges_ind]
                    #
                    # # TODO error non_existeed_edges does not exist
                    # ## validate that train_pos_edges are test_pos_edges are targeted edges
                    # # train_set = np.concatenate([train_pos_edges, selected_edges_with_label_dict['non_existed_edges'].iloc[train_neg_edges_ind]], axis=0)
                    # train_set = np.concatenate(
                    #     [train_pos_edges, selected_train_neg_edges], axis=0)
                    # np.random.shuffle(train_set)
                    # shuffled_train_set = train_set
                    #
                    # test_set = np.concatenate(
                    #     [test_pos_edges, selected_test_neg_edges], axis=0)
                    # # test_set = np.concatenate([test_pos_edges, selected_edges_with_label_dict['non_existed_edges'][test_neg_edges_ind]], axis=0)
                    # np.random.shuffle(test_set)
                    # shuffled_test_set = test_set
                    #
                    # # validate that test and test have no common edges.
                    # tmp = np.concatenate((train_set, test_set), axis=0).astype(
                    #     str)
                    # uniq_tmp = np.unique(tmp, axis=0)
                    #
                    # assert tmp.shape == uniq_tmp.shape, 'DATA LEAKAGE! => train_set and test_set contain same edges'
                    #
                    # # favorite_color = { "lion": "yellow", "kitty": "red" }
                    # splitted_edges_dict[ind] = {
                    #     'edges_train_set': shuffled_train_set,
                    #     'edges_test_set': shuffled_test_set}
                    #
                    # print(shuffled_train_set.shape)
                    # print(shuffled_test_set.shape)
                    #
                    # self.stratified_k_fold.append(
                    #     {'train_set': shuffled_train_set,
                    #      'test_set': shuffled_test_set})
                    # # TODO how many edges are there for each fold? => selected neg_edges equal to number of pos_edges => return train_set(shuffled_pos_neg_edges) and test_set
                    #
                    # selected_pos_edges_with_label = pd.DataFrame(selected_x_with_label).groupby(
                    #     selected_x_with_label.index[-1])

                    # train_set = selected_x_with_label[
                    #     train_set_ind]
                    # test_set = selected_x_with_label[
                    #     test_set_ind]
                    # self.stratified_k_fold.append((train_set, test_set))
                # create directory if not alreayd exist

        else:
            raise ValueError(
                'not yet implemented; create_stratified_k_Fold with stratfied = False')

        splitted_edges_dir = '\\'.join(
            splitted_edges_file_path.split('\\')[:-1])
        if not os.path.exists(splitted_edges_dir):
            os.makedirs(splitted_edges_dir)

        import pickle
        pickle.dump(splitted_edges_dict,
                    open(splitted_edges_file_path, "wb"))

        return self.stratified_k_fold

    def get_all_pos_neg_edges_for_link_prediction(self):
        existed_gene_disease_edges, existed_gene_disease_edges_label, non_existed_gene_disease_edges, non_existed_gene_disease_edges_label = self.get_existed_and_non_existed_gene_disease_edges()

        pos_edges_with_label_np = np.concatenate((np.array(
            existed_gene_disease_edges), np.array(
            existed_gene_disease_edges_label)), axis=1)
        neg_edges_with_label_np = np.concatenate((np.array(
            non_existed_gene_disease_edges), np.array(
            non_existed_gene_disease_edges_label)), axis=1)

        np.random.shuffle(pos_edges_with_label_np)
        np.random.shuffle(neg_edges_with_label_np)


        return pos_edges_with_label_np, neg_edges_with_label_np



    def get_selected_pos_neg_edges_for_link_prediction(self):
        existed_gene_disease_edges, existed_gene_disease_edges_label, non_existed_gene_disease_edges, non_existed_gene_disease_edges_label = self.get_existed_and_non_existed_gene_disease_edges()

        pos_edges_with_label_np = np.concatenate((np.array(
            existed_gene_disease_edges), np.array(
            existed_gene_disease_edges_label)), axis=1)
        neg_edges_with_label_np = np.concatenate((np.array(
            non_existed_gene_disease_edges), np.array(
            non_existed_gene_disease_edges_label)), axis=1)

        train_test_size = pos_edges_with_label_np.shape[0]

        selected_neg_edges_with_label_ind = np.random.choice(
            neg_edges_with_label_np.shape[0],
            train_test_size,
            replace=False)
        selected_neg_edges_with_label = neg_edges_with_label_np[
                                        selected_neg_edges_with_label_ind,
                                        :]

        assert selected_neg_edges_with_label.shape[0] == \
               pos_edges_with_label_np.shape[
                   0], "ratio of pos_sample and neg_sample are 50:50"

        selected_pos_neg_edges_with_label = np.concatenate(
            [pos_edges_with_label_np, selected_neg_edges_with_label],
            axis=0)

        np.random.shuffle(selected_pos_neg_edges_with_label)
        selected_shuffled_pos_neg_edges_with_label = selected_pos_neg_edges_with_label

        return selected_shuffled_pos_neg_edges_with_label

    def run_split_cross_validation_for_link_prediction(self, data, k_fold,
                                                       stratify,
                                                       split_by_node=None,
                                                       nodes_to_be_splitted=None,
                                                       splitted_edges_file_path=None):
        assert isinstance(split_by_node, bool), ''
        assert splitted_edges_file_path is not None, "splitted_edges_file_path must be specified to avoid ambiguity"
        if split_by_node:
            assert nodes_to_be_splitted is not None, "nodes_to_be_splitted must be specified to avoid ambiguity"

        # TODO saved this edges to file
        edges_with_pos_neg_as_label = r'C:\Users\Anak\PycharmProjects\recreate_gene_disease\Data\interim\LinkPrediction\GeneDiseaseProject\Edges\edges_with_pos_neg_as_label.csv'
        # check that no files with the same name existed within these folder
        if path.exists(edges_with_pos_neg_as_label):
            raise ValueError('not yet implemented')
            # TODO what do I expect this file to do?
            ## what is the content of the file?
            ## what value do function expect to return

            d = pd.read_csv(edges_with_pos_neg_as_label,
                            sep=',')  # file is not expected to have index

        else:

            selected_shuffled_pos_neg_edges_with_label = self.get_selected_pos_neg_edges_for_link_prediction()
            selected_shuffled_pos_neg_edges, selected_shuffled_pos_neg_edges_label = selected_shuffled_pos_neg_edges_with_label[
                                                                                     :,
                                                                                     :-1], selected_shuffled_pos_neg_edges_with_label[
                                                                                           :,
                                                                                           -1]
            # selected_shuffled_pos_neg_edges_with_label_df = pd.DataFrame(x_np, )
            return self.create_stratified_k_fold(data,
                                                 selected_shuffled_pos_neg_edges,
                                                 selected_shuffled_pos_neg_edges_label,
                                                 k_fold,
                                                 selected_shuffled_pos_neg_edges_with_label,
                                                 stratify,
                                                 split_by_node=split_by_node,
                                                 nodes_to_be_splitted=nodes_to_be_splitted,
                                                 splitted_edges_file_path=splitted_edges_file_path)

    def run_split_train_test_for_node_classification(self, split, stratify,
                                                     is_input_numpy):

        convert_disease2class_id = np.vectorize(
            lambda z: self.disease2class_id_dict[z])
        disease_class = convert_disease2class_id(self.diseases_np)

        if not is_input_numpy:
            return self.run_split_train_test_df(self.disease_df, disease_class,
                                             split,
                                             stratify)
        else:
            return self.run_split_train_test_np(self.disease_df, disease_class,
                                                split,
                                                stratify)

    def convert_nodes_to_str(self):

        tmp = self.G.copy()
        relabel_func = lambda i: str(i) if isinstance(i, int) else i
        tmp = nx.relabel_nodes(tmp,relabel_func)

        return tmp


    def new_create_train_test_set(self,
                              split,
                              stratify,
                              split_by_node=None,
                              nodes_to_be_splitted=None,
                              splitted_edges_file_path=None):

        import pathlib
        saved_splitted_edges_folder = pathlib.Path(r'C:\Users\Anak\PycharmProjects\recreate_gene_disease\Data\interim\LinkPrediction\GeneDiseaseProject\Edges')
        train_edges_file = f'train_edges_file_with_train_split={1-split}.pickle'
        test_edges_file  = f'test_edges_file_with_test_split={split}.pickle'

        saved_train_path  = saved_splitted_edges_folder/train_edges_file
        saved_test_path  = saved_splitted_edges_folder/test_edges_file
        import pickle
        if path.exists(saved_test_path) and path.exists(saved_train_path):

            test_edges = pickle.load(open(saved_test_path, 'rb'))
            train_edges = pickle.load(open(saved_train_path, 'rb'))

        else:
            pos_edges_with_label_np, neg_edges_with_label_np =  self.get_all_pos_neg_edges_for_link_prediction()
            # for 40 percent of existed edges, remove 1 edge at a time
            # G_disease_np = self.diseases_np
            G_tmp = self.convert_nodes_to_str()
            num_subgraph = len(list(nx.connected_components(G_tmp)))
            num_removed_edges = int(len(G_tmp) * split)
            num_nodes_before_remove_edges = len(G_tmp)
            G_tmp_2 = G_tmp.copy()

            np.random.seed(100)
            count_removed_eges = 0

            # NOTE: for some reason, [C0085207, '197']does not exist, so I just give up on this edge bcause it is only 1 edge
            existing_pos_edges_with_label_np = []
            for i in pos_edges_with_label_np[:, :-1]:
                if G_tmp_2.has_edge(i[0], i[1]):
                    existing_pos_edges_with_label_np.append(i)
                else:
                    print(i)
            existing_pos_edges_with_label_np = np.array(existing_pos_edges_with_label_np)
            ind_pos_edges_with_label = np.arange(existing_pos_edges_with_label_np.shape[0])

            tmp = []
            count_tmp = 0
            all_removed_edges = []
            while count_removed_eges < num_removed_edges:
                removed_ind = np.random.choice(ind_pos_edges_with_label, 1)
                removed_edges = existing_pos_edges_with_label_np[removed_ind][0]
                tmp.append(','.join(list(removed_edges))) # here>> why same edges are chosen?
                count_tmp += 1
                existing_pos_edges_with_label_np = np.delete(existing_pos_edges_with_label_np, removed_ind, axis=0)
                ind_pos_edges_with_label = np.arange(
                    existing_pos_edges_with_label_np.shape[0])
                # np.delete(existing_pos_edges_with_label_np, removed_ind)
                G_tmp_2.remove_edge(removed_edges[0], removed_edges[1])
                print(count_removed_eges)
                if (nx.number_connected_components(G_tmp_2) == num_subgraph) and len(G_tmp_2) == num_nodes_before_remove_edges:
                    count_removed_eges += 1
                    G_tmp.remove_edge(removed_edges[0], removed_edges[1])
                    all_removed_edges.append(removed_edges)
                else:
                    G_tmp_2 = G_tmp.copy()

            train_pos_edges_without_weight = np.array(list(G_tmp.edges))

            weight_of_train_pos_edges = np.ones(train_pos_edges_without_weight.shape[0], dtype=str).reshape(-1,1)
            weight_of_test_pos_edges = np.ones(len(all_removed_edges), dtype=str).reshape(-1,1)

            train_pos_edges = np.hstack((train_pos_edges_without_weight, weight_of_train_pos_edges))
            selected_train_neg_edges_ind = np.random.choice(np.arange(neg_edges_with_label_np.shape[0]), len(train_pos_edges),
                             replace=False)
            train_neg_edges = neg_edges_with_label_np[selected_train_neg_edges_ind]
            test_pos_edges = np.hstack((np.array(all_removed_edges), weight_of_test_pos_edges))

            left_over_test_neg_edges_ind = np.delete(np.arange(neg_edges_with_label_np.shape[0]), selected_train_neg_edges_ind)
            selected_test_neg_edges_ind = np.random.choice(left_over_test_neg_edges_ind, len(test_pos_edges), replace=False)
            test_neg_edges = neg_edges_with_label_np[selected_test_neg_edges_ind]

            # assert len(selected_train_neg_edges_ind) +  len(selected_test_neg_edges_ind) == len(neg_edges_with_label_np), ''
            # assert len(np.unique(np.vstack((test_neg_edges, train_neg_edges)), axis=0)) == len(neg_edges_with_label_np)

            train_edges = np.vstack((train_pos_edges, train_neg_edges))
            # test_edges = test_pos_edges
            test_edges = np.vstack((test_pos_edges, test_neg_edges))

            import pathlib

            os.makedirs(saved_splitted_edges_folder, exist_ok=True)


            if not path.exists(saved_train_path):
                import pickle
                pickle.dump(train_edges,
                            open(saved_train_path, "wb"))

            if not path.exists(saved_test_path):
                import pickle
                pickle.dump(test_edges,
                            open(saved_test_path, "wb"))


        return train_edges, test_edges

        # select randomly from neg_edges to be the same amount as pos_edges that still remain after remove 40 percent


    def create_train_test_set(self, x_np,
                              y_np,
                              split,
                              selected_x_with_label,
                              stratify,
                              split_by_node=None,
                              nodes_to_be_splitted=None,
                              splitted_edges_file_path=None):
        # TODO look at create_stratified_k_fold for example. What else do i Need here?
        assert self.train_set is None and self.test_set is None, ''

        if not split_by_node:

            (x_train, y_train), (x_test, y_test) = self.run_split_train_test_np(
                x_np,
                y_np,
                split,
                stratify)
            self.train_set = np.concatenate((x_train, y_train[:,np.newaxis]), axis=1)
            self.test_set = np.concatenate((x_test, y_test[:,np.newaxis]), axis=1)

        else:
            selected_x_with_label_pd = pd.DataFrame(
                selected_x_with_label)
            selected_edges_with_label_dict = {}

            # TODO here>>-11 this check correctness of paragraph of code below
            for label, group in selected_x_with_label_pd.groupby(
                    selected_x_with_label_pd.columns[-1]):
                label = float(label)
                if label == 1:
                    selected_edges_with_label_dict['existed_edges'] = group
                elif label == 0:
                    selected_edges_with_label_dict[
                        'non_existed_edges'] = group
                else:
                    raise ValueError(
                        'label is expected to be 0 or 0 (existed_edges or non-existed_edges)')

            # train_set_ind, test_set_ind = self.run_split_train_test(x_np, y_np, split, stratify)
            # TODO how to make ind and non index compatible?
            ## Take a look into the create_split_by_nodes, What are the changes I have to make ?
            ##      > if it is too much, just do a quick patch and rewrite code
            # (x_train, y_train), (x_test, y_test) = self.run_split_train_test(x_np,
            #                                                              y_np,
            #                                                              split,
            #                                                              stratify)

            # (x_train, y_train), (x_test, y_test) = self.run_split_train_test(
            #     nodes_to_be_splitted,
            #     np.ones(nodes_to_be_splitted.shape[0]), # place holder with constanit that shape is euqla to data.disease_np.shape[0] and only contain 1 unique value
            #     split,
            #     stratify)
            #
            # train_set = np.concatenate((x_train, y_train[:, np.newaxis]),
            #                            axis=1)
            # test_set = np.concatenate((x_test, y_test[:, np.newaxis]), axis=1)

            (x_train_ind, _), (x_test_ind, _) = self.run_split_train_test_np(
                np.arange(nodes_to_be_splitted.shape[0]),
                np.ones(nodes_to_be_splitted.shape[0]), # place holder with constanit that shape is euqla to data.disease_np.shape[0] and only contain 1 unique value
                split,
                stratify)

            shuffled_train_set, shuffled_test_set = self.create_split_by_nodes(
                selected_edges_with_label_dict,
                nodes_to_be_splitted,
                train_set_ind = x_train_ind, test_set_ind=x_test_ind)
                # train_set=train_set, test_set=test_set)

            self.train_set, self.test_set = shuffled_train_set, shuffled_test_set

        splitted_edges_dir = '\\'.join(
            splitted_edges_file_path.split('\\')[:-1])
        if not os.path.exists(splitted_edges_dir):
            os.makedirs(splitted_edges_dir)

        import pickle

        # favorite_color = { "lion": "yellow", "kitty": "red" }
        splitted_edges_dict = {'edges_train_set': self.train_set,
                               'edges_test_set': self.test_set}

        pickle.dump(splitted_edges_dict,
                    open(splitted_edges_file_path, "wb"))

        return self.train_set, self.test_set

    def run_split_train_test_for_link_prediction(self, split, stratify,
                                                 split_by_node=None,
                                                 nodes_to_be_splitted=None,
                                                 splitted_edges_file_path=None):

        # TODO saved this edges to file
        edges_with_pos_neg_as_label = r'C:\Users\Anak\PycharmProjects\recreate_gene_disease\Data\interim\LinkPrediction\GeneDiseaseProject\Edges\edges_with_pos_neg_as_label.csv'
        # check that no files with the same name existed within these folder
        if path.exists(edges_with_pos_neg_as_label):
            raise ValueError('not yet implemented')
            # TODO what do I expect this file to do?
            ## what is the content of the file?
            ## what value do function expect to return

            d = pd.read_csv(edges_with_pos_neg_as_label,
                            sep=',')  # file is not expected to have index
        else:

            # selected_shuffled_pos_neg_edges_with_label = self.get_selected_pos_neg_edges_for_link_prediction()
            #
            # selected_shuffled_pos_neg_edges, selected_shuffled_pos_neg_edges_label = selected_shuffled_pos_neg_edges_with_label[
            #                                                                          :,
            #                                                                          :-1], selected_shuffled_pos_neg_edges_with_label[
            #                                                                                :,
            #                                                                                -1]

            return self.new_create_train_test_set(
                                              split,
                                              stratify,
                                              split_by_node=split_by_node,
                                              nodes_to_be_splitted=nodes_to_be_splitted,
                                              splitted_edges_file_path=splitted_edges_file_path)

            # selected_shuffled_pos_neg_edges_with_label = self.get_selected_pos_neg_edges_for_link_prediction()
            #
            # selected_shuffled_pos_neg_edges, selected_shuffled_pos_neg_edges_label = selected_shuffled_pos_neg_edges_with_label[
            #                                                                          :,
            #                                                                          :-1], selected_shuffled_pos_neg_edges_with_label[
            #                                                                                :,
            #                                                                                -1]
            #
            # self.train_set, self.test_set = None, None
            # return self.create_train_test_set(selected_shuffled_pos_neg_edges,
            #                                   selected_shuffled_pos_neg_edges_label,
            #                                   split,
            #                                   selected_shuffled_pos_neg_edges_with_label,
            #                                   stratify,
            #                                   split_by_node=split_by_node,
            #                                   nodes_to_be_splitted=nodes_to_be_splitted,
            #                                   splitted_edges_file_path=splitted_edges_file_path)
            #

            # TODO paragraph of code below is moved to sef.get_selected_pos_neg_edges_for_link_prediction() and self.create_train_test_set()
            ## return of self.create_train_test_set is expect to be the return from the paragraph below
            # # note: I use combination to produce non_directional edges
            # existed_gene_disease_edges, existed_gene_disease_edges_label, non_existed_gene_disease_edges, non_existed_gene_disease_edges_label = self.get_existed_and_non_existed_gene_disease_edges()
            #
            # pos_edges_np = np.array(existed_gene_disease_edges)
            # neg_edges_np = np.array(non_existed_gene_disease_edges)
            #
            # # spliiting pos
            # (pos_x_train, pos_y_train), (
            #     pos_x_test, pos_y_test) = self.run_split_train_test(
            #     pos_edges_np, np.ones(pos_edges_np.shape[0]), split,
            #     stratify)
            #
            # # splitting neg
            # # TODO for neg, just select neg of size pos_x_train + pos_x_test ==> pass it to self.split
            # # train_test_size = pos_x_train.shape[0] + pos_x_test.shape[0]
            # train_test_size = pos_edges_np.shape[0]
            # selected_neg_edges_ind = np.random.choice(neg_edges_np.shape[0],
            #                                           train_test_size,
            #                                           replace=False)
            # selected_neg_edges = neg_edges_np[selected_neg_edges_ind, :]
            #
            # (neg_x_train, neg_y_train), (
            #     neg_x_test, neg_y_test) = self.run_split_train_test(
            #     selected_neg_edges,
            #     np.zeros_like(np.array(selected_neg_edges_ind)), split,
            #     stratify)
            #
            # # TODO x and y have to be shuffled the same way
            # def concat_and_shuffle(pos_x, pos_y, neg_x, neg_y):
            #     x = np.concatenate([pos_x, neg_x])
            #     y = np.concatenate([pos_y, neg_y])
            #     x_y = np.array([[*i, j] for i, j in zip(x, y)])
            #
            #     np.random.shuffle(x_y)
            #     shuffled_x_np = x_y[:, :2]
            #     shuffled_y_np = x_y[:, -1]
            #     return shuffled_x_np, shuffled_y_np
            #
            # # TODO shuffled_y_train and shuffle_y_test only second half are shuffled (wtf?)
            # shuffled_x_train_np, shuffled_y_train_np = concat_and_shuffle(
            #     pos_x_train, pos_y_train, neg_x_train, neg_y_train)
            # shuffled_x_test_np, shuffled_y_test_np = concat_and_shuffle(
            #     pos_x_test,
            #     pos_y_test,
            #     neg_x_test,
            #     neg_y_test)
            #
            # self.train_set, self.test_set = np.concatenate(
            #     [shuffled_x_train_np,
            #      shuffled_y_train_np[:, np.newaxis]], axis=1), np.concatenate([
            #     shuffled_x_test_np,
            #     shuffled_y_test_np[:, np.newaxis]], axis=1)
            #
            # # create directory if not alreayd exist
            # if not os.path.exists(splitted_edges_file_path):
            #     os.makedirs(splitted_edges_file_path)
            #
            # splitted_edges_dir = '\\'.join(
            #     splitted_edges_file_path.split('\\')[:-1])
            # if not os.path.exists(splitted_edges_dir):
            #     os.makedirs(splitted_edges_dir)
            #
            # import pickle
            #
            # # favorite_color = { "lion": "yellow", "kitty": "red" }
            # splitted_edges_dict = {'edges_train_set': self.train_set,
            #                        'edges_test_set': self.test_set}
            #
            # pickle.dump(splitted_edges_dict,
            #             open(splitted_edges_file_path, "wb"))
            #
            # return self.train_set, self.test_set

    def is_train_test_split_set(self):
        return self.train_set is not None and self.test_set is not None, "check whether or not train_set and test_set has already been set"

    def is_cross_validation_split_set(self):
        return len(
            self.stratified_k_fold) > 0, "check whether or not stratifed_k_fold has already been set"

    def split_cross_validation(self, data, k_fold, stratify=None, task=None,
                               split_by_node = None,
                               # reset_cross_validation_split=None,
                               splitted_edges_dir=None,
                               ):
        # assert isinstance(reset_cross_validation_split,
        #                   bool), " reset_train_test_split must be boolean"
        assert task is not None, "task must be specified to avoid ambiguity"
        assert isinstance(stratify, bool), " stratify must be boolean"

        # # TODO to be deleted after finished getting result from k_fold = 101
        # splitted_edges_dir = r'C:\Users\Anak\PycharmProjects\recreate_gene_disease\Data\processed\LinkPrediction\GeneDiseaseProject\copd\PhenotypeGeneDisease\PGDP\Node2Vec\UnweightedEdges\NoAddedEdges\SplitByNode\KFold=101\DecendingOrder\\'

        splitted_edges_file = splitted_edges_dir + 'splitted_edges.bin'

        if len(self.stratified_k_fold) == 0:
            if path.exists(splitted_edges_dir) and path.exists(
                    splitted_edges_file):
                # TODO read the splitted_nodes.bin and return self.stratified_k_fold
                import pickle

                # splitted_edges_dict is expect to have the following use case
                ## eg. spliited_edges_dict[# of fold] = {train_set: numpy of shape (# of edges, 2) , test_set: numpy of shape (# of edges, 2)
                splitted_edges_dict = pickle.load(
                    open(splitted_edges_file, "rb"))

                for key, edges_train_test_set_dict in splitted_edges_dict.items():
                    edges_train_set = edges_train_test_set_dict[
                        'edges_train_set']
                    edges_test_set = edges_train_test_set_dict[
                        'edges_test_set']

                    if key not in self.stratified_k_fold:
                        # self.stratified_k_fold[key] = {'train_set': edges_train_set, 'test_set': edges_test_set}
                        self.stratified_k_fold.append(
                            {'train_set': edges_train_set,
                             'test_set': edges_test_set})
                    else:
                        raise ValueError(
                            f'key is already existed in self.stratified_k_fol  => There are dubplicate keys in {splitted_edges_file}')
                print()

            else:

                # if reset_cross_validation_split:

                if task == 'node_classification':
                    return self.run_split_cross_validation_for_node_classification(
                        k_fold, stratify)
                elif task == 'link_prediction':
                    # return self.run_split_cross_validation_for_link_prediction(
                    #     data,
                    #     k_fold, stratify, split_by_node=True,
                    #     nodes_to_be_splitted=data.diseases_np,
                    #     splitted_edges_file_path=splitted_edges_file
                    # )
                    return self.run_split_cross_validation_for_link_prediction(
                        data,
                        k_fold, stratify, split_by_node=split_by_node,
                        nodes_to_be_splitted=data.diseases_np,
                        splitted_edges_file_path=splitted_edges_file
                    )
                else:
                    raise ValueError(
                        'task only accept node_classification or link_prediction as its value')

                # else:
                #     if self.stratified_k_fold is None:
                #         raise ValueError(
                #             "please run split_cross_validation with reset_cross_validation_split = True.it will set self.stratified_k_fold value")
                #
        print()
        return self.stratified_k_fold

    def split_train_test(self, split, stratify=None, task=None,
                         splitted_edges_dir=None,
                         split_by_node=None,
                         is_input_numpy=True):
        # reset_train_test_split=None):
        """

        @param x: type = np; shape = (-1,1)
        @param y: type = np: shape = (-1,1)
        @param split:
        @return:
        """
        # assert isinstance(reset_train_test_split,
        #                   bool), " reset_train_test_split must be boolean"
        assert task is not None, "task must be specified to avoid ambiguity"
        assert isinstance(stratify, bool), " stratify must be boolean"


        splitted_edges_file = splitted_edges_dir + 'splitted_edges.bin' if splitted_edges_dir is not None else None

        # Note: Potential bu-g might exist here for (train_test vs k_Fold)
        if split is not None:
            if task == 'node_classification':
                return self.run_split_train_test_for_node_classification(
                    split,
                    stratify,
                    is_input_numpy=is_input_numpy)

            elif task == 'link_prediction':

                # return self.run_split_train_test_for_link_prediction(split,
                #                                                      stratify,
                #                                                      split_by_node=True,
                #                                                      nodes_to_be_splitted=self.diseases_np,
                #                                                      splitted_edges_file_path=splitted_edges_file,
                #                                                      )
                return self.run_split_train_test_for_link_prediction(split,
                                                                     stratify,
                                                                     split_by_node=split_by_node,
                                                                     nodes_to_be_splitted=self.diseases_np,
                                                                     splitted_edges_file_path=splitted_edges_file,
                                                                     )
            else:
                raise ValueError(
                    'task only accept node_classification or link_prediction as its value')
        else:
            if self.train_set is None and self.test_set is None:
                if path.exists(splitted_edges_dir) and path.exists(
                        splitted_edges_file):
                    # TODO read the splitted_nodes.bin and return self.stratified_k_fold
                    import pickle

                    # splitted_edges_dict is expect to have the following use case
                    ## eg. spliited_edges_dict = {train_set: numpy of shape (# of edges, 2) , test_set: numpy of shape (# of edges, 2)
                    edges_train_test_set_dict = pickle.load(
                        open(splitted_edges_file, "rb"))

                    # for key, edges_train_test_set_dict in splitted_edges_dict.items():
                    self.train_set = edges_train_test_set_dict[
                        'edges_train_set']
                    self.test_set = edges_train_test_set_dict[
                        'edges_test_set']

                else:
                    if task == 'node_classification':
                        return self.run_split_train_test_for_node_classification(
                            split,
                            stratify,
                        is_input_numpy=is_input_numpy)

                    elif task == 'link_prediction':

                        # return self.run_split_train_test_for_link_prediction(split,
                        #                                                      stratify,
                        #                                                      split_by_node=True,
                        #                                                      nodes_to_be_splitted=self.diseases_np,
                        #                                                      splitted_edges_file_path=splitted_edges_file,
                        #                                                      )
                        return self.run_split_train_test_for_link_prediction(split,
                                                                             stratify,
                                                                             split_by_node=split_by_node,
                                                                             nodes_to_be_splitted=self.diseases_np,
                                                                             splitted_edges_file_path=splitted_edges_file,
                                                                             )
                    else:
                        raise ValueError(
                            'task only accept node_classification or link_prediction as its value')
            else:
                if self.train_set is None and self.test_set is None:
                    raise ValueError(
                        'please run split_train_test with reset_Train_test_split=True. if you hvae not run it yet ')

        return self.train_set, self.test_set

    def add_weighted_qualified_edges_to_graph(self, weighted_qualified_edges):
        """

        @param weighted_qualified_edges: type = np; shape=(-1,3) where last column = weight
        @return:
        """
        self.G.add_weighted_edges_from(weighted_qualified_edges)


    def is_in_original_diseases(self, diseases_to_be_validated):
        """validate that list of diseases are in the originla diseases"""
        # TODO is self.diseases_np numpy array that contains all original diseases
        return np.all(
            [True if i in self.diseases_np.flatten() else False for i in
             diseases_to_be_validated])

    def is_isomorphic_to_original_GeneDisease_Graph(self, Graph_to_be_tested):
        """
            references url : https://stackoverflow.com/questions/17428516/test-graph-equality-in-networkx/26807248#26807248

            check whether graph is the unmodified original gene2disease edges

            note:
                > the function does not considiered edges attributed
        @param Graph_to_be_tested: type = nx.Graph();
        @return: boolean
        """

        # check if edges between two graph are the same
        return nx.is_isomorphic(self.original_GeneDisease_edges,
                                Graph_to_be_tested)

    def is_disease2disease_edges_added_to_graph(self, outside_graph=None,
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
        # assert is_type_float_vectorize(np.array(list(graph.edges.self('weight')))[:, 2]), "For consistency, all value of weighted graph of graph must be of type float "

        # check if weight type is str or number

        is_weighted_one_func = lambda x: x != 1
        # is_weighted_one_str_func = lambda x: int(x) != 1

        # is_weighted_one_vectorize = np.vectorize(is_weighted_one_func)

        # TODO fix: AttributeError: 'tuple' object has no attribute 'edges'
        graph_edges_with_weight = np.array(list(graph.edges.data('weight')))
        # graph_edges_with_weight[:, 2] = graph_edges_with_weight[:,2].astype(float)
        weight_np = graph_edges_with_weight[:, 2].astype(float)

        has_weighted_edges = np.vectorize(is_weighted_one_func)(weight_np).any()
        # has_str_weighted_edges = np.vectorize(is_weighted_one_str_func)(weight_np).any()
        # is_weighted = has_weighted_edges or has_str_weighted_edges

        return has_weighted_edges



def get_diseases_that_are_overlapped_between_GPSim_and_GeneDisease_graph():
    """

    @return: overlapped_disease_np; type = np ;shape = (-1,1)
    """
    overlapped_disease_file_path = r'C:\Users\Anak\PycharmProjects\recreate_gene_disease\Data\raw\GPSim\GPsim_overlapped_diseases.txt'
    overlapped_disease_pd = pd.read_csv(overlapped_disease_file_path,
                                        header=None)

    return overlapped_disease_pd.to_numpy()


if __name__ == '__main__':
    # =====================
    # ==genedisease dataset
    # =====================

    # -- datasets
    gene_disease_root = r'c:\users\anak\pycharmprojects\recreate_gene_disease\data'  # indicate where file should be stored
    gene_disease_data = GeneDiseaseGeometricDataset(gene_disease_root)

    # convert_node2class_id = np.vectorize(
    #     lambda x: gene_disease_data.node2class_id[x])
    # gene_disease_class_id = convert_node2class_id(
    #     gene_disease_data.diseases_np)
    # gene_disease_train_set, gene_disease_test_set = gene_disease_data.split_train_test(
    #     gene_disease_data.diseases_np, gene_disease_class_id, 0.6)
    from arg_parser import args

    gene_disease_train_set, gene_disease_test_set = gene_disease_data.split_train_test(
        0.6, stratify=True, task=args.task)
