import pandas as pd
import numpy as np
import networkx as nx

class Converter:
    def __init__(self, GeneDisease_data):
        # Read disease_mapping (cui and doid) => create another preprocessing function called (convert_cui2doid) and (convert_doid2cui)
        self._disease_mapping_df = None
        self._disease_mappings_file_path = r'C:\Users\Anak\PycharmProjects\recreate_gene_disease\Data\raw\DisGeNET\mapping\disease_mappings.tsv'
        self._disease_mapping_df = pd.read_csv(self._disease_mappings_file_path, sep='|')

        # only select row whose vabaulary == DO
        self._disease_mapping_df = self._disease_mapping_df[self._disease_mapping_df['vocabulary'].isin(['DO'])]
        self._disease_mapping_df['doid'] = self._disease_mapping_df['vocabulary'] + 'ID:' + self._disease_mapping_df['code']


        # cui2class_id_dict
        self._cui2class_id_dict = None

        self._file = r'C:\Users\Anak\PycharmProjects\recreate_gene_disease\Data\raw\GeneDiseaseProject\COPD\Nodes\copd_label_content07_14_19_46.txt'
        self._cui_and_class_id = pd.read_csv(self._file, sep='\t', header=None)
        self._cui2class_id_dict = self._cui_and_class_id.to_dict()
        self._cui2class_id_dict = {cui: cls for cui, cls in
                                   zip(self._cui2class_id_dict[0].values(), self._cui2class_id_dict[1].values())}

        #     disease_mapping_df_for_orignal_disease_101 = self.disease_mapping_df[
        #         self.disease_mapping_df['diseaseId'].isin(GeneDisease_data.diseases_np)]
        #     self._original_cui2doid_dict = {i: j for i, j in
        #                      zip(disease_mapping_df_for_orignal_disease_101['diseaseId'],
        #                          disease_mapping_df_for_orignal_disease_101[
        #                              'doid'])}
        # @property
        # def original_cui2doid_dict(self):
        #     return self._original_cui2doid_dict
        # @property
        # def original_doid2cui_dict(self):
        #     return {j:i for i,j in self._original_cui2doid_dict.items()}

        # create_disease_mapping_for_original_disease_101_dict
        self._disease_mapping_for_orignal_disease_101_dict = self._create_disease_mapping_for_orignal_disease_101_dict()

    def _create_disease_mapping_for_orignal_disease_101_dict(self):
        # get orginal 101 disease from GeneDisease
        original_disease_101 = np.array(
            list(self.cui2class_id_dict.keys()))

        # Create cui2doid_dict and doid2cui_dict
        disease_mapping_for_orignal_disease_101_df = \
            self.disease_mapping_df[
                self.disease_mapping_df['diseaseId'].isin(
                    original_disease_101)]

        # TODO currently I only use 130 diseases; use 139 doid disease;
        ## > fix all code that convert from cui2doid (make sure all of them use 139 doid disease mapping)

        # x = {}
        # for i, j in zip(disease_mapping_for_orignal_disease_101_df['doid'],
        #                 disease_mapping_for_orignal_disease_101_df[
        #                     'diseaseId']):
        #     x.setdefault(i, []).append(j)
        # print(len(list(x.keys())))
        # print(sum([len(i) for i in x.values()]))

        disease_mapping_df_for_orignal_disease_101_dict = {i: j for i, j in
                                                           zip(
                                                               disease_mapping_for_orignal_disease_101_df[
                                                                   'doid'],
                                                               disease_mapping_for_orignal_disease_101_df[
                                                                   'diseaseId'])}

        return disease_mapping_df_for_orignal_disease_101_dict

    def original_cui2doid_mapping(self, cui_diseases):
        cui2doid_dict = {j: i for i, j in
                         self._disease_mapping_for_orignal_disease_101_dict.items()}
        return np.vectorize(
            lambda x: cui2doid_dict[x])(cui_diseases)

    def original_doid2cui_mapping(self, doid_diseases):
        return np.vectorize(
            lambda x: self._disease_mapping_for_orignal_disease_101_dict[x])(doid_diseases)

    @property
    def disease_mapping_df(self):
        return self._disease_mapping_df

    @property
    def cui2class_id_dict(self):
        return self._cui2class_id_dict

# def create_grpah_from_emb_file(use_saved_emb_file,emb_file):
#     """
#
#     @param use_saved_emb_file: it should be passed in, just to make sure that the function is placed in the correct location
#     @param emb_file:
#     @return:
#     """
#     assert use_saved_emb_file, "to use create_graph_from_emb_file(), use_saved_emb_file has to be passed in "
#     graph_from_emb_file = nx.Graph()
#
#     # emb_pd = pd.read_csv(emb_file)
#     # TODO  how to get edges from emd_file
#
#
#     return

def convert2binary_classifier_prediction(multiclass_clf):
    """

    @param multiclass_clf:
    @return:
    """

    pass

# def convert_numpy_to_networkx_graph(x):
#     """
#
#     @param x: type numpy; shape = (-1,2)
#     @return:
#     """
#     g = nx.graph()
#     g.add_edges_from(x)
#     return G
