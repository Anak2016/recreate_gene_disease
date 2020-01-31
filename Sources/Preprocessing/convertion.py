import pandas as pd

class Converter:
    def __init__(self, ):
        # Read disease_mapping (cui and doid) => create another preprocessing function called (convert_cui2doid) and (convert_doid2cui)
        self._disease_mapping_df = None
        self._disease_mappings_file_path = r'C:\Users\Anak\PycharmProjects\recreate_gene_disease\Data\raw\DisGeNET\mapping\disease_mappings.tsv'
        self._disease_mapping_df = pd.read_csv(self._disease_mappings_file_path, sep='|')

        # only select row whose vabaulary == DO
        self._disease_mapping_df = self._disease_mapping_df[self._disease_mapping_df['vocabulary'].isin(['DO'])]
        self._disease_mapping_df['doid'] = self._disease_mapping_df['vocabulary'] + 'ID:' + self._disease_mapping_df['code']


        self._cui2class_id_dict = None

        self._file = r'C:\Users\Anak\PycharmProjects\recreate_gene_disease\Data\raw\GeneDiseaseProject\COPD\Nodes\copd_label_content07_14_19_46.txt'
        self._cui_and_class_id = pd.read_csv(self._file, sep='\t', header=None)
        self._cui2class_id_dict = self._cui_and_class_id.to_dict()
        self._cui2class_id_dict = {cui: cls for cui, cls in
                                   zip(self._cui2class_id_dict[0].values(), self._cui2class_id_dict[1].values())}

    @property
    def disease_mapping_df(self):
        return self._disease_mapping_df

    @property
    def cui2class_id_dict(self):
        return self._cui2class_id_dict

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
