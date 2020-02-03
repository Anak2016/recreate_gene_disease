import numpy as np
import pandas as pd
from sklearn import preprocessing
from Sources.Preparation.Features.get_qualified_edges import get_all_GeneDisease_unweighted_disease2disease_qualified_edges
from Sources.Preparation.Features.get_qualified_edges import get_GPSim_disease2disease_qualified_edges
# from Sources.Preparation.Features.test import get_all_GeneDisease_unweighted_disease2disease_qualified_edges
# from Sources.Preparation.Features.test import get_GPSim_disease2disease_qualified_edges


def onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def select_emb_save_path(emb_type=None, add_qualified_edges=None, dataset=None,
                         use_weighted_edges=None, edges_number=None,
                         edges_percent=None,
                         added_edges_percent_of=None):
    """

    @param add_qualified_edges: type = Boolean eg True or False:
    @param dataset: type = String eg. GPSim, GeneDisease
    @return: valid path to saved emb_file
    """

    # if added_edges_percent_of is not None:
    #     raise ValueError("not yet implemented ")

    assert emb_type is not None, "emb_type mst be specified to avoide ambiguity "
    # assert add_qualified_edges is not None, "added_qualified_edges must be specified to avoid ambiguity"
    assert dataset is not None, "dataset must be specified to avoid ambiguity"
    assert use_weighted_edges is not None, "use_weighted_edges must be specified to avoid ambiguity "

    if add_qualified_edges is not None:
        assert dataset != "no", "NoAddedEdges Folder can be accessed only when add_qualified_edges is False"
        if edges_percent is not None:
            # note that input of added_edges_percent_of must be valid folder naming (look at notion/principle/ for more information of folder naming)
            assert added_edges_percent_of in ['GeneDisease', 'GPSim'], ""
            number_of_added_edges = f"EdgesPercent\\{added_edges_percent_of}\\"
        elif edges_number is not None:
            number_of_added_edges = f"EdgesNumber\\"
        else:
            raise ValueError(
                " when add_qualified_edges is true, only edges_percent and edges_nuber id acceptable")

    save_path_base = f"C:\\Users\\Anak\\PycharmProjects\\recreate_gene_disease\\Data\\processed\\"

    if dataset == 'GeneDisease' or dataset == 'no':
        dataset_processed_dir = f"GeneDiseaseProject\\copd\\"
    elif dataset == "GPSim":
        dataset_processed_dir = f"GPSim\\"
    else:
        raise ValueError("please specified dataset that existed or available ")

    if emb_type == 'node2vec':
        emb_type_dir = "Node2Vec\\"
    else:
        raise ValueError("please specified em_type correctly")

    if use_weighted_edges:
        assert dataset != 'no', "not yet implmented: dataset == 'no' and use_weighted_edges is True implies adding weighted gene2disease edges which is not yet implmented "
        weighted_status_dir = "WeightedEdges\\"
    else:
        weighted_status_dir = "UnweightedEdges\\"

    if add_qualified_edges is not None:
        add_edges_status = "AddedEdges\\"
    else:
        add_edges_status = "NoAddedEdges\\"

    embedding_model_file_path = save_path_base + dataset_processed_dir + emb_type_dir + weighted_status_dir + add_edges_status + number_of_added_edges

    return embedding_model_file_path


def split_train_test(data, split):
    # split into train_set and test_set
    convert_disease2class_id = np.vectorize(
        lambda x: data.disease2class_id_dict[x])
    disease_class = convert_disease2class_id(data.diseases_np)

    train_set, test_set = data.split_train_test(data.diseases_np,
                                                disease_class, split,
                                                stratify=disease_class)
    (x_train, y_train), (x_test, y_test) = train_set, test_set
    return train_set, test_set


def apply_normalization(data_with_features, use_saved_emb_file,
                        normalized_weighted_edges):
    """

    @param data_with_features: type = pd.DataFrame
    @param normalized_weighted_edges:
    @return:
    """

    if normalized_weighted_edges:
        assert (
            not use_saved_emb_file), "Cannot normalized embedding data: used_saved_emb_file  = True means data is embededing which implied that it is already normalied."
    # else:
    #     assert ( use_saved_emb_file) , "Cannot normalized embedding data: used_saved_emb_file  = True means data is embededing which implied that it is already normalied."

    if normalized_weighted_edges:
        # x = df.values  # returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(data_with_features)
        normalized_data_with_features = pd.DataFrame(x_scaled,
                                                     columns=data_with_features.columns,
                                                     index=data_with_features.index)

        return normalized_data_with_features
        # return  x_scaled
    else:
        return data_with_features  # not apply normalization


def multiply_constant_multiplier_to_weighted_disease2disease_edges(
        constant_multiplier=10,
        weighted_disease2disease_edges=None):
    """

    @param constant_multiplier: type = float/int
    @param weighted_disease2disease_edges: type = np; shape = (# disease2disease edges, 3) where the last coloumns = weight
    @return:
    """

    assert weighted_disease2disease_edges is not None, "weighted_disease2disease_edges must be specified to avoid ambiguity"

    # multiply weighted diseease2disease edges with constant multiplier
    multiplied_weighted_disease2disease_edges = constant_multiplier * weighted_disease2disease_edges[
                                                                      :,
                                                                      2].astype(
        float)

    # clips max value to 1
    ## note: min should not be specified in clippping to detect weighted edges value error at validation step
    multiplied_weighted_disease2disease_edges = multiplied_weighted_disease2disease_edges.clip(
        max=1)  # output is expected to be converted to str

    # validate that min value = 0
    assert multiplied_weighted_disease2disease_edges.astype(
        float).min() > 0, "no edges should indicate the least weighted value of 0 (no relationship) "

    weighted_disease2disease_edges[:,
    2] = multiplied_weighted_disease2disease_edges

    return weighted_disease2disease_edges


def get_number_of_added_edges(data, add_qualified_edges, edges_percent,
                              edges_number, added_edges_percent_of):
    if edges_percent is not None:
        if added_edges_percent_of == 'GeneDisease':
            all_qualified_GeneDisease_disease2disease_edges_df = get_all_GeneDisease_unweighted_disease2disease_qualified_edges(data)

            # qualified_edges_df = get_all_GeneDisease_qualified_edges(data,
            #                                     data.G,
            #                                     data.diseases_np)

            number_of_qualified_edges = all_qualified_GeneDisease_disease2disease_edges_df.shape[0]
            number_of_added_edges = int(edges_percent * number_of_qualified_edges)
            return number_of_added_edges

            # raise ValueError(
            #     " working on creating added_edges_percent_of == 'GeneDisease' rightnow!")
        else:
            all_qualified_GPSim_weighted_disease2disease_edges_pd = get_GPSim_disease2disease_qualified_edges(data, False) # what is the expected output of this
            number_of_qualfied_edges = all_qualified_GPSim_weighted_disease2disease_edges_pd.shape[0]
            number_of_added_edges = int(edges_percent * number_of_qualfied_edges)
            return number_of_added_edges
            # raise ValueError(
            #     "not yet implemented: when edges_percent is not None, I must also specified 'percentage of which set of qualified edges; (this question has not yet asnwer)  ")
        # return int(edges_without_weight * edges_percent)
    if edges_number is not None:
        return edges_number


def get_saved_file_name_for_emb(add_qualified_edges, edges_percent,
                                edges_number, dim, walk_len, num_walks,
                                window):
    """This function is create in case that file name getting increasingly more complicated as development process continues"""

    # assert (edges_number is not None) or (edges_percent is not None), ""
    if add_qualified_edges is not None:
        if edges_number is not None:
            return f'{add_qualified_edges}={edges_number}_dim{dim}_walk_len{walk_len}_num_walks{num_walks}_window{window}.txt'
        elif edges_percent is not None:
            return f'{add_qualified_edges}={edges_percent}_dim{dim}_walk_len{walk_len}_num_walks{num_walks}_window{window}.txt'
        else:
            raise ValueError(
                "only edges_number and edges_pecent is acceptable as subparser for add qulified_edges")
    else:
        return f'dim{dim}_walk_len{walk_len}_num_walks{num_walks}_window{window}.txt'

# def get_all_GPsim_weighted_disease2disease_qualified_edges():
#     # assert isinstance( with_weight , bool), "weith_weight is expected to be type bool"
#     GPSim_cui_qualified_edges_files_path = r'C:\Users\Anak\PycharmProjects\recreate_gene_disease\Data\raw\GPSim\Edges\cui_edges_weight.csv'
#     GPSim_cui_qualified_edges_pd = pd.read_csv(GPSim_cui_qualified_edges_files_path, sep = ',')
#     return GPSim_cui_qualified_edges_pd
#     # if with_weight:
#     #    pass


# def get_all_GeneDisease_unweighted_disease2disease_qualified_edges(data):
#
#     nodes = data.diseases_np
#
#     all_disease2disease_edges = list(combinations(nodes, 2))
#
#     all_disease2disease_qualified_edges = np.array(
#         [edge for edge in all_disease2disease_edges if
#          len(list(nx.common_neighbors(data.G, edge[0], edge[1]))) > 0])
#
#     G = nx.Graph()
#     G.add_edges_from(all_disease2disease_qualified_edges, weight=1)
#     all_qualified_disease2disease_edges_df = nx.to_pandas_edgelist(G)
#
#     return all_qualified_disease2disease_edges_df

# def get_all_disease2disease_GeneDisease_qualified_edges(G, nodes):
# 
#     all_disease2disease_edges = list(combinations(nodes, 2))
#     disease2disease_qualified_edges = np.array(
#         [edge for edge in all_disease2disease_edges if
#          len(list(nx.common_neighbors(G, edge[0], edge[1]))) > 0])
# 
#     return disease2disease_qualified_edges
# 
# def get_all_GeneDisease_qualified_edges(data, G, nodes):
# 
#     all_disease2disease_qualified_edges = get_all_disease2disease_GeneDisease_qualified_edges(G,nodes)
# 
#     G = nx.Graph()
#     G.add_edges_from(all_disease2disease_qualified_edges, weight=1)
#     all_qualified_edges_df = nx.to_pandas_edgelist(G)
# 
#     # # TODO below is the code that produces the the following
#     # # : processed\GeneDiseaseProject\copd\Node2Vec\UnweightedEdges\AddedEdges\EdgesNumber
#     # # : processed\GeneDiseaseProject\copd\Node2Vec\UnweightedEdges\NoAddedEdges\EdgesNumber
#     # all_disease2disease_edges = list(combinations(nodes, 2))
#     # all_disease2disease_qualified_edges = np.array(
#     #     [edge for edge in all_disease2disease_edges if
#     #      len(list(nx.common_neighbors(G, edge[0], edge[1]))) > 0])
#     # G = nx.Graph()
#     # G.add_edges_from(all_disease2disease_edges, weight=1) # I added "all_disease2disease_edges" instead of "all_disease2disease_qualified_edges"
#     # all_qualified_edges_df = nx.to_pandas_edgelist(G)
# 
#     assert not (data.is_graph_edges_weighted(G,
#                                              use_outside_graph=True)), "use_weighted_edges is True, but all edges in graph has weight == 1"
#     return all_qualified_edges_df
