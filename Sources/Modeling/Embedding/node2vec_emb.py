import os
from os import path

from node2vec import Node2Vec

from Sources.Preparation.Data import GeneDiseaseGeometricDataset
from Sources.Preparation.Features.build_features import \
    get_data_without_using_emb_as_feat
# from Sources.Preprocessing.preprocessing import add_disease2disease_edges
from Sources.Preprocessing.preprocessing import select_emb_save_path
from arg_parser import args
from Sources.Preprocessing.preprocessing import get_saved_file_name_for_emb

def run_node2vec(data=None, G=None, embedding_model_file_path=None, add_qualified_edges = None,
                 use_weighted_edges=None,
                 edges_percent = None,
                 edges_number = None,
                 dim=64, walk_len=30, num_walks=200,
                window=10,
                 added_edges_percent_of = None):
    """

    @param data:
    @param G:
    @param embedding_model_file_path:
    @param add_qualified_edges: type = str ;  it specified name of edges adding strategy being used
    @param use_weighted_edges:
    @param dim:
    @param walk_len:
    @param num_walks:
    @param window:
    @return:
    """

    # # catching not yet implemented error
    # if added_edges_percent_of is not None:
    #     raise ValueError("not yet implemented")

    assert data is not None, "dataset Class must be explitcitly specified to avoid ambiguity"
    assert G is not None, "Graph of type nx.Graph() must be explicitly specified to avoide ambiguity"
    assert embedding_model_file_path is not None, "please specifiied embedding_model_file_path  to save emb_file "
    assert add_qualified_edges is not None, "add_qualified_edges must be specified to avoid ambiguity"
    assert use_weighted_edges is not None, "use_weighted_edges must be explicitly specified to avoide ambiguity"

    file_name = get_saved_file_name_for_emb(add_qualified_edges, edges_percent, edges_number, dim, walk_len, num_walks, window)
    # file_name = f'{add_qualified_edges}={}_dim{dim}_walk_len{walk_len}_num_walks{num_walks}_window{window}.txt'
    save_path = embedding_model_file_path + file_name

    print("save emb_file to " + save_path)

    # check that no files with the same name existed within these folder
    assert not path.exists(
        save_path), "emb_file already exist, Please check if you argument is correct"

    # create dir if not alreayd existed
    if not os.path.exists(embedding_model_file_path):
        os.makedirs(embedding_model_file_path)

    if use_weighted_edges:
        assert data.is_graph_edges_weighted(G,
                                            use_outside_graph=True), "use_weighted_edges is True, but graph contains no weighted edges (defined as edges with weight != 1)"
    else:
        # TODO fix error here>>
        assert not data.is_graph_edges_weighted(G,
                                                use_outside_graph=True), "use_weighted_edges is True, but graph contains no weighted edges (defined as edges with weight != 1)"

    node2vec = Node2Vec(G, weight_key='weight', dimensions=dim,
                        walk_length=walk_len, num_walks=num_walks, workers=4)
    model = node2vec.fit(window=window, min_count=1, batch_words=4)

    # save model to
    model.wv.save_word2vec_format(save_path)


if __name__ == '__main__':
    # =====================
    # == Datsets
    # =====================
    ## GeneDisease
    GeneDisease_root = r'c:\users\anak\pycharmprojects\recreate_gene_disease\data'  # indicate where file should be stored
    data = GeneDiseaseGeometricDataset(GeneDisease_root)

    # # split into train_set and test_set
    # convert_disease2class_id = np.vectorize(lambda x: data.disease2class_id_dict[x])
    # disease_class = convert_disease2class_id(data.diseases_np)
    # train_set, test_set = data.split_train_test(data.diseases_np,
    #                                             disease_class, 0.4)
    #
    # (x_train, y_train), (x_test, y_test) = train_set, test_set

    # =====================
    # == Preprocessing
    # =====================
    graph_with_added_qualified_edges, data_with_features = get_data_without_using_emb_as_feat(
        data=data,
        add_qualified_edges=args.add_qualified_edges,
        dataset=args.dataset,
        use_weighted_edges=args.use_weighted_edges,
        normalized_weighted_edges=args.normalized_weighted_edges,
        return_graph_and_data_with_features = True,
        use_saved_emb_file=args.use_saved_emb_file,
        edges_number = args.edges_number,
        edges_percent= args.edges_percent,
        added_edges_percent_of= args.added_edges_percent_of)

    # TODO add path for added_edges_percent_of GeneDisease and GPsim
    emb_type = "node2vec"
    embedding_model_file_path = select_emb_save_path(emb_type=emb_type,
                                                     add_qualified_edges=args.add_qualified_edges,
                                                     dataset=args.dataset,
                                                     use_weighted_edges=args.use_weighted_edges,
                                                     edges_percent = args.edges_percent,
                                                     edges_number = args.edges_number,
                                                     added_edges_percent_of = args.added_edges_percent_of)

    # =====================
    # == run embedding (it should save result in appropriate folder within Data
    # =====================
    run_node2vec(data=data, G=graph_with_added_qualified_edges,
                 embedding_model_file_path=embedding_model_file_path,
                 edges_percent= args.edges_percent,
                 edges_number = args.edges_number,
                 use_weighted_edges=args.use_weighted_edges,
                 add_qualified_edges = args.add_qualified_edges,
                 added_edges_percent_of=args.added_edges_percent_of
                 )


