import os
from os import path

from node2vec import Node2Vec

from Sources.Preparation.Data import GeneDiseaseGeometricDataset
from Sources.Preparation.Features.build_features import \
    get_data_without_using_emb_as_feat
# from Sources.Preprocessing.preprocessing import add_disease2disease_edges
from Sources.Preprocessing.apply_preprocessing import select_emb_save_path
from arg_parser import args
from arg_parser import apply_parser_constraint
from arg_parser import run_args_conditions
from Sources.Preprocessing.apply_preprocessing import get_saved_file_name_for_emb
from Models.train_model import run_train_model

def run_node2vec_emb(data=None, G=None, embedding_model_file_path=None, add_qualified_edges = None,
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
        assert not data.is_graph_edges_weighted(G,
                                                use_outside_graph=True), "use_weighted_edges is Flase, but graph contains weighted edges (defined as edges with weight != 1)"

    node2vec = Node2Vec(G, weight_key='weight', dimensions=dim,
                        walk_length=walk_len, num_walks=num_walks, workers=4)
    model = node2vec.fit(window=window, min_count=1, batch_words=4)

    # save model to
    model.wv.save_word2vec_format(save_path)

def run_node2vec():
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
        added_edges_percent_of= args.added_edges_percent_of,
        use_shared_phenotype_edges=args.use_shared_phenotype_edges,
        use_shared_gene_edges = args.use_shared_gene_edges,
        use_shared_gene_and_phenotype_edges=args.use_shared_gene_and_phenotype_edges,
        use_shared_gene_but_not_phenotype_edges=args.use_shared_gene_but_not_phenotype_edges,
        use_shared_phenotype_but_not_gene_edges=args.use_shared_phenotype_but_not_gene_edges)

    emb_type = "node2vec"
    embedding_model_file_path = select_emb_save_path(save_path_base = 'data',
                                                     emb_type=emb_type,
                                                     add_qualified_edges=args.add_qualified_edges,
                                                     dataset=args.dataset,
                                                     use_weighted_edges=args.use_weighted_edges,
                                                     edges_percent = args.edges_percent,
                                                     edges_number = args.edges_number,
                                                     added_edges_percent_of = args.added_edges_percent_of,
                                                     use_shared_phenotype_edges=args.use_shared_phenotype_edges,
                                                     use_shared_gene_edges=args.use_shared_gene_edges,
                                                     use_shared_gene_and_phenotype_edges=args.use_shared_gene_and_phenotype_edges,
                                                     use_shared_gene_but_not_phenotype_edges=args.use_shared_gene_but_not_phenotype_edges,
                                                     use_shared_phenotype_but_not_gene_edges=args.use_shared_phenotype_but_not_gene_edges)

    # =====================
    # == run embedding (it should save result in appropriate folder within Data
    # =====================
    run_node2vec_emb(data=data, G=graph_with_added_qualified_edges,
                 embedding_model_file_path=embedding_model_file_path,
                 edges_percent= args.edges_percent,
                 edges_number = args.edges_number,
                 use_weighted_edges=args.use_weighted_edges,
                 add_qualified_edges = args.add_qualified_edges,
                 added_edges_percent_of=args.added_edges_percent_of
                 )

if __name__ == '__main__':
    """
    example of running node2vec_emb.py
       --add_qualified_edges top_k --dataset GeneDisease --edges_percent 0.05 --added_edges_percent_of GeneDisease --use_shared_phenotype_edges 
       --run_multiple_args_conditions node2vec
    """

    if args.run_multiple_args_conditions is not None:
        assert args.run_multiple_args_conditions == 'node2vec', "you can only run args.run_multiple_args_conditions == 'train_model' in train_model.py "
        run_args_conditions(run_node2vec, apply_parser_constraint,
                            args.run_multiple_args_conditions)
    else:
        apply_parser_constraint()
        run_node2vec()


    # # =====================
    # # == Datsets
    # # =====================
    # ## GeneDisease
    # GeneDisease_root = r'c:\users\anak\pycharmprojects\recreate_gene_disease\data'  # indicate where file should be stored
    # data = GeneDiseaseGeometricDataset(GeneDisease_root)
    #
    # # # split into train_set and test_set
    # # convert_disease2class_id = np.vectorize(lambda x: data.disease2class_id_dict[x])
    # # disease_class = convert_disease2class_id(data.diseases_np)
    # # train_set, test_set = data.split_train_test(data.diseases_np,
    # #                                             disease_class, 0.4)
    # #
    # # (x_train, y_train), (x_test, y_test) = train_set, test_set
    #
    # # =====================
    # # == Preprocessing
    # # =====================
    # graph_with_added_qualified_edges, data_with_features = get_data_without_using_emb_as_feat(
    #     data=data,
    #     add_qualified_edges=args.add_qualified_edges,
    #     dataset=args.dataset,
    #     use_weighted_edges=args.use_weighted_edges,
    #     normalized_weighted_edges=args.normalized_weighted_edges,
    #     return_graph_and_data_with_features = True,
    #     use_saved_emb_file=args.use_saved_emb_file,
    #     edges_number = args.edges_number,
    #     edges_percent= args.edges_percent,
    #     added_edges_percent_of= args.added_edges_percent_of,
    #     use_shared_phenotype_edges=args.use_shared_phenotype_edges,
    #     use_shared_gene_edges = args.use_shared_gene_edges,
    #     use_shared_gene_and_phenotype_edges=args.use_shared_gene_and_phenotype_edges,
    #     use_shared_gene_but_not_phenotype_edges=args.use_shared_gene_but_not_phenotype_edges,
    #     use_shared_phenotype_but_not_gene_edges=args.use_shared_phenotype_but_not_gene_edges)
    #
    # emb_type = "node2vec"
    # embedding_model_file_path = select_emb_save_path(emb_type=emb_type,
    #                                                  add_qualified_edges=args.add_qualified_edges,
    #                                                  dataset=args.dataset,
    #                                                  use_weighted_edges=args.use_weighted_edges,
    #                                                  edges_percent = args.edges_percent,
    #                                                  edges_number = args.edges_number,
    #                                                  added_edges_percent_of = args.added_edges_percent_of,
    #                                                  use_shared_phenotype_edges=args.use_shared_phenotype_edges,
    #                                                  use_shared_gene_edges=args.use_shared_gene_edges,
    #                                                  use_shared_gene_and_phenotype_edges=args.use_shared_gene_and_phenotype_edges,
    #                                                  use_shared_gene_but_not_phenotype_edges=args.use_shared_gene_but_not_phenotype_edges,
    #                                                  use_shared_phenotype_but_not_gene_edges=args.use_shared_phenotype_but_not_gene_edges)
    #
    # # =====================
    # # == run embedding (it should save result in appropriate folder within Data
    # # =====================
    # run_node2vec_emb(data=data, G=graph_with_added_qualified_edges,
    #              embedding_model_file_path=embedding_model_file_path,
    #              edges_percent= args.edges_percent,
    #              edges_number = args.edges_number,
    #              use_weighted_edges=args.use_weighted_edges,
    #              add_qualified_edges = args.add_qualified_edges,
    #              added_edges_percent_of=args.added_edges_percent_of
    #              )


