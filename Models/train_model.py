from Sources.Modeling.Classifier.SVM import run_svm
from Sources.Preparation.Data import GeneDiseaseGeometricDataset
from Sources.Preparation.Features.build_features import get_data_feat
from arg_parser import args
from Sources.Preprocessing.preprocessing import get_number_of_added_edges

if __name__ == '__main__':
    # =====================
    # == Datsets
    # =====================
    ## GeneDisease
    GeneDisease_root = r'c:\users\anak\pycharmprojects\recreate_gene_disease\data'  # indicate where file should be stored
    data = GeneDiseaseGeometricDataset(GeneDisease_root)

    # =====================
    # == Preprocessing
    # =====================

    # TODO add args.added_edges_percent_of when edges_percent is not None
    # x_train_with_features, x_test_with_features = get_train_test_feat(data, x_train, x_test, use_saved_emb_file = args.use_saved_emb_file ,add_qualified_edges= args.add_qualified_edges, dataset=args.dataset)
    data_with_features = get_data_feat(data = data,
                                       use_saved_emb_file=args.use_saved_emb_file,
                                       add_qualified_edges=args.add_qualified_edges,
                                       dataset=args.dataset,
                                       use_weighted_edges=args.use_weighted_edges,
                                       normalized_weighted_edges=args.normalized_weighted_edges,
                                       edges_percent= args.edges_percent,
                                       edges_number = args.edges_number,
                                       added_edges_percent_of=args.added_edges_percent_of)

    # =====================
    # == run classifier
    # =====================
    run_svm(data=data, x_with_features=data_with_features,
            cross_validation=args.cv, k_fold=args.k_fold, split=args.split)
