from Sources.Modeling.Classifier.SVM import run_svm
from Sources.Preparation.Data import GeneDiseaseGeometricDataset
from Sources.Preparation.Features.build_features import get_data_feat
from arg_parser import args
from arg_parser import run_args_conditions
from arg_parser import apply_parser_constraint
from Sources.Preprocessing.apply_preprocessing import get_number_of_added_edges

def run_train_model():

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
                                       added_edges_percent_of=args.added_edges_percent_of,
                                       use_shared_gene_edges = args.use_shared_gene_edges,
                                       use_shared_phenotype_edges=args.use_shared_phenotype_edges,
                                       use_shared_gene_and_phenotype_edges=args.use_shared_gene_and_phenotype_edges,
                                       use_shared_gene_but_not_phenotype_edges=args.use_shared_gene_but_not_phenotype_edges,
                                       use_shared_phenotype_but_not_gene_edges=args.use_shared_phenotype_but_not_gene_edges)

    # =====================
    # == run classifier
    # =====================
    run_svm(data=data, x_with_features=data_with_features,
            cross_validation=args.cv, k_fold=args.k_fold, split=args.split,
            use_saved_emb_file=args.use_saved_emb_file,
            add_qualified_edges=args.add_qualified_edges,
            dataset=args.dataset,
            use_weighted_edges=args.use_weighted_edges,
            normalized_weighted_edges=args.normalized_weighted_edges,
            edges_percent=args.edges_percent,
            edges_number=args.edges_number,
            added_edges_percent_of=args.added_edges_percent_of,
            use_shared_gene_edges=args.use_shared_gene_edges,
            use_shared_phenotype_edges=args.use_shared_phenotype_edges,
            use_shared_gene_and_phenotype_edges=args.use_shared_gene_and_phenotype_edges,
            use_shared_gene_but_not_phenotype_edges=args.use_shared_gene_but_not_phenotype_edges,
            use_shared_phenotype_but_not_gene_edges=args.use_shared_phenotype_but_not_gene_edges)

if __name__ == '__main__':
    """
    example of args for train_model.py
        --use_saved_emb_file --add_qualified_edges top_k --dataset GeneDisease --edges_percent 0.05 --added_edges_percent_of GeneDisease --use_shared_gene_edges --cv --k_fold 4
        --run_multiple_args_conditions train_model
    """
    # TODO I just can't run the file for some unknown reason. here ever get printed

    if args.run_multiple_args_conditions is not None:
        assert args.run_multiple_args_conditions == 'train_model' ,"you can only run args.run_multiple_args_conditions == 'train_model' in train_model.py "
        run_args_conditions(run_train_model, apply_parser_constraint, args.run_multiple_args_conditions)
    else:
        apply_parser_constraint()
        run_train_model()
