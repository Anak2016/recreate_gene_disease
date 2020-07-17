# from Sources.Modeling.Classifier.SVM import run_svm
from Sources.Modeling.Task import run_task
from Sources.Preparation.Data import GeneDiseaseGeometricDataset
from Sources.Preparation.Features.build_features import get_data_feat
from arg_parser import args
from arg_parser import run_args_conditions
from arg_parser import apply_parser_constraint
from Sources.Preprocessing.apply_preprocessing import get_number_of_added_edges
from global_param import GENEDISEASE_ROOT



def run_train_model():

    # =====================
    # == Datsets
    # =====================
    ## GeneDisease
    # GeneDisease_root = r'c:\users\anak\pycharmprojects\recreate_gene_disease\data'  # indicate where file should be stored
    # data = GeneDiseaseGeometricDataset(GeneDisease_root)

    data = GeneDiseaseGeometricDataset(GENEDISEASE_ROOT)

    # =====================
    # == Preprocessing
    # =====================

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
                                       use_shared_phenotype_but_not_gene_edges=args.use_shared_phenotype_but_not_gene_edges,
                                       use_gene_disease_graph=args.use_gene_disease_graph,
                                       use_phenotype_gene_disease_graph=args.use_phenotype_gene_disease_graph,
                                       use_shared_gene_or_phenotype_edges=args.use_shared_gene_or_phenotype_edges,
                                       graph_edges_type = args.graph_edges_type,
                                       task = args.task,
                                       enforce_end2end= args.enforce_end2end,
                                       cross_validation = args.cv,
                                       k_fold = args.k_fold,
                                       split=args.split,
                                       split_by_node=args.split_by_node,
                                       emb_type=args.emb_type
                                       )

    # #=====================
    # #== run task
    # #=====================
    # run_task(data=data, x_with_features=data_with_features,
    #         cross_validation=args.cv, k_fold=args.k_fold, split=args.split,
    #         use_saved_emb_file=args.use_saved_emb_file,
    #         add_qualified_edges=args.add_qualified_edges,
    #         dataset=args.dataset,
    #         use_weighted_edges=args.use_weighted_edges,
    #         normalized_weighted_edges=args.normalized_weighted_edges,
    #         edges_percent=args.edges_percent,
    #         edges_number=args.edges_number,
    #         added_edges_percent_of=args.added_edges_percent_of,
    #         use_shared_gene_edges=args.use_shared_gene_edges,
    #         use_shared_phenotype_edges=args.use_shared_phenotype_edges,
    #         use_shared_gene_and_phenotype_edges=args.use_shared_gene_and_phenotype_edges,
    #         use_shared_gene_but_not_phenotype_edges=args.use_shared_gene_but_not_phenotype_edges,
    #         use_shared_phenotype_but_not_gene_edges=args.use_shared_phenotype_but_not_gene_edges,
    #          use_shared_gene_or_phenotype_edges=args.use_shared_gene_or_phenotype_edges,
    #         use_gene_disease_graph=args.use_gene_disease_graph,
    #         use_phenotype_gene_disease_graph=args.use_phenotype_gene_disease_graph,
    #         graph_edges_type=args.graph_edges_type,
    #         task=args.task,
    #         split_by_node = args.split_by_node,
    #          classifier_name = args.classifier_name,
    #          emb_type=args.emb_type
    #          )


    # # =====================
    # # == run classifier
    # # =====================

    # run_svm(data=data, x_with_features=data_with_features,
    #         cross_validation=args.cv, k_fold=args.k_fold, split=args.split,
    #         use_saved_emb_file=args.use_saved_emb_file,
    #         add_qualified_edges=args.add_qualified_edges,
    #         dataset=args.dataset,
    #         use_weighted_edges=args.use_weighted_edges,
    #         normalized_weighted_edges=args.normalized_weighted_edges,
    #         edges_percent=args.edges_percent,
    #         edges_number=args.edges_number,
    #         added_edges_percent_of=args.added_edges_percent_of,
    #         use_shared_gene_edges=args.use_shared_gene_edges,
    #         use_shared_phenotype_edges=args.use_shared_phenotype_edges,
    #         use_shared_gene_and_phenotype_edges=args.use_shared_gene_and_phenotype_edges,
    #         use_shared_gene_but_not_phenotype_edges=args.use_shared_gene_but_not_phenotype_edges,
    #         use_shared_phenotype_but_not_gene_edges=args.use_shared_phenotype_but_not_gene_edges,
    #         use_gene_disease_graph=args.use_gene_disease_graph,
    #         use_phenotype_gene_disease_graph=args.use_phenotype_gene_disease_graph,
    #         graph_edges_type=args.graph_edges_type,
    #         )

if __name__ == '__main__':
    """
    example of args for train_model.py
        --task node_classification --enforce_end2end --use_phenotype_gene_disease_graph --graph_edges_type phenotype_gene_disease_phenotype --use_saved_emb_file --add_qualified_edges top_k --dataset GeneDisease --edges_percent 0.05 --added_edges_percent_of GeneDisease --use_shared_gene_edges --cv --k_fold 4
        --task link_prediction --enforce_end2end --use_phenotype_gene_disease_graph --graph_edges_type phenotype_gene_disease_phenotype --use_saved_emb_file --dataset no --cv --k_fold 4j
        --run_multiple_args_conditions train_model
        --run_multiple_args_conditions train_model --classifier_name svm --task node_classification  --enforce_end2end --use_phenotype_gene_disease_graph --graph_edges_type phenotype_gene_disease_phenotype --use_saved_emb_file --dataset no --cv --k_fold 5
    """
    # TODO I just can't run the file for some unknown reason. here ever get printed ??
    # TODO
    #   > curently, we are using 3 type of graph with svm classifer and 1 edge selected strategies ( top-percent)
    #       > goal is to use it with all of the edge selected strategies cases


    if args.run_multiple_args_conditions is not None:
        assert args.run_multiple_args_conditions == 'train_model' ,"you can only run args.run_multiple_args_conditions == 'train_model' in train_model.py "
        run_args_conditions(run_train_model, apply_parser_constraint, args.run_multiple_args_conditions)
    else:
        apply_parser_constraint()
        run_train_model()

