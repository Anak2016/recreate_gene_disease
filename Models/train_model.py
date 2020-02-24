from Sources.Modeling.Classifier.SVM import run_svm
from Sources.Modeling.Task.link_prediction import run_link_prediction
from Sources.Modeling.Task.node_classification import run_node_classification
from Sources.Preparation.Data import GeneDiseaseGeometricDataset
from Sources.Preparation.Features.build_features import get_data_feat
from arg_parser import args
from arg_parser import run_args_conditions
from arg_parser import apply_parser_constraint
from Sources.Preprocessing.apply_preprocessing import get_number_of_added_edges
from global_param import GENEDISEASE_ROOT

def run_task(data=None, x_with_features=None, cross_validation=None,
             k_fold=None, split=None,
             use_saved_emb_file=None,
             add_qualified_edges=None,
             dataset=None, use_weighted_edges=None,
             normalized_weighted_edges=None,
             edges_percent=None,
             edges_number=None,
             added_edges_percent_of=None,
             use_shared_gene_edges=None,
             use_shared_phenotype_edges=None,
             use_shared_gene_and_phenotype_edges=None,
             use_shared_gene_but_not_phenotype_edges=None,
             use_shared_phenotype_but_not_gene_edges=None,
             use_gene_disease_graph=None,
             use_phenotype_gene_disease_graph=None,
             graph_edges_type=None,
             task=None
             ):
    '''

    @param task: type str; eg. link prediciton or node classification
    @return:
    '''
    assert data is not None, ''
    assert x_with_features is not None, ''
    assert cross_validation is not None, ''
    assert use_saved_emb_file is not None, 'use_saved_emb_file must be explicitly stated to avoid ambiguity'
    # assert add_qualified_edges is not None, 'add_qualified_edges must be explicitly stated to avoid ambiguity'
    assert dataset is not None, 'dataset must be explicitly stated to avoid ambiguity'
    assert use_weighted_edges is not None, 'use_weighted_edges must be explicitly stated to avoid ambiguity'
    assert normalized_weighted_edges is not None, "normalized_weighted_edges must be specified to avoid ambiguity"
    # assert (edges_percent is not None) or (
    #         edges_number is not None), "either edges_percent or edges_number must be specified to avoid ambiguity"
    assert use_shared_gene_edges is not None, "use_shared_gene_edges must be specified to avoid ambiguity"
    assert use_shared_phenotype_edges is not None, "use_shared_phenotype_edges must be specified to avoid ambiguity"
    assert use_shared_gene_and_phenotype_edges is not None, "use_shared_gene_and_phenotype_edges must be specified to avoid ambiguity"
    assert use_shared_gene_but_not_phenotype_edges is not None, "use_shared_gene_but_not_phenotype_edges must be specified to avoid ambiguity"
    assert use_shared_phenotype_but_not_gene_edges is not None, "use_shared_phenotype_but_not_gene_edges must be specified to avoid ambiguity"
    assert graph_edges_type is not None, "graph_edges_type must be specified to avoid ambiguity"
    assert task is not None, "task must be specified to avoid ambiguity"
    assert task is not None, "task must be specified to avoid ambiguity"

    if task == 'link_prediction':
        run_link_prediction()
    elif task == 'node_classification':
        run_node_classification()
    else:
        raise ValueError("Please selection link_prediction or node_classicition")


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
                                       graph_edges_type = args.graph_edges_type
                                       )

    #=====================
    #==select task
    #=====================
    # TODO here>>

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
    #         use_gene_disease_graph=args.use_gene_disease_graph,
    #         use_phenotype_gene_disease_graph=args.use_phenotype_gene_disease_graph,
    #         graph_edges_type=args.graph_edges_type,
    #         task = args.task
    #         )

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
            use_shared_phenotype_but_not_gene_edges=args.use_shared_phenotype_but_not_gene_edges,
            use_gene_disease_graph=args.use_gene_disease_graph,
            use_phenotype_gene_disease_graph=args.use_phenotype_gene_disease_graph,
            graph_edges_type=args.graph_edges_type,
            task = args.task
            )

if __name__ == '__main__':
    """
    example of args for train_model.py
        --use_phenotype_gene_disease_graph --graph_edges_type phenotype_gene_disease_phenotype --use_saved_emb_file --add_qualified_edges top_k --dataset GeneDisease --edges_percent 0.05 --added_edges_percent_of GeneDisease --use_shared_gene_edges --cv --k_fold 4
        --run_multiple_args_conditions train_model
    """
    # TODO I just can't run the file for some unknown reason. here ever get printed

    if args.run_multiple_args_conditions is not None:
        assert args.run_multiple_args_conditions == 'train_model' ,"you can only run args.run_multiple_args_conditions == 'train_model' in train_model.py "
        run_args_conditions(run_train_model, apply_parser_constraint, args.run_multiple_args_conditions)
    else:
        apply_parser_constraint()
        run_train_model()
