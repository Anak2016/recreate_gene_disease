from Sources.Modeling.Classifier.LR import run_lr
from Sources.Modeling.Classifier.RF import run_rf
from Sources.Modeling.Classifier.SVM import run_svm
from Sources.Modeling.Classifier.NN import run_neural_network
def run_link_prediction(data=None, x_with_features=None,
                            cross_validation=None,
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
                            use_shared_gene_or_phenotype_edges=None,
                            use_gene_disease_graph=None,
                            use_phenotype_gene_disease_graph=None,
                            classifier_name=None,
                            graph_edges_type=None,
                            task=None ,
                            split_by_node=None,
                            emb_type=None,
                        splitted_edges_dir=None
                        ):


    assert classifier_name is not None, "classifier_name must be specified to avoid ambiguity"
    assert emb_type is not None, "emb_type must be specified to avoid ambiguity"


    if classifier_name == 'mlp':
        run_neural_network(data,
                           x_with_features,
                           cross_validation,
                           split,
                           k_fold,
                           task,
                           split_by_node,
                           splitted_edges_dir
                           )

    elif classifier_name == 'svm':
        run_svm(data=data, x_with_features=x_with_features,
                cross_validation=cross_validation, k_fold=k_fold, split=split,
                use_saved_emb_file=use_saved_emb_file,
                add_qualified_edges=add_qualified_edges,
                dataset=dataset,
                use_weighted_edges=use_weighted_edges,
                normalized_weighted_edges=normalized_weighted_edges,
                edges_percent=edges_percent,
                edges_number=edges_number,
                added_edges_percent_of=added_edges_percent_of,
                use_shared_gene_edges=use_shared_gene_edges,
                use_shared_phenotype_edges=use_shared_phenotype_edges,
                use_shared_gene_and_phenotype_edges=use_shared_gene_and_phenotype_edges,
                use_shared_gene_but_not_phenotype_edges=use_shared_gene_but_not_phenotype_edges,
                use_shared_phenotype_but_not_gene_edges=use_shared_phenotype_but_not_gene_edges,
                use_gene_disease_graph=use_gene_disease_graph,
                use_phenotype_gene_disease_graph=use_phenotype_gene_disease_graph,
                graph_edges_type=graph_edges_type,
                task=task,
                split_by_node=split_by_node,
                splitted_edges_dir=splitted_edges_dir
                )
    elif classifier_name == 'lr':
        run_lr(data=data, x_with_features=x_with_features,
               cross_validation=cross_validation, k_fold=k_fold, split=split,
               use_saved_emb_file=use_saved_emb_file,
               add_qualified_edges=add_qualified_edges,
               dataset=dataset,
               use_weighted_edges=use_weighted_edges,
               normalized_weighted_edges=normalized_weighted_edges,
               edges_percent=edges_percent,
               edges_number=edges_number,
               added_edges_percent_of=added_edges_percent_of,
               use_shared_gene_edges=use_shared_gene_edges,
               use_shared_phenotype_edges=use_shared_phenotype_edges,
               use_shared_gene_and_phenotype_edges=use_shared_gene_and_phenotype_edges,
               use_shared_gene_but_not_phenotype_edges=use_shared_gene_but_not_phenotype_edges,
               use_shared_phenotype_but_not_gene_edges=use_shared_phenotype_but_not_gene_edges,
               use_gene_disease_graph=use_gene_disease_graph,
               use_phenotype_gene_disease_graph=use_phenotype_gene_disease_graph,
               graph_edges_type=graph_edges_type,
               task=task,
                split_by_node = split_by_node,
               splitted_edges_dir=splitted_edges_dir
                )
    elif classifier_name == 'rf':
        run_rf(data=data, x_with_features=x_with_features,
               cross_validation=cross_validation, k_fold=k_fold, split=split,
               use_saved_emb_file=use_saved_emb_file,
               add_qualified_edges=add_qualified_edges,
               dataset=dataset,
               use_weighted_edges=use_weighted_edges,
               normalized_weighted_edges=normalized_weighted_edges,
               edges_percent=edges_percent,
               edges_number=edges_number,
               added_edges_percent_of=added_edges_percent_of,
               use_shared_gene_edges=use_shared_gene_edges,
               use_shared_phenotype_edges=use_shared_phenotype_edges,
               use_shared_gene_and_phenotype_edges=use_shared_gene_and_phenotype_edges,
               use_shared_gene_but_not_phenotype_edges=use_shared_gene_but_not_phenotype_edges,
               use_shared_phenotype_but_not_gene_edges=use_shared_phenotype_but_not_gene_edges,
               use_gene_disease_graph=use_gene_disease_graph,
               use_phenotype_gene_disease_graph=use_phenotype_gene_disease_graph,
               graph_edges_type=graph_edges_type,
               task=task,
               split_by_node=split_by_node,
               splitted_edges_dir=splitted_edges_dir
               )
    else:
        raise NotImplementedError
