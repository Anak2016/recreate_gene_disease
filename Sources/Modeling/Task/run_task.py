from Sources.Modeling.Embedding.gcn_emb import run_gcn_with_specified_classifier
from Sources.Modeling.Task.link_prediction import run_link_prediction
from Sources.Modeling.Task.node_classification import run_node_classification_with_unsupervised_emb


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
             use_shared_gene_or_phenotype_edges=None,
             use_gene_disease_graph=None,
             use_phenotype_gene_disease_graph=None,
             graph_edges_type=None,
             task=None,
             split_by_node=None,
             classifier_name = None,
             emb_type=None
             ):
    '''

    @param task: type str; eg. link prediciton or node classification
    @return:
    '''
    assert data is not None, ''
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
    assert use_shared_gene_or_phenotype_edges is not None, "use_shared_gene_or_phenotype_edges must be specified to avoid ambiguity"
    # assert graph_edges_type is not None, "graph_edges_type must be specified to avoid ambiguity"
    assert task is not None, "task must be specified to avoid ambiguity"
    assert split_by_node is not None, "split_by_node must be specified to avoid ambiguity"
    assert classifier_name is not None, "classifier_name must be specified to avoid ambiguity"
    assert x_with_features is not None, ''

    if emb_type not in ['node2vec', 'gcn']:
        raise NotImplementedError


    if task == 'link_prediction':
        run_link_prediction(data=data, x_with_features=x_with_features,
                                cross_validation=cross_validation,
                                k_fold=k_fold, split=split,
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
                                use_shared_gene_or_phenotype_edges=use_shared_gene_or_phenotype_edges,
                                use_gene_disease_graph=use_gene_disease_graph,
                                use_phenotype_gene_disease_graph=use_phenotype_gene_disease_graph,
                                graph_edges_type=graph_edges_type, task=task,
                                split_by_node=split_by_node,
                            )

    elif task == 'node_classification':
        # TODO I haven't add shared_gene_or_phenotype_edges in here yet
        if emb_type == 'node2vec':
            run_node_classification_with_unsupervised_emb(data=data, x_with_features=x_with_features,
                                                          cross_validation=cross_validation,
                                                          k_fold=k_fold, split=split,
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
                                                          graph_edges_type=graph_edges_type, task=task, classifier_name=classifier_name,
                                                          emb_type=emb_type)
        elif emb_type == 'gcn':
                run_gcn_with_specified_classifier(data=data, x_with_features=x_with_features,
                                                          cross_validation=cross_validation,
                                                          k_fold=k_fold, split=split,
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
                                                          graph_edges_type=graph_edges_type, task=task, classifier_name=classifier_name,
                                                          emb_type=emb_type)
        else:
            raise NotImplementedError

    else:
        raise ValueError(
            "Please selection link_prediction or node_classicition")
