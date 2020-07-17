
import numpy as np
from sklearn.linear_model import LogisticRegression

from Sources.Evaluation.evaluation import report_performance
from Sources.Evaluation.evaluation import run_clf_using_cross_validation


# from Sources.Evaluation.evaluation import run_cross_validation
from Utilities.saver import save2file


def run_lr_using_test_train_split(data, data_with_features, split, task):
    """
#
    @param X: numpy
    @param y: numpy
    @return:
    """

    # split training set and test set
    (x_train, y_train), (x_test, y_test) = data.split_train_test(split,
                                                                 stratify=True,
                                                                 task=task)
                                                                 # reset_train_test_split=True,
                                                                 # split_by_node=True)
    x_train_with_features, x_test_with_features = data_with_features.loc[
                                                      x_train], \
                                                  data_with_features.loc[
                                                      x_test]

    # train model
    clf = LogisticRegression(random_state=0)
    clf.fit(x_train_with_features, y_train)


    ## train set
    ### use decision_function
    y_train_pred = clf.decision_function(x_train_with_features).argmax(1)
    y_train_pred_proba = clf.decision_function(x_train_with_features)

    ## test set
    ### use decision function
    y_test_pred = clf.decision_function(x_test_with_features).argmax(1)
    y_test_pred_proba = clf.decision_function(x_test_with_features)

    print('=======training set=======')
    train_report_np, train_columns, train_index = report_performance(y_train, y_train_pred, y_train_pred_proba,
                                                                     np.unique(y_train), plot=True, verbose=True, return_value_for_cv=True)
    print('=======test set=======')
    test_report_np, test_columns,  test_index = report_performance(y_test, y_test_pred, y_test_pred_proba,
                                                                   np.unique(y_test), plot=True, verbose=True, return_value_for_cv=True)

    #========= save to file=========
    save2file(train_report_np, train_columns, train_index,
              test_report_np, test_columns, test_index)


def run_lr_for_each_fold(x_train_with_features,
                          x_test_with_features,
                          y_train,
                          y_test):
    """expected to output y_train_pred, y_train_pred_proba, y_test_pred, y_test_pred_proba"""

    # TODO debugging paramter of clr to make it binary clasiifier
    # train model
    # clf = svm.SVC(gamma='scale', decision_function_shape='ovr',
    #               probability=True)
    # clf = svm.SVC(kernel='linear', probability=True)  # her

    clf = LogisticRegression()
    clf.fit(x_train_with_features, y_train)

    ## train set
    ### use decision_function
    y_train_pred = clf.decision_function(x_train_with_features).argmax(1)
    y_train_pred_proba = clf.decision_function(x_train_with_features)

    ## test set
    ### use decision function
    y_test_pred = clf.decision_function(x_test_with_features).argmax(1)
    y_test_pred_proba = clf.decision_function(x_test_with_features)

    return y_train_pred, y_train_pred_proba, y_test_pred, y_test_pred_proba


def run_lr_using_cross_validation(data, data_with_features, k_fold,
                                   show_only_average_result=False,
                                   save_report_performance=None,
                                   report_performance_file_path=None,
                                   task=None,
                                   split_by_node=None
                                   ):
    """

    @param X: numpy
    @param y: numpy
    @return:
    """

    # TODO code below run_clf_using_cross_validation has been moved to cross_validation()
    run_clf_using_cross_validation(data, data_with_features, k_fold,
                                   show_only_average_result=show_only_average_result,
                                   save_report_performance=save_report_performance,
                                   report_performance_file_path=report_performance_file_path,
                                   run_clf_for_each_fold=run_lr_for_each_fold,
                                   task=task,
                                   edges_as_data=False,
                                   split_by_node=split_by_node
                                   )

def run_lr_node_classification(data,
                                x_with_features,
                                cross_validation,
                                split,
                                k_fold,
                                task,
                                ):
    if cross_validation:
        assert split is None, "split have to be None, if corss_validation is True ( Prevent subtle error and encorage more explicit command argument) "
        assert k_fold is not None, "if cross_validation is True, k_fold must be specified "

        run_lr_using_cross_validation(data, x_with_features, k_fold,
                                       show_only_average_result=True,
                                       save_report_performance=False,
                                       task=task,
                                       split_by_node=True)


    else:
        assert split is not None, "split have to be explicitly specify in command argument (prevent sbutle error and encorgae more explicit command arugment)"
        run_lr_using_test_train_split(data, x_with_features, split, task)


def run_lr(data=None, x_with_features=None, cross_validation=None,
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
            task=None,
            ):
    """

    @param X: numpy
    @param y: numpy
    @return:
    """
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
    # assert graph_edges_type is not None, "graph_edges_type must be specified to avoid ambiguity"
    assert task is not None, "task must be specified to avoid ambiguity"

    run_lr_node_classification(data,
                                x_with_features,
                                cross_validation,
                                split,
                                k_fold,
                                task,
                                )



