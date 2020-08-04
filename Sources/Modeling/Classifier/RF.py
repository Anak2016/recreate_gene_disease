
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from Sources.Evaluation.evaluation import report_performance
from Sources.Evaluation.evaluation import run_clf_using_cross_validation


# from Sources.Evaluation.evaluation import run_cross_validation
from Utilities.saver import save2file


def run_rf_using_test_train_split(data, data_with_features, split, task):
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
    clf = RandomForestClassifier(max_depth=2,random_state=0)
    clf.fit(x_train_with_features, y_train)


    ## train set
    ### use decision_function
    y_train_pred = clf.predict_proba(x_train_with_features).argmax(1)
    y_train_pred_proba = clf.predict_proba(x_train_with_features)

    ## test set
    ### use decision function
    y_test_pred = clf.predict_proba(x_test_with_features).argmax(1)
    y_test_pred_proba = clf.predict_proba(x_test_with_features)

    # report performance of model
    print('=======training set=======')
    train_report_np, train_columns, train_index = report_performance(y_train, y_train_pred, y_train_pred_proba,
                                                                     np.unique(y_train), plot=True, verbose=True, return_value_for_cv=True)
    print('=======test set=======')
    test_report_np, test_columns,  test_index = report_performance(y_test, y_test_pred, y_test_pred_proba,
                                                                   np.unique(y_test), plot=True, verbose=True, return_value_for_cv=True)

    #========= save to file=========
    save2file(train_report_np, train_columns, train_index,
              test_report_np, test_columns, test_index)


def run_rf_for_each_fold(x_train_with_features,
                         x_test_with_features,
                         y_train,
                         y_test):
    """expected to output y_train_pred, y_train_pred_proba, y_test_pred, y_test_pred_proba"""

    # TODO debugging paramter of clr to make it binary clasiifier
    # train model
    # clf = mlp.SVC(gamma='scale', decision_function_shape='ovr',
    #               probability=True)
    # clf = mlp.SVC(kernel='linear', probability=True)  # her

    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(x_train_with_features, y_train)

    ## train set
    ### use decision_function
    y_train_pred = clf.predict_proba(x_train_with_features).argmax(1)
    y_train_pred_proba = clf.predict_proba(x_train_with_features)

    ## test set
    ### use decision function
    y_test_pred = clf.predict_proba(x_test_with_features).argmax(1)
    y_test_pred_proba = clf.predict_proba(x_test_with_features)

    return y_train_pred, y_train_pred_proba, y_test_pred, y_test_pred_proba


def run_rf_using_cross_validation(data, data_with_features, k_fold,
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
                                   run_clf_for_each_fold=run_rf_for_each_fold,
                                   task=task,
                                   edges_as_data=False,
                                   split_by_node=split_by_node
                                   )

def run_rf_node_classification(data,
                               x_with_features,
                               cross_validation,
                               split,
                               k_fold,
                               task,
                               ):
    if cross_validation:
        assert split is None, "split have to be None, if corss_validation is True ( Prevent subtle error and encorage more explicit command argument) "
        assert k_fold is not None, "if cross_validation is True, k_fold must be specified "

        run_rf_using_cross_validation(data, x_with_features, k_fold,
                                      show_only_average_result=True,
                                      save_report_performance=False,
                                      task=task,
                                      split_by_node=True)


    else:
        assert split is not None, "split have to be explicitly specify in command argument (prevent sbutle error and encorgae more explicit command arugment)"
        run_rf_using_test_train_split(data, x_with_features, split, task)


def run_rf_using_test_train_split_with_link_prediction(data, x_with_features,
                                                       task, splitted_edges_dir,
                                                       split, split_by_node):

    train_set_np, test_set_np = data.split_train_test(split,
                                                      stratify=True,
                                                      task=task,
                                                      splitted_edges_dir=splitted_edges_dir,
                                                      split_by_node=split_by_node
                                                      )
    # train_set_np, test_set_np = data.train_set, data.test_set

    # x_train,y_train, x_test,y_test have to be used as row index to get access to x_with_features
    x_train_ind, y_train = train_set_np[:, :2], train_set_np[:, -1].astype(
        float)
    x_test_ind, y_test = test_set_np[:, :2], test_set_np[:, -1].astype(float)

    # from keras.utils import to_categorical
    # y_train = to_categorical(y_train.astype(int))
    # y_test = to_categorical(y_test.astype(int))

    import numpy as np

    # what should train and test of link prediction be?
    x_with_features.index = x_with_features.index.map(str)
    tmp = x_with_features.reindex(x_train_ind[:,0]).dropna().to_numpy()
    tmp_1 = x_with_features.reindex(x_train_ind[:,1]).dropna().to_numpy()
    x_train = np.concatenate([tmp, tmp_1], axis=1)

    tmp = x_with_features.reindex(x_test_ind[:,0]).dropna().to_numpy()
    tmp_1 = x_with_features.reindex(x_test_ind[:,1]).dropna().to_numpy()
    x_test = np.concatenate([tmp, tmp_1], axis=1)

    # x_train = np.concatenate([x_with_features.loc[x_train_ind[:, 0]],
    #                           x_with_features.loc[x_train_ind[:, 1]]], axis=1)
    #
    # x_test = np.concatenate([x_with_features.loc[x_test_ind[:, 0]],
    #                          x_with_features.loc[x_test_ind[:, 1]]], axis=1)

    assert x_train.shape[0] == x_train_ind.shape[0], ''

    # train model
    clf = RandomForestClassifier(max_depth=2,random_state=0)
    clf.fit(x_train, y_train)

    ## train set
    ### use decision_function
    y_train_pred = clf.predict_proba(x_train).argmax(1)
    y_train_pred_proba = clf.predict_proba(x_train)

    ## test set
    ### use decision function
    y_test_pred = clf.predict_proba(x_test).argmax(1)
    y_test_pred_proba = clf.predict_proba(x_test)

    # report performance of model
    print('=======training set=======')
    train_report_np, train_columns, train_index = report_performance(y_train, y_train_pred, y_train_pred_proba,
                                                                     np.unique(y_train), plot=True, verbose=True, return_value_for_cv=True)
    print('=======test set=======')
    test_report_np, test_columns,  test_index = report_performance(y_test, y_test_pred, y_test_pred_proba,
                                                                   np.unique(y_test), plot=True, verbose=True, return_value_for_cv=True)

    #========= save to file=========
    save2file(train_report_np, train_columns, train_index,
              test_report_np, test_columns, test_index)


def run_rf_link_prediction(data, x_with_features, cross_validation, split,
                           k_fold, task, split_by_node, splitted_edges_dir):

    if cross_validation:
        raise ValueError('cross_validation will not be used from this point on')
        assert split is None, "split have to be None, if corss_validation is True ( Prevent subtle error and encorage more explicit command argument) "
        assert k_fold is not None, "if cross_validation is True, k_fold must be specified "

        run_rf_using_cross_validation(data, x_with_features, k_fold,
                                      show_only_average_result=True,
                                      save_report_performance=False,
                                      task=task,
                                      split_by_node=True)

    else:
        assert split is not None, "split have to be explicitly specify in command argument (prevent sbutle error and encorgae more explicit command arugment)"
        run_rf_using_test_train_split_with_link_prediction(data, x_with_features, task,splitted_edges_dir,
                                                           split=split,split_by_node=split_by_node)


        # run_neural_network_using_train_test_split(data, x_with_features, task,splitted_edges_dir,
        #                                           split=split,split_by_node=split_by_node)


def run_rf(data=None, x_with_features=None, cross_validation=None,
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
           split_by_node=None,
           splitted_edges_dir=None
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
    assert split_by_node is not None, "split_by_node must be specified to avoid ambiguity"


    if task =='node_classification':
        run_rf_node_classification(data,
                                   x_with_features,
                                   cross_validation,
                                   split,
                                   k_fold,
                                   task,
                                   )
    else:
        run_rf_link_prediction(data,
                               x_with_features,
                               cross_validation,
                               split,
                               k_fold,
                               task,
                               split_by_node,
                               splitted_edges_dir
                               )




