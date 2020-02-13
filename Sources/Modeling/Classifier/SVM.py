import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import StratifiedKFold

from Sources.Evaluation.evaluation import report_performance
from Sources.Preprocessing.apply_preprocessing import select_emb_save_path
from Sources.Preprocessing.apply_preprocessing import split_train_test
from Sources.Preprocessing.apply_preprocessing import get_saved_file_name_for_emb


def run_svm_using_test_train_split(data, data_with_features, split):
    """

    @param X: numpy
    @param y: numpy
    @return:
    """

    # split training set and test set
    (x_train, y_train), (x_test, y_test) = split_train_test(data, split)
    x_train_with_features, x_test_with_features = data_with_features.loc[
                                                      x_train], \
                                                  data_with_features.loc[
                                                      x_test]

    # train model 
    clf = svm.SVC(gamma='scale', decision_function_shape='ovr',
                  probability=True)
    clf.fit(x_train_with_features, y_train)

    ## train set
    ### use decision_function
    y_train_pred = clf.decision_function(x_train_with_features).argmax(1)
    y_train_pred_proba = clf.decision_function(x_train_with_features)

    ## test set
    ### use decision function
    y_test_pred = clf.decision_function(x_test_with_features).argmax(1)
    y_test_pred_proba = clf.decision_function(x_test_with_features)

    # report performance of model 
    print('=======training set=======')
    report_performance(y_train, y_train_pred, y_train_pred_proba,
                       np.unique(y_train), plot=True)
    print('=======test set=======')
    report_performance(y_test, y_test_pred, y_test_pred_proba,
                       np.unique(y_test), plot=True)


def run_svm_using_cross_validation(data, data_with_features, k_fold,
                                   only_show_average_result=False,
                                   save_report_performance=None,
                                   report_performance_file_path=None):
    """

    @param X: numpy
    @param y: numpy
    @return:
    """
    assert save_report_performance is not None, "save_report_performance must be specified to avoid ambiguity"
    if save_report_performance:
        assert report_performance_file_path is not None, "report_performance_file_path must be specified to avoid ambiguity"

    convert_disease2class_id = np.vectorize(
        lambda x: data.disease2class_id_dict[x])
    disease_class = convert_disease2class_id(data.diseases_np)

    x = data.diseases_np
    skf = StratifiedKFold(n_splits=k_fold)

    sum_final_train_performance_report_np = None
    sum_final_test_performance_report_np = None

    # declare variables for roc_curve
    ## train set
    tprs_train = []
    aucs_train = []
    mean_fpr_train = np.linspace(0, 1, 100)
    fig_train, ax_train = plt.subplots()

    ## test_set
    tprs_test = []
    aucs_test = []
    mean_fpr_test = np.linspace(0, 1, 100)
    fig_test, ax_test = plt.subplots()

    columns_of_performance_metric = None
    indices_of_performance_metric = None

    for i, (train_ind, test_ind) in enumerate(skf.split(x, disease_class)):
        x_train, y_train = x[train_ind], disease_class[train_ind]
        x_test, y_test = x[test_ind], disease_class[test_ind]

        x_train_with_features, x_test_with_features = data_with_features.loc[
                                                          x_train], \
                                                      data_with_features.loc[
                                                          x_test]

        # TODO debugging paramter of clr to make it binary clasiifier
        # train model
        clf = svm.SVC(gamma='scale', decision_function_shape='ovr',
                      probability=True)
        # clf = svm.SVC(kernel='linear', probability=True)  # her

        clf.fit(x_train_with_features, y_train)

        ## train set
        ### use decision_function
        y_train_pred = clf.decision_function(x_train_with_features).argmax(1)
        y_train_pred_proba = clf.decision_function(x_train_with_features)

        ## test set
        ### use decision function
        y_test_pred = clf.decision_function(x_test_with_features).argmax(1)
        y_test_pred_proba = clf.decision_function(x_test_with_features)

        if not only_show_average_result:
            print(f"================training cv ={i}==================")

        report_final_train_performance_report_np, columns_of_performance_metric, indices_of_performance_metric = report_performance(
            y_train, y_train_pred, y_train_pred_proba, np.unique(y_train),
            verbose=not only_show_average_result,
            plot=False, return_value=True)

        if not only_show_average_result:
            print(f"================test cv ={i}==================")

        report_final_test_performance_report_np, columns_of_performance_metric, indices_of_performance_metric = report_performance(
            y_test, y_test_pred, y_test_pred_proba, np.unique(y_test),
            verbose=not only_show_average_result,
            plot=False, return_value=True)

        if sum_final_test_performance_report_np is None:
            sum_final_test_performance_report_np = report_final_test_performance_report_np
            sum_final_train_performance_report_np = report_final_train_performance_report_np
        else:
            sum_final_test_performance_report_np = sum_final_test_performance_report_np + report_final_test_performance_report_np
            sum_final_train_performance_report_np = sum_final_train_performance_report_np + report_final_train_performance_report_np

        # = convert2binary_classifier_prediction(x_train_with_features, y_train)

        # TODO convert clf to binary blf (refer to notebook for more information)
        # # visualize_roc_curve with cross validation
        # ## train set
        # ax_train, tprs_train, aucs_train = visualize_roc_curve_with_cross_validation_1(
        #     clf, i, x_train_with_features, y_train, mean_fpr_train, tprs_train,
        #     aucs_train, ax_train)
        # ## test_set
        # ax_test, tprs_test, aucs_test = visualize_roc_curve_with_cross_validation_1(
        #     clf, i, x_test_with_features, y_test, mean_fpr_test, tprs_test,
        #     aucs_test, ax_test)

    avg_final_train_performance_report_np = sum_final_train_performance_report_np / k_fold
    avg_final_test_performance_report_np = sum_final_test_performance_report_np / k_fold

    avg_final_train_performance_report_pd = pd.DataFrame(
        avg_final_train_performance_report_np,
        columns=columns_of_performance_metric,
        index=indices_of_performance_metric)
    avg_final_test_performance_report_pd = pd.DataFrame(
        avg_final_test_performance_report_np,
        columns=columns_of_performance_metric,
        index=indices_of_performance_metric)

    # report performance of model
    print(f'=======training set cv k_fold={k_fold}=======')
    print(avg_final_train_performance_report_pd)
    # report_performance(  y_train,  y_train_pred, y_train_pred_proba,  np.unique(y_train), plot=True)

    print(f'=======test set cv k_fold={k_fold}=======')
    print(avg_final_test_performance_report_pd)
    # report_performance(  y_test,  y_test_pred, y_test_pred_proba,  np.unique(y_test), plot=True )

    # TODO finish save_report_performance
    # if save_report_performance:
    #     import os
    #     from os import path
    #
    #     # check that no files with the same name existed within these folder
    #     assert not path.exists(
    #         report_performance_file_path), "emb_file already exist, Please check if you argument is correct"
    #
    #     # create dir if not alreayd existed
    #     report_performance_dir = '\\'.join(report_performance_file_path.split('\\')[:-1])
    #     if not os.path.exists(report_performance_dir):
    #         os.makedirs(report_performance_dir)
    #
    #     avg_final_train_test_performance_report_pd = pd.concat([avg_final_train_performance_report_pd, avg_final_test_performance_report_pd], axis =0)
    #     avg_final_train_test_performance_report_pd.to_csv(report_performance_file_path)


# # visualized roc_curve with std + mean
# ## train_set
# visualize_roc_curve_with_cross_validation_2(ax_train, mean_fpr_train,
#                                             tprs_train, aucs_train)
# ## test set
# visualize_roc_curve_with_cross_validation_2(ax_test, mean_fpr_test,
#                                             tprs_test, aucs_test)


def run_svm(data=None, x_with_features=None, cross_validation=None,
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
            use_shared_phenotype_but_not_gene_edges=None):
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

    path_to_saved_emb_dir = select_emb_save_path(save_path_base='report_performance',
                                                 emb_type='node2vec',
                                                 add_qualified_edges=add_qualified_edges,
                                                 dataset=dataset,
                                                 use_weighted_edges=use_weighted_edges,
                                                 edges_percent=edges_percent,
                                                 edges_number=edges_number,
                                                 added_edges_percent_of=added_edges_percent_of,
                                                 use_shared_gene_edges=use_shared_gene_edges,
                                                 use_shared_phenotype_edges=use_shared_phenotype_edges,
                                                 use_shared_gene_and_phenotype_edges=use_shared_gene_and_phenotype_edges,
                                                 use_shared_gene_but_not_phenotype_edges=use_shared_gene_but_not_phenotype_edges,
                                                 use_shared_phenotype_but_not_gene_edges=use_shared_phenotype_but_not_gene_edges)

    file_name = get_saved_file_name_for_emb(add_qualified_edges, edges_percent,
                                            edges_number, 64, 30, 200, 10)
    report_performance_file_path = path_to_saved_emb_dir + file_name

    if cross_validation:
        assert split is None, "split have to be None, if corss_validation is True ( Prevent subtle error and encorage more explicit command argument) "
        assert k_fold is not None, "if cross_validation is True, k_fold must be specified "
        run_svm_using_cross_validation(data, x_with_features, k_fold,
                                       only_show_average_result=True,
                                       save_report_performance=True,
                                       report_performance_file_path=report_performance_file_path)
    else:
        assert split is not None, "split have to be explicitly specify in command argument (prevent sbutle error and encorgae more explicit command arugment)"
        run_svm_using_test_train_split(data, x_with_features, split)
