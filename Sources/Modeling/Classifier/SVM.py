import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import StratifiedKFold

from Sources.Evaluation import report_performance
from Sources.Preprocessing import split_train_test
from Sources.Visualization.visualization import visualize_roc_curve_with_cross_validation_1
from Sources.Visualization.visualization import visualize_roc_curve_with_cross_validation_2
from Sources.Preprocessing.convertion import convert2binary_classifier_prediction


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


def run_svm_using_cross_validation(data, data_with_features, k_fold):
    """

    @param X: numpy
    @param y: numpy
    @return:
    """

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

        print(f"================training cv ={i+1}==================")
        report_final_train_performance_report_np, columns_of_performance_metric, indices_of_performance_metric = report_performance(
            y_train, y_train_pred, y_train_pred_proba, np.unique(y_train),
            plot=False, return_value=True)
        print(f"================test cv ={i+1}==================")
        report_final_test_performance_report_np, columns_of_performance_metric, indices_of_performance_metric = report_performance(
            y_test, y_test_pred, y_test_pred_proba, np.unique(y_test),
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
    print(f'=======training set cv={k_fold}=======')
    print(avg_final_train_performance_report_pd)
    # report_performance(  y_train,  y_train_pred, y_train_pred_proba,  np.unique(y_train), plot=True)

    print(f'=======test set cv={k_fold}=======')
    print(avg_final_test_performance_report_pd)
    # report_performance(  y_test,  y_test_pred, y_test_pred_proba,  np.unique(y_test), plot=True )

    # # visualized roc_curve with std + mean
    # ## train_set
    # visualize_roc_curve_with_cross_validation_2(ax_train, mean_fpr_train,
    #                                             tprs_train, aucs_train)
    # ## test set
    # visualize_roc_curve_with_cross_validation_2(ax_test, mean_fpr_test,
    #                                             tprs_test, aucs_test)


def run_svm(data=None, x_with_features=None, cross_validation=None,
            k_fold=None, split=None):
    """

    @param X: numpy
    @param y: numpy
    @return:
    """
    assert data is not None, ''
    assert x_with_features is not None, ''
    assert cross_validation is not None, ''

    if cross_validation:
        assert split is None, "split have to be None, if corss_validation is True ( Prevent subtle error and encorage more explicit command argument) "
        assert k_fold is not None, "if cross_validation is True, k_fold must be specified "
        run_svm_using_cross_validation(data, x_with_features, k_fold)
    else:
        assert split is not None, "split have to be explicitly specify in command argument (prevent sbutle error and encorgae more explicit command arugment)"
        run_svm_using_test_train_split(data, x_with_features, split)
