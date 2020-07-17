import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve

from Sources.Visualization.visualization import visualize_roc_curve


def report_performance(y_true, y_pred, y_score, labels, verbose=None,
                       plot=False, return_value_for_cv=False):
    """

    @param y_true: shape = (# of instance,); type = np.array or list
    @param y_pred:  shape = (# of instance,); type = np.array or list
    @param y_score: shape = (# of instance, # of class); type = np.array
    @param labels:
    @param plot:
    @param return_value_for_cv:
    @return: desc = return value if return_value_for_cv is True;
           report_final_performance_report_np: type = numpy
           columns_of_performance_metric:  type = list: desc = list of performance metrics name
    """
    assert verbose is not None, "verbose must be specified to avoid ambiguity"

    assert len(labels) > 1 , "minimum label = 2 (aka binary classification)"

    plot = False

    report_sklearn_classification_report = classification_report(y_true,
                                                                 y_pred,
                                                                 labels,
                                                                 output_dict=True)

    report_dict = copy.deepcopy(report_sklearn_classification_report)
    del report_dict['accuracy']
    report_dict['accuracy'] = {'precision': None, 'recall': None,
                               'f1-score':
                                   report_sklearn_classification_report[
                                       'accuracy'], 'support': None}
    report_df = pd.DataFrame(report_dict).round(2).transpose()
    # print(report_df)
    # print(report_sklearn_classification_report)

    # show support class and predicted classes
    ## note: np.unique output sorted value
    supported_class_np, supported_class_freq_np = np.unique(y_true,
                                                            return_counts=True)
    predicted_class_np, predicted_class_freq_np = np.unique(y_pred,
                                                            return_counts=True)

    support_class_df = pd.DataFrame(supported_class_freq_np,
                                    columns=['support'],
                                    index=supported_class_np)
    predicted_class_df = pd.DataFrame(predicted_class_freq_np,
                                      columns=['predicted'],
                                      index=predicted_class_np)

    report_support_pred_class = pd.concat(
        [support_class_df, predicted_class_df], axis=1)
    # print(report_support_pred_class)

    # show AUC
    # TODO here>>
    ## normalized to probability: ( This is a hack; because roc_auc_score only accept probaility like y_score .
    from sklearn.metrics import roc_auc_score
    normalized_row = np.apply_along_axis(lambda x: [i / sum(x) for i in x], 1,
                                         y_score)

    # ## create total_roc_auc_score; output shape = 1
    # if 2 == np.unique(y_true).shape[0] and 2 == normalized_row.shape[1]:
    #     total_roc_auc_score = roc_auc_score(y_true, normalized_row[:, 1],
    #                                         multi_class='ovo')
    # else:
    #     total_roc_auc_score = roc_auc_score(y_true, normalized_row,
    #                                         multi_class='ovo')

    # print(roc_auc_score(y_true, normalized_row, multi_class='ovo'))

    # TODO figure out why roc score is very low? what did I do wrong?
    ## read how get_roc_curve() works

    ##  create per class roc_auc_score; output shape = [# of instances]

    tmp = pd.get_dummies(y_true).to_numpy()
    tmp = np.hstack((tmp, np.zeros(tmp.shape[0]).reshape(-1, 1)))
    fpr, tpr, roc_auc = get_roc_curve(tmp,
                                      y_score, labels.shape[0])

    roc_auc = {i: [j] for i, j in roc_auc.items()}
    # todo this is just avg not micro avg
    roc_auc['micro_avg'] = np.array(
        [j for i in roc_auc.values() for j in i]).mean()
    # roc_auc['total_auc'] = total_roc_auc_score
    roc_auc_df = pd.DataFrame.from_dict(roc_auc).transpose()
    roc_auc_df.columns = ['AUC']

    if plot:
        # fpr, tpr, roc_auc = get_roc_curve(pd.get_dummies(y_true).to_numpy(), y_score, labels.shape[0])
        visualize_roc_curve(fpr, tpr, roc_auc)

    return combine_report_and_auc(report_df, report_support_pred_class,
                                  roc_auc_df, verbose=verbose,
                                  return_value_for_cv=return_value_for_cv)

    # if return_value_for_cv:
    #     return report_performance_for_cv(report_df, report_support_pred_class,
    #                                      roc_auc_df, verbose=verbose)


# def report_performance_for_cv(report_df,report_support_pred_class, roc_auc_df, verbose):
def combine_report_and_auc(report_df, report_support_pred_class,
                           roc_auc_df, verbose, return_value_for_cv):
    # create mask
    ## create mask for report_support_pred_class that have the smae index as report_df.index (fill with nan)
    na_np = np.tile(np.nan, (
        report_df.shape[0], report_support_pred_class.shape[1]))

    report_support_pred_class_mask_with_nan_df = pd.DataFrame(na_np,
                                                              index=report_df.index,
                                                              columns=report_support_pred_class.columns)
    report_support_pred_class_mask_with_nan_df.loc[
    :report_support_pred_class.shape[0],
    :] = report_support_pred_class.values
    # print(na_df)

    report_support_pred_class_mask_with_nan_with_predicted_col_df = \
        report_support_pred_class_mask_with_nan_df[['predicted']]

    ## create maks for roc_auc_df to have same index as report_df.index (fill with nan)

    na_np = np.tile(np.nan, (
        report_df.shape[0], roc_auc_df.shape[1]))

    roc_auc_mask_with_nan_df = pd.DataFrame(na_np, index=report_df.index,
                                            columns=roc_auc_df.columns)
    roc_auc_mask_with_nan_df.loc[:roc_auc_df.shape[0],
    :] = roc_auc_df.values
    # print(na_df)

    merged_report_df = pd.concat([report_df,
                                  report_support_pred_class_mask_with_nan_with_predicted_col_df,
                                  roc_auc_mask_with_nan_df], axis=1)
    # merged_report_df = report_df.merge(report_support_pred_class, how='outer', on=['support'], copy=False, right_index=True)
    if verbose:
        print(merged_report_df)

    if return_value_for_cv:
        return merged_report_df.to_numpy(), merged_report_df.columns, merged_report_df.index


def get_roc_curve(y_true, y_score, n_classes):
    """
    refer back to the following link : https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py

    @param y_true: type = numpy; desc = onehot vector
    @param y_score: type = numpy; desc = onehot vector
    @return:
    """

    # Compute ROC curve and ROC area for each clasu
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    return fpr, tpr, roc_auc


def report_average_performance_of_cross_validation(k_fold,
                                                   sum_final_train_performance_report_np,
                                                   sum_final_test_performance_report_np,
                                                   columns_of_performance_metric,
                                                   indices_of_performance_metric):
    """for example code, look at SVM/run_svm_using_cross_validation for example"""

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
    print(f'=======avg of training set cv k_fold={k_fold}=======')
    print(avg_final_train_performance_report_pd)
    # report_performance(  y_train,  y_train_pred, y_train_pred_proba,  np.unique(y_train), plot=True)

    print(f'=======avg of test set cv k_fold={k_fold}=======')
    print(avg_final_test_performance_report_pd)
    # report_performance(  y_test,  y_test_pred, y_test_pred_proba,  np.unique(y_test), plot=True )


def run_clf_using_cross_validation(data, data_with_features_dict, k_fold,
                                   show_only_average_result=False,
                                   save_report_performance=None,
                                   report_performance_file_path=None,
                                   run_clf_for_each_fold=None,
                                   task=None,
                                   edges_as_data=None,
                                   splitted_edges_dir=None,
                                   split_by_node=None
                                   ):

    assert save_report_performance is not None, "save_report_performance must be specified to avoid ambiguity"
    assert run_clf_for_each_fold is not None, "run_clf_for_each_fold must be specified to avoid ambiguity"
    assert isinstance(edges_as_data, bool), ''

    if save_report_performance:
        assert report_performance_file_path is not None, "report_performance_file_path must be specified to avoid ambiguity"

    # TODO here>>-2 paragraph of code below will be move to run_node2vec with task == 'node_classification'
    # convert_disease2class_id = np.vectorize(
    #     lambda x: data.disease2class_id_dict[x])
    # disease_class = convert_disease2class_id(data.diseases_np)
    #
    # x = data.diseases_np
    # skf = StratifiedKFold(n_splits=k_fold)

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

    # run_cross_validation(y_train_pred)
    columns_of_performance_metric = None
    indices_of_performance_metric = None

    # TODO here>-6 create cross_Validation for scenario either node or edges are used as row's name

    # skf = StratifiedKFold(n_splits=k_fold)
    # for i, (train_set, test_set) in enumerate(
    #         data.split_cross_validation(data, k_fold, stratify=True, task=task,
    #                                     # reset_cross_validation_split=False,
    #                                     splitted_edges_dir=splitted_edges_dir
    #                                     )):

    # TODO to be deleted after finished getting result from k_fold = 101

    # for i, (train_test_dict) in enumerate(
    #         data.split_cross_validation(data, k_fold, stratify=True, task=task,
    #                                     split_by_node=split_by_node,
    #                                     # reset_cross_validation_split=False,
    #                                     splitted_edges_dir=splitted_edges_dir,
    #                                     )):
    #     train_set = train_test_dict['train_set']
    #     test_set = train_test_dict['test_set']
    #     # for i, (train_ind, test_ind) in enumerate(skf.split(x, disease_class)):
    #
    #     x_train, y_train = train_set[:, :-1], train_set[:, -1]
    #     x_test, y_test = test_set[:, :-1], test_set[:, -1]
    #
    #     # TODO here>>-2 paragraph of code below will be move to run_node2vec with task == 'node_classification'
    #     # x_train, y_train = x[train_ind], disease_class[train_ind]
    #     # x_test, y_test = x[test_ind], disease_class[test_ind]
    #     if edges_as_data:
    #         train_instances = []
    #         test_instances = []
    #
    #         for node1, node2, _ in train_set:
    #             edge_instance = f'{node1}_{node2}'
    #             train_instances.append(edge_instance)
    #         for node1, node2, _ in test_set:
    #             edge_instance = f'{node1}_{node2}'
    #             test_instances.append(edge_instance)
    #
    #     # x = f'C:\\Users\\Anak\\PycharmProjects\\recreate_gene_disease\\Data\\processed\\LinkPrediction\\GeneDiseaseProject\\copd\\PhenotypeGeneDisease\\PGDP\\Node2Vec\\UnweightedEdges\\NoAddedEdges\\SplitByNode\\KFold=101\\DecendingOrder\\{i}\\dim64_walk_len30_num_walks200_window10.txt'
    #     # from Sources.Preparation.Features import get_instances_with_features
    #     # data_with_features_dict[i] = get_instances_with_features(
    #     #     use_saved_emb_file=True,
    #     #     path_to_saved_emb_file=x,
    #     #     normalized_weighted_edges=False
    #     # )
    #
    #     # TODO here>>-9 validate that slicing dataframe output desired outcome.
    #     x_train_with_features, x_test_with_features = \
    #         data_with_features_dict[i].loc[
    #             train_instances], \
    #         data_with_features_dict[i].loc[
    #             test_instances]
    #
    #     # x_train_with_features, x_test_with_features = data_with_features.loc[
    #     #                                                   x_train], \
    #     #                                               data_with_features.loc[
    #     #                                                   x_test]
    #     y_train = y_train.astype(float)
    #     y_train = y_train.astype(int)
    #     y_test = y_test.astype(float)
    #     y_test = y_test.astype(int)
    #
    #     # # garantee that
    #     # assert y_train[0] == x_train_with_features.shape[0], 'Dataleakage! x_test and x_train contain the same nonexisted_edges '
    #     # assert y_test[0] == x_test_with_features.shape[0], 'Dataleakage! x_test and x_train contain the same nonexisted_edges '
    #
    #     # TODO here>>-8 this is a quickfix to run the model => Goal is to see whether there are any more error other thna one cause by input dataset
    #     ## does this effect the outcome in unpredicatable way
    #     x_train_with_features = x_train_with_features.drop_duplicates()
    #     x_test_with_features = x_test_with_features.drop_duplicates()
    #
    #
    #     y_train_pred, y_train_pred_proba, y_test_pred, y_test_pred_proba = run_clf_for_each_fold(
    #         x_train_with_features,
    #         x_test_with_features,
    #         y_train, y_test)
    #
    #     # TODO paragraph of code below is moved to run_svm_for_each_fold()
    #     ## Goal is to make the function reuseable by different classifer( which is library/function dependent)
    #     # # TODO debugging paramter of clr to make it binary clasiifier
    #     # # train model
    #     # clf = svm.SVC(gamma='scale', decision_function_shape='ovr',
    #     #               probability=True)
    #     # # clf = svm.SVC(kernel='linear', probability=True)  # her
    #     #
    #     # clf.fit(x_train_with_features, y_train)
    #     #
    #     # ## train set
    #     # ### use decision_function
    #     # y_train_pred = clf.decision_function(x_train_with_features).argmax(1)
    #     # y_train_pred_proba = clf.decision_function(x_train_with_features)
    #     #
    #     # ## test set
    #     # ### use decision function
    #     # y_test_pred = clf.decision_function(x_test_with_features).argmax(1)
    #     # y_test_pred_proba = clf.decision_function(x_test_with_features)
    #
    #     if not show_only_average_result:
    #         print(f"================training cv ={i}==================")
    #
    #     report_final_train_performance_report_np, columns_of_performance_metric, indices_of_performance_metric = report_performance(
    #         y_train, y_train_pred, y_train_pred_proba, np.unique(y_train),
    #         verbose=not show_only_average_result,
    #         plot=False, return_value_for_cv=True)
    #
    #     if not show_only_average_result:
    #         print(f"================test cv ={i}==================")
    #
    #     report_final_test_performance_report_np, columns_of_performance_metric, indices_of_performance_metric = report_performance(
    #         y_test, y_test_pred, y_test_pred_proba, np.unique(y_test),
    #         verbose=not show_only_average_result,
    #         plot=False, return_value_for_cv=True)
    #
    #     if sum_final_test_performance_report_np is None:
    #         sum_final_test_performance_report_np = report_final_test_performance_report_np
    #         sum_final_train_performance_report_np = report_final_train_performance_report_np
    #     else:
    #         sum_final_test_performance_report_np = sum_final_test_performance_report_np + report_final_test_performance_report_np
    #         sum_final_train_performance_report_np = sum_final_train_performance_report_np + report_final_train_performance_report_np
    #
    #     # = convert2binary_classifier_prediction(x_train_with_features, y_train)
    #
    #     # TODO convert clf to binary blf (refer to notebook for more information)
    #     # # visualize_roc_curve with cross validation
    #     # ## train set
    #     # ax_train, tprs_train, aucs_train = visualize_roc_curve_with_cross_validation_1(
    #     #     clf, i, x_train_with_features, y_train, mean_fpr_train, tprs_train,
    #     #     aucs_train, ax_train)
    #     # ## test_set
    #     # ax_test, tprs_test, aucs_test = visualize_roc_curve_with_cross_validation_1(
    #     #     clf, i, x_test_with_features, y_test, mean_fpr_test, tprs_test,
    #     #     aucs_test, ax_test)

    splitted_edges_dir = r'C:\Users\Anak\PycharmProjects\recreate_gene_disease\Data\processed\LinkPrediction\GeneDiseaseProject\copd\PhenotypeGeneDisease\PGDP\Node2Vec\UnweightedEdges\NoAddedEdges\SplitByNode\KFold=101\DecendingOrder\\'
    for i, (train_test_dict) in enumerate(
            data.split_cross_validation(data, 101, stratify=True, task=task,
                                        split_by_node=split_by_node,
                                        # reset_cross_validation_split=False,
                                        splitted_edges_dir=splitted_edges_dir,
                                        )):
        train_set = train_test_dict['train_set']
        test_set = train_test_dict['test_set']
        # for i, (train_ind, test_ind) in enumerate(skf.split(x, disease_class)):

        x_train, y_train = train_set[:, :-1], train_set[:, -1]
        x_test, y_test = test_set[:, :-1], test_set[:, -1]

        # TODO here>>-2 paragraph of code below will be move to run_node2vec with task == 'node_classification'
        # x_train, y_train = x[train_ind], disease_class[train_ind]
        # x_test, y_test = x[test_ind], disease_class[test_ind]
        # TODO here>>-10 create another variable type = dataframe whose row = edges
        if edges_as_data:
            train_instances = []
            test_instances = []

            for node1, node2, _ in train_set:
                edge_instance = f'{node1}_{node2}'
                train_instances.append(edge_instance)
            for node1, node2, _ in test_set:
                edge_instance = f'{node1}_{node2}'
                test_instances.append(edge_instance)

        x = f'C:\\Users\\Anak\\PycharmProjects\\recreate_gene_disease\\Data\\processed\\LinkPrediction\\GeneDiseaseProject\\copd\\PhenotypeGeneDisease\\PGDP\\Node2Vec\\UnweightedEdges\\NoAddedEdges\\SplitByNode\\KFold=101\\DecendingOrder\\{i}\\dim64_walk_len30_num_walks200_window10.txt'
        from Sources.Preparation.Features import get_instances_with_features
        data_with_features_dict[i] = get_instances_with_features(
            use_saved_emb_file=True,
            path_to_saved_emb_file=x,
            normalized_weighted_edges=False
        )

        # TODO here>>-9 validate that slicing dataframe output desired outcome.
        x_train_with_features, x_test_with_features = \
            data_with_features_dict[i].loc[
                train_instances], \
            data_with_features_dict[i].loc[
                test_instances]

        # x_train_with_features, x_test_with_features = data_with_features.loc[
        #                                                   x_train], \
        #                                               data_with_features.loc[
        #                                                   x_test]
        y_train = y_train.astype(float)
        y_train = y_train.astype(int)
        y_test = y_test.astype(float)
        y_test = y_test.astype(int)

        # # garantee that
        # assert y_train[0] == x_train_with_features.shape[0], 'Dataleakage! x_test and x_train contain the same nonexisted_edges '
        # assert y_test[0] == x_test_with_features.shape[0], 'Dataleakage! x_test and x_train contain the same nonexisted_edges '

        # TODO here>>-8 this is a quickfix to run the model => Goal is to see whether there are any more error other thna one cause by input dataset
        ## does this effect the outcome in unpredicatable way
        x_train_with_features = x_train_with_features.drop_duplicates()
        x_test_with_features = x_test_with_features.drop_duplicates()


        y_train_pred, y_train_pred_proba, y_test_pred, y_test_pred_proba = run_clf_for_each_fold(
            x_train_with_features,
            x_test_with_features,
            y_train, y_test)

        # TODO paragraph of code below is moved to run_svm_for_each_fold()
        ## Goal is to make the function reuseable by different classifer( which is library/function dependent)
        # # TODO debugging paramter of clr to make it binary clasiifier
        # # train model
        # clf = svm.SVC(gamma='scale', decision_function_shape='ovr',
        #               probability=True)
        # # clf = svm.SVC(kernel='linear', probability=True)  # her
        #
        # clf.fit(x_train_with_features, y_train)
        #
        # ## train set
        # ### use decision_function
        # y_train_pred = clf.decision_function(x_train_with_features).argmax(1)
        # y_train_pred_proba = clf.decision_function(x_train_with_features)
        #
        # ## test set
        # ### use decision function
        # y_test_pred = clf.decision_function(x_test_with_features).argmax(1)
        # y_test_pred_proba = clf.decision_function(x_test_with_features)

        if not show_only_average_result:
            print(f"================training cv ={i}==================")

        report_final_train_performance_report_np, columns_of_performance_metric, indices_of_performance_metric = report_performance(
            y_train, y_train_pred, y_train_pred_proba, np.unique(y_train),
            verbose=not show_only_average_result,
            plot=False, return_value_for_cv=True)

        if not show_only_average_result:
            print(f"================test cv ={i}==================")

        report_final_test_performance_report_np, columns_of_performance_metric, indices_of_performance_metric = report_performance(
            y_test, y_test_pred, y_test_pred_proba, np.unique(y_test),
            verbose=not show_only_average_result,
            plot=False, return_value_for_cv=True)

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

    report_average_performance_of_cross_validation(k_fold,
                                                   sum_final_train_performance_report_np,
                                                   sum_final_test_performance_report_np,
                                                   columns_of_performance_metric,
                                                   indices_of_performance_metric)
