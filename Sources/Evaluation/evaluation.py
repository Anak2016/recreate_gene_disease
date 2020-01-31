import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix # todo check this next
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report
from Sources.Visualization.visualization import visualize_roc_curve
import copy


def report_performance(  y_true, y_pred, y_score, labels , plot = False, return_value = False):
    """

    @param y_true:
    @param y_pred:
    @param y_score:
    @param labels:
    @param plot:
    @param return_value:
    @return: desc = return value if return_value is True;
           report_final_performance_report_np: type = numpy
           columns_of_performance_metric:  type = list: desc = list of performance metrics name
    """
    report_sklearn_classification_report = classification_report(y_true, y_pred, labels, output_dict=True)
    report_dict = copy.deepcopy(report_sklearn_classification_report)
    del report_dict['accuracy']
    report_dict['accuracy'] = {'precision': None, 'recall': None,
                               'f1-score': report_sklearn_classification_report['accuracy'], 'support': None}
    report_df = pd.DataFrame(report_dict).round(2).transpose()
    # print(report_df)
    # print(report_sklearn_classification_report)


    # show support class and predicted classes
    ## note: np.unique output sorted value
    supported_class_np, supported_class_freq_np = np.unique(y_true, return_counts=True)
    predicted_class_np, predicted_class_freq_np = np.unique(y_pred, return_counts=True)

    support_class_df = pd.DataFrame(supported_class_freq_np, columns = ['support'], index=supported_class_np)
    predicted_class_df = pd.DataFrame(predicted_class_freq_np, columns = ['predicted'], index=predicted_class_np)

    report_support_pred_class = pd.concat([support_class_df, predicted_class_df], axis=1)
    # print(report_support_pred_class)

    if plot:
        fpr, tpr, roc_auc = get_roc_curve(pd.get_dummies(y_true).to_numpy(), y_score, labels.shape[0])
        visualize_roc_curve(fpr,tpr,roc_auc )

    if return_value:
        return report_performance_for_cv(report_df,report_support_pred_class)

def report_performance_for_cv(report_df,report_support_pred_class):
    # create mask for report_support_pred_class that have the smae index as report_df.index (fill with nan)
    na_np = np.tile(np.nan, (
        report_df.shape[0], report_support_pred_class.shape[1]))

    report_support_pred_class_df = pd.DataFrame(na_np, index=report_df.index,
                         columns=report_support_pred_class.columns)
    report_support_pred_class_df.loc[:report_support_pred_class.shape[0],
    :] = report_support_pred_class.values
    # print(na_df)

    report_support_pred_class_df_without_predicted = report_support_pred_class_df[['predicted']]

    merged_report_df = pd.concat([report_df, report_support_pred_class_df_without_predicted], axis= 1)
    # merged_report_df = report_df.merge(report_support_pred_class, how='outer', on=['support'], copy=False, right_index=True)
    print(merged_report_df)
    return merged_report_df.to_numpy(), merged_report_df.columns, merged_report_df.index

def get_roc_curve(y_true, y_score, n_classes ):
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

    return  fpr, tpr, roc_auc

