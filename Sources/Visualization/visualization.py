import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
from sklearn.metrics import auc
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import plot_roc_curve


def visualize_roc_curve(fpr, tpr, roc_auc):
    """
    refer back to the following link : https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
    @param fpr: type = dict
    @param tpr: type = dict
    @param roc_auc: type = dict
    @return:
    """

    plt.figure()
    lw = 2
    colors = ['red', 'blue', 'black', 'green', 'yellow']
    for i in range(len(list(fpr.keys()))):
        plt.plot(fpr[i], tpr[i], color=colors[i],
                 lw=lw,
                 label=f'Class = {i}; ROC curve (area = {roc_auc[i][0]: .2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")  # todo show legen of all class_roc_curve
    plt.show()


def visualize_roc_curve_with_cross_validation_1(clf, index_of_curve,
                                                x_with_features, y, mean_fpr,
                                                tprs, aucs, ax):

    ## train set
    # TODO follow my note documentatation
    viz = plot_roc_curve(clf,
                         x_with_features,
                         y,
                         name='ROC fold {}'.format(index_of_curve),
                         alpha=0.3,
                         lw=1,
                         ax=ax)  # How do i know if it use decision_funciton, predict proba or other?

    ### visualized roc cove of train set
    interp_tpr = interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)
    return ax, tprs, aucs


def visualize_roc_curve_with_cross_validation_2(ax, mean_fpr, tprs, aucs):
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title="Receiver operating characteristic example")
    ax.legend(loc="lower right")
    plt.show()


def visualize_confusion_matrix(classifier, X_test, y_test, class_names):
    np.set_printoptions(precision=2)
    # Plot non-normalized confusion matrix
    titles_options = [("Confusion matrix, without normalization", None),
                      ("Normalized confusion matrix", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(classifier, X_test, y_test,
                                     display_labels=class_names,
                                     cmap=plt.cm.Blues,
                                     normalize=normalize)
        disp.ax_.set_title(title)

        print(title)
        print(disp.confusion_matrix)
