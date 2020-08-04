import numpy as np
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential

from Sources.Evaluation.evaluation import report_performance
from Sources.Evaluation.evaluation import run_clf_using_cross_validation


# def run_neural_network_for_each_fold():
#
#     raise ValueError('run_neural_network_for_each_fold() is not yet implemented')
from Utilities.saver import save2file


def run_neural_network_using_cross_validation(data,
                                              data_with_features_with_dict,
                                              k_fold,
                                              split_by_node,
                                              show_only_average_result=False,
                                              save_report_performance=None,
                                              report_performance_file_path=None,
                                              task=None,
                                              splitted_edges_dir=None
                                              ):
    if task == 'link_prediction':
        # splitted_edges_dir = r'C:\Users\Anak\PycharmProjects\recreate_gene_disease\Data\processed\LinkPrediction\GeneDiseaseProject\copd\PhenotypeGeneDisease\PGDP\Node2Vec\UnweightedEdges\NoAddedEdges\\'
        # if not split_by_node:
        #     splitted_edges_dir = splitted_edges_dir + f'SplitByEdge\\'
        # else:
        #     splitted_edges_dir = splitted_edges_dir + f'SplitByNode\\'
        splitted_edges_file_path = splitted_edges_dir + f'\KFold={k_fold}\\splitted_edges.bin'

        # TODO code below run_clf_using_cross_validation has been moved to cross_validation()
        run_clf_using_cross_validation(data, data_with_features_with_dict,
                                       k_fold,
                                       # show_only_average_result=show_only_average_result,
                                       show_only_average_result=False,
                                       save_report_performance=save_report_performance,
                                       report_performance_file_path=report_performance_file_path,
                                       run_clf_for_each_fold=run_neural_network_for_each_fold,
                                       task=task,
                                       edges_as_data=True,
                                       splitted_edges_dir=splitted_edges_dir,
                                       split_by_node=split_by_node
                                       )
    else:
        raise ValueError('NN is only implemented for link_prediction')
    # stratifiedKFold


# def run_neural_network_for_each_fold(x_train, y_train, x_test, y_test):
# TODO return 4 values ( look at mark(a)) and figure out where to put report_performance eg. inside or outside of ruN-neural_network_for_each_Fold
# def run_neural_network_for_each_fold(x_train,
#                                      y_train ,
#                                      x_test,
#                                      y_test):
def run_neural_network_for_each_fold(x_train,
                                     x_test,
                                     y_train,
                                     y_test):
    epochs = 3

    feat_dim = x_train.shape[1]

    # define the keras model
    print(f'feat_dim={feat_dim}')
    # exit()
    model = Sequential()
    model.add(Dense(feat_dim, input_dim=feat_dim, activation='relu'))
    model.add(Dense(int(feat_dim / 2), activation='relu'))
    model.add(Dropout(0.5, seed=69))
    # model.add(Dense(feat_dim/4, input_dim=feat_dim/4, activation = 'relu'))
    model.add(Dense(int(feat_dim / 4), activation='relu'))
    # model.add(Dense(feat_dim/8, input_dim=feat_dim/8, activation = 'relu'))
    model.add(Dropout(0.5, seed=69))
    # model.add(Dense(2, activation='sigmoid'))
    # model.add(Dense(2, activation='softmaxk'))
    model.add(Dense(1, activation='sigmoid'))

    # model.add(Dense(1, activation='sigmoid'))

    import tensorflow as tf
    from keras import backend as K

    def auroc(y_true, y_pred):
        auc = tf.metrics.AUC(y_true, y_pred)[1]
        K.get_session().run(tf.local_variables_initializer())
        return auc

    # Build Model...
    # model.compile(loss='binary_crossentropy', optimizer='adam',
    #               metrics=['accuracy', auroc])
    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

    # fit the keras model on the dataset

    model.fit(x_train, y_train, epochs=epochs, batch_size=10)

    # # TODO validate that acc and auc below output correct value.
    # # evaluate the keras model using training set
    # _, accuracy, auc = model.evaluate(x_train, y_train)
    # print('training Accuracy: %.2f' % (accuracy * 100))
    # print('training aUC: %.2f' % (auc * 100))
    # # evaluate the keras model using test set
    # _, accuracy, auc = model.evaluate(x_test, y_test)
    # print('test Accuracy: %.2f' % (accuracy * 100))
    # print('test aUC: %.2f' % (auc * 100))

    y_label_1_train_pred_proba = model.predict(
        x_train)  # this implies that if the value is more than 0.5 it will be assigned to 1
    y_label_0_train_pred_proba = 1 - y_label_1_train_pred_proba  # opposite to y_label_1_test_pred_proba => if value > 0.5, predict label 0
    y_train_pred_proba = np.concatenate(
        [y_label_0_train_pred_proba, y_label_1_train_pred_proba], axis=1)
    y_train_pred = y_train_pred_proba.argmax(axis=1)
    # y_train_pred = np.where(y_label_0_train_pred_proba < 0.5, 0, 1).flatten()

    y_label_1_test_pred_proba = model.predict(
        x_test)  # this implies that if the value is more than 0.5 it will be assigned to 1
    y_label_0_test_pred_proba = 1 - y_label_1_test_pred_proba  # opposite to y_label_1_test_pred_proba => if value > 0.5, predict label 0
    y_test_pred_proba = np.concatenate(
        [y_label_0_test_pred_proba, y_label_1_test_pred_proba], axis=1)
    y_test_pred = y_test_pred_proba.argmax(axis=1)
    # y_test_pred = np.where(y_label_0_test_pred_proba < 0.5, 0, 1).flatten()

    # report_performance(y_test, y_test_pred, y_test_pred_proba,
    #                    np.unique(y_test), plot=True, verbose=True)

    return y_train_pred, y_train_pred_proba, y_test_pred, y_test_pred_proba

    # report_performance(y_test, y_test_pred, y_test_pred_proba[:,0],
    #                    np.unique(y_test), plot=True, verbose=True)

    # import pandas as pd
    # y_test_pred_one_hot = pd.get_dummies(y_test_pred).to_numpy()
    # y_test_one_hot = pd.get_dummies(y_test).to_numpy()
    #
    # report_performance(y_test_one_hot, y_test_pred_one_hot, y_test_pred_proba,
    #                    np.unique(y_test), plot=True, verbose=True)


def run_neural_network_using_train_test_split(data, x_with_features,
                                              task, splitted_edges_dir,
                                              split=None,
                                              split_by_node=None):
    # =====================
    # ==parameter setting
    # =====================
    # epochs = 1
    # epochs = 10
    # epochs = 150
    # TODO make sure that x_with_features have row = edges and columns = features dimension

    train_set_np, test_set_np = data.split_train_test(split,
                                                      stratify=True,
                                                      task=task,
                                                      splitted_edges_dir=splitted_edges_dir,
                                                      split_by_node=split_by_node
                                                      )

    def get_train_test_set():
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

        return (y_train, x_train, x_train_ind), (y_test, x_test, x_test_ind)
    (y_train, x_train, x_train_ind), (y_test, x_test, x_test_ind) = get_train_test_set()

    # x_train = np.concatenate([x_with_features.loc[x_train_ind[:, 0]],
    #                           x_with_features.loc[x_train_ind[:, 1]]], axis=1)
    #
    # x_test = np.concatenate([x_with_features.loc[x_test_ind[:, 0]],
    #                          x_with_features.loc[x_test_ind[:, 1]]], axis=1)
    assert x_train.shape[0] == x_train_ind.shape[0], ''

    # TODO pass the correct value in
    y_train_pred, y_train_pred_proba, y_test_pred, y_test_pred_proba = run_neural_network_for_each_fold(
        x_train, x_test, y_train, y_test)

    print(f"================training set==================")

    report_final_train_performance_report_np, columns_of_performance_metric, indices_of_performance_metric = report_performance(
        y_train, y_train_pred, y_train_pred_proba, np.unique(y_train),
        verbose=True,
        plot=False, return_value_for_cv=True)

    print(f"================test set==================")

    report_final_test_performance_report_np, columns_of_performance_metric, indices_of_performance_metric = report_performance(
        y_test, y_test_pred, y_test_pred_proba, np.unique(y_test),
        verbose=True,
        plot=False, return_value_for_cv=True)


    save2file(report_final_train_performance_report_np, columns_of_performance_metric, indices_of_performance_metric,
              report_final_test_performance_report_np, columns_of_performance_metric, indices_of_performance_metric)

    # # x_train,y_train, x_test,y_test have to be used as row index to get access to x_with_features
    # x_train_ind, y_train = train_set_np[:, :2], train_set_np[:, -1].astype(
    #     float)
    # x_test_ind, y_test = test_set_np[:, :2], test_set_np[:, -1].astype(float)
    #
    # # from keras.utils import to_categorical
    # # y_train = to_categorical(y_train.astype(int))
    # # y_test = to_categorical(y_test.astype(int))
    #
    # import numpy as np
    # x_train = np.concatenate([x_with_features.loc[x_train_ind[:, 0]],
    #                           x_with_features.loc[x_train_ind[:, 1]]], axis=1)
    # x_test = np.concatenate([x_with_features.loc[x_test_ind[:, 0]],
    #                          x_with_features.loc[x_test_ind[:, 1]]], axis=1)
    # assert x_train.shape[0] == x_train_ind.shape[0], ''
    #
    # feat_dim = x_train.shape[1]
    #
    # # define the keras model
    # model = Sequential()
    # model.add(Dense(feat_dim, input_dim=feat_dim, activation='relu'))
    # model.add(Dense(int(feat_dim / 2), activation='relu'))
    # model.add(Dropout(0.5, seed=69))
    # # model.add(Dense(feat_dim/4, input_dim=feat_dim/4, activation = 'relu'))
    # model.add(Dense(int(feat_dim / 4), activation='relu'))
    # # model.add(Dense(feat_dim/8, input_dim=feat_dim/8, activation = 'relu'))
    # model.add(Dropout(0.5, seed=69))
    # # model.add(Dense(2, activation='sigmoid'))
    # # model.add(Dense(2, activation='softmaxk'))
    # model.add(Dense(1, activation='sigmoid'))
    #
    # # model.add(Dense(1, activation='sigmoid'))
    #
    # import tensorflow as tf
    # from keras import backend as K
    #
    # def auroc(y_true, y_pred):
    #     auc = tf.metrics.auc(y_true, y_pred)[1]
    #     K.get_session().run(tf.local_variables_initializer())
    #     return auc
    #
    # # Build Model...
    # model.compile(loss='binary_crossentropy', optimizer='adam',
    #               metrics=['accuracy', auroc])
    #
    # # fit the keras model on the dataset
    # model.fit(x_train, y_train, epochs=epochs, batch_size=10)
    #
    # # evaluate the keras model using training set
    # _, accuracy, auc = model.evaluate(x_train, y_train)
    # print('training Accuracy: %.2f' % (accuracy * 100))
    # print('training aUC: %.2f' % (auc * 100))
    #
    # # evaluate the keras model using test set
    # _, accuracy, auc = model.evaluate(x_test, y_test)
    # print('test Accuracy: %.2f' % (accuracy * 100))
    # print('test aUC: %.2f' % (auc * 100))
    #
    # y_label_1_test_pred_proba = model.predict(x_test) # this implies that if the value is more than 0.5 it will be assigned to 1
    # y_label_0_test_pred_proba = 1 - y_label_1_test_pred_proba # opposite to y_label_1_test_pred_proba => if value > 0.5, predict label 0
    # y_test_pred_proba = np.concatenate(
    #     [y_label_0_test_pred_proba, y_label_1_test_pred_proba], axis=1)
    # y_test_pred = np.where(y_label_0_test_pred_proba < 0.5, 0, 1).flatten()
    #
    # report_performance(y_test, y_test_pred, y_test_pred_proba,
    #                    np.unique(y_test), plot=True, verbose=True)
    #
    # # report_performance(y_test, y_test_pred, y_test_pred_proba[:,0],
    # #                    np.unique(y_test), plot=True, verbose=True)
    #
    # # import pandas as pd
    # # y_test_pred_one_hot = pd.get_dummies(y_test_pred).to_numpy()
    # # y_test_one_hot = pd.get_dummies(y_test).to_numpy()
    # #
    # # report_performance(y_test_one_hot, y_test_pred_one_hot, y_test_pred_proba,
    # #                    np.unique(y_test), plot=True, verbose=True)


def run_neural_network(data,
                       x_with_features,
                       cross_validation,
                       split,
                       k_fold,
                       task,
                       split_by_node,
                        splitted_edges_dir
                       ):

    if cross_validation:
        # raise ValueError('not yet implemented')
        assert isinstance(x_with_features,
                          dict), 'x_with_features is expected to be dict where key = number of fold and value = x_with_feature given number of fold = key'

        # if task == 'link_prediction':
        #     spliited_edges_file_path =
        x_with_features_dict = x_with_features

        run_neural_network_using_cross_validation(data, x_with_features_dict,
                                                  k_fold,
                                                  split_by_node,
                                                  show_only_average_result=True,
                                                  save_report_performance=False,
                                                  task=task,
                                                  splitted_edges_dir=splitted_edges_dir)
    else:
        # if split_by_node:
        #     raise ValueError('not yet implemented')
        run_neural_network_using_train_test_split(data, x_with_features, task,splitted_edges_dir,
                                                  split=split,split_by_node=split_by_node)
