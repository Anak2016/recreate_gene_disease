import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv  # noqa

from Sources.Evaluation import report_performance
from Utilities.saver import save2file


def run_train_gcn(model, train_mask_bool, train_y, optimizer,
                  data_with_features, edges, edges_weight):
    """

    @param mask: ind of splitted data
    @return:
    """

    def train():
        model.train()
        optimizer.zero_grad()
        # HERE check out does output of model returns? (the same order or not?)
        logit = model(data_with_features, edges, edges_weight)
        F.nll_loss(
            logit[train_mask_bool],
            train_y).backward()
        optimizer.step()
        return model

    return train()


def run_test_gcn(model, mask_bool, y, optimizer, data_with_features,
                 edges, edges_weight):
    def performance_per_epoch(logits):
        train_test_pred = []
        train_test_accs = []
        train_test_pred_prob = []
        for mask, label in zip(mask_bool, y):
            pred_proba = logits[mask]
            pred = pred_proba.max(1)[1]
            acc = pred.eq(label).sum().item() / mask.sum().item()

            train_test_accs.append(acc)
            train_test_pred.append(pred)
            train_test_pred_prob.append(pred_proba)
        return train_test_accs, train_test_pred, train_test_pred_prob

    @torch.no_grad()
    def test():
        model.eval()
        logits, accs = model(data_with_features, edges, edges_weight), []
        return performance_per_epoch(logits)

    return test()


def run_gcn(data, data_with_features, split, task):
    (x_train, y_train), (x_test, y_test) = data.split_train_test(split,
                                                                 stratify=True,
                                                                 task=task,
                                                                 is_input_numpy=False)

    def convert_to_numpy_for_indexing(x):
        return x.to_numpy().reshape(-1)

    x_train_np = convert_to_numpy_for_indexing(x_train)
    x_test_np = convert_to_numpy_for_indexing(x_test)

    # ==This is only for disease data, data_with_features contain both disease and gene
    # boolean_np = np.zeros(data.diseases_np.shape[0])
    # for i in list(x_train.index):
    #     boolean_np[i] = 1
    #
    # train_mask_bool = (boolean_np == 1)
    # test_mask_bool = (boolean_np == 0)
    nodes_name_in_order = np.array(list(data_with_features.index))
    nodes_name_index = np.arange(nodes_name_in_order.shape[0])

    def convert_index_list_to_boolean(index_list):
        """

        @param index_list: type = np. is a list of index name, in this case it is nodes' name
        @return:
        """
        boolean_np = np.zeros(nodes_name_index.shape[0])
        for i in list(index_list.index):
            boolean_np[i] = 1
        return boolean_np

    train_boolean_np = convert_index_list_to_boolean(x_train)
    test_boolean_np = convert_index_list_to_boolean(x_test)

    train_mask_bool = (train_boolean_np == 1)
    test_mask_bool = (test_boolean_np == 1)

    assert sum(train_mask_bool) + sum(test_mask_bool) == data.diseases_np.shape[
        0]

    # x_train_with_features, x_test_with_features = data_with_features.loc[
    #                                                   x_train_np], \
    #                                               data_with_features.loc[
    #                                                   x_test_np]

    def prepare_argument_data(data_with_features, y_train, y_test):
        num_feature = data_with_features.shape[1]
        num_label = np.unique(y_train).shape[0]
        edges = np.array(list(data.G.edges)).T

        get_edges_weight = lambda x: x['weight']
        edges_weight = list(map(get_edges_weight,
                                np.array(list(data.G.edges(data=True)))[:,
                                -1].tolist()))

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = Net(num_feature, num_label).to(device)
        optimizer = torch.optim.Adam([
            dict(params=model.reg_params, weight_decay=5e-4),
            dict(params=model.non_reg_params, weight_decay=0)
        ], lr=0.01)

        def convert_node_name_to_its_index(nodes_with_features):
            # convert nodes to its index
            node_num = np.unique(list(nodes_with_features.index)).shape[0]
            assert node_num == nodes_with_features.shape[0], ' '

            node_ind = np.arange(node_num)
            node2ind_dict = {}
            for val in zip(node_ind, list(nodes_with_features.index)):
                if isinstance(val[1], int):
                    node2ind_dict[str(val[1])] = val[0]
                else:
                    node2ind_dict[val[1]] = val[0]

            assert len(node2ind_dict.keys()) == len(data.G.nodes), ''
            assert np.unique(list(data.G.edges)).shape[0] == len(
                data.G.nodes), ''
            assert np.unique(list(data.G.edges)).shape[0] == \
                   nodes_with_features.shape[0], ''

            convert_node_name2ind = lambda x: node2ind_dict[x]
            edges_ind_1 = np.array(list(map(convert_node_name2ind, edges[0])))
            edges_ind_2 = np.array(list(map(convert_node_name2ind, edges[1])))
            edges_ind = np.vstack((edges_ind_1, edges_ind_2))

            return edges_ind

        edges_ind = convert_node_name_to_its_index(data_with_features)

        # convert to tensor
        data_with_features = torch.tensor(data_with_features.to_numpy()).float()
        edges_ind = torch.tensor(edges_ind).long()
        edges_weight = torch.tensor(edges_weight)
        y_train = torch.tensor(y_train).long()
        y_test = torch.tensor(y_test).long()

        return data_with_features, y_train,y_test, model, optimizer, edges_ind, edges_weight

    data_with_features_ts, y_train_ts, y_test_ts, model, optimizer, edges_ind_ts, edges_weight_ts = prepare_argument_data(
        data_with_features, y_train, y_test)

    train_acc_hist = []
    test_acc_hist = []
    for i in range(1, 500):
        model = run_train_gcn(model,
                              train_mask_bool,
                              y_train_ts, optimizer,
                              data_with_features_ts,
                              edges_ind_ts,
                              edges_weight_ts)

        (train_acc, tmp_test_acc), (y_train_pred, y_test_pred), (
        y_train_pred_proba, y_test_pred_proba) = run_test_gcn(model,
                                                              [train_mask_bool,
                                                               test_mask_bool],
                                                              [y_train_ts, y_test_ts],
                                                              optimizer,
                                                              data_with_features_ts,
                                                              edges_ind_ts,
                                                              edges_weight_ts)
        test_acc = tmp_test_acc
        log = 'Epoch: {:03d}, Train: {:.4f}, Test: {:.4f}'
        # print(log.format(i, train_acc, test_acc))
        train_acc_hist.append(train_acc)
        test_acc_hist.append(test_acc)


    def convert_to_numpy(y_train, y_train_pred, y_test_pred, y_train_pred_proba,
                         y_test_pred_proba):
        return y_train.numpy(), y_train_pred.numpy(), y_test_pred.numpy(), y_train_pred_proba.detach().numpy(), y_test_pred_proba.detach().numpy()

    y_train_np, y_train_pred_np, y_test_pred_np, y_train_pred_proba_np, y_test_pred_proba_np = convert_to_numpy(
        y_train_ts, y_train_pred, y_test_pred, y_train_pred_proba,
        y_test_pred_proba)

    # report performance of model
    print('=======training set=======')
    train_report_np, train_columns, train_index = report_performance(y_train_np,
                                                                     y_train_pred_np,
                                                                     y_train_pred_proba_np,
                                                                     np.unique(
                                                                         y_train_np),
                                                                     plot=True,
                                                                     verbose=True,
                                                                     return_value_for_cv=True)

    print('=======test set=======')
    test_report_np, test_columns, test_index = report_performance(y_test,
                                                                  y_test_pred_np,
                                                                  y_test_pred_proba_np,
                                                                  np.unique(
                                                                      y_test),
                                                                  plot=True,
                                                                  verbose=True,
                                                                  return_value_for_cv=True)

    # ========= save to file=========
    save2file(train_report_np, train_columns, train_index,
              test_report_np, test_columns, test_index)


def run_gcn_with_specified_classifier(data=None, x_with_features=None,
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
                                      use_gene_disease_graph=None,
                                      use_phenotype_gene_disease_graph=None,
                                      graph_edges_type=None,
                                      task=None, classifier_name=None,
                                      emb_type=None):
    assert classifier_name is not None, "classifier_name must be specified to avoid ambiguity"
    assert emb_type is not None, "emb_type must be specified to avoid ambiguity"
    run_gcn(data, x_with_features, split, task)


class Net(torch.nn.Module):
    def __init__(self, num_feature, num_classes):
        super(Net, self).__init__()
        # hidden2 = 32
        hidden3 = 16
        self.conv1 = GCNConv(num_feature, hidden3, cached=True)
        # self.conv2 = GCNConv(hidden2, hidden3, cached=True)
        self.conv3 = GCNConv(hidden3, num_classes, cached=True)

        self.reg_params = self.conv1.parameters()
        self.non_reg_params = self.conv3.parameters()

    def forward(self, x, edge_index, edge_weight):
        # edge_weight = edge_weight.type(torch.double)
        x = F.relu(self.conv1(x.float(), edge_index.long(), edge_weight))
        x = F.dropout(x, training=self.training)
        # x = self.conv2(x.float(), edge_index.long(), edge_weight)
        # x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x.float(), edge_index.long(), edge_weight)
        return F.log_softmax(x, dim=1)
