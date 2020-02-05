import networkx as nx
import numpy as np

def apply_edges_adding_strategies( add_qualified_edges, G,edges_with_weight, number_of_added_edges):

    # TODO add arg.strategy to function argument
    # assert number_of_added_edges is not None,"when add_qualified_edges is not None, it implies that some edges is expected to be added"
    if add_qualified_edges == 'top_k':
        return select_edges_with_top_alpha(G,edges_with_weight, number_of_added_edges)
    else:
        raise ValueError('only top_k strategy is implemented')

def select_edges_with_top_alpha(G,edges_with_weight, number_of_added_edges):
    """
    @param G: nx.Graph() object
    @param edges: type np.ndarray or list: shape = (-1, 3) where last columns is weight
    @return: selected_edges_with_weight; type = np;
        desc:
            > for use_weighted_edges = False; weight of edges are expected to be all 1's
    """
    edges_without_weight = edges_with_weight[:,:2] # exclude weight

    # remove selfloop before calculate jaccard_coefficient
    G.remove_edges_from(nx.selfloop_edges(G))
    assert len(list(nx.selfloop_edges(G))) == 0, 'there are self loop'

    jaccoff = np.array(list(nx.jaccard_coefficient(G,edges_without_weight)))

    # select only none zero value edges
    weighted_value = jaccoff[:, 2]
    non_zero_weighted_value_ind = weighted_value.nonzero()

    ## use non_zero weighted_value to select row index from jaccoff
    ## note: I decided (for now ) to not change name jaccoff to reflect state of value it contains for consistency reason
    jaccoff = jaccoff[non_zero_weighted_value_ind]

    # sorted GeneDisease_jaccoff by jaccoff value
    index_sorted_by_jaccoff = np.argsort(jaccoff[:, 2])[::-1]
    jaccoff_sorted = jaccoff[index_sorted_by_jaccoff]

    selected_edges = jaccoff_sorted[:number_of_added_edges, :2]

    # select edges_with_weight that have edges in selected_edges
    get_match_edges = lambda x: x[:2].tolist() in selected_edges[:, :2].tolist()
    selected_edges_with_weight_boolean = np.apply_along_axis(get_match_edges, 1, edges_with_weight)

    selected_edges_with_weight = edges_with_weight[selected_edges_with_weight_boolean]

    # validation step before returning value
    assert selected_edges_with_weight.shape[0] == selected_edges.shape[0], "selected_edges_with_weight is not implemented correctly "

    nodes_of_added_edges = selected_edges_with_weight[:,
                           :2].flatten()
    assert np.array([True for i in nodes_of_added_edges if
                     i in list(
                         G.nodes)]).all(), "nodes of edges to be added involved nodes that do not exist within the original graph"

    # assert data.is_disease2disease_edges_added_to_graph(
    #     disease2disease_edges_to_be_added,
    #     use_outside_graph=True), "added_qualified_edges is True, but no disease2disease are added "

    return selected_edges_with_weight
