from typing import List

import networkx as nx
import numpy as np



def _get_nodes_prob_dist(jaccoff: np.ndarray ) -> List:
    sum_val = jaccoff[:, 2].astype(float).sum()
    normalization = lambda x: x / sum_val
    prob_dist = list(map(normalization, jaccoff[:, 2].astype(float)))
    return prob_dist

def _get_nodes_inverse_prob_dist(jaccoff: np.ndarray) -> List:
    # validate  that this inverse implementation is correct
    inverse_func = lambda x: 1/x
    inverse =  list(map(inverse_func, list(map(float, jaccoff[:,2])) ))
    sum_val = sum(inverse)
    normalization = lambda x: x / sum_val
    prob_dist = list(map(normalization, inverse))
    return prob_dist


def select_edges_with_top_bottom_alpha(G, edges_with_weight,
                                       number_of_added_edges, random):
    """
    @param G: nx.Graph() object
    @param edges: type np.ndarray or list: shape = (-1, 3) where last columns is weight
    @return: selected_edges_with_weight; type = np;
        desc:
            > for use_weighted_edges = False; weight of edges are expected to be all 1's
    """
    assert isinstance(random, bool), f'random must have type = boolean '

    edges_without_weight = edges_with_weight[:, :2]  # exclude weight

    # remove selfloop before calculate jaccard_coefficient
    G.remove_edges_from(nx.selfloop_edges(G))
    assert len(list(nx.selfloop_edges(G))) == 0, 'there are self loop'

    jaccoff = np.array(list(nx.jaccard_coefficient(G, edges_without_weight)))

    # select only none zero value edges
    weighted_value = jaccoff[:, 2]
    non_zero_weighted_value_ind = weighted_value.nonzero()

    ## use non_zero weighted_value to select row index from jaccoff
    ## note: I decided (for now ) to not change name jaccoff to reflect state of value it contains for consistency reason
    jaccoff = jaccoff[non_zero_weighted_value_ind]

    # sorted GeneDisease_jaccoff by jaccoff value
    index_sorted_by_jaccoff = np.argsort(jaccoff[:, 2])[::-1]
    jaccoff_sorted = jaccoff[index_sorted_by_jaccoff]


    if random:
        half_top = int(number_of_added_edges/2)
        half_down = number_of_added_edges - half_top

        selected_top_ind = np.random.choice(jaccoff.shape[0], half_top, p=_get_nodes_prob_dist(jaccoff))
        selected_bottom_ind = np.random.choice(jaccoff.shape[0], half_down, p=_get_nodes_inverse_prob_dist(jaccoff))

        selected_ind = selected_top_ind.tolist()
        selected_ind.extend( selected_bottom_ind)
        selected_edges = jaccoff_sorted[selected_ind, :2]
    else:
        selected_edges = jaccoff_sorted[:number_of_added_edges, :2]
    #=====================
    #==version 1
    #=====================

    # select edges_with_weight that have edges in selected_edges
    # BUG: I am not sure why the code below some time produce error,
    #  try to run the whole function multiple time, sometimes, the error is raised
    # get_match_edges = lambda x: x[:2].tolist() in selected_edges[:, :2].tolist()
    # selected_edges_with_weight_boolean = np.apply_along_axis(get_match_edges, 1,
    #                                                          edges_with_weight)
    # ================== end of version 1 ==============

    #=====================
    #==version 2
    #=====================

    # test that jaccoff and edges_with_weight have the exact same set of edges
    # note: if the code paragraph below end up raising error in the future. just use the b-u-g code above.
    #   > This is good enough at the time of implementation and I don't expect to need anything more in the near future

    tmp = set(list(map(tuple, edges_with_weight[:, :2].tolist())))
    tmp1 = set(list(map(tuple, jaccoff[:, :2].tolist())))
    assert len(tmp.intersection(tmp1)) == len(tmp) and len(tmp) == len(tmp1), 'not all selected edges are in edges_with_edges'

    # note: This is really error prone to positional mismatch. Even error occur, no error will be raise
    selected_edges = np.array(selected_edges)
    edges_weight = np.ones(selected_edges.shape[0]).reshape(-1,1)
    selected_edges_with_weight = np.hstack(( selected_edges, edges_weight))
    # selected_edges_with_weight = selected_edges
    # ================== end of version 2 ==============
    # validation step before returning value
    assert selected_edges_with_weight.shape[0] == selected_edges.shape[
        0], "selected_edges_with_weight is not implemented correctly "

    nodes_of_added_edges = selected_edges_with_weight[:,
                           :2].flatten()
    assert np.array([True for i in nodes_of_added_edges if
                     i in list(
                         G.nodes)]).all(), "nodes of edges to be added involved nodes that do not exist within the original graph"

    # assert data.is_disease2disease_edges_added_to_graph(
    #     disease2disease_edges_to_be_added,
    #     use_outside_graph=True), "added_qualified_edges is True, but no disease2disease are added "

    return selected_edges_with_weight


def select_edges_with_shared_nodes_random(data, G, edges_with_weight,
                                          number_of_added_edges, random):
    """
    @param G: nx.Graph() object
    @param edges: type np.ndarray or list: shape = (-1, 3) where last columns is weight
    @return: selected_edges_with_weight; type = np;
        desc:
            > for use_weighted_edges = False; weight of edges are expected to be all 1's
    """
    assert isinstance(random, bool), f'random must have type = boolean '

    edges_without_weight = edges_with_weight[:, :2]  # exclude weight
    # non_zero_weighted_value_ind = edges_with_weight.nonzero()

    # remove selfloop before calculate jaccard_coefficient
    G.remove_edges_from(nx.selfloop_edges(G))
    assert len(list(nx.selfloop_edges(G))) == 0, 'there are self loop'

    jaccoff = np.array(list(nx.jaccard_coefficient(G, edges_without_weight)))

    # select only none zero value edges
    weighted_value = jaccoff[:, 2]
    non_zero_weighted_value_ind = weighted_value.nonzero()

    ## use non_zero weighted_value to select row index from jaccoff
    ## note: I decided (for now ) to not change name jaccoff to reflect state of value it contains for consistency reason
    jaccoff = jaccoff[non_zero_weighted_value_ind]

    selected_ind = np.random.choice(edges_without_weight.shape[0], number_of_added_edges)
    selected_edges = edges_without_weight[selected_ind]

    #=====================
    #==version 1
    #=====================

    # select edges_with_weight that have edges in selected_edges
    # BUG: I am not sure why the code below some time produce error,
    #  try to run the whole function multiple time, sometimes, the error is raised
    # get_match_edges = lambda x: x[:2].tolist() in selected_edges[:, :2].tolist()
    # selected_edges_with_weight_boolean = np.apply_along_axis(get_match_edges, 1,
    #                                                          edges_with_weight)
    # ================== end of version 1 ==============

    #=====================
    #==version 2
    #=====================

    # test that jaccoff and edges_with_weight have the exact same set of edges
    # note: if the code paragraph below end up raising error in the future. just use the b-u-g code above.
    #   > This is good enough at the time of implementation and I don't expect to need anything more in the near future

    tmp = set(list(map(tuple, edges_with_weight[:, :2].tolist())))
    tmp1 = set(list(map(tuple, jaccoff[:, :2].tolist())))
    assert len(tmp.intersection(tmp1)) == len(tmp) and len(tmp) == len(tmp1), 'not all selected edges are in edges_with_edges'

    # note: This is really error prone to positional mismatch. Even error occur, no error will be raise
    selected_edges = np.array(selected_edges)
    edges_weight = np.ones(selected_edges.shape[0]).reshape(-1,1)

    selected_edges_with_weight = np.hstack(( selected_edges, edges_weight))
    # ================== end of version 2 ==============

    # validation step before returning value
    assert selected_edges_with_weight.shape[0] == selected_edges.shape[
        0], "selected_edges_with_weight is not implemented correctly "

    nodes_of_added_edges = selected_edges[:,
                           :2].flatten()
    # nodes_of_added_edges = selected_edges_with_weight[:,
    #                        :2].flatten()
    assert np.array([True for i in nodes_of_added_edges if
                     i in list(
                         G.nodes)]).all(), "nodes of edges to be added involved nodes that do not exist within the original graph"

    # assert data.is_disease2disease_edges_added_to_graph(
    #     disease2disease_edges_to_be_added,
    #     use_outside_graph=True), "added_qualified_edges is True, but no disease2disease are added "

    return selected_edges_with_weight


def select_edges_with_all_nodes_random(data, G, edges_with_weight,
                                       number_of_added_edges, random):
    """
    @param G: nx.Graph() object
    @param edges: type np.ndarray or list: shape = (-1, 3) where last columns is weight
    @return: selected_edges_with_weight; type = np;
        desc:
            > for use_weighted_edges = False; weight of edges are expected to be all 1's
    """
    assert isinstance(random, bool), f'random must have type = boolean '

    np.random.seed(100)
    from itertools import product
    np.random.shuffle(data.diseases_np)
    selected_edges = []
    # Note: this way of selected edges from all node pair forcefully assinged weight as 1 to all edge pair
    for ind,i in enumerate(product(data.diseases_np, data.diseases_np)):
        if ind < number_of_added_edges:
            selected_edges.append(i)

    selected_edges = np.array(selected_edges)
    edges_weight = np.ones(selected_edges.shape[0]).reshape(-1,1)

    selected_edges_with_weight = np.hstack(( selected_edges, edges_weight))

    nodes_of_added_edges = selected_edges_with_weight[:,
                           :2].flatten()
    assert np.array([True for i in nodes_of_added_edges if
                     i in list(
                         G.nodes)]).all(), "nodes of edges to be added involved nodes that do not exist within the original graph"

    # assert data.is_disease2disease_edges_added_to_graph(
    #     disease2disease_edges_to_be_added,
    #     use_outside_graph=True), "added_qualified_edges is True, but no disease2disease are added "

    return selected_edges_with_weight


def apply_edges_adding_strategies(data, add_qualified_edges, G, edges_with_weight,
                                  number_of_added_edges):
    # assert number_of_added_edges is not None,"when add_qualified_edges is not None, it implies that some edges is expected to be added"
    # if add_qualified_edges == 'top_k':
    #     return select_edges_with_top_alpha(G,edges_with_weight, number_of_added_edges)
    # elif add_qualified_edges == 'bottom_k':
    #     return select_edges_with_bottom_alpha(G,edges_with_weight, number_of_added_edges)
    # else:
    #     raise ValueError('only top_k strategy is implemented')

    # HERE    raise NotImplementedError:
    #   >change from top_k_deterministic ( check all of the places that this words are replaced)
    #   > run and check random option that it works as expected
    #   >add random option to each of the select_edges_with function
    if add_qualified_edges == 'top_k':
        return select_edges_with_top_alpha(G, edges_with_weight,
                                           number_of_added_edges,
                                           random=False)
    elif add_qualified_edges == 'bottom_k_deterministic':
        return select_edges_with_bottom_alpha(G, edges_with_weight,
                                              number_of_added_edges,
                                              random=False)
    elif add_qualified_edges == 'top_bottom_k_deterministic':
        return select_edges_with_top_bottom_alpha(G, edges_with_weight,
                                                  number_of_added_edges,
                                                  random=True)
    if add_qualified_edges == 'top_k_random':
        return select_edges_with_top_alpha(G, edges_with_weight,
                                           number_of_added_edges, random=True)
    elif add_qualified_edges == 'bottom_k_random':
        return select_edges_with_bottom_alpha(G, edges_with_weight,
                                              number_of_added_edges,
                                              random=True)
    elif add_qualified_edges == 'top_bottom_k_random':
        return select_edges_with_top_bottom_alpha(G, edges_with_weight,
                                                  number_of_added_edges,
                                                  random=True)
    elif add_qualified_edges == 'shared_nodes_random':
        return select_edges_with_shared_nodes_random(data, G, edges_with_weight,
                                                     number_of_added_edges,
                                                     random=True)
    elif add_qualified_edges == 'all_nodes_random':
        return select_edges_with_all_nodes_random(data, G, edges_with_weight,
                                                  number_of_added_edges,
                                                  random=True)
    else:
        raise ValueError('only top_k strategy is implemented')




def select_edges_with_bottom_alpha(G, edges_with_weight, number_of_added_edges,
                                   random):
    """
    @param G: nx.Graph() object
    @param edges: type np.ndarray or list: shape = (-1, 3) where last columns is weight
    @return: selected_edges_with_weight; type = np;
        desc:
            > for use_weighted_edges = False; weight of edges are expected to be all 1's
    """
    assert isinstance(random, bool), f'random must have type = boolean '
    # TODO when select n edges. check if there are entry of the same values, if there are select index from it with uniform probability

    edges_without_weight = edges_with_weight[:, :2]  # exclude weight

    # remove selfloop before calculate jaccard_coefficient
    G.remove_edges_from(nx.selfloop_edges(G))
    assert len(list(nx.selfloop_edges(G))) == 0, 'there are self loop'

    jaccoff = np.array(list(nx.jaccard_coefficient(G, edges_without_weight)))

    # select only none zero value edges
    weighted_value = jaccoff[:, 2]
    non_zero_weighted_value_ind = weighted_value.nonzero()

    ## use non_zero weighted_value to select row index from jaccoff
    ## note: I decided (for now ) to not change name jaccoff to reflect state of value it contains for consistency reason
    jaccoff = jaccoff[non_zero_weighted_value_ind]

    # sorted GeneDisease_jaccoff by jaccoff value
    index_sorted_by_jaccoff = np.argsort(jaccoff[:, 2])[::-1]
    jaccoff_sorted = jaccoff[index_sorted_by_jaccoff]

    if random:
        # TODO this is not correct
        selected_ind = np.random.choice(jaccoff.shape[0], number_of_added_edges, p=_get_nodes_inverse_prob_dist(jaccoff))
        selected_edges = jaccoff_sorted[selected_ind, :2]
    else:
        selected_edges = jaccoff_sorted[:number_of_added_edges, :2]


    #=====================
    #==version 1
    #=====================

    # select edges_with_weight that have edges in selected_edges
    # BUG: I am not sure why the code below some time produce error,
    #  try to run the whole function multiple time, sometimes, the error is raised
    # get_match_edges = lambda x: x[:2].tolist() in selected_edges[:, :2].tolist()
    # selected_edges_with_weight_boolean = np.apply_along_axis(get_match_edges, 1,
    #                                                          edges_with_weight)
    # ================== end of version 1 ==============

    #=====================
    #==version 2
    #=====================

    # test that jaccoff and edges_with_weight have the exact same set of edges
    # note: if the code paragraph below end up raising error in the future. just use the b-u-g code above.
    #   > This is good enough at the time of implementation and I don't expect to need anything more in the near future

    tmp = set(list(map(tuple, edges_with_weight[:, :2].tolist())))
    tmp1 = set(list(map(tuple, jaccoff[:, :2].tolist())))
    assert len(tmp.intersection(tmp1)) == len(tmp) and len(tmp) == len(tmp1), 'not all selected edges are in edges_with_edges'

    # note: This is really error prone to positional mismatch. Even error occur, no error will be raise
    selected_edges = np.array(selected_edges)
    edges_weight = np.ones(selected_edges.shape[0]).reshape(-1,1)
    selected_edges_with_weight = np.hstack(( selected_edges, edges_weight))
    # selected_edges_with_weight = selected_edges
    # ================== end of version 2 ==============

    # validation step before returning value
    assert selected_edges_with_weight.shape[0] == selected_edges.shape[
        0], "selected_edges_with_weight is not implemented correctly "

    nodes_of_added_edges = selected_edges_with_weight[:,
                           :2].flatten()
    assert np.array([True for i in nodes_of_added_edges if
                     i in list(
                         G.nodes)]).all(), "nodes of edges to be added involved nodes that do not exist within the original graph"

    # assert data.is_disease2disease_edges_added_to_graph(
    #     disease2disease_edges_to_be_added,
    #     use_outside_graph=True), "added_qualified_edges is True, but no disease2disease are added "

    return selected_edges_with_weight


def select_edges_with_top_alpha(G, edges_with_weight, number_of_added_edges,
                                random):
    """
    @param G: nx.Graph() object
    @param edges: type np.ndarray or list: shape = (-1, 3) where last columns is weight
    @return: selected_edges_with_weight; type = np;
        desc:
            > for use_weighted_edges = False; weight of edges are expected to be all 1's
    """
    assert isinstance(random, bool), f'random must have type = boolean '

    edges_without_weight = edges_with_weight[:, :2]  # exclude weight

    # remove selfloop before calculate jaccard_coefficient
    G.remove_edges_from(nx.selfloop_edges(G))
    assert len(list(nx.selfloop_edges(G))) == 0, 'there are self loop'

    jaccoff = np.array(list(nx.jaccard_coefficient(G, edges_without_weight)))

    # select only none zero value edges
    weighted_value = jaccoff[:, 2]
    non_zero_weighted_value_ind = weighted_value.nonzero()

    ## use non_zero weighted_value to select row index from jaccoff
    ## note: I decided (for now ) to not change name jaccoff to reflect state of value it contains for consistency reason
    jaccoff = jaccoff[non_zero_weighted_value_ind]

    # sorted GeneDisease_jaccoff by jaccoff value
    index_sorted_by_jaccoff = np.argsort(jaccoff[:, 2])[::-1]
    jaccoff_sorted = jaccoff[index_sorted_by_jaccoff]


    if random:
        selected_ind = np.random.choice(jaccoff.shape[0], number_of_added_edges, p=_get_nodes_prob_dist(jaccoff))
        selected_edges = jaccoff_sorted[selected_ind, :2]
    else:
        selected_edges = jaccoff_sorted[:number_of_added_edges, :2]

    #=====================
    #==version 1
    #=====================

    # select edges_with_weight that have edges in selected_edges
    # BUG: I am not sure why the code below some time produce error,
    #  try to run the whole function multiple time, sometimes, the error is raised
    # get_match_edges = lambda x: x[:2].tolist() in selected_edges[:, :2].tolist()
    # selected_edges_with_weight_boolean = np.apply_along_axis(get_match_edges, 1,
    #                                                          edges_with_weight)
    # ================== end of version 1 ==============

    #=====================
    #==version 2
    #=====================

    # test that jaccoff and edges_with_weight have the exact same set of edges
    # note: if the code paragraph below end up raising error in the future. just use the b-u-g code above.
    #   > This is good enough at the time of implementation and I don't expect to need anything more in the near future

    tmp = set(list(map(tuple, edges_with_weight[:, :2].tolist())))
    tmp1 = set(list(map(tuple, jaccoff[:, :2].tolist())))
    assert len(tmp.intersection(tmp1)) == len(tmp) and len(tmp) == len(tmp1), 'not all selected edges are in edges_with_edges'

    # note: This is really error prone to positional mismatch. Even error occur, no error will be raise
    selected_edges = np.array(selected_edges)
    # selected_edges_with_weight = selected_edges

    edges_weight = np.ones(selected_edges.shape[0]).reshape(-1,1)

    selected_edges_with_weight = np.hstack(( selected_edges, edges_weight))

    # ================== end of version 2 ==============

    # validation step before returning value
    assert selected_edges_with_weight.shape[0] == selected_edges.shape[
        0], "selected_edges_with_weight is not implemented correctly "

    nodes_of_added_edges = selected_edges_with_weight[:,
                           :2].flatten()
    assert np.array([True for i in nodes_of_added_edges if
                     i in list(
                         G.nodes)]).all(), "nodes of edges to be added involved nodes that do not exist within the original graph"

    # assert data.is_disease2disease_edges_added_to_graph(
    #     disease2disease_edges_to_be_added,
    #     use_outside_graph=True), "added_qualified_edges is True, but no disease2disease are added "

    return selected_edges_with_weight
