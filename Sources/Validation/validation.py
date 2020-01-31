# import numpy as np
#
# # assert G.is_graph_edges_weighted(), "use_weighted_edges is True, but all edges in graph has weight == 1"
# # assert validation.is_graph_edges_weighted(G), "use_weighted_edges is True, but all edges in graph has weight == 1"
#
# def is_disease2disease_edges_added_to_graph(self):
#     """
#     check self.G whether the graph has disease2disease edges
#     @return: type = Boolean
#     """
#
#     is_node_disease_func = lambda x: x in self.disease_np
#     is_node_disease_vectorized = np.vectorize(is_node_disease_func)
#     graph_edges_np = np.array(list(self.G.edges))
#
#     has_disease2disease_edges = is_node_disease_vectorized(graph_edges_np)
#     has_disease2disease_edges = has_disease2disease_edges.all(axis=1).any()
#
#     return has_disease2disease_edges
#
#
# def is_graph_edges_weighted(G):
#     """
#     check self.G whether or not edges of graph is weighted
#     @param G: type = nx.Graph()
#     @return: type = Boolean
#     """
#     is_weighted_one_func = lambda x: x != 1
#     is_weighted_one_vectorized = np.vectorized(is_weighted_one_func)
#
#     graph_edges_with_weight = np.array(list(self.G.edges.data('weight')))
#     weight_np = graph_edges_with_weight[:,2]
#
#     has_weighted_edges = is_weighted_one_vectorized(weight_np).any()
#
#     return has_weighted_edges
