# from itertools import combinations
# import pandas as pd
# import numpy as np
# import networkx as nx
#
# def get_GPSim_disease2disease_qualified_edges(data, use_weight_edges):
#     """
#     note: self loop is not a qualified edges ( so it will be removed here)
#
#     @return:
#     """
#     # note: code that create cui_edges_weight.csv is in notebook/Dataset/GPSim.ipynb (which is where code support to be because it is a part of Exploratory Data Analysis aka eda)
#     file_name = r'C:\Users\Anak\PycharmProjects\recreate_gene_disease\Data\raw\GPSim\Edges\cui_edges_weight.csv'
#     non_zero_GPsim_disease_cui_disease_pair_with_no_self_loop_df = pd.read_csv(file_name, sep=",") # varaible name should reflect state of its content
#
#     return non_zero_GPsim_disease_cui_disease_pair_with_no_self_loop_df
#
# def get_all_GeneDisease_unweighted_disease2disease_qualified_edges(data):
#
#     nodes = data.diseases_np
#
#     all_disease2disease_edges = list(combinations(nodes, 2))
#
#     all_disease2disease_qualified_edges = np.array(
#         [edge for edge in all_disease2disease_edges if
#          len(list(nx.common_neighbors(data.G, edge[0], edge[1]))) > 0])
#
#     G = nx.Graph()
#     G.add_edges_from(all_disease2disease_qualified_edges, weight=1)
#     all_qualified_disease2disease_edges_df = nx.to_pandas_edgelist(G)

#     return all_qualified_disease2disease_edges_df
