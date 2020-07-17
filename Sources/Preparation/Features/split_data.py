from os import path

import pandas as pd
import numpy as np

from Sources.Preprocessing import remove_edges_from_graph_for_link_prediction
from Sources.Preprocessing import select_emb_save_path


def get_k_fold_data(run_node2vec_emb,data, graph_with_no_added_qualified_edges,
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
                 use_shared_gene_or_phenotype_edges=None,
                 use_gene_disease_graph=None,
                 use_phenotype_gene_disease_graph=None,
                 graph_edges_type=None,
                 task=None,
                 enforce_end2end=None,
                 cross_validation=None,
                 k_fold=None,
                 split=None,
                 get_data_with_emb_as_feat=None,
                 split_by_node = None):
    raise NotImplementedError()

    reset_graph = graph_with_no_added_qualified_edges
    edges_with_features_dict = {}

    splitted_edges_dir = f'C:\\Users\\Anak\\PycharmProjects\\recreate_gene_disease\\Data\\processed\\LinkPrediction\\GeneDiseaseProject\\copd\\PhenotypeGeneDisease\\PGDP\\Node2Vec\\UnweightedEdges\\NoAddedEdges\\'

    if split_by_node:
        splitted_edges_dir += f'SplitByNode\\KFold={k_fold}\\'
    else:
        splitted_edges_dir += f'SplitByEdge\\KFold={k_fold}\\'

    # for i, (train_set_np, test_set_np) in enumerate(
    #         data.split_cross_validation(data, k_fold, stratify=True, task=task,
    #                                     # reset_cross_validation_split=True,
    #                                     splitted_edges_dir=splitted_edges_dir
    #                                     )):
    for i, (train_test_dict) in enumerate(
            data.split_cross_validation(data, k_fold, stratify=True, task=task,
                                        split_by_node=split_by_node,
                                        # reset_cross_validation_split=True,
                                        splitted_edges_dir=splitted_edges_dir
                                        )):
        if i == 10:
            print('run node2vec without disease nodes of the highest degree')
            exit()
        train_set_np = train_test_dict['train_set']  # is it np or df?
        test_set_np = train_test_dict['test_set']
        embedding_model_file_path = select_emb_save_path(
            save_path_base='data',
            emb_type='node2vec',
            add_qualified_edges=add_qualified_edges,
            dataset=dataset,
            use_weighted_edges=use_weighted_edges,
            edges_percent=edges_percent,
            edges_number=edges_number,
            added_edges_percent_of=added_edges_percent_of,
            use_shared_phenotype_edges=use_shared_phenotype_edges,
            use_shared_gene_edges=use_shared_gene_edges,
            use_shared_gene_and_phenotype_edges=use_shared_gene_and_phenotype_edges,
            use_shared_gene_but_not_phenotype_edges=use_shared_gene_but_not_phenotype_edges,
            use_shared_phenotype_but_not_gene_edges=use_shared_phenotype_but_not_gene_edges,
            use_shared_gene_or_phenotype_edges=use_shared_gene_or_phenotype_edges,
            use_gene_disease_graph=use_gene_disease_graph,
            use_phenotype_gene_disease_graph=use_phenotype_gene_disease_graph,
            graph_edges_type=graph_edges_type,
            task=task,
            split=split,
            k_fold=k_fold,
            k_fold_ind=i,
            split_by_node=split_by_node
        )

        if not path.exists(
                embedding_model_file_path):
            graph_with_no_added_qualified_edges = reset_graph.copy()
            # print(len( reset_graph.edges ))

            graph_with_no_added_qualified_edges_with_removed_test_edges = remove_edges_from_graph_for_link_prediction(
                data, graph_with_no_added_qualified_edges, train_set_np,
                test_set_np)

            run_node2vec_emb(data=data,
                             G=graph_with_no_added_qualified_edges_with_removed_test_edges,
                             embedding_model_file_path=embedding_model_file_path,
                             # enforce_end2end=enforce_end2end,
                             edges_percent=edges_percent,
                             edges_number=edges_number,
                             use_weighted_edges=use_weighted_edges,
                             add_qualified_edges=add_qualified_edges,
                             added_edges_percent_of=added_edges_percent_of,
                             enforce_end2end=enforce_end2end
                             )

        node_with_features = get_data_with_emb_as_feat(data,
                                                       # use_saved_emb_file,
                                                       add_qualified_edges,
                                                       dataset,
                                                       use_weighted_edges,
                                                       normalized_weighted_edges,
                                                       edges_percent,
                                                       edges_number,
                                                       added_edges_percent_of,
                                                       use_shared_gene_edges,
                                                       use_shared_phenotype_edges,
                                                       use_shared_gene_and_phenotype_edges,
                                                       use_shared_gene_but_not_phenotype_edges,
                                                       use_shared_phenotype_but_not_gene_edges,
                                                       use_gene_disease_graph,
                                                       use_phenotype_gene_disease_graph,
                                                       graph_edges_type,
                                                       task,
                                                       split,
                                                       k_fold,
                                                       split_by_node,
                                                       k_fold_ind=i,
                                                       # need it to choose file name
                                                       )

        edges_with_features = []
        edges_label = []
        # BUG: there are duplicate value in edes_label? How come?
        ## even if there are duplicate index name, embedding value is unique. What is the reason
        for node1, node2, _ in np.concatenate([train_set_np, test_set_np],
                                              axis=0).tolist():
            edges_instance_with_features = np.concatenate(
                [node_with_features.loc[node1],
                 node_with_features.loc[node2]]).tolist()
            edges_instance_label = f'{node1}_{node2}'
            edges_with_features.append(edges_instance_with_features)
            edges_label.append(edges_instance_label)

        edges_with_features_df = pd.DataFrame(edges_with_features,
                                              index=edges_label)
        edges_with_features_dict[i] = edges_with_features_df

    print()
    return edges_with_features_dict

