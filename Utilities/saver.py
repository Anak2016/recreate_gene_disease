import pandas as pd
import os
import pathlib
import pickle

from Sources.Preprocessing import get_saved_file_name_for_emb
from Sources.Preprocessing import select_emb_save_path
from arg_parser import args
from global_param import DIM
from global_param import NUM_WALKS
from global_param import WALK_LEN
from global_param import WINDOW


def save2file(train_report_np, train_columns, train_index,
              test_report_np, test_columns, test_index) -> None:

    folder_path = select_emb_save_path(save_path_base='report_performance',
                                       emb_type=args.emb_type,
                                       add_qualified_edges=args.add_qualified_edges,
                                       dataset=args.dataset,
                                       use_weighted_edges=args.use_weighted_edges,
                                       edges_percent=args.edges_percent,
                                       edges_number=args.edges_number,
                                       added_edges_percent_of=args.added_edges_percent_of,
                                       use_shared_phenotype_edges=args.use_shared_phenotype_edges,
                                       use_shared_gene_edges=args.use_shared_gene_edges,
                                       use_shared_gene_and_phenotype_edges=args.use_shared_gene_and_phenotype_edges,
                                       use_shared_gene_but_not_phenotype_edges=args.use_shared_gene_but_not_phenotype_edges,
                                       use_shared_phenotype_but_not_gene_edges=args.use_shared_phenotype_but_not_gene_edges,
                                       use_shared_gene_or_phenotype_edges=args.use_shared_gene_or_phenotype_edges,
                                       use_gene_disease_graph=args.use_gene_disease_graph,
                                       use_phenotype_gene_disease_graph=args.use_phenotype_gene_disease_graph,
                                       graph_edges_type=args.graph_edges_type,
                                       task=args.task,
                                       split=args.split,
                                       k_fold=args.k_fold)

    file_name = get_saved_file_name_for_emb(args.add_qualified_edges,
                                            args.edges_percent,
                                            args.edges_number, dim=DIM,
                                            walk_len=WALK_LEN,
                                            num_walks=NUM_WALKS, window=WINDOW)

    import pathlib
    # folder_path = pathlib.Path(r'C:\Users\Anak\PycharmProjects\recreate_gene_disease')
    first_part = folder_path.split('\\')[:11]
    first_part = '\\'.join(first_part)
    first_part = pathlib.Path(first_part)

    second_part = folder_path.split('\\')[11:]
    second_part = '\\'.join(second_part)
    second_part = pathlib.Path(second_part)

    folder_path = first_part / args.classifier_name / second_part

    os.makedirs(folder_path, exist_ok=True)

    test_save_path = folder_path / f'train_{file_name}'
    train_save_path = folder_path / f'test_{file_name}'

    train_df = pd.DataFrame(train_report_np, index=train_index,
                            columns=train_columns)
    test_df = pd.DataFrame(test_report_np, index=test_index,
                           columns=test_columns)

    train_df.to_csv(train_save_path)
    print(f'save to {train_save_path}')

    test_df.to_csv(test_save_path)
    print(f'save to {test_save_path}')



