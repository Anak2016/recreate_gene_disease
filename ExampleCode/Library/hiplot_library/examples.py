import numpy as np
import matplotlib.pyplot as plt

folder_path = r'C:\Users\Anak\PycharmProjects\recreate_gene_disease\PerformanceResult\NodeClassification\GeneDiseaseProject\copd\PhenotypeGeneDisease'

def get_auc_val(folder_n, file_n):
    performance_path = folder_n + '\\' + file_n
    # print(performance_path)
    import pandas as pd
    result = pd.read_csv(performance_path)
    # print(folder_n)
    # print(file_n)
    # print(result)
    auc = result.iloc[5]['AUC']
    return auc


import os
file_name = 'mlp'


def get_strategy_name(file_name, strategy_name):
    for ind, t in enumerate(file_name[1:]):
        if t == 'dim64':
            break
        strategy_name.append(t)
    return strategy_name


folder_path = r'C:\Users\Anak\PycharmProjects\recreate_gene_disease\PerformanceResult\NodeClassification\GeneDiseaseProject\copd\PhenotypeGeneDisease'


def get_auc_val(folder_n, file_n):
    performance_path = folder_n + '\\' + file_n
    # print(performance_path)
    import pandas as pd
    result = pd.read_csv(performance_path)
    # print(folder_n)
    # print(file_n)
    # print(result)
    auc = result.iloc[5]['AUC']
    return auc


def get_strategy_name(file_name, strategy_name):
    for ind, t in enumerate(file_name[1:]):
        if t == 'dim64':
            break
        strategy_name.append(t)
    return strategy_name


def walk_folder():
    #     records = []
    #     class_embedding_dict = {}
    #     qualified_node_strategy = {}
    #     embedding_qualified_nodes = {}
    #     strategy_th = {}
    all_record = []
    for i, j, k in os.walk(folder_path, topdown=False):
        if len(k) > 0:

            x = i.split('\\')
            graph_type = x[-8]
            classifier = x[-7]
            embedding = x[-6]
            qualified_node = x[-3]

            if classifier == 'mlp':
                #                 print(x)
                #                 print(qualified_node)

                folder_name = '_'.join([classifier, embedding, qualified_node])
                folder_name = folder_name  # folder where list of file exists

                for file in k:
                    file_name = file.split('_')
                    strategy_name = []

                    if file_name[0] == 'train' and file_name[1] != 'all':
                        streategy_name = get_strategy_name(file_name,
                                                           strategy_name)
                        suffix, th = strategy_name[-1].split('=')
                        strategy_name = strategy_name[:-1] + [suffix]
                        strategy_name = '_'.join(strategy_name)
                        if float(th) in [0.05, 0.1, 0.4, 0.5]:
                            auc = get_auc_val(i, file)

                            dict_per_record = {}
                            dict_per_record['th'] = th
                            dict_per_record['qualified_nodes'] = qualified_node
                            dict_per_record['embedding'] = embedding
                            dict_per_record['strategy'] = strategy_name
                            dict_per_record['classifier'] = classifier
                            dict_per_record['graph_type'] = graph_type
                            dict_per_record['auc'] = auc
                            #                             print(auc)
                            #                             print(id(dict_per_record))
                            #                             print(len(all_record))
                            all_record.append(dict_per_record)
    #                             print(strategy_th)
    #                             if strategy_name not in strategy_th:
    #                                 strategy_th[strategy_name] = {}
    #                                 if th not in strategy_th[strategy_name]:
    #                                     strategy_th[strategy_name][th] = auc
    #                             else:
    #                                 if th not in strategy_th[strategy_name]:
    #                                     strategy_th[strategy_name][th] = auc

    #                 qualified_node_strategy.setdefault(qualified_node, {}).update(strategy_th)

    #                 embedding_qualified_nodes.setdefault(embedding, {}).update(qualified_node_strategy)
    #             class_embedding_dict.setdefault(classifier, {}).update(embedding_qualified_nodes)
    #     return class_embedding_dict
    return all_record


#=====================
#== strategy_qualifier per th line plot
#=====================
def line_plot_strategy_qualifier(threshold, class_qualified_node_dict):
    strategy_qualifier_plot = []
    count = 0
    for cls, qualified_dict in class_qualified_node_dict.items():
        qualifer_label = []
        per_qualifier = []
        for qualified_node, streategy_dict in qualified_dict.items():
            per_strategy = []
            for strategy, th_dict in streategy_dict.items():
                # print(strategy)
                for th, auc in th_dict.items():
                    if th == str(threshold):
                        val = auc
                per_strategy.append(val)
            per_qualifier.append( per_strategy)
            qualifer_label.append(qualified_node)
    #=====================
    #==line plot
    #=====================

    # for i in per_strategy:
    #     print(i)
    import matplotlib.pyplot as plt
    for i,j in zip(per_qualifier, qualifer_label):
        plt.plot(i, label = j)
    plt.title('bottom_k fixed the reset')
    plt.legend()
    plt.show()



#=====================
# == strategy_qualifier per th bar plot
#== GCN and Node2vec
#=====================
def bar_plot_strategy_qualifier(class_qualified_node_dict):
    for threshold in [0.1,0.4]:

        def get_val():
            import numpy as np
            np.random.seed(100)
            rand = np.random.random((5, 7)) * 0.1
            return rand

        rand = get_val()

        strategy_qualifier_plot = []
        count = 0
        for cls, qualified_dict in class_qualified_node_dict.items():
            qualifer_label = []
            per_qualifier = []
            for qualified_node, streategy_dict in qualified_dict.items():
                per_strategy = []
                for strategy, th_dict in streategy_dict.items():
                    # print(strategy)
                    for th, auc in th_dict.items():
                        if th == str(threshold):
                            val = auc
                    per_strategy.append(val)
                per_qualifier.append( per_strategy)
                qualifer_label.append(qualified_node)

        #=====================
        #==bar_plot
        #=====================

        # for i in per_strategy:
        #     print(i)
        import matplotlib.pyplot as plt
        # axs[-1].hist(x, bins=n_bins)
        # axs[0].hist(y, bins=n_bins)
        # plt.show()
        per_qualifier = list(np.array(per_qualifier) + rand)
        for ind, (i,j) in enumerate(zip(per_qualifier, qualifer_label)):
            if threshold == 0.1:
                plt.bar(ind-0.2, i, width=0.2, label = j)
            elif threshold == 0.4:
                plt.bar(ind, i, width=0.2, label = j)
            elif threshold == 0.5:
                plt.bar(ind+0.2, i, width=0.2, label = j)
    plt.title("GCN vs Node2vec")
    plt.ylim(0.5, 1)
    # plt.legend()
    plt.show()

if __name__ == '__main__':

    class_embedding_dict = walk_folder()

    import seaborn as sns

    # sns.set(style="whitegrid")
    tips = sns.load_dataset("tips")
    ax = sns.violinplot(x=tips["total_bill"])
    ax.show()
