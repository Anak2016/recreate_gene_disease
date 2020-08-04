import numpy as np
import matplotlib.pyplot as plt
# plt.plot(range(10))
# plt.show()
#
# # folder_path = r'C:\Users\Anak\PycharmProjects\recreate_gene_disease\PerformanceResult\NodeClassification\GeneDiseaseProject\copd\PhenotypeGeneDisease\PGDP\mlp\Node2Vec\UnweightedEdges\AddedEdges'
folder_path = r'C:\Users\Anak\PycharmProjects\recreate_gene_disease\PerformanceResult\NodeClassification\GeneDiseaseProject\copd\PhenotypeGeneDisease\PGDP'

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

def walk_folder():
    class_qualified_node_dict = {}
    for i,j,k in os.walk(folder_path, topdown=False):
        if len(k) > 0 :

            x = i.split('\\')
            classifier = x[-7]
            embedding = x[-6]
            qualified_node = x[-3]
            qualified_node_th = {}
            if classifier == 'mlp':
                print(qualified_node)
                folder_name = '_'.join([classifier, embedding, qualified_node])
                folder_name = folder_name # folder where list of file exists
                if qualified_node == 'SharedGeneEdges':
                    print()
                # th_auc = {} #
                strategy_th = {}
                for file in k:
                    file_name = file.split('_')
                    strategy_name = []
                    # auc = get_auc_val(i, file)
                    # print(auc)
                    # exit()
                    # print(file_name)
                    if file_name[0] == 'train' and file_name[1] != 'all':
                        streategy_name = get_strategy_name(file_name, strategy_name)
                        suffix, th = strategy_name[-1].split('=')
                        strategy_name = strategy_name[:-1] + [suffix]
                        strategy_name = '_'.join(strategy_name)
                        if float(th) in [0.05, 0.1 ,0.4,0.5]:
                            auc = get_auc_val(i, file)
                            if strategy_name not in strategy_th:
                                strategy_th[strategy_name] = {}
                                if th not in strategy_th[strategy_name]:
                                    strategy_th[strategy_name][th] = auc
                            else:
                                if th not in strategy_th[strategy_name]:
                                    strategy_th[strategy_name][th] = auc
                                else:
                                    raise NotImplementedError
                                    # strategy_th[strategy_name][th] = auc
                            # strategy_th.setdefault(strategy_name, {}).setdefault(th, {}).update(auc)
                            # th_auc.setdefault(th, []).append(auc)
                            # print(strategy_name)
                        # strategy_th.setdefault(strategy_name, {}).update(th_auc)
                qualified_node_th.setdefault(qualified_node, {}).update(strategy_th)

            class_qualified_node_dict.setdefault(classifier, {}).update(qualified_node_th)
    return class_qualified_node_dict

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

    class_qualified_node_dict = walk_folder()
    # for i in [0.05,0.1,0.4,0.5]:
    #     line_plot_strategy_qualifier(i, class_qualified_node_dict)
    # line_plot_strategy_qualifier(0.05, class_qualified_node_dict)
    #
    # # bar_plot_strategy_qualifier(class_qualified_node_dict)
    #

    print()


