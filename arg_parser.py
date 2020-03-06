import argparse

'''
example of a command argument assuming that current directory is my_utility/python_script
    python __init__.py --merge
'''

'''
    with default
        parser.add_argument('--dataset', type=str, default='gene_disease', help='specify type of dataset to be used')

    with action
        parser.add_argument('--subgraph', action="store_true", help='NOT CURRENTLY COMPATIBLE WITH THE PROGRAM;Use only node in the largest connected component instead of all nodes disconnected graphs')

    with nargs, this is used to extract provided arguments as a list 
        eg --something 1 2 3 4 5 
            args.something == [1,2, 3,4,5] is true
        parser.add_argument('--weighted_class', default=[1,1,1,1,1], nargs='+', help='list of weighted_class of diseases only in order <0,1,2,3,4,5>')
'''
"""
example of use case 
> for node2vec 
    --add_qualified_edges top_k --dataset GeneDisease --edges_percent 1 --added_edges_percent_of GPSim
> for train_model
    
    
"""
parser = argparse.ArgumentParser()

# TODO testing parser that required subparser

# --------main
## preparation
parser.add_argument('--dataset', type=str, default='no',
                    help='name of the dataset eg. GeneDisease or GPSim')
# parser.add_argument('--dataset', type=str, default=None, help='name of the dataset eg. GeneDisease or GPSim')

## preprocessing
### adding features
parser.add_argument('--use_saved_emb_file', action='store_true', help="")
# parser.add_argument('--add_qualified_edges', action='store_true', help="")
parser.add_argument('--add_qualified_edges', type=str, default=None,
                    help='specified strategy to be selected qualified edges to be added ')
parser.add_argument('--use_weighted_edges', action='store_true', help="")
parser.add_argument('--normalized_weighted_edges', action='store_true',
                    help="")
## choose strategies
# parser.add_argument('--strategy', type=str, default=None, help='choose edges adding strategy')
parser.add_argument('--edges_percent', type=float, default=None,
                    help='if strategy is invoked, either edges_number or edges_percent must also be passed ')
parser.add_argument('--added_edges_percent_of', type=str, default=None,
                    help='which dataset is the toal number of qualified edges in which to calculate percentage from ; when ars.edges_percent is not None, arg.added_edges_percent_of must not be NOne ')
parser.add_argument('--edges_number', type=int, default=None,
                    help='if strategy is invoked, either edges_number or edges_percent must also be passed ')
parser.add_argument('--use_shared_gene_edges', action='store_true', help="")
parser.add_argument('--use_shared_phenotype_edges', action='store_true',
                    help="")
parser.add_argument('--use_shared_gene_and_phenotype_edges',
                    action='store_true', help="")
parser.add_argument('--use_shared_gene_but_not_phenotype_edges',
                    action='store_true', help="")
parser.add_argument('--use_shared_phenotype_but_not_gene_edges',
                    action='store_true', help="")

parser.add_argument('--use_phenotype_gene_disease_graph', action='store_true',
                    help="")
parser.add_argument('--use_gene_disease_graph', action='store_true', help="")

# TODO convert use graph_type instead of --use_phenotype_gene_disease_graph and use_gene_disease_graph
## I may need to do this at some point just because it is cleaner, but due to amount of change and its tradeoff. its is considered a waste of time.
# parser.add_argument('--graph_type', type=str, default=None, help='there are 2 options: \n'
#                                                                  '1. phenotype_gene_disease\n'
#                                                                  '2. gene_disease\n')

parser.add_argument('--graph_edges_type', type=str, default=None, help='there are 3 options:\n'
                                                                       '1. phenotype_gene_disease\n'
                                                                       '2. phenotype_gene_disease_phenotype\n')

### spliting train and test dataset
#### train_test_split
parser.add_argument('--split', type=float, default=None,
                    help='split specify test_size as percentage of dataset size; split is used when args.cross_validation is False')
parser.add_argument('--split_by_node', action='store_true', help="split_by_node only used when task = 'link_prediction' ")

#### k fold cross validation
parser.add_argument('--cv', action='store_true',
                    help="activate cross_validation")
parser.add_argument('--k_fold', type=int, default=None,
                    help='k_fold is used when args.cross_validation is True')
## modeling

## task
parser.add_argument('--task', type=str, default=None, help='please select between \n'
                                                            '1. link prediction\n'
                                                            '2. nodes classification\n')
parser.add_argument('--enforce_end2end', action='store_true', help="apply emb then use emb as input to run_task. \n"
                                                                   "note: error will be raised as following\n"
                                                                   " 1. if --use_saved_emb_File is False, error is raised.\n "
                                                                   " 2. if --use_saved_emb_file is True and emb_file does not exist, error is raised.\n"
                                                                   "    error is raised to avoid unpredicted behavior\n")
## run multiple args condition
parser.add_argument('--run_multiple_args_conditions', type=str, default=None,
                    help='')
# -- utilities


args = parser.parse_args()
## parser constraint
assert args.run_multiple_args_conditions in [None, 'train_model',
                                             'node2vec'], "currently only 3 options are supported: None (implies not running multiple args conditions), train_model or node2vec"


def apply_parser_constraint():
    # section for subparser
    ## cross valdiation and train_tset_split
    if args.cv:
        assert args.k_fold is not None, "if cv is true, k_fold must be set"
    else:
        assert args.k_fold is None, "if cv is false, k_fold have to be None; this implies that train_test_split will be used in stead "
        assert  0< args.split < 1," split can only be between 0 and 1 "

    ## about task
    assert args.task in ['link_prediction', 'node_classification'], 'for args.task, please select link_repdcition or node_classification'

    if args.task == 'link_prediction':
        assert args.dataset == 'no' and  args.graph_edges_type == 'phenotype_gene_disease_phenotype' and args.use_phenotype_gene_disease_graph, 'arg.task == link prediction, only support  the following condition\n' \
                                                                                                                      '1. use_phenotype_gene_disease_graph is true\n' \
                                                                                                                      '2. args.graph_edges_type == "phenotype_gene_disease_phenotype"'

    ## about enforce_end2end
    if args.enforce_end2end:
        assert args.use_saved_emb_file, "args.enforce_end2end only apply with embedding"


    ## about starter graph
    starter_graph_constraint = [args.use_gene_disease_graph,
                                args.use_phenotype_gene_disease_graph]
    if sum(starter_graph_constraint) == 1:
        if args.use_gene_disease_graph:
            assert args.graph_edges_type is None, "for gene_disease graph, graph_edges_type is not supported"

        if args.use_phenotype_gene_disease_graph:
            if args.graph_edges_type not in ['phenotype_gene_disease','phenotype_gene_disease_phenotype']:
                raise ValueError(
                    'graph_edges_type only support for phenotype_gene_disease and phenotype_gene_disease_phenotype')

    else:
        print( "you must specify exactly 1 starter graph, including\n" 
                                        "1. use_gene_disease_graph\n" 
                                        "2. use_phenotype_disease_graph\n")

    ## about shared_nodes when adding qualified edges to graph
    if args.add_qualified_edges is None:
        assert not args.use_shared_gene_edges, "if add_qualified_edges is None, use_shared_shared_gene_should_be_False"
        assert not args.use_shared_phenotype_edges, "if add_qualified_edges is None, use_shared_shared_phenotype_should_be_False"
    else:
        ## about getting qualifed edges to be used with selected strategy (called it "use_initial_qualified_edges_option")
        use_initial_qualified_edges_option1 = [args.use_shared_gene_edges,
                                               args.use_shared_phenotype_edges]
        use_initial_qualified_edges_option2 = [
            args.use_shared_gene_and_phenotype_edges,
            args.use_shared_gene_but_not_phenotype_edges,
            args.use_shared_phenotype_but_not_gene_edges]
        assert sum(
            use_initial_qualified_edges_option2) <= 1, "no more than 1 of the following can be true at the same time:\n" \
                                                       "1. shared_gene_and_phenotype_edges OR\n" \
                                                       "2. use_shared_gene_but_not_phenotype_edges OR \n" \
                                                       "3. use_shared_phenotype_but_not_gene_edges \n"
        if sum(use_initial_qualified_edges_option2) > 0:

            assert sum(
                use_initial_qualified_edges_option1) == 0, "option1: use_shared_gene_edges or use_shared_phenotypes_edges are Ture or " \
                                                           "option2: one of the following is true \n" \
                                                           "         1. shared_gene_and_phenotype_edges OR\n" \
                                                           "         2. use_shared_gene_but_not_phenotype_edges OR \n" \
                                                           "         3. use_shared_phenotype_but_not_gene_edges \n"

        else:
            # print(use_initial_qualified_edges_option1)
            # print(use_initial_qualified_edges_option2)

            assert sum(
                use_initial_qualified_edges_option1) > 0, "option1: use_shared_gene_edges or use_shared_phenotypes_edges are Ture or \n" \
                                                          "option2: one of the following is true \n" \
                                                          "         1. shared_gene_and_phenotype_edges OR\n" \
                                                          "         2. use_shared_gene_but_not_phenotype_edges OR\n " \
                                                          "         3. use_shared_phenotype_but_not_gene_edges\n "

    ## on edges_percent cases
    ### note: added_number_of_edges is implemented in get_number_of_added_edges()
    if args.edges_percent is None:
        assert args.added_edges_percent_of is None, "added_edges_percent_of only need to be specified when edges_percent is not None"
    else:
        assert args.added_edges_percent_of is not None, "added_edges_percent_of need to be specified when edges_percent is not None"

        if args.added_edges_percent_of not in ["GeneDisease", "GPSim", "no"]:
            raise ValueError(
                "edges can only be added from extracted qualified edges of GeneDisease and GPSim dataset")

        assert args.edges_percent <= 1, "percent is expected to be between range of 0 to 1 "
        assert args.edges_percent > 0, "percent need to be more than 0. (please use dataset = 'no' and add_qulified_edges is None instead) "

    ## validation about use cases of add_qualified_edges
    if args.add_qualified_edges is not None:
        if args.add_qualified_edges not in ['top_k', 'bottom_k']:
            raise ValueError(
                "for add_qualified_edges, only top_k is implemented")

        assert args.dataset is not None, "when add_qulified_edges is true, dataset must be speicied "
        # if args.strategy is not None:
        if (args.edges_percent is not None) or (args.edges_number is not None):
            if (args.edges_percent is not None) and (
                    args.edges_number is not None):
                raise ValueError(
                    "only percent or edges_number can be used at one time")
        else:
            raise ValueError(
                'if strategy is invoked, percent argument must also be passed')


def run_args_conditions(run_model_func, apply_parser_constraint_func,
                        run_multiple_args_condition=None):
    """

    @param run_model_func: desc: func to be run that required input from args such as run_train_model and run_node2vec
    @param apply_parser_constraint_func: desc: constraint of parser and subparser of input from args
    @param run_multiple_args_condition: type: boolean
    @return:
    """
    assert run_multiple_args_condition is not None, "run_multiple_args_condition must be in 'train_model' or 'node2vec' "

    # --use_saved_emb_file - -add_qualified_edges
    # top_k - -dataset
    # GeneDisease - -edges_percent
    # 0.05 - -added_edges_percent_of
    # GeneDisease - -use_shared_gene_edges - -cv - -k_fold
    # 4

    if run_multiple_args_condition == 'train_model':
        # train_model
        use_run_train_model_args(run_model_func, apply_parser_constraint_func)

    elif run_multiple_args_condition == 'node2vec':
        # run_node2vec
        use_run_node2vec_args(run_model_func,
                              apply_parser_constraint_func)
    else:
        raise ValueError(
            "currently only 3 options are supported: None (implies not running multiple args conditions), train_model or node2vec")


def reset_args():
    args.use_shared_gene_edges = False
    args.use_shared_phenotype_edges = False
    args.use_shared_phenotype_edge = False
    args.use_share_gene_edges = False
    args.use_shared_gene_and_phenotype_edges = False
    args.use_shared_gene_but_not_phenotype_edges = False
    args.use_shared_phenotype_but_not_gene_edges = False


def use_run_train_model_args(run_train_model, apply_parser_constraint_func):
    """

    @param run_train_model: type = function()
    @param apply_constraint_func: type = function()
    @param run_multiples_args_conditions: type = boolean; for more information read arg_parser file on run_multiple_args_confitions
    @return:
    """

    # task
    # tasks = ['link_prediction', 'node_classification']
    tasks = [ 'node_classification']

    # starter_graph = ['use_gene_disease_graph','use_phenotype_gene_disease_graph']
    starter_graph = ['use_phenotype_gene_disease_graph']

    # graph_edges_types = ['phenotype_gene_disease_phenotype', 'phenotype_gene_disease']
    graph_edges_types = ['phenotype_gene_disease_phenotype']

    use_saved_emb_file = True
    # use_saved_emb_file = False

    # add_qualified_edges = ['top_k',
    #                        'bottom_k']  # bottom_k is not yet implemented
    # add_qualified_edges = ['bottom_k']
    add_qualified_edges = ['top_k']
    dataset = 'GeneDisease'
    edges_percent = [
          0.05, 0.1, 0.2, 0.3, 0.4,
        0.5]  # look at my paper and see what other percentages has
    # edges_percent = [0.05, 0.1]
    # edges_percent = [0.05]
    added_edges_percent_of = 'GeneDisease'
    added_edges_dataset = {
        # 'use_shared_gene_edges': True,
        # 'use_shared_phenotype_edges': True,
        # 'use_shared_gene_or_phenotype_edges': True,
        # 'use_shared_gene_and_phenotype_edges': True,
        'use_shared_gene_but_not_phenotype_edges': True,
        # 'use_shared_phenotype_but_not_gene_edges': True
    }
    cv = True
    k_fold = 4

    # assign value to args
    args.use_saved_emb_file = use_saved_emb_file
    args.dataset = dataset
    args.added_edges_percent_of = added_edges_percent_of
    args.cv = cv
    args.k_fold = k_fold

    try:
        for t in tasks:
            args.task = t
            for g in starter_graph:
                if g == 'use_phenotype_gene_disease_graph':
                    args.use_phenotype_gene_disease_graph = True
                elif g == 'use_gene_disease_graph':
                    args.use_gene_disease_graph = True
                else:
                    raise ValueError(
                        "Currently only 2 options for starter graphs are offered\n"
                        "1. use_phenotype_gene_disease_graph\n"
                        "2. use_gene_disease_graph\n")

                for g in graph_edges_types:
                    args.graph_edges_type = g

                    # iterate args of type list
                    for i in add_qualified_edges:
                        args.add_qualified_edges = i
                        for percent in edges_percent:
                            args.edges_percent = percent
                            for key, val in added_edges_dataset.items():

                                # reset args input for the next loop
                                reset_args()

                                if key == 'use_shared_gene_edges':
                                    args.use_shared_gene_edges = True
                                elif key == 'use_shared_phenotype_edges':
                                    args.use_shared_phenotype_edges = True
                                elif key == 'use_shared_gene_or_phenotype_edges':
                                    args.use_shared_phenotype_edge = True
                                    args.use_shared_gene_edges = True
                                elif key == 'use_shared_gene_and_phenotype_edges':
                                    args.use_shared_gene_and_phenotype_edges = True
                                elif key == 'use_shared_gene_but_not_phenotype_edges':
                                    args.use_shared_gene_but_not_phenotype_edges = True
                                elif key == 'use_shared_phenotype_but_not_gene_edges':
                                    args.use_shared_phenotype_but_not_gene_edges = True
                                else:
                                    raise ValueError(
                                        "non of avaliable option added edges are selected which include the follwoing:"
                                        "use_shared_gene_edges,\n"
                                        "use_shared_phenotyp_edges,\n"
                                        "use_shared_gene_or_phenotype_edges,\n"
                                        "use_shared_gene_and_phenotype_edges,\n"
                                        "use_shared_gene_but_not_phenotype_edges, and \n"
                                        "use_shared_phenotype_but_not_gene_edges.\n")
                                print(
                                    f"args.add_qualified_edges = {args.add_qualified_edges}")
                                print(f"args.graph_edges_type = {args.graph_edges_type}")
                                print(f"args.edges_percent = {args.edges_percent}")
                                print(
                                    f"args.add_qualified_edges = {args.add_qualified_edges}")
                                print(
                                    f"args.use_shared_gene_edges = {args.use_shared_gene_edges}")
                                print(
                                    f"args.use_shared_phenotype_edges = {args.use_shared_phenotype_edges} ")
                                print(
                                    f'args.use_shared_gene_and_phenotype_edges= {args.use_shared_gene_and_phenotype_edges}')
                                print(
                                    f'args.use_shared_gene_but_not_phenotype_edges = {args.use_shared_gene_but_not_phenotype_edges}')
                                print(
                                    f'args.use_shared_phenotype_but_not_gene_edges  = {args.use_shared_phenotype_but_not_gene_edges}')

                                if key in ['use_shared_gene_and_phenotype_edges'] and percent in [0.2,0.3,0.4,0.5]:
                                    pass
                                else:
                                    # aplly constrain
                                    apply_parser_constraint_func()

                                    # run input mode
                                    run_train_model()

                                # # aplly constrain
                                # apply_parser_constraint_func()


    except:
        print("The following args causes error")
        print(f"args.add_qualified_edges = {args.add_qualified_edges}")
        print(f"args.use_shared_gene_edges = {args.use_shared_gene_edges}")
        print(
            f"args.use_shared_phenotype_edges = {args.use_shared_phenotype_edges} ")
        print(
            f'args.use_shared_gene_and_phenotype_edges= {args.use_shared_gene_and_phenotype_edges}')
        print(
            f'args.use_shared_gene_but_not_phenotype_edges = {args.use_shared_gene_but_not_phenotype_edges}')
        print(
            f'args.use_shared_phenotype_but_not_gene_edges  = {args.use_shared_phenotype_but_not_gene_edges}')

        raise ValueError('args input error')


def use_run_node2vec_args(run_node2vec, apply_parser_constraint_func):

    # task
    # tasks = ['link_prediction', 'node_classification']
    tasks = [ 'node_classification']

    # starter_graph = ['use_gene_disease_graph','use_phenotype_gene_disease_graph']
    starter_graph = ['use_phenotype_gene_disease_graph']

    # graph_edges_types = ['phenotype_gene_disease_phenotype', 'phenotype_gene_disease']
    graph_edges_types = ['phenotype_gene_disease_phenotype']

    # add_qualified_edges = ['top_k',
    #                        'bottom_k']  # bottom_k is not yet implemented
    # add_qualified_edges = ['bottom_k']

    add_qualified_edges = ['top_k']

    dataset = 'GeneDisease'
    # edges_percent = [0.05, 0.1, 0.2, 0.3, 0.4,
    #                  0.5]  # look at my paper and see what other percentages has
    edges_percent = [0.05]
    added_edges_percent_of = 'GeneDisease'

    added_edges_dataset = {
        # 'use_shared_gene_edges': True,
        # 'use_shared_phenotype_edges': True,
        # 'use_shared_gene_or_phenotype_edges': True,
        # 'use_shared_gene_and_phenotype_edges': True, # only have 53 edges, so 0.1 percent of GeneDeisease edges = 79 which is more than 53 already
        # 'use_shared_gene_but_not_phenotype_edges': True,
        'use_shared_phenotype_but_not_gene_edges': True
    }

    # assign value to args
    args.dataset = dataset
    args.added_edges_percent_of = added_edges_percent_of

    try:
        for t in tasks:
            args.task = t
            for g in starter_graph:
                if g == 'use_phenotype_gene_disease_graph':
                    args.use_phenotype_gene_disease_graph = True
                elif g == 'use_gene_disease_graph':
                    args.use_gene_disease_graph = True
                else:
                    raise ValueError(
                        "Currently only 2 options for starter graphs are offered\n"
                        "1. use_phenotype_gene_disease_graph\n"
                        "2. use_gene_disease_graph\n")

                for g in graph_edges_types:
                    args.graph_edges_type = g
                    # iterate args of type list
                    for i in add_qualified_edges:
                        args.add_qualified_edges = i
                        for percent in edges_percent:
                            args.edges_percent = percent
                            for key, val in added_edges_dataset.items():

                                # reset args input for the next loop
                                reset_args()

                                if key == 'use_shared_gene_edges':
                                    args.use_shared_gene_edges = True
                                elif key == 'use_shared_phenotype_edges':
                                    args.use_shared_phenotype_edges = True
                                elif key == 'use_shared_gene_or_phenotype_edges':
                                    args.use_shared_phenotype_edges = True
                                    args.use_shared_gene_edges = True
                                elif key == 'use_shared_gene_and_phenotype_edges':
                                    args.use_shared_gene_and_phenotype_edges = True
                                elif key == 'use_shared_gene_but_not_phenotype_edges':
                                    args.use_shared_gene_but_not_phenotype_edges = True
                                elif key == 'use_shared_phenotype_but_not_gene_edges':
                                    args.use_shared_phenotype_but_not_gene_edges = True
                                else:
                                    raise ValueError(
                                        "non of avaliable option added edges are selected which include the follwoing:"
                                        "use_shared_gene_edges,\n"
                                        "use_shared_phenotype_edges,\n"
                                        "use_shared_gene_or_phenotype_edges,\n"
                                        "use_shared_gene_and_phenotype_edges,\n"
                                        "use_shared_gene_but_not_phenotype_edges, and\n "
                                        "use_shared_phenotype_but_not_gene_edges.\n")
                                print(
                                    f"args.use_gene_disease_graph = {args.use_gene_disease_graph}")
                                print(f"args.graph_edges_type = {args.graph_edges_type}")
                                print(
                                    f"args.use_phenotype_gene_disease_graph = {args.use_phenotype_gene_disease_graph}")
                                print(
                                    f"args.add_qualified_edges = {args.add_qualified_edges}")
                                print(f"args.edges_percent = {args.edges_percent}")
                                print(
                                    f"args.use_shared_gene_edges = {args.use_shared_gene_edges}")
                                print(
                                    f"args.use_shared_phenotype_edges = {args.use_shared_phenotype_edges} ")
                                print(
                                    f'args.use_shared_gene_and_phenotype_edges= {args.use_shared_gene_and_phenotype_edges}')
                                print(
                                    f'args.use_shared_gene_but_not_phenotype_edges = {args.use_shared_gene_but_not_phenotype_edges}')
                                print(
                                    f'args.use_shared_phenotype_but_not_gene_edges  = {args.use_shared_phenotype_but_not_gene_edges}')

                                if key in ['use_shared_gene_and_phenotype_edges'] and percent in [0.2,0.3,0.4,0.5]:
                                    pass
                                else:
                                    # aplly constrain
                                    apply_parser_constraint_func()

                                    # run input mode
                                    run_node2vec()

    except AssertionError as error:

        print("The following args causes error")
        print(f"args.add_qualified_edges = {args.add_qualified_edges}")
        print(f"args.use_shared_gene_edges = {args.use_shared_gene_edges}")
        print(
            f"args.use_shared_phenotype_edges = {args.use_shared_phenotype_edges} ")
        print(
            f'args.use_shared_gene_and_phenotype_edges= {args.use_shared_gene_and_phenotype_edges}')
        print(
            f'args.use_shared_gene_but_not_phenotype_edges = {args.use_shared_gene_but_not_phenotype_edges}')
        print(
            f'args.use_shared_phenotype_but_not_gene_edges  = {args.use_shared_phenotype_but_not_gene_edges}')
        print(error)
        # raise ValueError('args input error')  # aplly constrain
