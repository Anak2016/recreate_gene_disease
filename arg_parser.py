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
# parser.add_

### spliting train and test dataset
#### train_test_split
parser.add_argument('--split', type=float, default=None,
                    help='split is used when args.cross_validation is False')
#### k fold cross validation
parser.add_argument('--cv', action='store_true',
                    help="activate cross_validation")
parser.add_argument('--k_fold', type=int, default=None,
                    help='k_fold is used when args.cross_validation is True')
## modeling

# -- utilities


args = parser.parse_args()

# section for subparser

## on edges_percent cases
### note: added_number_of_edges is implemented in get_number_of_added_edges()
if args.edges_percent is None:
    assert args.added_edges_percent_of is None, "added_edges_percent_of only need to be specified when edges_percent is not None"
else:
    assert args.added_edges_percent_of is not None, "added_edges_percent_of need to be specified when edges_percent is not None"

    if args.added_edges_percent_of not in ["GeneDisease", "GPSim"]:
        raise ValueError(
            "edges can only be added from extracted qualified edges of GeneDisease and GPSim dataset")

    assert args.edges_percent <= 1, "percent is expected to be between range of 0 to 1 "
    assert args.edges_percent > 0, "percent need to be more than 0. (please use dataset = 'no' and add_qulified_edges is None instead) "

## validation about use cases of add_qualified_edges
if args.add_qualified_edges is not None:
    if args.add_qualified_edges not in ['top_k']:
        raise ValueError("for add_qualified_edges, only top_k is implemented")

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
