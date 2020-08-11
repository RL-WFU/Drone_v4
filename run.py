import argparse
from Training.train_search import *
from Training.train_tracing import *
from Training.train_full import *
from Training.train_selection import *
from Testing.full_testing import *
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_model', default='selection', type=str, help='model to train: search, trace, or full')
    parser.add_argument('--target_selection', default=True, type=bool, help='select target with agent')
    parser.add_argument('--search_weights', default="search_weights", type=str, help='weights to load for search model')
    parser.add_argument('--trace_weights', default="trace_weights", type=str, help='weights to load for trace model')
    parser.add_argument('--target_selection_weights', default=None, type=str, help='weights to load for selection model')
    parser.add_argument('--training', default=True, type=bool, help='training or testing model')
    args = parser.parse_args()

    if args.training:
        if args.train_model == 'search':
            train_search_agent(args.search_weights)

        elif args.train_model == 'trace':
            train_tracing_agent('Weights_final/trace_weights')

        elif args.train_model == 'selection':
            train_selection(not args.target_selection, args.search_weights, args.trace_weights)

        elif args.train_model == 'full':
            train_full_model(not args.target_selection, 'Training_results/Weights/search_full_model_weights_1',
                             'trace_weights', args.target_selection_weights)

    else:
        print('Testing...')
        test_full_model(not args.target_selection, 'search_weights',
                        'trace_weights')
