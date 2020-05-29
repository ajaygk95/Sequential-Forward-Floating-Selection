"""
Feature selection Algorithm
    -by Abhiram M Kaushik 112686262
        Ajay Gopal Krishna 112688765
"""
import argparse
import numpy as np

from sklearn.neighbors import KNeighborsClassifier

from SequentialForwardSelector import SequentialForwardSelector
from SequentialForwardFloatingSelector import SequentialForwardFloatingSelector
from util import timeit


class Distance:
    def __init__(self, distance):
        self.__distance = distance

    def get_significance(self, X, y):
        """
        Calculated significance using Mahalanobis Distance.
        Ref: https://aip.scitation.org/doi/pdf/10.1063/1.4915708.
        Can be overridden.
        :param X: Features
        :param y: Target variable
        :return: significance of features in X with respect to class in y
        """
        x0 = []
        x1 = []
        for i in range(len(y)):
            if y[i] == 1:
                x1.append(X[i])
            else:
                x0.append(X[i])

        x1 = np.array(x1)
        x0 = np.array(x0)

        x1_std = np.std(x1)
        x0_std = np.std(x0)

        if x0_std == 0 or x1_std == 0:
            return 0

        x1_mean = np.mean(x1, axis=0) / x1_std
        x0_mean = np.mean(x0, axis=0) / x0_std

        x1_diff = (x1 - x1_mean)
        x0_diff = (x0 - x0_mean)

        c = (np.dot(x1_diff.T, x1_diff) + np.dot(x0_diff.T, x0_diff))
        if len(X.shape) == 1:
            if c == 0:
                return 0
            ci = 1 / c
        else:
            ci = np.linalg.pinv(c)
        d2 = np.dot(np.dot((x1_mean - x0_mean), ci), (x1_mean - x0_mean))
        return d2


def _parse_args():
    """
    parse arguments
    :return: Parser object
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=False, help='Path to dataset', default='mushroom.csv', type=str)
    parser.add_argument('--objective_type', required=False, help='Wrapper or Filter objective functions. Wrappers '
                                                                 'must implement \'fit\' and \'predict\' functions.'
                                                                 'Wrappers for now support only \'KNN\'.'
                                                                 'Filters for now only support \'Mahalanobis\' '
                                                                 'distance.',
                        default='wrapper', type=str)
    parser.add_argument('--features', required=False, help='k most significance features', default=5, type=int)
    parser.add_argument('--folds', required=False, help='Total folds while performing cross-validation', default=5,
                        type=int)
    parser.add_argument('--floating', required=False, help='Choose between SFFS or SFS', nargs='?', const=True,
                        default=False)

    return parser.parse_known_args()


if __name__ == '__main__':
    options, _ = _parse_args()  # parse arguments from command line
    data = np.genfromtxt(options.dataset, delimiter=',', skip_header=True)
    X = data[:, 1:]  # features - columns 1 to end
    y = data[:, 0]  # target variable - column 0

    if options.objective_type == 'wrapper':
        model = KNeighborsClassifier(n_neighbors=1)  # Wrapper method - 1-NN classifier
    else:
        model = Distance('Mahalanobis')  # Filter method - mahalanobis distance

    if options.floating:
        selector = SequentialForwardFloatingSelector(model, objective_function_type=options.objective_type,
                                                     folds=options.folds)  # SFFS
    else:
        selector = SequentialForwardSelector(model, objective_function_type=options.objective_type,
                                             folds=options.folds)  # SFS

    print("Using ", selector.__class__.__name__, "with arguments: ")
    print("\t Model: ", model.__class__.__name__)
    for arg in vars(options):
        print('\t', arg, ': ', getattr(options, arg))


    @timeit
    def fit(selector):
        selector.fit(X, y, options.features)


    fit(selector)
    print("Best Feature Indices: ", selector.best_features)
