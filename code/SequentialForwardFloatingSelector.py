from SequentialBackwardSelector import SequentialBackwardSelector
from SequentialForwardSelector import SequentialForwardSelector


class SequentialForwardFloatingSelector(SequentialForwardSelector, SequentialBackwardSelector):

    def __init__(self, model, objective_function_type='wrapper', folds=5):
        super().__init__(model, objective_function_type, folds)

    def fit(self, X, y, k_features):
        """
        Run the SFFS algorithm
        :param X: Features
        :param y: target variable
        :param k_features: number of features to be selected
        :return: best features
        """
        self._X = X
        self._y = y
        self._best_features = []

        self._k_features = k_features
        argmax = [0] * (self._k_features + 1)
        k = 0

        while k < self._k_features:

            significance_score, i = self._most_significant(self._best_features)  # calls SFS
            self._best_features.append(i)

            if k < 2:
                k += 1
                argmax[k] = significance_score
            else:
                significance_score_r, r = self._least_significant(self._best_features)  # calls SBS
                if r == i:
                    k += 1
                    argmax[k] = significance_score
                else:
                    features = self._best_features.copy()
                    features.remove(r)
                    if significance_score_r > argmax[k]:
                        if k == 2:
                            argmax[k] = significance_score_r
                            k += 1
                        else:
                            stop = False
                            while not stop:
                                significance_score_s, s = self._least_significant(features)
                                if significance_score_s <= argmax[k - 1]:
                                    self._best_features = features.copy()
                                    argmax[k] = significance_score_r
                                    stop = True
                                else:
                                    features.remove(s)
                                    k -= 1
                                    if k == 2:
                                        self._best_features = features.copy()
                                        stop = True
                    else:
                        k += 1
                        argmax[k] = significance_score

            print("Iteration: ", k, "Best Features: ", self._best_features)