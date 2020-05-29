from FeatureSelector import FeatureSelector


class SequentialForwardSelector(FeatureSelector):

    def __init__(self, model, objective_function_type='wrapper', folds=5):
        super().__init__(model, objective_function_type, folds)
        self._k_features = None
        self._X = None
        self._y = None

    def _most_significant(self, best_features):
        """
        Find the most significant feature
        :param best_features:
        :return: return the significance and index of the most significant feature
        """
        significance_score = 0
        for j in range(self._X.shape[1]):
            if j in best_features:
                continue
            x = self._X[:, best_features + [j]]
            score = self._criterion(x, self._y)
            if significance_score < score:
                significance_score = score
                i = j

        return significance_score, i

    def fit(self, X, y, k_features):
        """
        Run the SFS algorithm
        :param X: Features
        :param y: target variable
        :param k_features: number of features to be selected
        :return: best features
        """
        self._X = X
        self._y = y
        self._best_features = []

        self._k_features = k_features
        k = 0
        self._k_features = self._k_features if X.shape[1] >= self._k_features else X.shape[1]

        while k < self._k_features:
            _, i = self._most_significant(self._best_features)
            self._best_features.append(i)
            k += 1

            print("Iteration: ", k, "Best Features: ", self._best_features)