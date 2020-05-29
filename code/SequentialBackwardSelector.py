from FeatureSelector import FeatureSelector


class SequentialBackwardSelector(FeatureSelector):  # SBS

    def __init__(self, model, objective_function_type='wrapper', folds=5):
        super().__init__(model, objective_function_type, folds)
        self._k_features = None
        self._X = None
        self._y = None

    def _least_significant(self, best_features):
        """
        Find the least significant feature
        :param best_features:
        :return: return the significance and index of the least significant feature
        """
        significance_score = 0
        for j in best_features:
            features = best_features.copy()
            features.remove(j)
            x = self._X[:, features]
            score = self._criterion(x, self._y)
            if significance_score < score:
                significance_score = score
                r = j

        return significance_score, r

    def fit(self, X, y, k_features):
        """
        Run the SBS algorithm
        :param X: Features
        :param y: target variable
        :param k_features: number of features to be selected
        :return: best features
        """
        self._X = X
        self._y = y
        self._best_features = []

        self._k_features = k_features
        self._k_features = self._k_features if X.shape[1] >= self._k_features else X.shape[1]
        self._best_features, k_features = ([0], 1) if len(X.shape) < 2 else ([i for i in range(X.shape[1]-1,-1,-1)], X.shape[1])

        while k_features > self._k_features:
            _, r = self._least_significant(self._best_features)
            self._best_features.remove(r)
            k_features -= 1
