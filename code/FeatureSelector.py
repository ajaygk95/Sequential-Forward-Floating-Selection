from abc import ABC, abstractmethod
import numpy as np
from sklearn.model_selection import cross_val_score


class FeatureSelector(ABC):  # Base class of feature selection
    def __init__(self, model, objective_function_type='wrapper', folds=5):
        self._best_features = []
        self._folds = folds
        self._objective_function_type = objective_function_type
        self._model = model
        self._criterion = self._get_criterion_function()

    def _get_criterion_function(self):
        """
        decide the criterion function based on objective criterion function
        :return: function pointer
        """
        if self._objective_function_type == 'wrapper':
            return lambda x, y: np.mean(cross_val_score(self._model, x, y, cv=self._folds))
        else:
            return self._model.get_significance

    @property
    def best_features(self):
        """
        Returns the best features
        """
        return sorted(self._best_features)

    @abstractmethod
    def fit(self, X, y, k_features):
        pass
