"""Assignment - making a sklearn estimator and cv splitter.

The goal of this assignment is to implement by yourself:

- a scikit-learn estimator for the KNearestNeighbors for classification
  tasks and check that it is working properly.
- a scikit-learn CV splitter where the splits are based on a Pandas
  DateTimeIndex.

Detailed instructions for question 1:
The nearest neighbor classifier predicts for a point X_i the target y_k of
the training sample X_k which is the closest to X_i. We measure proximity with
the Euclidean distance. The model will be evaluated with the accuracy (average
number of samples corectly classified). You need to implement the `fit`,
`predict` and `score` methods for this class. The code you write should pass
the test we implemented. You can run the tests by calling at the root of the
repo `pytest test_sklearn_questions.py`. Note that to be fully valid, a
scikit-learn estimator needs to check that the input given to `fit` and
`predict` are correct using the `validate_data, check_is_fitted` functions
imported in this file.
You can find more information on how they should be used in the following doc:
https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator.
Make sure to use them to pass `test_nearest_neighbor_check_estimator`.


Detailed instructions for question 2:
The data to split should contain the index or one column in
datatime format. Then the aim is to split the data between train and test
sets when for each pair of successive months, we learn on the first and
predict of the following. For example if you have data distributed from
november 2020 to march 2021, you have have 4 splits. The first split
will allow to learn on november data and predict on december data, the
second split to learn december and predict on january etc.

We also ask you to respect the pep8 convention: https://pep8.org. This will be
enforced with `flake8`. You can check that there is no flake8 errors by
calling `flake8` at the root of the repo.

Finally, you need to write docstrings for the methods you code and for the
class. The docstring will be checked using `pydocstyle` that you can also
call at the root of the repo.

Hints
-----
- You can use the function:

from sklearn.metrics.pairwise import pairwise_distances

to compute distances between 2 sets of samples.
"""
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin

from sklearn.model_selection import BaseCrossValidator

from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import validate_data
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils.multiclass import unique_labels


class KNearestNeighbors(ClassifierMixin, BaseEstimator):
    """KNearestNeighbors classifier."""

    def __init__(self, n_neighbors=1):  # noqa: D107
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        """Fitting function.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data to train the model.
        y : ndarray, shape (n_samples,)
            Labels associated with the training data.

        Returns
        -------
        self : instance of KNearestNeighbors
            The current instance of the classifier
        """
        X, y = validate_data(self, X, y)
        # Validate n_neighbors
        if not (isinstance(self.n_neighbors, (int, np.integer))
                and self.n_neighbors >= 1):
            raise ValueError("n_neighbors must be a positive integer")
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X):
        """Predict function.

        Parameters
        ----------
        X : ndarray, shape (n_test_samples, n_features)
            Data to predict on.

        Returns
        -------
        y : ndarray, shape (n_test_samples,)
            Predicted class labels for each test data sample.
        """
        # Input validation
        X = validate_data(self, X, reset=False)
        # Check if fit has been called
        check_is_fitted(self)
        # Estimation of the distances
        distances = pairwise_distances(X, self.X_, metric='euclidean')
        # Use at most the number of training samples when selecting neighbors
        k_eff = min(self.n_neighbors, self.X_.shape[0])
        nbr_index = np.argsort(distances, axis=1)[:, :k_eff]
        nbr_labels = self.y_[nbr_index]
        # Majority vote: find most common label (works for int and float)
        y_pred = np.array([
            np.unique(row, return_counts=True)[0][
                np.argmax(np.unique(row, return_counts=True)[1])
            ]
            for row in nbr_labels
        ])
        return y_pred

    def score(self, X, y):
        """Calculate the score of the prediction.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data to score on.
        y : ndarray, shape (n_samples,)
            target values.

        Returns
        ----------
        score : float
            Accuracy of the model computed for the (X, y) pairs.
        """
        y_pred = self.predict(X)
        # Calculation of the accuracy
        accurate = (y_pred == y)
        score = np.where(accurate, 1, 0)
        accuracy = score.sum()/len(y)
        return accuracy


class MonthlySplit(BaseCrossValidator):
    """CrossValidator based on monthly split.

    Split data based on the given `time_col` (or default to index). Each split
    corresponds to one month of data for the training and the next month of
    data for the test.

    Parameters
    ----------
    time_col : str, defaults to 'index'
        Column of the input DataFrame that will be used to split the data. This
        column should be of type datetime. If split is called with a DataFrame
        for which this column is not a datetime, it will raise a ValueError.
        To use the index as column just set `time_col` to `'index'`.
    """

    def __init__(self, time_col='index'):  # noqa: D107
        self.time_col = time_col

    def _get_time_index(self, X):
        """Return DatetimeIndex from X using ``time_col`` or the index.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
            can be Series or Dataframe

        Returns
        -------
        pd.DatetimeIndex
        a data frame which column time_col is datetime format
        """
        if type(X) is pd.DataFrame:
            idx_source = X.index if self.time_col == "index"\
                                 else X[self.time_col]

        elif type(X) is pd.Series:
            idx_source = X.index if self.time_col == "index" else X

        else:
            raise ValueError("X must be a pandas DataFrame or Series")

        if not pd.api.types.is_datetime64_any_dtype(idx_source):
            raise ValueError(
                f"The column '{self.time_col}' must be of datetime type."
            )

        return pd.DatetimeIndex(idx_source)

    def get_n_splits(self, X, y=None, groups=None):
        """Return the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.

        Returns
        -------
        n_splits : int
            The number of splits.
        """
        time_index = self._get_time_index(X)
        # Group data by months by year (BEFORE validate_data converts to array)
        months = time_index.to_period('M').unique().sort_values()

        # Count the number of successive months
        n_splits = len(months) - 1

        return max(0, n_splits)

    def split(self, X, y, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.

        Yields
        ------
        idx_train : ndarray
            The training set indices for that split.
        idx_test : ndarray
            The testing set indices for that split.
        """
        time_index = self._get_time_index(X)
        time_periods = time_index.to_period('M').unique().sort_values()
        months_for_train = time_periods[:-1]
        months_for_test = time_periods[1:]
        for month_train, month_test in zip(months_for_train, months_for_test):
            train = np.where(time_index.to_period("M") == month_train)[0]
            test = np.where(time_index.to_period("M") == month_test)[0]
            yield (train, test)
