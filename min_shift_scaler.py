#!/usr/bin/python3
# -*- coding: utf-8 -*-

from numbers import Real

import numpy as np

from scipy import sparse
from sklearn.utils import check_array
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import FLOAT_DTYPES, check_is_fitted
from sklearn.base import OneToOneFeatureMixin, TransformerMixin, BaseEstimator, _fit_context


class MinShiftScaler(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    _parameter_constraints: dict = {
        "offset": [Interval(Real, 0, None, closed="neither")], # (offset, +∞)
        "copy": ["boolean"],
        "clip": ["boolean"],
    }

    def __init__(self, offset=1, *, copy=True, clip=True):
        self.offset = offset
        self.copy = copy
        self.clip = clip

    def _reset(self):
        """Reset internal data-dependent state of the scaler, if necessary.

        __init__ parameters are not touched.
        """
        # Checking one attribute is enough, because they are all set together
        # in partial_fit
        if hasattr(self, "data_min_"):
            del self.data_min_
            del self.n_samples_seen_

    def fit(self, X, y=None):
        self._reset()
        return self.partial_fit(X, y)

    @_fit_context(prefer_skip_nested_validation=True)
    def partial_fit(self, X, y=None):
        """Online computation of min on X for later scaling.

        All of X is processed as a single batch. This is intended for cases
        when :meth:`fit` is not feasible due to very large number of
        `n_samples` or because X is read from a continuous stream.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to compute the min
            used for later scaling along the features axis.

        y : None
            Ignored.

        Returns
        -------
        self : object
            Fitted scaler.
        """
        if sparse.issparse(X):
            raise TypeError(
                "MinShiftScaler does not support sparse input."
            )
        first_pass = not hasattr(self, "n_samples_seen_")
        X = self._validate_data(
            X,
            reset=first_pass,
            dtype=FLOAT_DTYPES,
            force_all_finite="allow-nan",
        )
        data_min = np.nanmin(X, axis=0)
        if first_pass:
            self.n_samples_seen_ = X.shape[0]
        else:
            data_min = np.minimum(self.data_min_, data_min)
            self.n_samples_seen_ += X.shape[0]
        self.data_min_ = data_min
        return self

    def transform(self, X):
        """Scale features of X according to per feature minimums and offset.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data that will be transformed.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_features)
            Transformed data.
        """
        check_is_fitted(self)

        X = self._validate_data(
            X,
            copy=self.copy,
            dtype=FLOAT_DTYPES,
            force_all_finite="allow-nan",
            reset=False,
        )

        X = X - self.data_min_ + self.offset
        if self.clip:
            np.clip(X, 1e-15, np.inf, out=X)
        return X

    def inverse_transform(self, X):
        """Undo the scaling of X according to data minimum and offset.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data that will be transformed. It cannot be sparse.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_features)
            Transformed data.
        """
        check_is_fitted(self)

        X = check_array(
            X, copy=self.copy, dtype=FLOAT_DTYPES, force_all_finite="allow-nan"
        )

        X = X + self.min_ - self.offset
        return X

    def _more_tags(self):
        return {"allow_nan": True}


if __name__ == '__main__':
    scaler = MinShiftScaler(offset=1., clip=True, copy=True)
    print(scaler)
