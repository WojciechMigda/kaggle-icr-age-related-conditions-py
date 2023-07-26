#!/usr/bin/python3
# -*- coding: utf-8 -*-

import scipy as sp

from sklearn.base import RegressorMixin, _fit_context
from sklearn.linear_model import LinearModel, _preprocess_data


class IcrRegressor(RegressorMixin):
    _parameter_constraints: dict = {
        "fit_intercept": ["boolean"],
        "copy_X": ["boolean"],
    }

    def __init__(
        self,
        *,
        C=1.0,
        fit_intercept=True,
        copy_X=True,
        max_iter=1000,
        tol=1e-4,
        random_state=None
    ):
        self.C = C
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, sample_weight=None):
        X, y = self._validate_data(
            X, y, accept_sparse=accept_sparse, y_numeric=True, multi_output=True
        )
        has_sw = sample_weight is not None
        if has_sw:
            sample_weight = _check_sample_weight(
                sample_weight, X, dtype=X.dtype, only_non_negative=True
            )
        copy_X_in_preprocess_data = self.copy_X and not sp.issparse(X)

        X, y, X_offset, y_offset, X_scale = _preprocess_data(
            X,
            y,
            fit_intercept=self.fit_intercept,
            copy=copy_X_in_preprocess_data,
            sample_weight=sample_weight,
        )

        if has_sw:
            # Sample weight can be implemented via a simple rescaling. Note
            # that we safely do inplace rescaling when _preprocess_data has
            # already made a copy if requested.
            X, y, sample_weight_sqrt = _rescale_data(
                X, y, sample_weight, inplace=copy_X_in_preprocess_data
            )

        if sp.issparse(X):
            X_offset_scale = X_offset / X_scale

            if has_sw:

                def matvec(b):
                    return X.dot(b) - sample_weight_sqrt * b.dot(X_offset_scale)

                def rmatvec(b):
                    return X.T.dot(b) - X_offset_scale * b.dot(sample_weight_sqrt)

            else:

                def matvec(b):
                    return X.dot(b) - b.dot(X_offset_scale)

                def rmatvec(b):
                    return X.T.dot(b) - X_offset_scale * b.sum()

            X_centered = sparse.linalg.LinearOperator(
                shape=X.shape, matvec=matvec, rmatvec=rmatvec
            )

            if y.ndim < 2:
                self.coef_ = lsqr(X_centered, y)[0]
            else:
                # sparse_lstsq cannot handle y with shape (M, K)
                outs = Parallel(n_jobs=n_jobs_)(
                    delayed(lsqr)(X_centered, y[:, j].ravel())
                    for j in range(y.shape[1])
                )
                self.coef_ = np.vstack([out[0] for out in outs])
        else:
            self.coef_, _, self.rank_, self.singular_ = linalg.lstsq(X, y)
            self.coef_ = self.coef_.T

        if y.ndim == 1:
            self.coef_ = np.ravel(self.coef_)
        self._set_intercept(X_offset, y_offset, X_scale)
        return self

    def predict(self, X):
        return X
