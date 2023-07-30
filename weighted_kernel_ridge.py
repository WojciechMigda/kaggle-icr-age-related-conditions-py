#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np

from sklearn.kernel_ridge import KernelRidge
from sklearn.base import _fit_context
from sklearn.utils._param_validation import StrOptions
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import compute_class_weight
from sklearn.utils.validation import _check_sample_weight


class WeightedKernelRidge(KernelRidge):
    _parameter_constraints: dict = {
        "class_weight": [dict, StrOptions({"balanced"}), None],
    }

    def __init__(
        self,
        alpha=1,
        *,
        kernel="linear",
        gamma=None,
        degree=3,
        coef0=1,
        kernel_params=None,
        class_weight=None
    ):
        self.class_weight = class_weight
        super().__init__(alpha=alpha, kernel=kernel, gamma=gamma, degree=degree, coef0=coef0, kernel_params=kernel_params)

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, sample_weight=None):
        # Convert data
        X, y = self._validate_data(
            X, y, accept_sparse=("csr", "csc"), multi_output=True, y_numeric=True
        )

        # If sample weights exist, convert them to array (support for lists)
        # and check length
        # Otherwise set them to 1 for all examples
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype, copy=True)

        # If class_weights is a dict (provided by the user), the weights
        # are assigned to the original labels. If it is "balanced", then
        # the class_weights are assigned after masking the labels with a OvR.
        classes = np.unique(y)
        le = LabelEncoder()
        if isinstance(self.class_weight, dict):
            class_weight_ = compute_class_weight(self.class_weight, classes=classes, y=y)
            sample_weight *= class_weight_[le.fit_transform(y)]
        
        # For doing a ovr, we need to mask the labels first. For the
        # multinomial case this is not necessary.
        if classes.size > 2:
            raise ValueError("To fit OvR, use the pos_class argument")
        # np.unique(y) gives labels in sorted order.
        pos_class = classes[1]

        mask = y == pos_class
        y_bin = np.ones(y.shape, dtype=X.dtype)
        # HalfBinomialLoss, used for those solvers, represents y in [0, 1] instead
        # of in [-1, 1].
        mask_classes = np.array([0, 1])
        y_bin[~mask] = 0.0
        
        # for compute_class_weight
        if self.class_weight == "balanced":
            class_weight_ = compute_class_weight(
                self.class_weight, classes=mask_classes, y=y_bin
            )
            sample_weight *= class_weight_[le.fit_transform(y_bin)]

        super().fit(X, y, sample_weight)
