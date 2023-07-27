#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np

from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils.validation import _check_feature_names_in


class ProperPolynomialFeatures(PolynomialFeatures):
    """Difference wrt. PolynomialFeatures is ability to
    specify operator string (prod_op) used to concatenate feature
    names to produce new features.
    """
    def __init__(self, degree=2, *, interaction_only=False, include_bias=True, order="C", prod_op='×'):
        self.prod_op = prod_op
        super().__init__(degree=degree, interaction_only=interaction_only, include_bias=include_bias, order=order)
    def get_feature_names_out(self, input_features=None):
        powers = self.powers_
        input_features = _check_feature_names_in(self, input_features)
        feature_names = []
        for row in powers:
            inds = np.where(row)[0]
            if len(inds):
                name = self.prod_op.join(
                    (
                        "%s^%d" % (input_features[ind], exp)
                        if exp != 1
                        else input_features[ind]
                    )
                    for ind, exp in zip(inds, row[inds])
                )
            else:
                name = "1"
            feature_names.append(name)
        return np.asarray(feature_names, dtype=object)
