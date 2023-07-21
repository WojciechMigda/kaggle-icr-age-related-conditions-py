#!/usr/bin/python3
# -*- coding: utf-8 -*-

from sklearn.base import (
    OneToOneFeatureMixin, # Provides get_feature_names_out for simple transformers.
    TransformerMixin,
    BaseEstimator)


class IdentityTransformer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    def __init__(self):
        return

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X
