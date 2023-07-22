#!/usr/bin/python3
# -*- coding: utf-8 -*-

from sklearn.base import (
    OneToOneFeatureMixin, # Provides get_feature_names_out for simple transformers.
    TransformerMixin,
    BaseEstimator)
import pandas as pd


class MonitoringTransformer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    def __init__(self, tag, log_type):
        self.tag = tag
        self.log_type = log_type

    def select_log(self, X):
        to_show = None
        if self.log_type == 'head':
            to_show = pd.DataFrame(X).head()
        elif self.log_type == 'describe':
            to_show = pd.DataFrame(X).describe()
        return to_show

    def fit(self, X, y=None):
        to_show = self.select_log(X)
        print(f'[{self.__class__.__name__}][{self.tag}] Fitting dataframe\n{to_show}')
        return self

    def transform(self, X):
        to_show = self.select_log(X)
        print(f'[{self.__class__.__name__}][{self.tag}] Transforming dataframe\n{to_show}')
        return X
