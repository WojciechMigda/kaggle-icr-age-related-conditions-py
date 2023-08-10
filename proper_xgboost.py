#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np

from xgboost import XGBRegressor, XGBClassifier


class ProperXGBRegressor(XGBRegressor):
    """Apply rounding to input X for fit()
    """
    def __init__(self, decimals=6, avoid_rows=[], **kwargs):
        self.decimals = decimals
        self.avoid_rows = avoid_rows
        super().__init__(**kwargs)

    def fit(self, X, y, **kwargs):
        new_X = X.round(self.decimals).drop(self.avoid_rows, errors='ignore')
        super().fit(
            new_X,
            y[X.index.isin(self.avoid_rows) == False],
            **kwargs)
        return self


class ProperXGBClassifier(XGBClassifier):
    """Apply rounding to input X for fit()
    """
    def __init__(self, decimals=6, **kwargs):
        self.decimals = decimals
        super().__init__(**kwargs)

    def fit(self, X, y, **kwargs):
        return super().fit(X.round(self.decimals), y, **kwargs)

    def predict(self, *args, **kwargs):
        return super().predict_proba(*args, **kwargs)[:, 1]
