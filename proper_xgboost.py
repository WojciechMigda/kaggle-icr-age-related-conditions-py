#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np

from xgboost import XGBRegressor, XGBClassifier


class ProperXGBRegressor(XGBRegressor):
    """Apply rounding to input X for fit()
    """
    def __init__(self, decimals=6, **kwargs):
        self.decimals = decimals
        super().__init__(**kwargs)

    def fit(self, X, y, **kwargs):
        return super().fit(X.round(self.decimals), y, **kwargs)


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
