#!/usr/bin/python3
# -*- coding: utf-8 -*-

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd


class OutlierEncoder(TransformerMixin, BaseEstimator):
    def __init__(self, fill_na=np.nan):
        self.fill_na = fill_na

    def fit(self, X):
        return self

    @property
    def _to_floor(self):
        return (
            # AZ:
            #   [3.396778,  3.724482,  4.099451,  4.487024,  4.613064,  4.701292, ...]
            ('AZ', 3.56),
            # CW:
            #   [7.03064 , 14.310416, 14.987444, 15.063252, 15.59722 , 16.085096, ...]
            ('CW ', 7.38),
            # FI:
            #   [3.58345  ,  4.542712 ,  4.708102 ,  4.917596 ,  5.1960025, ...]
            ('FI', 3.76),
            # GL:
            #   [1.12927800e-03, 1.31707300e-03, 1.41428500e-03, 2.38554200e-03, ...]
            ('GL', 1.186e-3),
        )

    @property
    def _to_ceil(self):
        return (
            # BQ:
            #   [..., 320.006015 , 321.52994  , 329.31889  , 334.59662  , 344.644105]
            ('BQ', 340),
            # EL:
            #   [..., 88.648989 ,  90.11574  ,  90.3124365,  91.4737395,  94.230708 , 109.125159]
            ('EL', 103.9),
            # GL:
            #   [..., 1.54440000e+01, 1.60380000e+01, 2.19780000e+01]
            ('GL', 2.09),
        )

    @staticmethod
    def _floored_name(col: str):
        return f'{col.strip()}_floored'

    @staticmethod
    def _ceiled_name(col: str):
        return f'{col.strip()}_ceiled'

    def _do_floor(self, X: pd.DataFrame, col: str, thr: float):
        new_col: str = self._floored_name(col)
        X[new_col] = X[col] < thr
        if self.fill_na is not None:
            X.loc[X[new_col], col] = self.fill_na

    def _do_ceil(self, X: pd.DataFrame, col: str, thr: float):
        new_col: str = self._ceiled_name(col)
        X[new_col] = X[col] > thr
        if self.fill_na is not None:
            X.loc[X[new_col], col] = self.fill_na

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"{self.__class__.__name__} requires pandas DataFrame as input")
        # definitely
        #   floor: AZ, CW, FI, GL
        #   ceil: BQ, EL, GL
        # sus
        #   floor: CR
        #   ceil: AM, AR, CR
        for col, thr in self._to_floor:
            self._do_floor(X, col, thr)

        for col, thr in self._to_ceil:
            self._do_ceil(X, col, thr)
        return X

    @staticmethod
    def _get_feature_names_out(this, input_features: list):
        return (input_features + 
            [this._floored_name(col) for col, _ in this._to_floor] + 
            [this._ceiled_name(col) for col, _ in this._to_ceil])
