#!/usr/bin/python3
# -*- coding: utf-8 -*-

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd


class BinSplitter(TransformerMixin, BaseEstimator):
    def __init__(self, drop_source=True):
        self.drop_source = drop_source
        return

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"{self.__class__.__name__} requires pandas DataFrame as input")
        X = X.copy()
        for col_name, *pivots in self._to_bin:
            source_col = X[col_name].values
            bin_ix = np.digitize(source_col, pivots)
            for ix in range(len(pivots) + 1):
                new_col = np.full_like(source_col, np.nan)
                mask = (bin_ix == ix)
                new_col[mask] = source_col[mask]
                X[self._col_name(col_name, ix)] = new_col
        if self.drop_source:
            X.drop([c for c, *_ in self._to_bin], inplace=True, axis=1)
        return X

    @staticmethod
    def _col_name(col: str, ix: int):
        return f'{col.strip()}_{chr(97 + ix)}'

    @property
    def _to_bin(self):
        return (
            ('AF', 193),
            ('AH', 85.5),
            ('AR', 8.15),
            ('AX', 0.80),
            ('AY', 0.02565),
            ('AZ', 3.50),
            ('BC', 1.291e+00),
            ('BQ', 344.6),
            ('BR', 5.38e+01),
            ('BZ', 257.5),
            ('CB', 12.573),
            ('CD ', 23.6),
            ('CF', 0.52),
            ('CH', 0.007761),
            ('CL', 1.052),
            ('CR', 0.12225, 2.895),
            ('CS', 13.9),
            ('CW ', 7.1, 27.0),
            ('DF', 0.24),
            ('DI', 60.3),
            ('DL', 10.4),
            ('DU', 7.6e-03),
            ('DV', 1.745),
            ('DY', 0.82),
            ('EB', 4.93),
            ('EG', 187),
            ('EH', 3.1941e-03),
            ('EL', 109),
            ('EP', 78.6),
            ('EU', 3.9),
            ('FD ', 3.05e-01),
            ('FI', 3.6),
            ('FL', 0.1819),
            ('FR', 5.10e-01),
            ('FS', 0.069),
            ('GE', 72.7),
            ('GL', 21.9),
        )

    @staticmethod
    def _get_feature_names_out(this, input_features: list):
        return (input_features +
            [this._col_name(col, ix) for col, *pivots in this._to_bin for ix in range(len(pivots) + 1)])
