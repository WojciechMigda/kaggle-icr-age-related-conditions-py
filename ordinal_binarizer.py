#!/usr/bin/python3
# -*- coding: utf-8 -*-

from typing import Union
from collections.abc import Sequence

import numpy as np
from sklearn.preprocessing import FunctionTransformer

class OrdinalBinarizer(FunctionTransformer):
    from functools import lru_cache

    @staticmethod
    @lru_cache(maxsize=4096)
    def _as_bits(x, nbits):
        s = '1' * x + '0' * (nbits - x)
        return np.array([int(c) for c in s])

    def __init__(self, nbits: Union[int, Sequence]) -> None:
        self.nbits = nbits
        super().__init__(func=self._func, validate=True, feature_names_out=self._get_feature_names_out)

    def _func(self, X):
        if X.ndim > 2:
            raise ValueError(f"Input array cannot have more than 2 dimensions, got {X.ndim}")

        if isinstance(self.nbits, int):
            X = np.clip(X, 0, self.nbits)
            X_ = np.empty_like(X, dtype=np.uint64)
            np.rint(X, out=X_, casting='unsafe')
            F = np.frompyfunc(self._as_bits, 2, 1)
            rv = np.stack(F(X_.ravel(), self.nbits)).reshape(X.shape[0], -1)
            return rv
        else:
            import itertools
            ncol = X.shape[1]
            assert ncol == len(self.nbits), f"nbits arrray length ({len(self.nbits)}) and X number of columns ({ncol}) mismatch."
            X = np.clip(X, [0] * ncol, self.nbits)
            X_ = np.empty_like(X, dtype=int)
            np.rint(X, out=X_, casting='unsafe')
            rv = np.fromiter(itertools.chain.from_iterable(
                      [self._as_bits(val, self.nbits[ix[1]]) for ix, val in np.ndenumerate(X_)]
                  ), int).reshape(X.shape[0], -1)
            return rv

    @staticmethod
    def _get_feature_names_out(this, input_features):
        feature_names = []
        for i in range(len(input_features)):
            if isinstance(this.nbits, int):
                names = [input_features[i] + "_" + str(t) for t in range(this.nbits)]
            else:
                names = [input_features[i] + "_" + str(t) for t in range(this.nbits[i])]
            feature_names.extend(names)
        return feature_names
