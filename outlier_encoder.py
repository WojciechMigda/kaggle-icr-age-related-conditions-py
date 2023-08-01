#!/usr/bin/python3
# -*- coding: utf-8 -*-

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd


class OutlierEncoder(TransformerMixin, BaseEstimator):
    def __init__(self, fill_na=np.nan, version=1):
        self.fill_na = fill_na
        self.version = version

    def fit(self, X, y=None):
        return self

    @property
    def _to_floor(self):
        # AF AH AR AY BC BR BZ CB CD CF CL CS DF DI DU DV DY EB EH EP EU? FD FL FR FS GE GF 
        # moze AX CH CR EG 
        return (
            # AF:
            #   [192.59328,   372.04596,   505.09877,   562.28579,   751.02522, ...]
            ('AF', 210) if self.version == 1 else ('AF', 193), # v1, v2
            # AH:
            #   [85.200147 ,   86.349735 ,   87.7301115,   88.117662 , ...]
            ('AH', 85.8) if self.version == 1 else ('AH', 85.5), # v1, v2
            # AR:
            #   [8.138688,   8.204238,   8.23308 ,   8.254056,   8.2593  , ...]
            ('AR', 8.17) if self.version == 1 else ('AR', 8.15), # v1, v2
            # AY:
            #   [0.025578 ,  0.0258825,  0.026187 ,  0.0271005,  0.0277095, ...]
            ('AY', 0.0257) if self.version == 1 else ('AY', 0.02565), # v1, ???
            # BC:
            #   [1.22990000e+00, 1.99595200e+00, 2.05217600e+00, 2.09434400e+00, ...]
            ('BC', 1.291e+00), # v1, ???
            # BR:
            #   [5.12168830e+01, 1.53145557e+02, 1.58904205e+02, 1.81908553e+02, ...]
            ('BR', 5.38e+01), # v1, ???
            # BZ:
            #   [257.432377 ,   505.8933865,   525.1034205,   526.0421605, ...]
            ('BZ', 270.3) if self.version == 1 else ('BZ', 257.5), # v1, ???
            # CB:
            #   [12.49976  ,   12.646816 ,   12.665198 ,   12.701962 , ...]
            ('CB', 12.573), # v1, ???
            # CD:
            #   [23.3876  ,  26.820424,  31.783352,  32.845704,  34.24896 , ...]
            ('CD ', 24.56) if self.version == 1 else ('CD ', 23.6), # v1, v2
            # CF:
            #   [0.510888 ,   0.562585 ,   0.647733 ,   0.681184 ,   0.693348 , ...]
            ('CF', 0.5367) if self.version == 1 else ('CF', 0.52), # v1, ???
            # CL:
            #   [1.050225 ,  1.062955 ,  1.06932  ,  1.0725025,  1.08205  , ...]
            ('CL', 1.05659) if self.version == 1 else ('CL', 1.052), # v1, ???
            # CS:
            #   [13.784111 ,  17.0863   ,  17.12117  ,  17.3077245,  17.891797 , ...]
            ('CS', 14.47) if self.version == 1 else ('CS', 13.9), # v1, v2 ???
            # DF:
            #   [0.23868 ,  0.268866,  0.346788,  0.378729,  0.38961 ,  0.421551, ...]
            ('DF', 0.25377) if self.version == 1 else ('DF', 0.24), # v1, v2 ???
            # DI:
            #   [60.23247  ,   61.636815 ,   62.0302575,   62.984685 , ...]
            ('DI', 60.93) if self.version == 1 else ('DI', 60.3), # v1, v2

            (None, -1) if self.version == 1 else ('DL', 10.4), # v2

            # DU:
            #   [5.51760000e-03, 6.89700000e-03, 1.37940000e-02, 2.06910000e-02, 3.44850000e-02, ...]
            #('DU', 5.7935e-03),
            ('DU', 2.2e-02) if self.version == 1 else ('DU', 7.6e-03),
            # DV:
            #   [1.74307 ,  1.7612  ,  1.83372 ,  1.84408 ,  1.9166  ,  1.99948 , ...]
            ('DV', 1.7521) if self.version == 1 else ('DV', 1.745),
            # DY:
            #   [0.804068,   0.98824 ,   1.86418 ,   1.913592,   2.147176, ...]
            ('DY', 0.8442) if self.version == 1 else ('DY', 0.82),
            # EB:
            #   [4.926396,  5.011212,  5.110164,  5.301   ,  5.357544,  5.449428, ...]
            ('EB', 4.969) if self.version == 1 else ('EB', 4.93),
            # EH:
            #   [3.0420000e-03, 6.0840000e-03, 1.2168000e-02, 1.8252000e-02, ...]
            ('EH', 3.1941e-03),
            # EP:
            #   [78.526968 ,   79.778283 ,   79.815359 ,   80.26954  , ...]
            ('EP', 79.15) if self.version == 1 else ('EP', 78.6),
            # EU:
            #   [3.82838400e+00, 4.21071600e+00, 4.32465600e+00, 4.34997600e+00, ...]
            ('EU', 4.0196) if self.version == 1 else ('EU', 3.9),
            # FD:
            #   [2.96850000e-01, 3.26535000e-01, 3.32472000e-01, 3.74031000e-01, ...]
            ('FD ', 3.117e-01) if self.version == 1 else ('FD ', 3.05e-01),
            # FL:
            #   [0.173229  ,   0.2373    ,   0.265776  ,   0.280805  , ...]
            ('FL', 0.1819),
            # FR:
            #   [4.97060000e-01, 5.71590000e-01, 5.77100000e-01, 5.78840000e-01, ...]
            ('FR', 5.3432e-01) if self.version == 1 else ('FR', 5.10e-01),
            # FS:
            #   [0.06773 ,  0.074503,  0.081276,  0.088049,  0.094822,  0.101595, ...]
            ('FS', 0.07112) if self.version == 1 else ('FS', 0.069),
            # GE:
            #   [72.611063 ,   73.207358 ,   73.348702 ,   73.529799 , ...]
            ('GE', 72.909) if self.version == 1 else ('GE', 72.7),
            # GF:
            #   [1.30388940e+01, 1.42913889e+02, 1.60814484e+02, 1.97429238e+02, ...]
            ('GF', 1.369e+01) if self.version == 1 else (None, -1), # korekta

            # AX:
            #   [0.699861 ,  1.009926 ,  1.169388 ,  1.638915 ,  1.674351 , ...]
            ('AX', 0.8549) if self.version == 1 else ('AX', 0.80),
            # CH:
            #   [0.003184, 0.012338, 0.012736, 0.013134, 0.01393 , 0.014129, ...]
            ('CH', 0.007761),
            # CR:
            #   [0.069225 , 0.175275 , 0.20835  , 0.231825 , 0.2377875, 0.24015  , ...]
            ('CR', 0.12225),
            # EG:
            #   [185.5941   ,   193.3452625,   218.3701   ,   259.129225 , ...]
            ('EG', 189.47) if self.version == 1 else ('EG', 187),

            # AZ:
            #   [3.396778,  3.724482,  4.099451,  4.487024,  4.613064,  4.701292, ...]
            ('AZ', 3.56) if self.version == 1 else ('AZ', 3.50),
            # CW:
            #   [7.03064 , 14.310416, 14.987444, 15.063252, 15.59722 , 16.085096, ...]
            ('CW ', 7.38) if self.version == 1 else ('CW ', 7.1),
            # FI:
            #   [3.58345  ,  4.542712 ,  4.708102 ,  4.917596 ,  5.1960025, ...]
            ('FI', 3.76) if self.version == 1 else ('FI', 3.6),
            # GL:
            #   [1.12927800e-03, 1.31707300e-03, 1.41428500e-03, 2.38554200e-03, ...]
            ('GL', 1.186e-3) if self.version == 1 else (None, -1),
        )

    @property
    def _to_ceil(self):
        return (
            # BQ:
            #   [..., 320.006015 , 321.52994  , 329.31889  , 334.59662  , 344.644105]
            ('BQ', 340) if self.version == 1 else ('BQ', 344.6),
            # EL:
            #   [..., 88.648989 ,  90.11574  ,  90.3124365,  91.4737395,  94.230708 , 109.125159]
            ('EL', 103.9) if self.version == 1 else ('EL', 109),
            # GL:
            #   [..., 1.54440000e+01, 1.60380000e+01, 2.19780000e+01]
            ('GL', 2.09) if self.version == 1 else ('GL', 21.9),
            # CR:
            #   [..., 1.5825   , 1.6344   , 1.697025 , 1.780725 , 2.1472875, 2.404275 , 3.039675]
            ('CR', 2.895),
        )

    @staticmethod
    def _floored_name(col: str):
        return f'dummy {col.strip()}_floored'

    @staticmethod
    def _ceiled_name(col: str):
        return f'dummy {col.strip()}_ceiled'

    def _do_floor(self, X: pd.DataFrame, col: str, thr: float):
        new_col: str = self._floored_name(col)
        X[new_col] = (X[col] < thr).astype(int)
        if self.fill_na is not None:
            X.loc[X[new_col].astype(bool), col] = self.fill_na

    def _do_ceil(self, X: pd.DataFrame, col: str, thr: float):
        new_col: str = self._ceiled_name(col)
        X[new_col] = (X[col] > thr).astype(int)
        if self.fill_na is not None:
            X.loc[X[new_col].astype(bool), col] = self.fill_na

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"{self.__class__.__name__} requires pandas DataFrame as input")
        # definitely
        #   floor: AZ, CW, FI, GL
        #   ceil: BQ, EL, GL
        # sus
        #   floor: CR
        #   ceil: AM, AR, CR
        X = X.copy()
        for col, thr in self._to_floor:
            if col is not None:
                self._do_floor(X, col, thr)

        for col, thr in self._to_ceil:
            if col is not None:
                self._do_ceil(X, col, thr)
        return X

    @staticmethod
    def _get_feature_names_out(this, input_features: list):
        return (input_features + 
            [this._floored_name(col) for col, _ in this._to_floor] + 
            [this._ceiled_name(col) for col, _ in this._to_ceil])
