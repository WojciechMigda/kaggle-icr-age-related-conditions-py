#!/usr/bin/python3
# -*- coding: utf-8 -*-

import collections

from sklearn.base import (
    OneToOneFeatureMixin, # Provides get_feature_names_out for simple transformers.
    TransformerMixin,
    BaseEstimator)
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import MinMaxScaler
from .min_shift_scaler import MinShiftScaler
from sklearn.pipeline import make_pipeline
from sklearn.base import clone as clone_estimator

import numpy as np
from scipy import stats


try:
    from sklearn.utils._param_validation import Hidden, Interval, Integral
except ImportError:
    def Hidden(*args):
        "Dummy replacement for a class introduced in scikit-learn==1.1."
        return None
    def Interval(*args):
        "Dummy replacement for a class introduced in scikit-learn==1.1."
        return None
    def Integral(*args):
        "Dummy replacement for a class introduced in scikit-learn==1.1."
        return None


# https://github.com/scipy/scipy/blob/v1.11.1/scipy/stats/_morestats.py#L372
def _calc_uniform_order_statistic_medians(n):
    """Approximations of uniform order statistic medians.

    Parameters
    ----------
    n : int
        Sample size.

    Returns
    -------
    v : 1d float array
        Approximations of the order statistic medians.

    References
    ----------
    .. [1] James J. Filliben, "The Probability Plot Correlation Coefficient
           Test for Normality", Technometrics, Vol. 17, pp. 111-117, 1975.

    Examples
    --------
    Order statistics of the uniform distribution on the unit interval
    are marginally distributed according to beta distributions.
    The expectations of these order statistic are evenly spaced across
    the interval, but the distributions are skewed in a way that
    pushes the medians slightly towards the endpoints of the unit interval:

    >>> import numpy as np
    >>> n = 4
    >>> k = np.arange(1, n+1)
    >>> from scipy.stats import beta
    >>> a = k
    >>> b = n-k+1
    >>> beta.mean(a, b)
    array([0.2, 0.4, 0.6, 0.8])
    >>> beta.median(a, b)
    array([0.15910358, 0.38572757, 0.61427243, 0.84089642])

    The Filliben approximation uses the exact medians of the smallest
    and greatest order statistics, and the remaining medians are approximated
    by points spread evenly across a sub-interval of the unit interval:

    >>> from scipy.stats._morestats import _calc_uniform_order_statistic_medians
    >>> _calc_uniform_order_statistic_medians(n)
    array([0.15910358, 0.38545246, 0.61454754, 0.84089642])

    This plot shows the skewed distributions of the order statistics
    of a sample of size four from a uniform distribution on the unit interval:

    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(0.0, 1.0, num=50, endpoint=True)
    >>> pdfs = [beta.pdf(x, a[i], b[i]) for i in range(n)]
    >>> plt.figure()
    >>> plt.plot(x, pdfs[0], x, pdfs[1], x, pdfs[2], x, pdfs[3])

    """
    v = np.empty(n, dtype=np.float64)
    v[-1] = 0.5**(1.0 / n)
    v[0] = 1 - v[-1]
    i = np.arange(2, n)
    v[1:-1] = (i - 0.3175) / (n + 0.365)
    return v


class DistributionTransformer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    _parameter_constraints: dict = {
        "transforms": [list, Hidden(tuple)],
        "n_quantiles": [Interval(Integral, 1, None, closed="left")],
        "random_state": ["random_state"],
        "copy": ["boolean"],
        "verbose": ["boolean"],
    }

    def __init__(self, transforms=['original', 'yeo-johnson', 'box-cox-0-1', 'box-cox-1-2', 'box-cox-0', 'box-cox-1', 'quantile'], n_quantiles=1000, random_state=None, copy=True, verbose=False):
        self.transforms = transforms
        self.n_quantiles = n_quantiles
        self.random_state = random_state
        self.copy = copy
        self.verbose = verbose

    def _check_input(self, X, *, in_fit: bool):
        from sklearn.utils.validation import FLOAT_DTYPES
        return self._validate_data(X,
            ensure_2d=True,
            dtype=FLOAT_DTYPES,
            copy=self.copy,
            force_all_finite="allow-nan",
            reset=in_fit,
        )

    def _validate_transforms(self):
        if not self.transforms:
            return
        allowed = ('original', 'yeo-johnson', 'box-cox-0-1', 'box-cox-1-2', 'box-cox-0', 'box-cox-1', 'quantile')
        for xform in self.transforms:
            if xform not in allowed:
                raise TypeError(
                    f"Elements of transforms parameter must be one of {{{', '.join(allowed)}}}"
                )

    def fit(self, X, y=None):
        self._validate_params()
        self._validate_transforms()

        # reduce to unique elements
        self.transforms = list(collections.OrderedDict.fromkeys(self.transforms))

        Rs = np.empty((len(self.transforms), X.shape[1]), dtype=np.float64)
        dist = stats.distributions.norm
        transformers = {
            'yeo-johnson' : PowerTransformer(method="yeo-johnson"),
            'box-cox-0-1' : make_pipeline(
                # "Box-Cox requires input data to be strictly positive"
                MinMaxScaler((1e-15, 1), copy=True),
                FunctionTransformer(np.clip, kw_args={'a_min': 1e-15, 'a_max': None}),
                PowerTransformer(method="box-cox", copy=False)
            ),
            'box-cox-1-2' : make_pipeline(
                # "Box-Cox requires input data to be strictly positive"
                MinMaxScaler((1, 2), copy=True),
                FunctionTransformer(np.clip, kw_args={'a_min': 1e-15, 'a_max': None}),
                PowerTransformer(method="box-cox", copy=False)
            ),
            'box-cox-0' : make_pipeline(
                # "Box-Cox requires input data to be strictly positive"
                MinShiftScaler(offset=1e-15, clip=True, copy=True, verbose=self.verbose),
                PowerTransformer(method="box-cox", copy=False)
            ),
            'box-cox-1' : make_pipeline(
                # "Box-Cox requires input data to be strictly positive"
                MinShiftScaler(offset=1., clip=True, copy=True, verbose=self.verbose),
                PowerTransformer(method="box-cox", copy=False)
            ),
            'quantile' : QuantileTransformer(n_quantiles=self.n_quantiles, output_distribution="normal", random_state=self.random_state),
        }

        X = self._check_input(X, in_fit=True)

        for cix in range(X.shape[1]):
            Xi = X[:, cix]
            Xi = Xi[~np.isnan(Xi)]
            # check if input is not all equal values
            if len(Xi) == 0 or np.all(Xi == Xi[0]):
                # force-select 'original'
                for rix, xform in enumerate(self.transforms):
                    if xform == 'original':
                        Rs[rix, cix] = 1
                    else:
                        Rs[rix, cix] = 0
                continue

            osm_uniform = _calc_uniform_order_statistic_medians(len(Xi))
            Xi = np.atleast_2d(Xi).T
            osm = dist.ppf(osm_uniform)

            for rix, xform in enumerate(self.transforms):
                if xform == 'original':
                    osr = np.sort(Xi, axis=0)
                else:
                    osr = np.sort(transformers[xform].fit_transform(Xi), axis=0)

                Ri, _ = stats.pearsonr(osr.ravel(), osm)
                Rs[rix, cix] = Ri

        self.winners_ = np.argmax(Rs, axis=0)
        self.Rs_ = Rs

        estimators = [None] * X.shape[1]
        for cix in range(X.shape[1]):
            Xi = X[:, cix]
            Xi = np.atleast_2d(Xi).T
            xformer = transformers.get(self.transforms[self.winners_[cix]], None)
            if xformer is None:
                # already None in estimators[cix]
                continue
            xformer = clone_estimator(xformer)
            xformer.fit(Xi)
            estimators[cix] = xformer
        self.estimators_ = estimators

        return self

    def transform(self, X):
        if self.verbose:
            print(f"[{self.__class__.__name__}] Transforming matrix with shape {X.shape}")
        X = self._check_input(X, in_fit=False)

        for cix in range(X.shape[1]):
            xformer = self.estimators_[cix]
            if xformer is not None:
                try:
                    X[:, cix] = xformer.transform(np.atleast_2d(X[:, cix]).T).ravel()
                except ValueError as ex:
                    print(f"Transformer {xformer} failed for column {cix}, Xi={X[:, cix]}")
                    raise

        return X


def distribution_transformer_results(scaler, feature_names):
    if hasattr(scaler, "winners_"):
        import pandas as pd
        r_scores = pd.DataFrame(scaler.Rs_, index=scaler.transforms, columns=feature_names).T
        r_scores["Winner"] = r_scores.idxmax(axis=1)
        r_scores["R Best"] = r_scores.max(axis=1, numeric_only=True)
        return r_scores
    else:
        import warnings
        warnings.warn(f"{scaler.__class__.__name__} was not fitted.")
        return
