#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np

def make_lm_preprocessor(
    *,
    random_state=None,
    pre__verbose=False,
    oe__fill_na=None,
    ii__max_iter=50, ii__verbose=0,
    rfe__n_features_to_select=57, rfe__step=0.03, rfe__verbose=False, rfe__estimator=None,
):
    from sklearn.pipeline import Pipeline
    from .outlier_encoder import OutlierEncoder
    from sklearn.compose import ColumnTransformer, make_column_selector
    from .distribution_transformer import DistributionTransformer
    from sklearn.preprocessing import OneHotEncoder
    from .pickleable_bits import ohe_feature_name_combiner
    from sklearn.preprocessing import StandardScaler
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    from .proper_polynomial_features import ProperPolynomialFeatures
    from sklearn.pipeline import FeatureUnion
    from .identity_transformer import IdentityTransformer
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import RFE
    from sklearn import base
    from .dataset_columns import COL_X_CAT, COL_X_NUM

    preprocessor = Pipeline(
        [
            ('X encode outliers', OutlierEncoder(fill_na=oe__fill_na, version=2)),

            ('NUM re-distribute (1)', ColumnTransformer(transformers=[
                ("NUM DistributionTransformer", DistributionTransformer(transforms=['original', 'yeo-johnson', 'quantile'], random_state=random_state), COL_X_NUM),
            ], remainder='passthrough', verbose_feature_names_out=False, n_jobs=1).set_output(transform='pandas')),

            ('CAT one-hot', ColumnTransformer(transformers=[
                ("CAT OneHotEncoder", OneHotEncoder(drop=None, sparse_output=False, handle_unknown='ignore', feature_name_combiner=ohe_feature_name_combiner, dtype=int), COL_X_CAT),
                ("CAT drop", "drop", COL_X_CAT),
            ], remainder='passthrough', verbose_feature_names_out=False, n_jobs=1).set_output(transform='pandas')),

            ('NUM std scale', ColumnTransformer(transformers=[
                ("NUM StandardScaler", StandardScaler(), COL_X_NUM),
                ("DUMMY StandardScaler", StandardScaler(with_std=False), make_column_selector(pattern="dummy .*")),
            ], remainder='passthrough', verbose_feature_names_out=False, n_jobs=1).set_output(transform='pandas')),

            ('X impute', ColumnTransformer(transformers=[
                ('X IterativeImputer', IterativeImputer(verbose=ii__verbose, sample_posterior=True, max_iter=ii__max_iter, random_state=random_state), COL_X_NUM + ['dummy EJ_A', 'dummy EJ_B']),
            ], remainder='passthrough', verbose_feature_names_out=False, n_jobs=1).set_output(transform='pandas')),

            ('NUM re-distribute (2)', ColumnTransformer(transformers=[
                ("NUM DistributionTransformer", DistributionTransformer(transforms=['original', 'yeo-johnson', 'quantile'], random_state=random_state, verbose=False), COL_X_NUM),
            ], remainder='passthrough', verbose_feature_names_out=False, n_jobs=1).set_output(transform='pandas')),

            ('X poly', ColumnTransformer(transformers=[
                ('X PolynomialFeatures', ProperPolynomialFeatures(degree=2, interaction_only=True, include_bias=False, prod_op='×'), make_column_selector(pattern="(?!Class)"))
            ], remainder='passthrough', verbose_feature_names_out=False, n_jobs=1).set_output(transform='pandas')),
            
            #('X union', ColumnTransformer(transformers=[
            #    ('X PCA', FeatureUnion([('X', IdentityTransformer()), ('PCA', PCA(n_components=2, whiten=False, random_state=random_state))]), make_column_selector(pattern="(?!Class)"))
            #], remainder='passthrough', verbose_feature_names_out=False, n_jobs=1).set_output(transform='pandas')),

             ('X coarse feat selection', RFE(estimator=base.clone(rfe__estimator), n_features_to_select=0.5, step=rfe__step, verbose=rfe__verbose).set_output(transform='pandas')),
#             #('X coarse feat selection', RFE(estimator=ExtraTreesClassifier(random_state=random_state), n_features_to_select=0.5, step=rfe__step, verbose=rfe__verbose).set_output(transform='pandas')),
            
            ('POLY re-distribute (3)',
                ColumnTransformer(transformers=[
                    ("POLY DistributionTransformer", DistributionTransformer(transforms=['original', 'yeo-johnson', 'quantile'], random_state=random_state, verbose=False), make_column_selector(pattern=".*×.*")),
            ], remainder='passthrough', verbose_feature_names_out=False, n_jobs=1).set_output(transform='pandas')),
            
            #('PCA', PCA(n_components='mle', whiten=False, random_state=random_state)), # 42(+) 0.357410 43(+) 0.331301 44(+) 0.409705
            #('PCA', PCA(whiten=False, random_state=random_state)), # 42(+) 0.357410 43(+) 0.331301 44(+) 0.409705
            
            ('X feat selection', RFE(estimator=base.clone(rfe__estimator), n_features_to_select=rfe__n_features_to_select, step=rfe__step, verbose=rfe__verbose).set_output(transform='pandas')),
        ],
        verbose=pre__verbose,
    )

    return preprocessor


def make_xgbr_preprocessor(
    *,
    pre__verbose=False,
    bs__drop_source=False,
    mms__max=1,
):
    from sklearn.pipeline import Pipeline
    from .bin_splitter import BinSplitter
    from sklearn.compose import ColumnTransformer, make_column_selector
    from sklearn.preprocessing import OneHotEncoder
    from .pickleable_bits import ohe_feature_name_combiner
    from sklearn.preprocessing import MinMaxScaler
    #from sklearn.pipeline import FeatureUnion
    #from .identity_transformer import IdentityTransformer
    #from sklearn.decomposition import PCA
    #from sklearn.feature_selection import RFE
    #from sklearn import base
    from .dataset_columns import COL_X_CAT


    mms__max = np.clip(mms__max, 1, 10)
    
    preprocessor = Pipeline(
        [
            ('X split', BinSplitter(drop_source=bs__drop_source)),

            ('CAT one-hot', ColumnTransformer(transformers=[
                ("CAT OneHotEncoder", OneHotEncoder(drop=None, sparse_output=False, handle_unknown='ignore', feature_name_combiner=ohe_feature_name_combiner, dtype=int), COL_X_CAT),
                ("CAT drop", "drop", COL_X_CAT),
            ], remainder='passthrough', verbose_feature_names_out=False, n_jobs=1).set_output(transform='pandas')),

            ('X 0-n scale', ColumnTransformer(transformers=[
                ("NUM MinMaxScaler", MinMaxScaler(feature_range=(0, mms__max)), make_column_selector(pattern="(?!Class)")),
            ], remainder='passthrough', verbose_feature_names_out=False, n_jobs=1).set_output(transform='pandas')),

            #('X feat selection', RFE(estimator=base.clone(rfe__estimator), n_features_to_select=rfe__n_features_to_select, step=rfe__step, verbose=rfe__verbose).set_output(transform='pandas')),
        ],
        verbose=pre__verbose,
    )

    return preprocessor

###############################################################################

from sklearn.base import TransformerMixin, BaseEstimator
class BinarizingTransformer(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        *,
        kbd__n_bins,
        kbd__encode='ordinal',
        kbd__strategy='quantile',
    ):
        self.kbd__n_bins = kbd__n_bins
        self.kbd__encode = kbd__encode
        self.kbd__strategy = kbd__strategy

    def fit(self, X, y=None):
        from sklearn.preprocessing import KBinsDiscretizer
        from .ordinal_binarizer import OrdinalBinarizer
        self._kbd = KBinsDiscretizer(n_bins=self.kbd__n_bins, encode=self.kbd__encode, strategy=self.kbd__strategy)
        self._kbd.fit(X)
        self._obin = OrdinalBinarizer(nbits=[v - 1 for v in self._kbd.n_bins_])
        self._obin.fit(X)
        return self

    def transform(self, X):
        rv = self._obin.transform(self._kbd.transform(X))
        return rv

    def get_feature_names_out(self, input_features):
        return self._obin._get_feature_names_out(self._obin, input_features)


def make_tsr_preprocessor(
    *,
    random_state=None,
    pre__verbose=False,
    kbd__n_bins=20,
    ii__max_iter=50,
    ii__verbose=False,
):
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer, make_column_selector
    from sklearn.preprocessing import OneHotEncoder
    from .pickleable_bits import ohe_feature_name_combiner
    from sklearn.preprocessing import StandardScaler
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    from sklearn.preprocessing import KBinsDiscretizer
    from .ordinal_binarizer import OrdinalBinarizer
    from sklearn import base
    from .dataset_columns import COL_X_CAT, COL_X_NUM
    from sklearn.pipeline import FeatureUnion
    from .identity_transformer import IdentityTransformer

    preprocessor = Pipeline(
        [
            #('CAT one-hot 1', ColumnTransformer(transformers=[
            #    ("CAT OneHotEncoder", OneHotEncoder(drop=None, sparse_output=False, handle_unknown='ignore', feature_name_combiner=ohe_feature_name_combiner, dtype=int), COL_X_CAT),
            #    #("CAT drop", "drop", COL_X_CAT),
            #], remainder='passthrough', verbose_feature_names_out=False, n_jobs=1).set_output(transform='pandas')),

            ('CAT one-hot 1', FeatureUnion([
                ('ALL pass', IdentityTransformer()),
                ('CAT one hot w/ drop', ColumnTransformer(transformers=[
                    ("CAT OneHotEncoder", OneHotEncoder(drop=None, sparse_output=False, handle_unknown='ignore', feature_name_combiner=ohe_feature_name_combiner, dtype=int), COL_X_CAT),
                        ], remainder='drop', verbose_feature_names_out=False, n_jobs=1).set_output(transform='pandas'))
            ]).set_output(transform='pandas')),

            ('NUM std scale', ColumnTransformer(transformers=[
                ("NUM StandardScaler", StandardScaler(), COL_X_NUM),
                ("DUMMY StandardScaler", StandardScaler(with_std=False), make_column_selector(pattern="dummy .*")),
            ], remainder='passthrough', verbose_feature_names_out=False, n_jobs=1).set_output(transform='pandas')),

            ('X impute', ColumnTransformer(transformers=[
                ('X IterativeImputer', IterativeImputer(verbose=ii__verbose, sample_posterior=True, max_iter=ii__max_iter, random_state=random_state), COL_X_NUM + ['dummy EJ_A', 'dummy EJ_B']),
            ], remainder='passthrough', verbose_feature_names_out=False, n_jobs=1).set_output(transform='pandas')),

            #('X discretize', ColumnTransformer(transformers=[
            #    ('X KBinsDiscretizer', KBinsDiscretizer(n_bins=kbd__n_kbins, encode='ordinal', strategy='quantile'), COL_X_NUM),
            #], remainder='passthrough', verbose_feature_names_out=False, n_jobs=1).set_output(transform='pandas')),

            # drop scaled dummies used for imputing
            ('DUMMY drop', ColumnTransformer(transformers=[
                ('DUMMY drop', 'drop', make_column_selector(pattern="dummy .*"))
            ], remainder='passthrough', verbose_feature_names_out=False, n_jobs=1).set_output(transform='pandas')),
            
            # re-create unscales dummies
            ('CAT one-hot 2', ColumnTransformer(transformers=[
                ("CAT OneHotEncoder", OneHotEncoder(drop=None, sparse_output=False, handle_unknown='ignore', feature_name_combiner=ohe_feature_name_combiner, dtype=int), COL_X_CAT),
                ("CAT drop", "drop", COL_X_CAT),
            ], remainder='passthrough', verbose_feature_names_out=False, n_jobs=1).set_output(transform='pandas')),

            ('X binarize', ColumnTransformer(transformers=[
                ('X BinarizingTransformer', BinarizingTransformer(kbd__n_bins=kbd__n_bins, kbd__encode='ordinal', kbd__strategy='quantile'), COL_X_NUM),
            ], remainder='passthrough', verbose_feature_names_out=False, n_jobs=1).set_output(transform='pandas')),
        ],
        verbose=pre__verbose,
    )

    return preprocessor


def make_tsr_legacy_preprocessor(
    *,
    random_state=None,
    pre__verbose=False,
    kbd__n_bins=20,
    ii__max_iter=50,
    ii__verbose=False,
):
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer, make_column_selector
    from sklearn.preprocessing import OneHotEncoder
    from .pickleable_bits import ohe_feature_name_combiner
    from sklearn.preprocessing import StandardScaler
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    from sklearn.preprocessing import KBinsDiscretizer
    from sklearn.preprocessing import FunctionTransformer
    from .ordinal_binarizer import OrdinalBinarizer
    from sklearn import base
    from .dataset_columns import COL_X_CAT, COL_X_NUM
    from sklearn.pipeline import FeatureUnion
    from .identity_transformer import IdentityTransformer

    preprocessor = Pipeline(
        [
            ('CAT one-hot 1', ColumnTransformer(transformers=[
                ("CAT OneHotEncoder", OneHotEncoder(drop=None, sparse_output=False, handle_unknown='ignore', feature_name_combiner=ohe_feature_name_combiner, dtype=int), COL_X_CAT),
                #("CAT drop", "drop", COL_X_CAT),
            ], remainder='passthrough', verbose_feature_names_out=False, n_jobs=1).set_output(transform='pandas')),

            ('NUM std scale', ColumnTransformer(transformers=[
                ("NUM StandardScaler", StandardScaler(), COL_X_NUM),
            ], remainder='passthrough', verbose_feature_names_out=False, n_jobs=1).set_output(transform='pandas')),

            ('X impute', ColumnTransformer(transformers=[
                ('X IterativeImputer', IterativeImputer(verbose=ii__verbose, sample_posterior=True, max_iter=ii__max_iter, random_state=random_state), COL_X_NUM + ['dummy EJ_A', 'dummy EJ_B']),
            ], remainder='passthrough', verbose_feature_names_out=False, n_jobs=1).set_output(transform='pandas')),

            ('X binarize', ColumnTransformer(transformers=[
                ('X BinarizingTransformer', BinarizingTransformer(kbd__n_bins=kbd__n_bins, kbd__encode='ordinal', kbd__strategy='quantile'), COL_X_NUM),
            ], remainder='passthrough', verbose_feature_names_out=False, n_jobs=1).set_output(transform='pandas')),

            ('ALL int', FunctionTransformer(lambda a: a.astype(int)).set_output(transform='pandas')),
        ],
        verbose=pre__verbose,
    )

    return preprocessor
