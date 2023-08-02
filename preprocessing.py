#!/usr/bin/python3
# -*- coding: utf-8 -*-

def make_lm_preprocessor(
    *,
    random_state=None,
    pre__verbose=False,
    oe__fill_na=None,
    ii__max_iter=50, ii__verbose=0,
    rfe__n_features_to_select=57, rfe__step=0.03, rfe__verbose=False, rfe__estimator=None,
):
    from sklearn.pipeline import Pipeline
    from pyicr import OutlierEncoder
    from sklearn.compose import ColumnTransformer, make_column_selector
    from pyicr import DistributionTransformer
    from sklearn.preprocessing import OneHotEncoder
    from pyicr import ohe_feature_name_combiner
    from sklearn.preprocessing import StandardScaler
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    from pyicr import ProperPolynomialFeatures
    from sklearn.pipeline import FeatureUnion
    from pyicr import IdentityTransformer
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import RFE
    from sklearn import base
    from pyicr import COL_X_CAT, COL_X_NUM

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
