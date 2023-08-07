#!/usr/bin/python3
# -*- coding: utf-8 -*-

from multiprocessing.sharedctypes import Value
from sklearn import base

from .proper_xgboost import ProperXGBRegressor
from .proper_logistic_regression import ProperLogisticRegression
from .weighted_kernel_ridge import WeightedKernelRidge
from .tsetlin_regressor import TsetliniRegressor


def make_xgbr_single_regressor(
    *,
    random_state,
    xgb__decimals,
    xgb__params,
):
    regressor = ProperXGBRegressor(
        validate_parameters=False,
        seed=random_state,
        decimals=xgb__decimals,

        objective="reg:logistic",

        booster=xgb__params['booster'],
        n_estimators=xgb__params['n_estimators'],
        max_depth=xgb__params['max_depth'],
        min_child_weight=xgb__params['min_child_weight'],
        gamma=xgb__params['gamma'],
        scale_pos_weight=xgb__params['scale_pos_weight'],
        learning_rate=xgb__params['learning_rate'],
        subsample=xgb__params['subsample'],
        colsample_bytree=xgb__params['colsample_bytree'],
        colsample_bylevel=xgb__params['colsample_bylevel'],
        colsample_bynode=xgb__params['colsample_bynode'],
        max_delta_step=xgb__params['max_delta_step'],
        reg_lambda=xgb__params['lambda'],
        reg_alpha=xgb__params['alpha'],
    )
    return regressor


def make_xgbr_single_estimator(
    regressor,
    *,
    est__verbose=False,
    pre__verbose=False,
    bs__drop_source=False,
    mms__max=1,
):
    from sklearn.pipeline import Pipeline
    from sklearn import base
    from .preprocessing import make_xgbr_preprocessor

    estimator = Pipeline(
        [
            ('Preprocess', make_xgbr_preprocessor(
                pre__verbose=pre__verbose,
                bs__drop_source=bs__drop_source,
                mms__max=mms__max,
            )),
            ('XGB', base.clone(regressor)),
        ],
        verbose=est__verbose,
    )

    return estimator


def make_xgbr_model(
    *,
    est__verbose=False,
    pre__verbose=False,
    vr__verbose=False,
    xgbr__params_list=[{}]
):
    from sklearn.ensemble import VotingRegressor

    return VotingRegressor(
        [
            (f"XGBR_{i + 1}",
            make_xgbr_single_estimator(
                make_xgbr_single_regressor(
                    random_state=params["random_state"],
                    xgb__decimals=params["xgb__decimals"],
                    xgb__params=params,
                ),
                est__verbose=est__verbose,
                pre__verbose=pre__verbose,
                bs__drop_source=params["bs__drop_source"],
                mms__max=params["mms__max"]
            ))
            for i, params in enumerate(xgbr__params_list)
        ],
        verbose=vr__verbose,
    )

###############################################################################

def make_lr_single_regressor(
    *,
    lr__class_weight,
    lr__max_iter,
    lr__C,
):
    regressor = ProperLogisticRegression(class_weight={0: 1, 1: lr__class_weight}, max_iter=lr__max_iter, C=lr__C)
    return regressor


def make_lr_single_estimator(
    regressor,
    *,
    random_state=None,
    est__verbose=False,
    pre__verbose=False,
    oe__fill_na=None,
    ii__max_iter=50,
    ii__verbose=0,
    rfe__n_features_to_select=57,
    rfe__step=0.03,
    rfe__verbose=False,
    rfe__estimator=None,
):
    from sklearn.pipeline import Pipeline
    from sklearn import base
    from .preprocessing import make_lm_preprocessor

    estimator = Pipeline(
        [
            ('Preprocess', make_lm_preprocessor(
                random_state=random_state,
                pre__verbose=pre__verbose,
                oe__fill_na=oe__fill_na,
                ii__max_iter=ii__max_iter,
                ii__verbose=ii__verbose,
                rfe__n_features_to_select=rfe__n_features_to_select,
                rfe__step=rfe__step,
                rfe__verbose=rfe__verbose,
                rfe__estimator=rfe__estimator,
            )),
            ('LR', base.clone(regressor)),
        ],
        verbose=est__verbose,
    )

    return estimator


def make_lr_model(
    *,
    est__verbose=False,
    pre__verbose=False,
    vr__verbose=False,
    lr__params_list=[{}]
):
    from sklearn.ensemble import VotingRegressor

    return VotingRegressor(
        [
            (f"LR_{i + 1}",
            make_lr_single_estimator(
                regressor=make_lr_single_regressor(
                    lr__class_weight=params["lr__class_weight"],
                    lr__max_iter=params["lr__max_iter"],
                    lr__C=params["lr__C"],
                ),
                random_state=params["random_state"],
                est__verbose=est__verbose,
                pre__verbose=pre__verbose,
                oe__fill_na=params["oe__fill_na"],
                ii__max_iter=params["ii__max_iter"],
                ii__verbose=params["ii__verbose"],
                rfe__n_features_to_select=params["rfe__n_features_to_select"],
                rfe__verbose=params["rfe__verbose"],
                rfe__estimator=make_lr_single_regressor(
                    lr__class_weight=params["lr__class_weight"],
                    lr__max_iter=params["lr__max_iter"],
                    lr__C=params["lr__C"],
                ),
            ))
            for i, params in enumerate(lr__params_list)
        ],
        verbose=vr__verbose,
    )

###############################################################################

def make_krr_single_regressor(
    *,
    krr__params,
):
    kernel = krr__params['kernel']

    if kernel == 'poly':
        regressor = WeightedKernelRidge(
            kernel=kernel,
            alpha=krr__params['poly:alpha'],
            degree=krr__params['poly:degree'],
            kernel_params={
                'gamma': krr__params['poly:gamma'],
                'coef0': krr__params['poly:coef0']
            },
            class_weight={0: 1, 1: krr__params['poly:class_weight']}
        )
    elif kernel == 'laplacian':
        regressor = WeightedKernelRidge(
            kernel=kernel,
            alpha=krr__params['laplacian:alpha'],
            kernel_params={
                'gamma': krr__params['laplacian:gamma']
            },
            class_weight={0: 1, 1: krr__params['laplacian:class_weight']}
        )
    else:
        raise ValueError(f'Unknown KRR kernel: {kernel}')
    return regressor


def make_krr_single_estimator(
    regressor,
    *,
    random_state=None,
    est__verbose=False,
    pre__verbose=False,
    oe__fill_na=None,
    ii__max_iter=50,
    ii__verbose=0,
    rfe__n_features_to_select=57,
    rfe__step=0.03,
    rfe__verbose=False,
    rfe__estimator=None,
):
    from sklearn.pipeline import Pipeline
    from sklearn import base
    from .preprocessing import make_lm_preprocessor

    estimator = Pipeline(
        [
            ('Preprocess', make_lm_preprocessor(
                random_state=random_state,
                pre__verbose=pre__verbose,
                oe__fill_na=oe__fill_na,
                ii__max_iter=ii__max_iter,
                ii__verbose=ii__verbose,
                rfe__n_features_to_select=rfe__n_features_to_select,
                rfe__step=rfe__step,
                rfe__verbose=rfe__verbose,
                rfe__estimator=rfe__estimator,
            )),
            ('KRR', base.clone(regressor)),
        ],
        verbose=est__verbose,
    )

    return estimator


def make_krr_model(
    *,
    est__verbose=False,
    pre__verbose=False,
    vr__verbose=False,
    krr__params_list=[{}]
):
    from sklearn.ensemble import VotingRegressor

    return VotingRegressor(
        [
            (f"KRR_{i + 1}",
            make_krr_single_estimator(
                regressor=make_krr_single_regressor(krr__params=params),
                random_state=params["random_state"],
                est__verbose=est__verbose,
                pre__verbose=pre__verbose,
                oe__fill_na=params["oe__fill_na"],
                ii__max_iter=params["ii__max_iter"],
                ii__verbose=params["ii__verbose"],
                rfe__n_features_to_select=params["rfe__n_features_to_select"],
                rfe__verbose=params["rfe__verbose"],
                rfe__estimator=make_lr_single_regressor(
                    lr__class_weight=params["lr__class_weight"],
                    lr__max_iter=params["lr__max_iter"],
                    lr__C=params["lr__C"],
                ),
            ))
            for i, params in enumerate(krr__params_list)
        ],
        verbose=vr__verbose,
    )

###############################################################################

def make_tsr_single_regressor(
    *,
    random_state,
    subprocess__shell,
    subprocess__check_call,
    tsr__app_path,
    tsr__C,
    tsr__T,
    tsr__s,
    tsr__epochs,
    tsr__C1,
    tsr__C2,
    tsr__n_jobs,
    tsr__tile_size=64,
    tsr__boost_tpf=False,
    tsr__verbose=False
):
    regressor = TsetliniRegressor(
        random_state=random_state,
        app_path=tsr__app_path,
        C=tsr__C,
        T=tsr__T,
        s=tsr__s,
        epochs=tsr__epochs,
        C1=tsr__C1,
        C2=tsr__C2,
        n_jobs=tsr__n_jobs,
        tile_size=tsr__tile_size,
        verbose=tsr__verbose,
        boost_tpf=tsr__boost_tpf,
        subprocess__shell=subprocess__shell,
        subprocess__check_call=subprocess__check_call,
    )
    return regressor

def make_tsr_single_estimator(
    regressor,
    *,
    random_state=None,
    est__verbose=False,
    pre__verbose=False,
    kbd__n_bins=20,
    ii__max_iter=50,
    ii__verbose=False,
):
    from sklearn.pipeline import Pipeline
    from sklearn import base
    from .preprocessing import make_tsr_preprocessor

    estimator = Pipeline(
        [
            ('Preprocess', make_tsr_preprocessor(
                random_state=random_state,
                pre__verbose=pre__verbose,
                kbd__n_bins=kbd__n_bins,
                ii__max_iter=ii__max_iter,
                ii__verbose=ii__verbose,
            )),
            ('TSR', base.clone(regressor)),
        ],
        verbose=est__verbose,
    )

    return estimator

def make_tsr_model(
    *,
    est__verbose=False,
    pre__verbose=False,
    vr__verbose=False,
    ii__verbose=False,
    kbd__n_bins=20,
    subprocess__shell=False,
    subprocess__check_call=False,
    tsr__app_path='/kaggle/working/app-build/app/main',
    tsr__boost_tpf=False,
    tsr__n_jobs=2,
    tsr__tile_size=64,
    tsr__verbose=False,
    tsr__params_list=[{}]
):
    from sklearn.ensemble import VotingRegressor

    return VotingRegressor(
        [
            (
                f"TSR_{i + 1}",
                make_tsr_single_estimator(
                    regressor=make_tsr_single_regressor(
                        random_state=params['random_state'],
                        subprocess__shell=subprocess__shell,
                        subprocess__check_call=subprocess__check_call,
                        tsr__app_path=tsr__app_path,
                        tsr__C=params['C'],
                        tsr__T=params['T'],
                        tsr__s=params['s'],
                        tsr__epochs=params['epochs'],
                        tsr__C1=params['C1'],
                        tsr__C2=params['C1'] / params['C2 factor'],
                        tsr__n_jobs=tsr__n_jobs,
                        tsr__tile_size=tsr__tile_size,
                        tsr__boost_tpf=tsr__boost_tpf,
                        tsr__verbose=tsr__verbose,
                    ),
                    random_state=params["random_state"],
                    est__verbose=est__verbose,
                    pre__verbose=pre__verbose,
                    kbd__n_bins=kbd__n_bins,

                    ii__max_iter=params["ii__max_iter"],
                    ii__verbose=ii__verbose,
                ),
            )
            for i, params in enumerate(tsr__params_list)
        ],
        verbose=vr__verbose,
    )
