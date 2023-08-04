#!/usr/bin/python3
# -*- coding: utf-8 -*-

from sklearn import base

from .proper_xgboost import ProperXGBRegressor
from .proper_logistic_regression import ProperLogisticRegression


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
                make_lr_single_regressor(
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
