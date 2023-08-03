#!/usr/bin/python3
# -*- coding: utf-8 -*-

from sklearn import base

from .proper_xgboost import ProperXGBRegressor


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
