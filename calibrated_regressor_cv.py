#!/usr/bin/python3
# -*- coding: utf-8 -*-

from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import RegressorMixin


class CalibratedRegressorCV(RegressorMixin, CalibratedClassifierCV):
    """CalibratedRegressorCV morphs CalibratedClassifierCV
    to return positive label probability as result of predict().
    """
    def predict(self, *args, **kwargs):
        return super().predict_proba(*args, **kwargs)[:, 1]
