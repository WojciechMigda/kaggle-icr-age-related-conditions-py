#!/usr/bin/python3
# -*- coding: utf-8 -*-

from sklearn.linear_model import LogisticRegression
from sklearn.base import RegressorMixin


class ProperLogisticRegression(RegressorMixin, LogisticRegression):
    """ProperLogisticRegression improves over LogisticRegression
    (which is a classifier) to return positive label probability
    as result of predict().
    """
    def predict(self, *args, **kwargs):
        return super().predict_proba(*args, **kwargs)[:, 1]


if __name__ == '__main__':
    lr = ProperLogisticRegression(C=1.234)
    print(lr)
