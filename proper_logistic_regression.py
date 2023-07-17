#!/usr/bin/python3
# -*- coding: utf-8 -*-

from sklearn.linear_model import LogisticRegression


class ProperLogisticRegression(LogisticRegression):
    def predict(self, *args, **kwargs):
        return super().predict_proba(*args, **kwargs)[:, 1]


if __name__ == '__main__':
    lr = ProperLogisticRegression(C=1.234)
    print(lr)
