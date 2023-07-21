#!/usr/bin/python3
# -*- coding: utf-8 -*-

from sklearn.pipeline import Pipeline
from .identity_transformer import IdentityTransformer


def make_cached_estimator(estimator, *, step_name, memory, verbose=False):
    pipeline = Pipeline([
        (step_name, estimator),
        ('nop', IdentityTransformer()),
    ], memory=memory, verbose=verbose)
    return pipeline
