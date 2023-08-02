#!/usr/bin/python3
# -*- coding: utf-8 -*-

def split_gen(X, split_targets, *, cv):
    """Wrapper for regular cv splitters to be used
    with cross_val_predict, which allows to split
    on other vector than y.
    """
    for train_index, test_index in cv.split(X, split_targets):
        yield train_index, test_index

