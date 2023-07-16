#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np

def balanced_log_loss(*, y_true, y_hat) -> float:
    if y_hat.ndim != 2 or y_hat.shape[1] != 2:
        raise TypeError("y_hat must be a 2d array with two columns for class 0 and 1, respectively.")
    # Extracting class labels from y_true
    y_true = y_true.astype(int)

    # Computing the number of observations for each class
    N0 = np.sum(y_true == 0)
    N1 = np.sum(y_true == 1)

    # Calculating the inverse prevalence weights
    w0 = 1 / N0
    w1 = 1 / N1

    # Rescaling the predicted probabilities
    y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)
    y_hat /= y_hat.sum(axis=1, keepdims=True)

    # Calculating the logarithmic loss for each class
    log_loss_0 = np.sum((1 - y_true) * np.log(y_hat[:, 0]))
    log_loss_1 = np.sum(y_true * np.log(y_hat[:, 1]))

    # Computing the balanced logarithmic loss
    balanced_log_loss = (-w0 * log_loss_0 - w1 * log_loss_1) / 2

    return balanced_log_loss
