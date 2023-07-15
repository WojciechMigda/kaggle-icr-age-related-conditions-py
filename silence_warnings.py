#!/usr/bin/python3
# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*Bins whose width are too small.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*n_quantiles \([0-9]+\) is greater than the total number of samples.*")