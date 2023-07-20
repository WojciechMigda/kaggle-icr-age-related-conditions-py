#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
os.environ['PYTHONWARNINGS']='ignore::DeprecationWarning,ignore:n_quantiles :UserWarning,ignore:Persisting input arguments took:UserWarning'

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*Bins whose width are too small.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*n_quantiles \([0-9]+\) is greater than the total number of samples.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Persisting input arguments took.*")
