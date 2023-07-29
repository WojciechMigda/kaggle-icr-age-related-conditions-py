#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
os.environ['PYTHONWARNINGS']=",".join([
    'ignore::DeprecationWarning',
    'ignore:n_quantiles :UserWarning',
    'ignore:Persisting input arguments took:UserWarning',
    'ignore:Setting the eps parameter is deprecated and will be removed in 1.5. Instead eps will always havea default value of:FutureWarning',
    'ignore:suggest_loguniform has been deprecated in:FutureWarning',
    'ignore:X has feature names, but :UserWarning'
])

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*Bins whose width are too small.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*n_quantiles \([0-9]+\) is greater than the total number of samples.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Persisting input arguments took.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*Setting the eps parameter is deprecated and will be removed in 1.5. Instead eps will always havea default value of.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*suggest_loguniform has been deprecated in.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*X has feature names, but .*")
