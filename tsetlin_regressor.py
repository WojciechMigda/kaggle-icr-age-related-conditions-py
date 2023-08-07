#!/usr/bin/python3
# -*- coding: utf-8 -*-

import subprocess
import tempfile
import shutil
import os

import pandas as pd
from sklearn.base import RegressorMixin, BaseEstimator


class TsetliniRegressor(RegressorMixin, BaseEstimator):
    def __init__(
        self,
        *,
        app_path: str,
        C: int,
        T: int,
        s: float,
        epochs: int,
        C1: float,
        C2: float,
        boost_tpf: bool,
        tile_size: int,
        random_state: int,
        n_jobs: int,
        verbose: bool,
        subprocess__shell: bool,
        subprocess__check_call: bool,
    ):
        self.C = C
        self.T = T
        self.s = s
        self.epochs = epochs
        self.C1 = C1
        self.C2 = C2
        self.boost_tpf = boost_tpf
        self.tile_size = tile_size
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.app_path = app_path
        self.verbose = verbose
        self.subprocess__shell = subprocess__shell
        self.subprocess__check_call = subprocess__check_call

    def __del__(self):
        if hasattr(self, '_tmp_path'):
            shutil.rmtree(self._tmp_path)
        pass

    @property
    def _train_Xy_csv(self):
        if hasattr(self, '_tmp_path'):
            return os.path.join(self._tmp_path, 'train_Xy.csv')
        else:
            raise RuntimeError(f'_train_Xy_csv called before fit()')

    @property
    def _infer_X_csv(self):
        if hasattr(self, '_tmp_path'):
            return os.path.join(self._tmp_path, 'infer_X.csv')
        else:
            raise RuntimeError(f'_infer_X_csv called before fit()')

    @property
    def _infer_y_csv(self):
        if hasattr(self, '_tmp_path'):
            return os.path.join(self._tmp_path, 'infer_y.csv')
        else:
            raise RuntimeError(f'_infer_y_csv called before fit()')

    @property
    def _model_json(self):
        if hasattr(self, '_tmp_path'):
            return os.path.join(self._tmp_path, 'model.json')
        else:
            raise RuntimeError(f'_model_json called before fit()')

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            raise ValueError(f'{self.__class__.__name__} requires pandas DataFrame as X input')
        if not isinstance(y, pd.DataFrame):
            #raise ValueError(f'{self.__class__.__name__} requires pandas DataFrame as y input')
            y = pd.DataFrame(y, columns=['Class']).set_index(X.index)
        tmp_path = tempfile.mkdtemp(self.__class__.__name__)
        self._tmp_path = tmp_path

        pd.concat([X, y], axis=1).astype(int).to_csv(self._train_Xy_csv)

        app_args_train = [
            'train',
            '--tsetlini-regressor',
        ]
        train_args = app_args_train
        train_args.extend(['-C', str(self.C)])
        train_args.extend(['-T', str(self.T)])
        train_args.extend(['-s', str(self.s)])
        train_args.extend(['--nepochs', str(self.epochs)])
        train_args.extend(['-C1', str(self.C1)])
        train_args.extend(['-C2', str(self.C2)])
        train_args.extend(['--tsetlini-random-state', str(self.random_state)])
        train_args.extend(['--tsetlini-tile-size', str(self.tile_size)])
        train_args.extend(['--tsetlini-boost-tpf'] if self.boost_tpf else ['--tsetlini-no-boost-tpf'])
        train_args.extend(['-j', str(self.n_jobs)])
        train_args.extend(['-d', self._train_Xy_csv])
        train_args.extend(['-o', self._model_json])

        if self.subprocess__check_call:
            from tempfile import NamedTemporaryFile
            with NamedTemporaryFile() as f:
                subprocess.check_call([self.app_path] + train_args, stdout=f, stderr=subprocess.STDOUT)
                if self.verbose:
                    f.seek(0)
                    print(f.read())
        elif self.subprocess__shell:
            subprocess.run(' '.join([self.app_path] + train_args), capture_output=self.verbose, check=True, shell=True)
        else:
            subprocess.run([self.app_path] + train_args, capture_output=self.verbose, check=True, shell=False)
        return self

    def predict(self, X):
        if not hasattr(self, '_tmp_path'):
            raise RuntimeError(f'predict() called before fit()')

        X.astype(int).to_csv(self._infer_X_csv)

        app_args_infer = [
            'infer',
        ]
        infer_args = app_args_infer

        infer_args.extend(['-d', self._infer_X_csv])
        infer_args.extend(['-m', self._model_json])
        infer_args.extend(['-o', self._infer_y_csv])

        if self.subprocess__check_call:
            from tempfile import NamedTemporaryFile
            with NamedTemporaryFile() as f:
                subprocess.check_call([self.app_path] + infer_args, stdout=f, stderr=subprocess.STDOUT)
                if self.verbose:
                    f.seek(0)
                    print(f.read())
        elif self.subprocess__shell:
            subprocess.run(' '.join([self.app_path] + infer_args), capture_output=self.verbose, check=True, shell=True)
        else:
            subprocess.run([self.app_path] + infer_args, capture_output=self.verbose, check=True, shell=False)

        return pd.read_csv(self._infer_y_csv, index_col="Id").class_1.values
