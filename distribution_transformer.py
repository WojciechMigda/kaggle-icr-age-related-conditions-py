#!/usr/bin/python3
# -*- coding: utf-8 -*-

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer


FEATURES_BY_POLICY = {
    'with_quantile_xformer' : {
        'prior_imputation' : {
            'YeoJohnson' : ["AF", "AR", "AX", "AZ", "BD ", "BN", "BP", "BZ", "CD ", "CF", "CH", "CS", "DA", "DF", "DH", "DI", "DN", "DY", "EB", "EE", "EG", "EH", "EP", "FI", "FS", "GB", "GH", "GL"],
            'BoxCox' : ["BQ", "DE", "DV", "EL", "GE", "GF", "GI"],
            'Quantile' : ["AB", "AH", "AM", "AY", "BC", "BR", "CB", "CC", "CL", "CR", "CU", "CW ", "DL", "DU", "EU", "FC", "FD ", "FE", "FL", "FR"],
        },
        'post_imputation' : {

        },
    },
    'without_quantile_xformer' : {
        'prior_imputation' : {
            'YeoJohnson' : ["AB", "AF", "AH", "AR", "AX", "AZ", "BD ", "BN", "BP", "BR", "BZ", "CC", "CD ", "CF", "CH", "CR", "CS", "CU", "DA", "DF", "DH", "DI", "DL", "DN", "DY", "EB", "EE", "EG", "EH", "EP", "FC", "FD ", "FE", "FI", "FL", "FR", "FS", "GB", "GH", "GL"],
            'BoxCox' : ["AM", "AY", "BC", "BQ", "CB", "CL", "CW ", "DE", "DU", "DV", "EL", "EU", "GE", "GF", "GI"],
        },
        'post_imputation' : {
            
        },
    },
}


class DistributionTransformer(ColumnTransformer):
    def __init__(self, n_quantiles: int, policy: str, random_state: int):
        self.n_quantiles = n_quantiles
        if policy not in ('prior_imputation', 'post_imputation'):
            raise RuntimeError(f"{self.__class__.__name__} policy parameter must be one of: prior_imputation, post_imputation")
        self.policy = policy
        self.random_state = int(random_state)
        if self.n_quantiles is not None:
            self.n_quantiles = int(self.n_quantiles)
            yeo_johnson_step = (
                "YeoJohnson xformer",
                PowerTransformer(method="yeo-johnson"),
                FEATURES_BY_POLICY['with_quantile_xformer'][self.policy]['YeoJohnson'])
            box_cox_step = (
                "BoxCox xformer",
                PowerTransformer(method="box-cox"),
                FEATURES_BY_POLICY['with_quantile_xformer'][self.policy]['BoxCox'])
            quantile_step = (
                "Quantile xformer",
                QuantileTransformer(n_quantiles=self.n_quantiles, output_distribution="normal", random_state=self.random_state),
                FEATURES_BY_POLICY['with_quantile_xformer'][self.policy]['Quantile'])
            mapper_steps = (yeo_johnson_step, box_cox_step, quantile_step)
        else:
            yeo_johnson_step = (
                "YeoJohnson xformer",
                PowerTransformer(method="yeo-johnson"),
                FEATURES_BY_POLICY['without_quantile_xformer'][self.policy]['YeoJohnson'])
            box_cox_step = (
                "BoxCox xformer",
                PowerTransformer(method="box-cox"),
                FEATURES_BY_POLICY['without_quantile_xformer'][self.policy]['BoxCox'])
            mapper_steps = (yeo_johnson_step, box_cox_step)
        super().__init__(transformers=mapper_steps, remainder='passthrough', verbose_feature_names_out=False)
