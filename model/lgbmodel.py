#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
@project: project201901
@time: 2019-01-04  12-52-43
@description: 

"""

import lightgbm as lgb

from lib.dataio import DataBox, prepare_data_pipline


def lgb_model(data_box):
    assert isinstance(data_box, DataBox)
    param = {'num_leaves': 120,
             'min_data_in_leaf': 30,
             'objective': 'regression',
             'max_depth': -1,
             'learning_rate': 0.01,
             "min_child_samples": 30,
             "boosting": "gbdt",
             "feature_fraction": 0.9,
             "bagging_freq": 1,
             "bagging_fraction": 0.9,
             "bagging_seed": 11,
             "metric": 'mse',
             "lambda_l1": 0.1,
             "verbosity": -1}

    dtrain = lgb.Dataset(data_box.train_df, label=data_box.train_label)
    dvali = lgb.Dataset(data_box.vali_df, label=data_box.vali_label)
    dtest = lgb.Dataset(data_box.test_df)

    num_round = 10000
    clf = lgb.train(param, dtrain, num_round, valid_sets=[dtrain, dvali], verbose_eval=200,
                    early_stopping_rounds=100)
    data_box.submit_result = clf.predict(data_box.test_df, num_iteration=clf.best_iteration)
    data_box.saving_submit_result()


if __name__ == "__main__":
    data_box = prepare_data_pipline()
    lgb_model(data_box)