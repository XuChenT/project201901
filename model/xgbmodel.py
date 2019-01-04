#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
@project: project201901
@time: 2019-01-04  09-50-39
@description: 
    XGBoost Model
"""

from lib.dataio import DataBox, prepare_data_pipline

import xgboost as xgb


def xgboost_model(data_box):
    assert isinstance(data_box, DataBox)

    dtrain = xgb.DMatrix(data_box.train_df, label=data_box.train_label)
    dvali = xgb.DMatrix(data_box.vali_df, label=data_box.vali_label)
    dtest = xgb.DMatrix(data_box.test_df)

    param = {
        'objective': 'reg:linear',
        'eta': 0.05,
        'max_depth': 6,
        'gamma': 0,
        'min_child_weight': 1,
        'tree_method': 'gpu_hist'
    }
    num_round = 5000
    evallist = [(dtrain, 'train'), (dvali, 'vali')]
    xgb_module = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=100)
    data_box.submit_result = xgb_module.predict(dtest, ntree_limit=xgb_module.best_ntree_limit)
    data_box.saving_submit_result()


if __name__ == "__main__":
    data_box = prepare_data_pipline()
    xgboost_model(data_box)
