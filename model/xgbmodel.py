#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
@project: project201901
@time: 2019-01-04  09-50-39
@description: 
    XGBoost Model
"""

from lib.dataio import DataBox, prepare_data_pipline
from lib.datacleaning import data_cleaning_pipline

import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np


def xgboost_model(data_box, k_fold=5):
    assert isinstance(data_box, DataBox)

    # dtrain = xgb.DMatrix(data_box.train_df, label=data_box.train_label)
    # dvali = xgb.DMatrix(data_box.vali_df, label=data_box.vali_label)
    dtest = xgb.DMatrix(data_box.test_df)

    # param = {'eta': 0.005,
    #          'max_depth': 10,
    #          'subsample': 0.8,
    #          'colsample_bytree': 0.8,
    #          'objective': 'reg:linear',
    #          'eval_metric': 'rmse',
    #          'silent': True,
    #          'nthread': 4}
    # num_round = 5000
    # evallist = [(dtrain, 'train'), (dvali, 'vali')]
    # xgb_module = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=100)

    predictions_xgb = np.zeros(data_box.test_df.shape[0])
    for train_df, train_label, vali_df, vali_label in data_box.k_folds(k_fold, shuffle=True):
        dtrain = xgb.DMatrix(train_df, label=train_label)
        dvali = xgb.DMatrix(vali_df, label=vali_label)
        # dtest = xgb.DMatrix(data_box.test_df)

        param = {'eta': 0.005,
                 'max_depth': 10,
                 'subsample': 0.8,
                 'colsample_bytree': 0.8,
                 'objective': 'reg:linear',
                 'eval_metric': 'rmse',
                 'silent': True,
                 'nthread': 4}
        num_round = 5000
        evallist = [(dtrain, 'train'), (dvali, 'vali')]
        xgb_module = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=100)
        predictions_xgb += xgb_module.predict(dtest, ntree_limit=xgb_module.best_ntree_limit)/k_fold

    data_box.submit_result = predictions_xgb
    plt.plot(data_box.vali_label.tolist())
    plt.plot(xgb_module.predict(dtest, ntree_limit=xgb_module.best_ntree_limit))
    plt.show()
    data_box.saving_submit_result()


if __name__ == "__main__":
    all_df = prepare_data_pipline()
    all_df = data_cleaning_pipline(all_df)
    data_box = DataBox(all_df)
    xgboost_model(data_box)
