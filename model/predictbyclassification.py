#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
@project: project201901
@time: 2019-01-07  21-20-47
@description: 

"""
from lib.datacleaning import data_cleaning_pipline
from lib.dataio import DataBox, prepare_data_pipline

import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


def xgb_classification(data_box):
    assert isinstance(data_box, DataBox)

    encoder = LabelEncoder()
    dtrain = xgb.DMatrix(data_box.train_df, label=encoder.fit_transform(data_box.train_label.round(1)))
    dvali = xgb.DMatrix(data_box.vali_df, label=encoder.transform(data_box.vali_label.round(1)))
    dtest = xgb.DMatrix(data_box.test_df)

    param = {'eta': 0.005,
                 'max_depth': 10,
                 'subsample': 0.8,
                 'colsample_bytree': 0.8,
                 'objective': 'multi:softmax',
                 # 'eval_metric': 'rmse',
                 'silent': True,
                 'nthread': 4,
                 'num_class': data_box.all_df['Yield'].round(1).unique().shape[0]-1
             }
    print(data_box.all_df['Yield'].round(1).unique().shape[0]-1)
    num_round = 5000
    evallist = [(dtrain, 'train'), (dvali, 'vali')]
    xgb_module = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=100)
    data_box.submit_result = xgb_module.predict(dtest, ntree_limit=xgb_module.best_ntree_limit)
    print(xgb_module.predict(dtrain, ntree_limit=xgb_module.best_ntree_limit))
    # plt.plot(data_box.vali_label.tolist())
    # plt.plot(xgb_module.predict(dtest, ntree_limit=xgb_module.best_ntree_limit))
    # plt.show()
    # data_box.saving_submit_result()


if __name__ == "__main__":
    all_df = prepare_data_pipline()
    all_df = data_cleaning_pipline(all_df)
    data_box = DataBox(all_df)
    xgb_classification(data_box)