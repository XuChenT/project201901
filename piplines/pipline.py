#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
@project: project201901
@time: 2019-01-06  10-37-10
@description: 

"""


from lib.datacleaning import data_cleaning_pipline
from lib.dataio import prepare_data_pipline, DataBox
from model.lgbmodel import lgb_model

if __name__ == "__main__":
    # preparing data set
    all_df = prepare_data_pipline()
    all_df = data_cleaning_pipline(all_df)
    data_box = DataBox(all_df)
    # launching model
    lgb_model(data_box)
