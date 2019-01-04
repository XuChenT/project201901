#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
@project: project201901
@time: 2019-01-04  09-53-39
@description: 
    Pipline of whole project
"""

from lib.dataio import prepare_data_pipline
from model.lgbmodel import lgb_model

if __name__ == "__main__":
    data_box = prepare_data_pipline()
    lgb_model(data_box)

