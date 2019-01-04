#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
@project: project201901
@time: 2019-01-03  22-04-47
@description: 
    All the constants in project should be writen here
"""
import os


class Constants:
    # data path
    data_path = '../data'
    train_data_path = os.path.join(data_path, 'train_a.csv')
    test_data_path = os.path.join(data_path, 'test_a.csv')
    submit_sample_path = os.path.join(data_path, 'submit.csv')

    results_path = '../data/results'
    submit_path = os.path.join(results_path, 'submit_result.csv')

    # time features
    timestamp_features = ['A5', 'A7', 'A9', 'A11', 'A14', 'A16', 'A24', 'A26', 'B5', 'B7']
    time_interval_features = ['A20', 'A28', 'B4', 'B9', 'B10', 'B11']


