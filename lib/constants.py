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
    ROOT_PATH = '..'
    DATA_PATH = os.path.join(ROOT_PATH, 'data')
    TRAIN_DATA_PATH = os.path.join(DATA_PATH, 'train_a.csv')
    TEST_DATA_PATH = os.path.join(DATA_PATH, 'test_a.csv')
    SUBMIT_SAMPLE_PATH = os.path.join(DATA_PATH, 'submit.csv')

    RESULTS_PATH = os.path.join(ROOT_PATH, 'data', 'results')
    SUBMIT_PATH = os.path.join(RESULTS_PATH, 'submit_result.csv')

    # time features
    TIMESTAMP_FEATURES = ['A5', 'A7', 'A9', 'A11', 'A14', 'A16', 'A24', 'A26', 'B5', 'B7']
    TIME_INTERVAL_FEATURES = ['A20', 'A28', 'B4', 'B9', 'B10', 'B11']

    TIMESTAMP_CASE_A = ['A5', 'A9', 'A11', 'A14', 'A16', 'A24', 'A26']
    # GROUP_1: A5 - A18,
    # NOTES: A7(A8) is ignored cause the number of NONE
    TEMPERATURE_STATUS_CASE_A = {'A5': 'A6', 'A9': 'A10',
                                 'A11': 'A12', 'A14': 'A15',
                                 'A16': 'A17', 'A24': 'A25', 'A26': 'A27'}

    TIMESTAMP_CASE_B = ['B5', 'B7']

    # material features
    MATERIAL = ['A1', 'A2', 'A3', 'A4', 'A19', 'B1', 'B12', 'B14']
    MATERIAL_A_GROUP_1 = ['A1', 'A19']
    MATERIAL_A_GROUP_2 = ['A2', 'A3', 'A4']
    MATERIAL_B = ['B1', 'B12', 'B14']

    INDEX_AND_LABEL = ['Sample_id', 'Yield']

    USELESS_COL = ['A7', 'A8', 'A18']

