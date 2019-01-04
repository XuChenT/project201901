#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
@project: project201901
@time: 2019-01-04  09-48-43
@description: 
    Static methods of data cleaning
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from lib.constants import Constants


def data_encoder(all_df, encoder_way='LabelEncoder'):
    """
    Encode features
    Notes: train/vali/test data should be included in all_df, because the encoder will not be saved.
    :param encoder_way:
    :param all_df:
    :return:
    """
    assert encoder_way in ['LabelEncoder', 'OneHotEncoder']

    for col in all_df.columns:
        if col in Constants.time_interval_features or col in Constants.timestamp_features:
            if encoder_way == 'LabelEncoder':
                le = LabelEncoder()
                all_df[col] = le.fit_transform(all_df[col])
            # elif encoder_way == "OneHotEncoder":
                # on = OneHotEncoder()
                # all_df[col] = on.fit_transform(all_df[col])
        elif all_df[col].dtype == 'object' and col != 'Sample_id':
            all_df[col] = all_df[col].astype(int)
    return all_df


def fillna_strategy(all_df):
    """
    V1:
        if feature's dtype is object ====> fill 'NONE'
        keep numerical features
    :param all_df:
    :return:
    """
    for col in all_df.columns:
        if col in Constants.time_interval_features or col in Constants.timestamp_features:
            all_df[col].fillna('NONE', inplace=True)
    return all_df


def exception_handling(all_data):
    # A25:
    def handling_A25(x):
        try:
            x = int(x)
        except Exception:
            x = all_data.loc[all_data.index.tolist()[0], 'A25']
        return x
    all_data['A25'] = all_data['A25'].apply(handling_A25)
    return all_data


def data_cleaning_pipline(all_df):
    all_df = fillna_strategy(all_df)
    all_df = exception_handling(all_df)
    all_df = data_encoder(all_df)

