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
            elif encoder_way == "OneHotEncoder":
                one_hot_buf = pd.get_dummies(all_df[col])
                one_hot_buf.columns = [col+'_'+str(i) for i in range(all_df[col].unique().shape[0])]
                assert one_hot_buf.shape[0] == all_df.shape[0]
                all_df = pd.concat([all_df, one_hot_buf], axis=1)
        elif all_df[col].dtype == 'object' and col != 'Sample_id':
            all_df[col] = all_df[col].astype(int)
    all_df.drop(Constants.timestamp_features, axis=1, inplace=True)
    all_df.drop(Constants.time_interval_features, axis=1, inplace=True)
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


def exception_handling(all_df):
    # A25:
    def handling_A25(x):
        try:
            x = int(x)
        except Exception:
            x = int(all_df.loc[all_df.index.tolist()[0], 'A25'])
        return x

    all_df['A25'] = all_df['A25'].apply(handling_A25)
    return all_df


def delete_useless_features(all_df):
    for col in all_df.columns:
        if all_df[col].unique().shape[0] < 3 and col not in Constants.index_and_label:
            all_df.drop(col, axis=1, inplace=True)
            print(col)
    return all_df


def data_cleaning_pipline(all_df):
    all_df = fillna_strategy(all_df)
    all_df = exception_handling(all_df)
    all_df = data_encoder(all_df, encoder_way='OneHotEncoder')
    # all_df = delete_useless_features(all_df)
    return all_df
