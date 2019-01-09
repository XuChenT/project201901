#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
@project: project201901
@time: 2019-01-04  09-48-43
@description: 
    Static methods of data cleaning
"""
import datetime
import itertools

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, PolynomialFeatures

from lib.constants import Constants
import re


def data_encoder(all_df, encoder_way='LabelEncoder'):
    """
    Encode features
    Notes: train/vali/test data should be included in all_df, because the encoder will not be saved.
    :param encoder_way:
    :param all_df:
    :return:
    """
    assert encoder_way in ['LabelEncoder', 'OneHotEncoder']

    encode_col = []
    for col in all_df.columns:
        if col in Constants.TIME_INTERVAL_FEATURES or col in Constants.TIMESTAMP_FEATURES:
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
    all_df.drop(Constants.TIMESTAMP_FEATURES, axis=1, inplace=True)
    all_df.drop(Constants.TIME_INTERVAL_FEATURES, axis=1, inplace=True)
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
        if col in Constants.TIME_INTERVAL_FEATURES or col in Constants.TIMESTAMP_FEATURES:
            all_df[col].fillna('NONE', inplace=True)

    # features: A1-A4 filling None strategy:
    for col in Constants.MATERIAL_A_GROUP_1:
        all_df[col].fillna(0, inplace=True)

    return all_df


def repair_timestamp(all_df):
    """
    fill NONE for time stamp features
    Notes:  should be run after adding time features
    :param all_df:
    :return:
    """

    # A7/A8 will be handled alone
    def repair_A7(x):
        if x['A7'] == 'NONE':
            x['A7'] = str((pd.to_datetime(x['A5']) +
                           datetime.timedelta(hours=x['A9A5']/2)).time())
            x['A8'] = x['A6'] + x['A9A5'] / 2 * x['A10A6rate']
        return x

    all_df = all_df.apply(repair_A7, axis=1)
    return all_df


def exception_handling(all_df):
    """
    repair specific features
    :param all_df:
    :return:
    """
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
    # for col in all_df.columns:
    #     if all_df[col].unique().shape[0] < 3 and col not in Constants.INDEX_AND_LABEL:
    #         all_df.drop(col, axis=1, inplace=True)
    #         print(col)

    all_df.drop(Constants.USELESS_COL, axis=1, inplace=True)

    return all_df


def add_time_A_feature(all_df):
    def time_handle(x):
        pattern = re.compile(r'\d+:\d+:\d+')
        for item in Constants.TIMESTAMP_CASE_A:
            if item == Constants.TIMESTAMP_CASE_A[0]:
                item_last = item
                continue
            elif pattern.match(x[item]) is not None and pattern.match(x[item_last]) is not None:
                tmpcur = x[item].split(':')
                tmplast = x[item_last].split(':')
                if int(tmpcur[0]) < int(tmplast[0]):
                    tmpcur[0] = str(24+int(tmpcur[0]))

                x[item+item_last] = (int(tmpcur[0]) - int(tmplast[0])) + \
                                     (int(tmpcur[1]) - int(tmplast[1]))/60 +\
                                     (int(tmpcur[2]) - int(tmplast[2]))/3600

            else:
                x[item+item_last] = 0
            item_last = item
        return x
    all_df = all_df.apply(time_handle, axis=1)

    pre_time = Constants.TIMESTAMP_CASE_A[0]

    all_df['A16A5'] = 0
    for time_index in range(1, len(Constants.TIMESTAMP_CASE_A)):
        time = Constants.TIMESTAMP_CASE_A[time_index]
        pre_tem = Constants.TEMPERATURE_STATUS_CASE_A[pre_time]
        tem = Constants.TEMPERATURE_STATUS_CASE_A[time]

        # change in unit time
        all_df[tem + pre_tem + 'rate'] = (all_df[tem] - all_df[pre_tem])/all_df[time + pre_time]

        # total change
        all_df[tem + pre_tem] = (all_df[tem] - all_df[pre_tem])

        # Total reaction time
        all_df['A16A5'] += all_df[time + pre_time]
        pre_time = time
    return all_df


def add_time_B_feature(all_df):
    def time_handle(x):
        pattern = re.compile(r'\d+:\d+:\d+')
        if pattern.match(x['B5']) is not None and pattern.match(x['B7']) is not None:
            tmpcur = x['B7'].split(':')
            tmplast = x['B5'].split(':')
            if int(tmpcur[0]) < int(tmplast[0]):
                tmpcur[0] = str(24+int(tmpcur[0]))
            x['B7B5'] = (int(tmpcur[0]) - int(tmplast[0])) + \
                        (int(tmpcur[1]) - int(tmplast[1]))/60 +\
                        (int(tmpcur[2]) - int(tmplast[2]))/3600
        else:
            x['B7B5'] = 0
        return x
    all_df = all_df.apply(time_handle,axis=1)
    return all_df


def add_timeinterval_features(all_df):
    def time_interval_handle(x):
        pattern = re.compile(r'\d+:\d+-\d+:\d+')
        for item in Constants.TIME_INTERVAL_FEATURES:
            if pattern.match(x[item]) is not None:
                cols = re.split('-|:', x[item])
                if cols[3][-1] == '分':
                    cols[3] = cols[3][0:-1]
                if int(cols[0]) > int(cols[2]):
                    cols[2] = str(24+int(cols[2]))  # 处理case  23：00-00：00
                x[item + 'delta'] = int(cols[2]) - int(cols[0]) + \
                                           (int(cols[3]) - int(cols[1])) / 60
            else:
                x[item+'delta'] = 0
        return x
    all_df = all_df.apply(time_interval_handle, axis=1)
    return all_df


def adding_material_A_group_1_features(all_df):
    """
    encode material features
    :param all_df:
    :return:
    """
    # numerical new features
    all_df['A_1_2_3_4'] = all_df['A1'] + all_df['A2'] + all_df['A3'] + all_df['A4']
    all_df['A_2_3_4'] = all_df['A2'] + all_df['A3'] + all_df['A4']

    # handling as scatter features
    for col in itertools.combinations(Constants.MATERIAL, 2):
        combine_col = all_df[col[0]].astype(str) + all_df[col[1]].astype(str)
        encoder = LabelEncoder()
        combine_col = encoder.fit_transform(combine_col)
        all_df[col[0] + '_' + col[1]] = combine_col
        all_df[col[0] + '_' + col[1]] = all_df[col[0] + '_' + col[1]].astype(str)

    for col in itertools.combinations(Constants.MATERIAL, 3):
        combine_col = all_df[col[0]].astype(str) + all_df[col[1]].astype(str) + all_df[col[2]].astype(str)
        encoder = LabelEncoder()
        combine_col = encoder.fit_transform(combine_col)
        all_df[col[0] + '_' + col[1] + '_' + col[2]] = combine_col
        all_df[col[0] + '_' + col[1] + '_' + col[2]] = all_df[col[0] + '_' + col[1] + '_' + col[2]].astype(str)

    for col in Constants.MATERIAL:
        all_df[col].fillna(all_df[col].value_counts().index.tolist()[0], inplace=True)
    poly = PolynomialFeatures(2)
    test = poly.fit_transform(all_df[Constants.MATERIAL])
    test = test[:, len(Constants.MATERIAL):]
    for i in range(test.shape[1]):
        encoder = LabelEncoder()
        all_df['Poly_'+str(i)] = test[:, i]
        all_df['Poly_' + str(i)] = encoder.fit_transform(all_df['Poly_' + str(i)])
    return all_df


def test_feature(all_df):
    """
    features from forum
    :param all_df:
    :return:
    """
    train_df = all_df[all_df['Yield'] != -1]
    train_df['intYield'] = pd.cut(train_df['Yield'], 5, labels=False)
    train_df = pd.get_dummies(train_df, columns=['intYield'])
    li = ['intYield_0', 'intYield_1', 'intYield_2', 'intYield_3', 'intYield_4']
    mean_features = []

    for f1 in train_df.columns:
        if f1 not in ['Sample_id', 'Yield']:
            rate = train_df[f1].value_counts(normalize=True, dropna=False).values[0]
            if rate < 0.50:
                for f2 in li:
                    col_name = f1+"_"+f2+'_mean'
                    mean_features.append(col_name)
                    order_label = train_df.groupby([f1])[f2].mean()
                    # for df in [train, test]:
                    all_df[col_name] = all_df[f1].map(order_label)
    return all_df


def data_cleaning_pipline(all_df):
    all_df = fillna_strategy(all_df)
    all_df = exception_handling(all_df)
    # all_df = test_feature(all_df)
    all_df = add_time_A_feature(all_df)
    all_df = add_time_B_feature(all_df)
    all_df = add_timeinterval_features(all_df)
    all_df = repair_timestamp(all_df)
    # all_df = delete_useless_features(all_df)
    all_df = adding_material_A_group_1_features(all_df)
    all_df = data_encoder(all_df, encoder_way='OneHotEncoder')
    return all_df
