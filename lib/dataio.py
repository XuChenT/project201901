#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
@project: project201901
@time: 2019-01-04  09-55-21
@description: 
    Static methods about data IO
"""
import os
import random
import pandas as pd

from lib.constants import Constants
from lib.datacleaning import data_cleaning_pipline


class DataBox:
    """
    Provide various data format for Prediction models
    """
    def __init__(self, all_data):
        self.all_df = all_data

        self.train_df_all = self.all_df.loc[self.all_df['Yield'] != -1]
        self.test_df = self.all_df.loc[self.all_df['Yield'] == -1].drop(['Sample_id', 'Yield'], axis=1)
        self.test_label = self.all_df.loc[self.all_df['Yield'] == -1, 'Yield']

        # split train/validation
        self.train_index, self.vali_index = split_df(self.train_df_all)

        self.train_df = self.train_df_all.loc[self.train_index, :].drop(['Sample_id', 'Yield'], axis=1)
        self.vali_df = self.train_df_all.loc[self.vali_index, :].drop(['Sample_id', 'Yield'], axis=1)
        self.train_label = self.train_df_all.loc[self.train_index, 'Yield']
        self.vali_label= self.train_df_all.loc[self.vali_index, 'Yield']

        # cache sample ID
        self.train_df_sample_id = self.train_df_all.loc[self.train_index, 'Sample_id']
        self.vali_df_sample_id = self.train_df_all.loc[self.vali_index, 'Sample_id']
        self.test_df_sample_id = self.all_df.loc[self.all_df['Yield'] == -1, 'Sample_id']

        self._submit_result = self.all_df.loc[self.all_df['Yield'] == -1, ['Sample_id', 'Yield']]

    @property
    def submit_result(self):
        return self._submit_result

    @submit_result.setter
    def submit_result(self, value):
        try:
            if isinstance(value, pd.Series):
                value = value.tolist()
            self._submit_result['Yield'] = value
            assert self._submit_result.shape[0] == self.test_df.shape[0]
            assert self._submit_result.shape[1] == 2
        except ValueError:
            print('Length of Model results dose not match length of test data set')

    def saving_submit_result(self):
        self._submit_result.to_csv(Constants.SUBMIT_PATH, index=False, header=False)


def read_raw_data():
    """
    Read raw train/test data set
    :return: train data , test data
    """
    print(os.getcwd())
    train_df = pd.read_csv(Constants.TRAIN_DATA_PATH, encoding='gb18030', sep=',')
    test_df = pd.read_csv(Constants.TEST_DATA_PATH, encoding='gb18030', sep=',')

    train_df.rename(columns={'样本id': 'Sample_id', '收率': 'Yield'}, inplace=True)
    test_df.rename(columns={'样本id': 'Sample_id'}, inplace=True)
    return train_df, test_df


def combine_data(train_df, test_df):
    """
    Combine data set;
    Notes: Labels in test_df replaced by -1

    :param train_df:
    :param test_df:
    :return:
    """
    test_df['Yield'] = -1
    all_data = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    assert all_data.shape[1] == train_df.shape[1]
    return all_data


def split_df(train_df, test_size=0.1, shuffle=True):
    """
     Split raw train_df to train_df and vali_df
    :param train_df:
    :param test_size: 0 ~ 1
    :param shuffle:
    :return: train/validation data INDEX
    """

    assert 1 > test_size > 0

    test_size = int(train_df.shape[0] * test_size)
    train_df_index = train_df.index.tolist()

    if shuffle:
        random.seed(666)
        vali_index = random.sample(range(len(train_df_index)), test_size)
        train_index = list(set([i for i in range(len(train_df_index))]) - set(vali_index))
        assert len(set(vali_index)) == test_size
        assert len(set(vali_index)) + len(set(train_index)) == len(train_df_index)
    else:
        train_index = [i for i in range(len(train_df_index))][:-test_size]
        vali_index = list(set([i for i in range(len(train_df_index))]) - set(train_index))

    return [train_df_index[i] for i in train_index], [train_df_index[i] for i in vali_index]


def prepare_data_pipline():
    train_df, test_df = read_raw_data()
    all_df = combine_data(train_df, test_df)
    return all_df


if __name__ == "__main__":
    prepare_data_pipline()
