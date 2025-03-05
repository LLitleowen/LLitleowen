import warnings
warnings.simplefilter('ignore')

import os
import re
import gc
import glob
from multiprocessing import Pool

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
from tqdm import tqdm

DATA_SUFFIX = 'feather'
INTERVAL = 2727  # 聚合窗口大小 3600s
USECOLS = ['LogTime', 'deviceID','ChannelId', 'DimmId', 'BankId', 'ColumnId', 'RowId', 
           'MciAddr', 'RetryRdErrLog', 'RetryRdErrLogParity', 'error_type_full_name']
DATA_PATH = '/data1/xiaoyulin/code'
FEATURE_PATH = './feature'
WORKER_NUM = 64

os.makedirs(f'{FEATURE_PATH}/type_A', exist_ok=True)
os.makedirs(f'{FEATURE_PATH}/type_B', exist_ok=True)

def window_gather_feathers(sn_file):
    # 读取数据
    df = pd.read_feather(sn_file)
    # 排除无关原始特征
    df = df[USECOLS]
    # 缺失值填充
    df['deviceID'] = df['deviceID'].fillna(-1).astype(int)
    df['RetryRdErrLogParity'] = df['RetryRdErrLogParity'].fillna(0).replace("", 0).astype(np.int64)
    # 按时间排序
    df = df.sort_values(['LogTime'], ascending=True).reset_index(drop=True)
    
    # 按时间划分聚合窗口
    df['time_index'] = df['LogTime'] // INTERVAL
    
    # 报错位置
    df['locale'] = df['ChannelId'].astype(str) + '_' + df['DimmId'].astype(str) + '_' +\
                   df['BankId'].astype(str) + '_' + df['RowId'].astype(str) + '_' + df['ColumnId'].astype(str)
    df['CellId'] = df['RowId'].astype(str) + '_' + df['ColumnId'].astype(str)
    
    # 日志类型
    df['error_type_CE.READ'] = (df['error_type_full_name'] == 'CE.READ').astype(int)
    df['error_type_CE.SCRUB'] = (df['error_type_full_name'] == 'CE.SCRUB').astype(int)
    
    # bit_dq_burst 特征 (抄自官方 baseline)
    df['bin_parity'] = df['RetryRdErrLogParity'].apply(lambda x: bin(x)[2:].zfill(32))
    df['bin_count'] = df['bin_parity'].apply(lambda x: x.count("1"))
    df['binary_row_array'] = df['bin_parity'].apply(lambda x: [x[i:i+4].count("1") for i in range(0,32,4)])
    df['binary_row_array_indices'] = df['binary_row_array'].apply(lambda x: [idx for idx, value in enumerate(x) if value > 0])
    df['burst_count'] = df['binary_row_array_indices'].apply(len)
    df['max_burst_interval'] = df['binary_row_array_indices'].apply(lambda x: x[-1] - x[0] if x else 0)
    df['binary_column_array'] = df['bin_parity'].apply(lambda x: [x[i::4].count("1") for i in range(4)])
    df['binary_column_array_indices'] = df['binary_column_array'].apply(lambda x: [idx for idx, value in enumerate(x)if value > 0])
    df['dq_count'] = df['binary_column_array_indices'].apply(len)
    df['max_dq_interval'] = df['binary_column_array_indices'].apply(lambda x: x[-1] - x[0] if x else 0)
    
    # 每个 window 里面的最后一个时间作为 LogTime 记录
    df_ret = df.groupby('time_index')['LogTime'].last().to_frame().reset_index()
    
    # 统计特征
    mapping = df.groupby(['time_index'])['LogTime'].count().to_dict()
    df_ret['window_logs_count'] = df_ret['time_index'].map(mapping)
    mapping = df.groupby(['time_index'])['error_type_CE.READ'].sum().to_dict()
    df_ret['window_read_error_logs_count'] = df_ret['time_index'].map(mapping)
    mapping = df.groupby(['time_index'])['error_type_CE.SCRUB'].sum().to_dict()
    df_ret['window_scrub_error_logs_count'] = df_ret['time_index'].map(mapping)
    mapping = df.groupby(['time_index'])['burst_count'].sum().to_dict()
    df_ret['window_burst_count'] = df_ret['time_index'].map(mapping)
    mapping = df.groupby(['time_index'])['dq_count'].sum().to_dict()
    df_ret['window_dq_count'] = df_ret['time_index'].map(mapping)
    mapping = df.groupby(['time_index'])['max_burst_interval'].max().to_dict()
    df_ret['window_max_burst_interval'] = df_ret['time_index'].map(mapping)
    mapping = df.groupby(['time_index'])['max_dq_interval'].sum().to_dict()
    df_ret['window_max_dq_interval'] = df_ret['time_index'].map(mapping)

    # 类别个数特征
    for col in ['deviceID', 'ChannelId', 'BankId', 'DimmId', 'ColumnId', 'ColumnId', 'RowId', 
                'MciAddr', 'RetryRdErrLogParity', 'RetryRdErrLog', 'locale', 'CellId']:
        mapping = df.groupby(['time_index'])[col].nunique().to_dict()
        df_ret[f'window_{col}_nunique'] = df_ret['time_index'].map(mapping)

    # 故障模式特征 (近似, 不如官方 baseline 严谨)
    df_ret['fault_mode_others'] = df_ret['window_deviceID_nunique'].apply(lambda x: 1 if x > 1 else 0)
    df_ret['fault_mode_device'] = df_ret.apply(
        lambda row: 1 if row['fault_mode_others'] == 0 and row['window_BankId_nunique'] > 1 else 0, axis=1)
    df_ret['fault_mode_bank'] = df_ret.apply(
        lambda row: 1 if row['fault_mode_device'] == 0 and \
        (row['window_ColumnId_nunique'] > 1 or row['window_RowId_nunique'] > 1) else 0, axis=1)
    df_ret['fault_mode_row'] = df_ret.apply(
        lambda row: 1 if row['window_RowId_nunique'] == 1 and row['window_ColumnId_nunique'] > 1 else 0, axis=1)
    df_ret['fault_mode_column'] = df_ret.apply(
        lambda row: 1 if row['window_ColumnId_nunique'] == 1 and row['window_RowId_nunique'] > 1 else 0, axis=1)
    df_ret['fault_mode_cell'] = df_ret['window_CellId_nunique'].apply(lambda x: 1 if x > 1 else 0)

    # 变化特征
    for col in ['LogTime', 'burst_count', 'dq_count']:
        aggs = df.groupby(['time_index'])[col].agg(list).to_frame().reset_index()
        aggs[f'window_{col}_diff'] = aggs[col].apply(np.diff)
        aggs[f'window_{col}_diff_mean'] = aggs[f'window_{col}_diff'].apply(
            lambda x: -1 if len(x) == 0 else np.mean(x))
        aggs[f'window_{col}_diff_max'] = aggs[f'window_{col}_diff'].apply(
            lambda x: -1 if len(x) == 0 else np.max(x))
        aggs[f'window_{col}_diff_std'] = aggs[f'window_{col}_diff'].apply(
            lambda x: -1 if len(x) == 0 else np.std(x))
        aggs.drop([col, f'window_{col}_diff'], axis=1, inplace=True)
        df_ret = df_ret.merge(aggs, on='time_index', how='left')
        
    save_path = sn_file.replace(DATA_PATH, FEATURE_PATH).replace("csv", "feather")
    df_ret.to_feather(save_path)

    sn_files = glob.glob(f'{DATA_PATH}/type_[AB]/*.{DATA_SUFFIX}')
    sn_files.sort()
    with Pool(WORKER_NUM) as pool:
        list(
            tqdm(
                pool.imap(window_gather_feathers, sn_files),
                total=len(sn_files),
                desc="Generating features",
            )
        )