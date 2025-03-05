import warnings
warnings.simplefilter('ignore')

import os
import gc
import glob
import math
import random
from multiprocessing import Pool

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
from tqdm.auto import tqdm

from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
import lightgbm as lgb
DATA_PATH = './feature'
FEATURE_PATH = './feature'
WORKER_NUM = 64
NEGATIVE_SAMPLE_RATIO = 0.1  # 增加负样本比例
# 内存故障维修记录单 (2024/01/01 - 2024/05/31) 
# 记录的是该内存在这五个月内第一次故障的时间点 
# 如果没出现在记录单内的内存，说明在这五个月内没有发生故障

ticket = pd.read_csv(f'{DATA_PATH}/ticket.csv')
# display(ticket)

# 内存 - 第一次故障时间 映射表
ticket_alarm_time_map = ticket[['sn_name', 'alarm_time']].set_index('sn_name').to_dict()['alarm_time']
# 所有内存 sn
all_sn_names = [x.split('/')[-1].replace('.feather', '') for x in glob.glob(f'{DATA_PATH}/type_[AB]/sn_*.feather')]

# 在这五个月内发生过故障的内存 sn
positive_sn_names = ticket['sn_name'].values.tolist()

# 在这五个月内没发生过故障的内存 sn
negative_sn_names = list(set(all_sn_names) - set(positive_sn_names))

assert len(all_sn_names) == len(positive_sn_names) + len(negative_sn_names)
# 这里用回归来做, 毕竟从官方定义来说, 第一次出现故障是最重要的, 需要有个体现
# 线性指标变换: 第一次出现故障为 1, 往后的正样本进行标签衰减, 最小值截断为 0.5

def get_label_by_index(idx):
    label = 1 - (idx / 250) * 0.5
    return max(label, 0.5)
def get_positive_data(sn_name):
    # 获取文件路径
    if os.path.exists(f'{FEATURE_PATH}/type_A/{sn_name}.feather'):
        filepath = f'{FEATURE_PATH}/type_A/{sn_name}.feather'
    else:
        filepath = f'{FEATURE_PATH}/type_B/{sn_name}.feather'
    # 读取文件内容
    df = pd.read_feather(filepath)
    # 添加 sn
    df['sn_name'] = sn_name
    # 从映射表获取该 sn 第一次发生故障的时间点 (每个 sn 都只有一次)
    alarm_time = ticket_alarm_time_map[sn_name]
    # 训练集 timestamp < 1717171200 (2024/06/01 00:00:00) & alarm_time 之后的不要了
    df = df[(df['LogTime'] < 1717171200)&(df['LogTime'] <= alarm_time - 15*60)].sort_values('LogTime').reset_index(drop=True) 
    labels = []
    idx = 0   # 记录离第一次故障信息的相对位置
    for ts in df['LogTime'].values:
        # 落在范围内的才能是正样本 Tl(15minutes)+Tp(7days)
        if ts + 15*60 <= alarm_time <= ts + 15*60 + 7*24*60*60:
            labels.append(get_label_by_index(idx))
            idx += 1
        else:
            labels.append(0)
    df['label'] = labels
    df['serial_number_type'] = df_sn_type[df_sn_type['sn_name'] == sn_name]['serial_number_type'].values[0]
    return df

with Pool(WORKER_NUM) as pool:
    res = list(
        tqdm(
            pool.imap(get_positive_data, positive_sn_names),
            total=len(positive_sn_names),
            desc="Generating positive data",
        )
    )

df_positive = pd.concat(res).reset_index(drop=True)
df_positive['label'].value_counts(dropna=False)
def add_time_features(df):
    """添加时间相关特征"""
    df['hour'] = pd.to_datetime(df['LogTime'], unit='s').dt.hour
    df['weekday'] = pd.to_datetime(df['LogTime'], unit='s').dt.weekday
    df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
    return df

def add_rolling_features(df):
    """添加滚动统计特征"""
    windows = [3, 5, 7]
    for window in windows:
        # 错误日志数量的滚动统计
        df[f'window_logs_count_rolling_mean_{window}'] = df.groupby('sn_name')['window_logs_count'].rolling(window).mean().reset_index(0, drop=True)
        df[f'window_logs_count_rolling_std_{window}'] = df.groupby('sn_name')['window_logs_count'].rolling(window).std().reset_index(0, drop=True)
        
        # DQ和burst相关的滚动统计
        for col in ['window_dq_count', 'window_burst_count']:
            df[f'{col}_rolling_mean_{window}'] = df.groupby('sn_name')[col].rolling(window).mean().reset_index(0, drop=True)
            df[f'{col}_rolling_max_{window}'] = df.groupby('sn_name')[col].rolling(window).max().reset_index(0, drop=True)
    
    df = df.fillna(-1)
    return df

def get_negative_data(sn_name):
    # 获取文件路径
    if os.path.exists(f'{FEATURE_PATH}/type_A/{sn_name}.feather'):
        filepath = f'{FEATURE_PATH}/type_A/{sn_name}.feather'
    else:
        filepath = f'{FEATURE_PATH}/type_B/{sn_name}.feather'
    # 读取文件内容
    df = pd.read_feather(filepath)
    # 添加 sn
    df['sn_name'] = sn_name
    # 负样本只用训练集的最后两个月 训练集 (2024/04/01 00:00:00) 1711900800 < timestamp < 1717171200 (2024/06/01 00:00:00)
    df = df[df['LogTime'] < 1717171200].sort_values('LogTime').reset_index(drop=True)
    df['label'] = 0
    df['serial_number_type'] = df_sn_type[df_sn_type['sn_name'] == sn_name]['serial_number_type'].values[0]
    return df

with Pool(WORKER_NUM) as pool:
    res = list(
        tqdm(
            pool.imap(get_negative_data, negative_sn_names),
            total=len(negative_sn_names),
            desc="Generating negative data",
        )
    )

df_negative = pd.concat(res).reset_index(drop=True)
df_data = pd.concat([df_positive, df_negative]).sort_values('sn_name').reset_index(drop=True)
feature_names = [c for c in df_data.columns if c not in ['LogTime', 'sn_name', 'label', 'time_index']]
df_data
kf = GroupKFold(n_splits=5)
models = []
oof_pred = np.zeros(len(df_data))
for i, (train_index, valid_index) in enumerate(kf.split(df_data, groups=df_data['sn_name'])):
    print(f'Fold {i} ...')
    x_valid = df_data.loc[valid_index, feature_names].copy()
    y_valid = df_data.loc[valid_index, 'label']
    
    # 训练集负采样，但保持一定比例的难例
    train_df = df_data.loc[train_index, :].copy()
    pos_df = train_df[train_df['label'] != 0].reset_index(drop=True)
    neg_df = train_df[train_df['label'] == 0].reset_index(drop=True)
    
    # 保留预测值较高的难例
    if i > 0:  # 从第二折开始使用难例挖掘
        neg_df['pred_temp'] = model.predict(neg_df[feature_names])
        hard_neg = neg_df[neg_df['pred_temp'] > 0.3]  # 保留预测值较高的样本
        easy_neg = neg_df[neg_df['pred_temp'] <= 0.3].sample(
            frac=NEGATIVE_SAMPLE_RATIO, 
            random_state=42
        )
        neg_df = pd.concat([hard_neg, easy_neg])
    else:
        neg_df = neg_df.sample(frac=NEGATIVE_SAMPLE_RATIO, random_state=42)
    
    train_df = pd.concat([pos_df, neg_df]).reset_index(drop=True)
    x_train = train_df[feature_names].copy()
    y_train = train_df['label']
    
    model = lgb.LGBMRegressor(
        max_depth=8, 
        num_leaves=64,
        min_child_samples=64,
        n_estimators=500, 
        learning_rate=0.1, 
        verbose=-1
    )
    model.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)]
    )
    oof_pred[valid_index] = model.predict(x_valid)
    models.append(model)
    del model; gc.collect()
df_data['pred'] = oof_pred
df_data['pred'].describe()
# sn wise 指标 (不是很高效, 需改进)
def calc_score(df_data, ticket, threshold=0.5):
    """计算 sn wise 指标的优化版本
    
    优化点：
    1. 使用 groupby + agg 替代循环
    2. 使用向量化操作替代逐行判断
    3. 减少 DataFrame 的复制操作
    """
    # 合并数据，只保留需要的列
    result_df = pd.merge(
        df_data[['sn_name', 'LogTime', 'pred']], 
        ticket[['sn_name', 'alarm_time']], 
        on='sn_name', 
        how='left'
    )
    
    # 处理正样本
    pos_df = result_df[result_df['alarm_time'].notna()].copy()
    
    # 计算时间窗口条件
    pos_df['time_diff'] = pos_df['alarm_time'] - pos_df['LogTime']
    pos_df['in_window'] = (pos_df['time_diff'] >= 15*60) & \
                         (pos_df['time_diff'] <= 15*60 + 7*24*60*60)
    
    # 计算每个 sn 是否有预测成功的样本
    pos_pred = pos_df.groupby('sn_name').agg({
        'pred': lambda x: (x >= threshold).any(),
        'in_window': 'any'
    })
    pos_pred['pred'] = (pos_pred['pred'] & pos_pred['in_window']).astype(int)
    pos_pred['label'] = 1
    
    # 处理负样本
    neg_pred = result_df[result_df['alarm_time'].isna()].groupby('sn_name')['pred'].max()
    neg_pred = pd.DataFrame({
        'pred': (neg_pred >= threshold).astype(int),
        'label': 0
    })
    
    # 合并计算 F1 分数
    metric_df = pd.concat([
        pos_pred[['pred', 'label']], 
        neg_pred
    ]).reset_index()
    
    score = f1_score(metric_df['label'], metric_df['pred'])
    return score, metric_df

# 测试不同阈值
for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
    score, _ = calc_score(df_data, ticket, threshold)
    print(f'threshold: {threshold:.3f} score: {score:.6f}')
feature_importances = np.mean([m.feature_importances_ for m in models], axis=0)
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": feature_importances
}).sort_values(by="Importance", ascending=False)

# display(importance_df)
# 测试集
def get_test_data(sn_name):
    # 获取文件路径
    if os.path.exists(f'{FEATURE_PATH}/type_A/{sn_name}.feather'):
        filepath = f'{FEATURE_PATH}/type_A/{sn_name}.feather'
    else:
        filepath = f'{FEATURE_PATH}/type_B/{sn_name}.feather'
    # 读取文件内容
    df = pd.read_feather(filepath)
    # 添加 sn
    df['sn_name'] = sn_name
    df = df[df['LogTime'] >= 1717171200].sort_values('LogTime').reset_index(drop=True)
    return df

# 已经故障过的正样本就不用做 infer 了，定义是第一次故障才会记录
with Pool(WORKER_NUM) as pool:
    res = list(
        tqdm(
            pool.imap(get_test_data, negative_sn_names),
            total=len(negative_sn_names),
            desc="Generating test data",
        )
    )

df_test = pd.concat(res).reset_index(drop=True)
df_test

pred_test = np.zeros(len(df_test))
for model in tqdm(models):
    pred_test += model.predict(df_test[feature_names]) / kf.n_splits
# 生成特征时忘了 ...
df_sn_type = pd.concat([
    pd.DataFrame({
        'sn_name': [i.split('/')[-1].replace('.feather', '') for i in glob.glob(f'{DATA_PATH}/type_A/*.feather')],
        'serial_number_type': ['A'] * len(glob.glob(f'{DATA_PATH}/type_A/*.feather'))
    }),
    pd.DataFrame({
        'sn_name': [i.split('/')[-1].replace('.feather', '') for i in glob.glob(f'{DATA_PATH}/type_B/*.feather')],
        'serial_number_type': ['B'] * len(glob.glob(f'{DATA_PATH}/type_B/*.feather'))
    })
]).reset_index(drop=True)

df_sub = df_test[['sn_name', 'LogTime']].copy()
df_sub = df_sub.merge(df_sn_type, on='sn_name', how='left')
df_sub['pred'] = pred_test

df_sub = df_sub[df_sub['pred'] >= 0.5].reset_index(drop=True)
df_sub.drop('pred', axis=1, inplace=True)
df_sub.columns = ['sn_name', 'prediction_timestamp', 'serial_number_type']
print(df_sub['sn_name'].nunique())
# display(df_sub)
df_sub.to_csv('submission.csv', index=False)