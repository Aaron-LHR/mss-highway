import os
import pandas as pd
import numpy as np
import random
import joblib
import warnings

from config import cur_config as cfg


def remove_duplicates_and_save(df, target_col, feature_col, save_path):
    '''
    :param df: 历史数据
    :type df:
    :param target_col: 表名字
    :type target_col:  str
    :param feature_col: 保存的特征
    :type feature_col:  str of list
    :param save_path: 要保存的路径
    :type save_path: str
    :return: 历史特征表
    '''

    sub = df[target_col + feature_col]
    sub.drop_duplicates(subset=target_col, keep='first', inplace=True)
    sub.to_csv(save_path, index=False)
    print('saving history features to... ', save_path)
    return sub

def aftertreatment_rule(df):
    '''

    :param df: 历史数据
    :type df:
    :return: qujian_xs_rule.csv, stn_xs_rule.csv
    :rtype:
    '''

    # df['FORMER_STN'] = df['STN_NAME'].shift(1)  # 上一站
    # df['FORMER_DPT_DELAY'] = df['DPT_DELAY'].shift(1)  # 上一站
    df['former_stn']='深圳'
    df['former_dpt_delay']=0

    for i in range(len(df)):
        if i==0:
            df['former_stn'].iat[i]=df['stn_name'].iat[i]
            df['former_dpt_delay'].iat[i]=df['arr_delay'].iat[i]
        else:
            if df['statis_id'].iat[i]==df['statis_id'].iat[i-1]:
                df['former_stn'].iat[i] = df['stn_name'].iat[i-1]
                df['former_dpt_delay'].iat[i] = df['dpt_delay'].iat[i-1]

            else:
                df['former_stn'].iat[i] = df['stn_name'].iat[i]
                df['former_dpt_delay'].iat[i] = df['arr_delay'].iat[i]


    # mask = df['STATIS_ID'].shift(1) == df['STATIS_ID']  # 始发站没有上一站
    # df = df[mask]

    # 区间吸收规则表
    df = df[(df['qujian_tdtl_time'] != 0) & (df['qujian_xs_time'] >= 0)]
    df['qujian_xs_pct'] = df['qujian_xs_time'] / df['qujian_tdtl_time']
    qujian_xs_rule = df.groupby(['stn_name', 'former_stn', 'arr_name'])['qujian_xs_pct'].agg([('max_qujian_xs_pct', 'max')]).reset_index()

    # 站点吸收规则表
    df = df[(df['stn_tdtl_time'] != 0) & (df['stn_xs_time'] >= 0)]
    df['stn_xs_pct'] = df['stn_xs_time'] / df['stn_tdtl_time']
    stn_xs_rule = df.groupby(['stn_name', 'arr_name'])['stn_xs_pct'].agg([('max_stn_xs_pct', 'max')]).reset_index()

    return qujian_xs_rule, stn_xs_rule

if __name__ == '__main__':

    '''读取全年历史序列数据进行统计，使用时需要修改相应路径'''
    # history_feature_data = pd.read_csv(cfg.HISTORY_FEATURE_DATA_PATH)
    history_feature_data = pd.read_csv('2020_rnn_jhdwandian.csv')
    history_feature_data.columns = [i.lower() for i in history_feature_data.columns]
    print(history_feature_data['stn_name'].head(),history_feature_data['former_stn'].head())
    # 调用去重保存方法保存历史相关特征
    # print(history_feature_data[cfg.STN_COL])
    # history_feature_data['former_stn_col']='深圳'
    # for i in range(len(history_feature_data)):
    #     if i==0:
    #         history_feature_data['former_stn_col'].iat[i]=history_feature_data['stn_name'].iat[i]
    #     else:
    #         if history_feature_data['statis_id'].iat[i]==history_feature_data['statis_id'].iat[i-1]:
    #             history_feature_data['former_stn_col'].iat[i] = history_feature_data['stn_name'].iat[i-1]
    #
    #         else:
    #             history_feature_data['former_stn_col'].iat[i] = history_feature_data['stn_name'].iat[i]

    # history_feature_data['former_stn_col'] = history_feature_data[cfg.STN_COL].shift(1)
    # mark = history_feature_data['statis_id'].shift(1) == history_feature_data['statis_id']  # 始发站没有上一站
    # print(history_feature_data)
    if not os.path.exists(cfg.SAVE_HISTORY_FEATURE_PATH):
        os.makedirs(cfg.SAVE_HISTORY_FEATURE_PATH)
    # TODO 新建qujian_train_table将plan_speed特征放入
    # 根据站点统计的变量，站表
    # stn_base_feature_path = os.path.join(cfg.SAVE_HISTORY_FEATURE_PATH, 'stn_base_feature_test.csv')
    stn_base = remove_duplicates_and_save(history_feature_data,  cfg.STN_COL, cfg.STN_BASE_COLS, 'stn_base_feature.csv')
    # 根据车次统计的变量，车次表
    # train_base_feature_path = os.path.join(cfg.SAVE_HISTORY_FEATURE_PATH, 'train_base_feature_test.csv')
    train_base = remove_duplicates_and_save(history_feature_data, cfg.TRAIN_COL, cfg.TRAIN_BASE_COLS,
                                            'train_base_feature.csv')
    # 根据区间统计的变量， 区间表
    # qujian_base_feature_path = os.path.join(cfg.SAVE_HISTORY_FEATURE_PATH, 'qujian_base_feature_test.csv')
    qujian_base = remove_duplicates_and_save(history_feature_data, cfg.QUJIAN_COL, cfg.QUJIAN_BASE_COLS,
                                             'qujian_base_feature.csv')
    # 根据车站和车次统计的变量， 车次站表
    # stn_train_base_feature_path = os.path.join(cfg.SAVE_HISTORY_FEATURE_PATH, 'stn_train_base_feature_test.csv')
    stn_train_base = remove_duplicates_and_save(history_feature_data, cfg.STN_COL + cfg.TRAIN_COL,
                                                     cfg.STN_TRAIN_BASE_COLS, 'stn_train_base_feature.csv')
    # 根据车次和区间统计的变量，车次区间表
    # qujian_train_base_feature_path = os.path.join(cfg.SAVE_HISTORY_FEATURE_PATH, 'qujian_train_base_feature_test.csv')
    qujian_train_base = remove_duplicates_and_save(history_feature_data, cfg.QUJIAN_COL + cfg.TRAIN_COL,
                                                        cfg.TRAIN_QUJIAN_BASE_COLS, 'train_qujian_base_feature.csv')


    # 统计后处理逻辑表
    # aftertreatment_rule_data = pd.read_csv(cfg.HISTORY_FEATURE_DATA_PATH)
    aftertreatment_rule_data = pd.read_csv('2020_ql_dwandian.csv')
    aftertreatment_rule_data.columns = [i.lower() for i in aftertreatment_rule_data.columns]
    aftertreatment_rule_data = aftertreatment_rule_data[cfg.AFTERTREATMENT_DEAL_COLUMNS]

    qujian_xs_rule, stn_xs_rule = aftertreatment_rule(aftertreatment_rule_data)
    if not os.path.exists(cfg.SAVE_POST_PROCESS_PATH):
        os.makedirs(cfg.SAVE_POST_PROCESS_PATH)
    #
    # qujian_xs_rule.to_csv(os.path.join(cfg.SAVE_POST_PROCESS_PATH, 'qujian_xs_rule_test.csv'), index=False)
    # stn_xs_rule.to_csv(os.path.join(cfg.SAVE_POST_PROCESS_PATH, 'stn_xs_rule_test.csv'), index=False)

    qujian_xs_rule.to_csv(os.path.join('qujian_xs_rule.csv'), index=False)
    stn_xs_rule.to_csv(os.path.join('stn_xs_rule.csv'), index=False)





