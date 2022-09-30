# -*- coding: utf-8 -*-
''' 
    @date        : 2020/7/15 10:32
    @Author      : Zyy 
    @File Name   : rest_pre.py
    @Description :  解析传入的json数据并返回预测结果
'''
import json

import numpy as np
from config import cur_config as cfg
import datetime
import pandas as pd
pd.set_option("display.max_columns", None)
import os
import time
from multiprocessing.pool import Pool

# warnings.filterwarnings("ignore")

# qujian_base_feature_table = pd.read_csv(os.path.join(cfg.SAVE_HISTORY_FEATURE_PATH, 'qujian_base_feature.csv'))
qujian_base_feature_table = pd.read_csv(os.path.join('qujian_base_feature.csv'))


# stn_base_feature_table = pd.read_csv(os.path.join(cfg.SAVE_HISTORY_FEATURE_PATH, 'stn_base_feature.csv'))
stn_base_feature_table = pd.read_csv(os.path.join('stn_base_feature.csv'))

train_qujian_base_feature_table=pd.read_csv(os.path.join('train_qujian_base_feature.csv'))

# stn_train_base_feature_table = pd.read_csv(os.path.join(cfg.SAVE_HISTORY_FEATURE_PATH, 'stn_train_base_feature.csv'))

stn_train_base_feature_table = pd.read_csv(os.path.join('stn_train_base_feature.csv'))


# train_base_feature_table = pd.read_csv(os.path.join(cfg.SAVE_HISTORY_FEATURE_PATH, 'train_base_feature.csv'))
train_base_feature_table = pd.read_csv(os.path.join('train_base_feature.csv'))
# cfg.SAVE_POST_PROCESS_PATH,
reg_xs_rule_table = pd.read_csv(os.path.join( 'qujian_xs_rule.csv'))


stn_xs_rule_table = pd.read_csv(os.path.join( 'stn_xs_rule.csv'))


def what_time_api(hour_time):
    '''
    [通过传入的小时时间判断当前列车属于一天中那个时间段并返回由历史数据独热编码后的column名称]
    :param hour_time:  图定时间提取的小时
    :return:  独热编码的某一列列名
    '''

    if hour_time >= 0 and hour_time < 6:
        return 'EarlyMorning'
    elif hour_time >= 6 and hour_time < 12:
        return 'Morning'
    elif hour_time >= 12 and hour_time < 18:
        return 'Afternoon'
    else:
        return 'Night'


def stn_train_state_api(arrive_time, depart_time, former_arrive_time, former_depart_time):
    '''
    [统计列车在站与站之间的运行状态，返回根据历史数据独热编码的column名称]
    :param arrive_time:  当前站的图定到达时间
    :param depart_time:  当前站的图定出发时间
    :param former_arrive_time:  上一站的图定到达时间
    :param former_depart_time:  上一站的图定出发时间
    :return:  独热编码的某一列列名
    '''

    if arrive_time == depart_time:
        if former_arrive_time == former_depart_time:
            return 'STN_TRAIN_STATE_1'
        else:
            return 'STN_TRAIN_STATE_3'
    else:
        if former_arrive_time != former_depart_time:
            return 'STN_TRAIN_STATE_2'
        else:
            return 'STN_TRAIN_STATE_3'


def history_feature_api(curr_stn_name, former_stn_name, arr_name,):
    """
    获取历史统计特征值.
    Args:
        curr_stn_name (string): 当前站站名
        former_stn_name (string): 前一站站名
        arr_name (string): 列车号
    Returns:
        list: 历史统计特征值列表, 依次为:
            'distance',  	 # 站距
            'plan_speed',    # 区间计划运行速度
            'capacity',      # 运力（车站历史被经过的次数）
            'stn_arr_delay_pct',  		# 车站历史晚到发生概率
            'stn_arr_early_pct',  		# 车站历史早到发生概率
            'stn_arr_delay_ave',  		# 车站历史晚到时间均值
            'stn_arr_early_ave',  		# 车站历史早到时间均值
            'train_arr_delay_ave',  	# 车次历史晚到时间均值
            'stn_xs_max',        		# 车站历史吸收最大值
            'stn_train_arr_delay_pct',  # 车站车次历史晚到发生概率
            'qujian_xs_ave',  	# 区间历史吸收时间均值
            'qujian_ks_ave',  	# 区间历史扩散时间均值
            'qujian_xs_pct',  	# 区间历史吸收发生概率
            'qujian_ks_pct',  	# 区间历史扩散发生概率
            'qujian_xs_max',  	# 区间历史吸收最大值
    """
    feature_map = {}

    # 提取区间历史特征
    qujian_data = qujian_base_feature_table[
        (qujian_base_feature_table['stn_name'] == curr_stn_name) &
        (qujian_base_feature_table['former_stn'] == former_stn_name)
        ]
    #查不到初始化为0
    if len(qujian_data) == 0:
        qujian_xs_ave=qujian_ks_ave=qujian_xs_max = qujian_xs_pct = qujian_ks_pct = distance = plan_speed = 0
    else:
        qujian_xs_ave = qujian_data['qujian_xs_ave'].values[0]
        qujian_ks_ave = qujian_data['qujian_ks_ave'].values[0]
        qujian_xs_max = qujian_data['qujian_xs_max'].values[0]
        qujian_xs_pct = qujian_data['qujian_xs_pct'].values[0]
        qujian_ks_pct = qujian_data['qujian_ks_pct'].values[0]
        distance = qujian_data['distance'].values[0]
        plan_speed = qujian_data['plan_speed'].values[0]
    feature_map.update(
        {
            'qujian_xs_ave': qujian_xs_ave,
            'qujian_ks_ave': qujian_ks_ave,
            'qujian_xs_max': qujian_xs_max,
            'qujian_xs_pct': qujian_xs_pct,
            'qujian_ks_pct': qujian_ks_pct,
            'distance': distance,
            'plan_speed': plan_speed,
        }
    )

    # 提取车站历史特征
    stn_data = stn_base_feature_table[stn_base_feature_table['stn_name'] == curr_stn_name]

    # 查不到初始化为0
    if len(stn_data) == 0:
        capacity = stn_arr_delay_ave = stn_arr_early_pct = stn_xs_max = stn_arr_delay_pct = stn_arr_early_ave = 0
    else:
        capacity = stn_data['capacity'].values[0]
        stn_arr_delay_ave = stn_data['stn_arr_delay_ave'].values[0]
        stn_arr_early_pct = stn_data['stn_arr_early_pct'].values[0]
        stn_xs_max = stn_data['stn_xs_max'].values[0]
        stn_arr_delay_pct = stn_data['stn_arr_delay_pct'].values[0]
        stn_arr_early_ave = stn_data['stn_arr_early_ave'].values[0]

    feature_map.update(
        {
            'capacity': capacity,
            'stn_arr_delay_ave': stn_arr_delay_ave,
            'stn_arr_early_pct': stn_arr_early_pct,
            'stn_xs_max': stn_xs_max,
            'stn_arr_delay_pct': stn_arr_delay_pct,
            'stn_arr_early_ave': stn_arr_early_ave,
        }
    )

    # 提取提取车站车次历史特征
    stn_train_data = stn_train_base_feature_table[
        (stn_train_base_feature_table['stn_name'] == curr_stn_name) &
        (stn_train_base_feature_table['arr_name'] == arr_name)
    ]

    # 查不到初始化为0
    if len(stn_train_data) == 0:
        stn_train_arr_delay_pct=stn_train_arr_early_pct=stn_train_arr_delay_ave=stn_train_arr_early_ave =stn_train_xs_ave=stn_train_ks_ave=stn_train_xs_pct=stn_train_ks_pct=stn_train_xs_max= 0
    else:
        stn_train_arr_delay_pct = stn_train_data['stn_train_arr_delay_pct'].values[0]
        stn_train_arr_early_pct = stn_train_data['stn_train_arr_early_pct'].values[0]
        stn_train_arr_delay_ave = stn_train_data['stn_train_arr_delay_ave'].values[0]
        stn_train_arr_early_ave = stn_train_data['stn_train_arr_early_ave'].values[0]
        stn_train_xs_ave = stn_train_data['stn_train_xs_ave'].values[0]
        stn_train_ks_ave = stn_train_data['stn_train_ks_ave'].values[0]
        stn_train_xs_pct = stn_train_data['stn_train_xs_pct'].values[0]
        stn_train_ks_pct = stn_train_data['stn_train_ks_pct'].values[0]
        stn_train_xs_max = stn_train_data['stn_train_xs_max'].values[0]

    feature_map.update(
        {
            'stn_train_arr_delay_pct': stn_train_arr_delay_pct,
            'stn_train_arr_early_pct': stn_train_arr_early_pct,
            'stn_train_arr_delay_ave': stn_train_arr_delay_ave,
            'stn_train_arr_early_ave': stn_train_arr_early_ave,
            'stn_train_xs_ave': stn_train_xs_ave,
            'stn_train_ks_ave': stn_train_ks_ave,
            'stn_train_xs_pct': stn_train_xs_pct,
            'stn_train_ks_pct': stn_train_ks_pct,
            'stn_train_xs_max': stn_train_xs_max,

        }
    )

    # 提取车次历史特征
    train_data = (train_base_feature_table[train_base_feature_table['arr_name'] == arr_name])

    # 查不到初始化为0
    if len(train_data) == 0:
        train_arr_delay_ave=train_arr_early_ave=train_arr_delay_pct=train_arr_early_pct = 0
    else:
        train_arr_delay_ave = train_data['train_arr_delay_ave'].values[0]
        train_arr_early_ave = train_data['train_arr_early_ave'].values[0]
        train_arr_delay_pct = train_data['train_arr_delay_pct'].values[0]
        train_arr_early_pct = train_data['train_arr_early_pct'].values[0]

    feature_map.update(
        {
            'train_arr_delay_ave': train_arr_delay_ave,
            'train_arr_early_ave': train_arr_early_ave,
            'train_arr_delay_pct': train_arr_delay_pct,
            'train_arr_early_pct': train_arr_early_pct,
        }
    )

    train_qujian_data=train_qujian_base_feature_table[
        (train_qujian_base_feature_table['stn_name'] == curr_stn_name) &
        (train_qujian_base_feature_table['arr_name'] == arr_name)&
        (train_qujian_base_feature_table['former_stn'] == former_stn_name)
        ]
    # 查不到初始化为0

    if len(train_qujian_data) == 0:
        train_qujian_xs_ave = train_qujian_ks_ave = train_qujian_xs_pct = train_qujian_ks_pct=train_qujian_xs_max = 0

    else:
        train_qujian_xs_ave = train_qujian_data['train_qujian_xs_ave'].values[0]
        train_qujian_ks_ave = train_qujian_data['train_qujian_ks_ave'].values[0]
        train_qujian_xs_pct = train_qujian_data['train_qujian_xs_pct'].values[0]
        train_qujian_ks_pct = train_qujian_data['train_qujian_ks_pct'].values[0]
        train_qujian_xs_max = train_qujian_data['train_qujian_xs_max'].values[0]

    feature_map.update(
        {
            'train_qujian_xs_ave': train_qujian_xs_ave,
            'train_qujian_ks_ave': train_qujian_ks_ave,
            'train_qujian_xs_pct': train_qujian_xs_pct,
            'train_qujian_ks_pct': train_qujian_ks_pct,
            'train_qujian_xs_max': train_qujian_xs_max,
        }
    )

    return feature_map


def data_pre_api(id, meta_data, former_dpt_delay, former_stn_name, former_depart_time, former_arrival_time):
    datas_dict = {}
    arr_name = meta_data['train_num']  # 当前列车号
    arrival_time = datetime.datetime.strptime(meta_data['arrival_time'], "%Y-%m-%d %H:%M:%S")
    depart_time = datetime.datetime.strptime(meta_data['depart_time'], "%Y-%m-%d %H:%M:%S")
    # 列车图定停留时间
    td_yx_time = (arrival_time - former_depart_time).seconds
    # 与前车图定发车时间间隔
    # arr_diff = meta_data['arr_diff'] * 60
    
    # 统计处于一天中哪个时间段
    time_on_day = what_time_api(arrival_time.hour)
    # time_dic = {
    #     "WHAT_TIME_Afternoon": 0, "WHAT_TIME_EarlyMorning": 0, "WHAT_TIME_Morning": 0, "WHAT_TIME_Night": 0,
    # }
    time_dic = {
        "Afternoon": 0, "EarlyMorning": 0, "Morning": 0, "Night": 0,
    }
    time_dic[time_on_day] = 1
    whattime_list = [
        time_dic['Afternoon'], time_dic['EarlyMorning'], time_dic['Morning'],
        time_dic['Night']
    ]
    
    # 发晚模型标记 todo:要做shift(-1)
    if arrival_time == depart_time:
        dpt_model_flag = 0
    else:
        dpt_model_flag = 1
        
    # 发晚模型站点图定停留时间
    stn_tdtl_time = (depart_time - arrival_time).seconds

    # 获取到发晚模型相关的历史特征
    curr_stn_name = meta_data['stn_name']
    feature_map = history_feature_api(curr_stn_name, former_stn_name, arr_name)
    # print(feature_map)
    # 如果为初始晚点站重新判断
    if not meta_data['pred_flag']:
        # 统计到发晚时间--label
        arr_delay = meta_data['arr_delay'] * 60
        dpt_delay = meta_data['dpt_delay'] * 60
        # stn_state_one = 'STN_TRAIN_STATE_3'
    else:
        # 发晚模型到达晚点时间、出发晚点时间
        arr_delay = dpt_delay = 0
        # stn_state_one = stn_train_state_api(arrival_time, depart_time, former_arrival_time, former_depart_time)

    # state_dic = {"STN_TRAIN_STATE_1": 0, "STN_TRAIN_STATE_2": 0, "STN_TRAIN_STATE_3": 0}
    # state_dic[stn_state_one] = 1
    # stn_train_state_list = [
    #     state_dic['STN_TRAIN_STATE_1'], state_dic['STN_TRAIN_STATE_2'], state_dic['STN_TRAIN_STATE_3']
    # ]

    # 每条数据生成字典保存
    datas_dict.update(
        {



        # 到晚模型
        'former_dpt_delay': former_dpt_delay, 'capacity': feature_map['capacity'],'td_yx_time': td_yx_time,
        'qujian_xs_max': feature_map['qujian_xs_max'],'qujian_xs_ave': feature_map['qujian_xs_ave'],
        'qujian_ks_ave': feature_map['qujian_ks_ave'],'train_qujian_xs_ave': feature_map['train_qujian_xs_ave'],
        'train_qujian_ks_ave': feature_map['train_qujian_ks_ave'],'train_qujian_xs_pct': feature_map['train_qujian_xs_pct'],
        'train_qujian_ks_pct': feature_map['train_qujian_ks_pct'],'train_qujian_xs_max': feature_map['train_qujian_xs_max'],
        'qujian_xs_pct': feature_map['qujian_xs_pct'], 'qujian_ks_pct': feature_map['qujian_ks_pct'],
        'distance': feature_map['distance'], 'plan_speed': feature_map['plan_speed'],


        # 发晚模型
        'arr_delay': arr_delay, 'dpt_delay': dpt_delay, 'stn_arr_delay_ave': feature_map['stn_arr_delay_ave'],
        'stn_arr_early_pct': feature_map['stn_arr_early_pct'], 'stn_xs_max': feature_map['stn_xs_max'],
        'stn_tdtl_time': stn_tdtl_time, 'stn_train_arr_delay_pct': feature_map['stn_train_arr_delay_pct'],
        'stn_arr_delay_pct': feature_map['stn_arr_delay_pct'], 'stn_arr_early_ave': feature_map['stn_arr_early_ave'],
        'train_arr_delay_ave': feature_map['train_arr_delay_ave'],'train_arr_delay_pct': feature_map['train_arr_delay_pct'],
        'train_arr_early_pct': feature_map['train_arr_early_pct'],'stn_train_arr_early_pct': feature_map['stn_train_arr_early_pct'],
        'stn_train_arr_early_ave': feature_map['stn_train_arr_early_ave'],'stn_train_arr_delay_ave': feature_map['stn_train_arr_delay_ave'],
        'train_arr_early_ave': feature_map['train_arr_early_ave'],'stn_train_xs_max': feature_map['stn_train_xs_max'],
        'stn_train_ks_ave': feature_map['stn_train_ks_ave'],'stn_train_ks_pct': feature_map['stn_train_ks_pct'],
        'stn_train_xs_ave': feature_map['stn_train_xs_ave'],'stn_train_xs_pct': feature_map['stn_train_xs_pct'],
        
        # 滚动预测
        'stn_name': curr_stn_name, 'arrival_time': arrival_time, 'depart_time': depart_time,
        'dpt_model_flag': dpt_model_flag, 'pred_flag': meta_data['pred_flag'], 'arr_name': arr_name
        }
    )

    # 将数据转为字符串，后续以json格式返回
    for key, value in datas_dict.items():
        datas_dict.update(
            {key: str(value)}
        )
    return (id,datas_dict)


def data_parse(json_data):

    '''
    [解析json数据]
    :param json_data: 得到列车已经到达站的相关信息和要预测站的信息
    :return:  入模的dict_of_list
    '''
    try:
        
        datas_dict_of_list = [] # 每条数据存放
        initial_data = json_data['datas'][0] # 初始晚点站的上一站（模型有上一站相关特征）
        meta_datas = json_data['datas'][1:]# 列车相关数据

        if initial_data['dpt_delay'] is not None:
            former_dpt_delay_empty = initial_data['dpt_delay'] * 60
        else:
            former_dpt_delay_empty = 0

        if initial_data is not None:
            former_stn_name_empty = initial_data['stn_name']
            former_depart_time_empty = datetime.datetime.strptime(initial_data['depart_time'], "%Y-%m-%d %H:%M:%S")
            former_arrival_time_empty = datetime.datetime.strptime(initial_data['arrival_time'], "%Y-%m-%d %H:%M:%S")
        else:

            former_stn_name_empty = former_depart_time_empty = former_arrival_time_empty = None

        former_dpt_delay_list = [former_dpt_delay_empty] + [data["dpt_delay"]*60 if data["dpt_delay"] is not None else 0 for data in meta_datas ]
        former_stn_name_list = [former_stn_name_empty] + [data["stn_name"] for data in meta_datas]
        former_depart_time_list = [former_depart_time_empty] + [datetime.datetime.strptime(data["depart_time"], "%Y-%m-%d %H:%M:%S") for data in meta_datas]
        former_arrival_time_list = [former_arrival_time_empty] + [datetime.datetime.strptime(data["arrival_time"], "%Y-%m-%d %H:%M:%S") for data in meta_datas]
        # print(former_depart_time_list)
        # 循环进行数据处理
        p = Pool()
        for i, (meta_data, former_dpt_delay, former_stn_name, former_depart_time, former_arrival_time) in enumerate(
            zip(
                meta_datas, former_dpt_delay_list, former_stn_name_list,
                former_depart_time_list, former_arrival_time_list
            )

        ):
            # print(111111)
            datas_dict_of_list.append(
                p.apply_async(
                    data_pre_api,
                    args=(
                        i, meta_data, former_dpt_delay, former_stn_name,
                        former_depart_time, former_arrival_time
                    )
                )
            )
        p.close()
        p.join()
        # 获取进程的结果并重排序
        datas_dict_of_list = [res.get() for res in datas_dict_of_list]
        sort_dict = sorted(datas_dict_of_list, key=lambda k: k[0], reverse=False)  # 根据i的从小到大
        datas_dict_of_list = list(map(lambda k : k[1], sort_dict))


        return datas_dict_of_list, "success"

    except Exception:
        print("Json data not valid!\n")
        return [], None, "Fail"


def get_data_api(datas_of_dict):
    '''
    [拼接每条数据，让特征有序]
    :param datas_of_dict:  每条数据的数据字典
    :return:  array类型的到晚模型预测数据和发晚模型预测数据
    '''
    x_arr_data = []
    x_dpt_data = []

    # 已经做独热编码排除做独热编码的原始列名
    # arr_columns_list = [col for col in cfg.X_ARR_COLUMNS if col not in cfg.ONE_HOT_ARR_COLUMNS]
    # dpt_columns_list = [col for col in cfg.X_DPT_COLUMNS if col not in cfg.ONE_HOT_DPT_COLUMNS]
    arr_columns_list = [col for col in cfg.X_ARR_COLUMNS]
    dpt_columns_list = [col for col in cfg.X_DPT_COLUMNS]
    # print(arr_columns_list)
    x_arr_data += [datas_of_dict.get(columns) for columns in arr_columns_list]
    # x_arr_data += datas_of_dict['stn_train_state_list']
    # x_arr_data += datas_of_dict['whattime_list']
    # print(x_arr_data)
    # 将每条数据根据发晚的特征拼接成数组
    x_dpt_data += [datas_of_dict.get(columns) for columns in dpt_columns_list]
    # print(x_dpt_data)
    return np.array(x_arr_data).reshape(1,-1), np.array(x_dpt_data).reshape(1,-1)

def rout_prep_data(id, dict, curr_stn_name):
    '''
      [获取数据中所用的后处理截断数据：区间吸收最大比例和站点吸收最大比例]
      :dict:  数据字典
      :return: dict类型的区间吸收最大比例和站点吸收最大比例
      '''

    dict_res = {}
    reg_xs_rule = reg_xs_rule_table[
        (reg_xs_rule_table['stn_name'] == curr_stn_name) & (reg_xs_rule_table['former_stn'] == dict['stn_name']) &
        (reg_xs_rule_table['arr_name'] == dict['arr_name'])
        ]

    # 表中查不到默认吸收概率为百分之二十
    if len(reg_xs_rule) > 0:
        max_reg_xs_pct = reg_xs_rule['max_qujian_xs_pct'].values[0]
    else:
        max_reg_xs_pct = 0.2

    # 查询站点截断规则表
    stn_xs_rule = stn_xs_rule_table[
        (stn_xs_rule_table['stn_name'] == curr_stn_name) & (stn_xs_rule_table['arr_name'] ==  dict['arr_name'])
        ]

    # 表中查不到默认吸收概率为百分之二十
    if len(stn_xs_rule) > 0:
        max_stn_xs_pct = stn_xs_rule['max_stn_xs_pct'].values[0]
    else:
        max_stn_xs_pct = 0.2

    dict_res.update(
        {
            'max_reg_xs_pct' : max_reg_xs_pct,
            'max_stn_xs_pct' : max_stn_xs_pct,
        }
    )
    return (id, dict_res)


def type_transform(datas_dict_of_list):
    '''
        [将json数据的字符串格式转换成原有的数据格式]
        :datas_dict_of_list:  数据字典
        :return: datas_dict_of_list类型，值从str个数转换为原始数据格式
    '''

    # 根据数据类型的不同，分别将字典的键放入对应的列表中
    # list_type_list = ['stn_train_state_list', 'whattime_list']
    # list_type_list +
    datetime_type_list = ['arrival_time', 'depart_time']
    bool_type_list = ['pred_flag']
    str_type_list = ['stn_name', 'arr_name']
    int_type_list = [name for name in datas_dict_of_list[0]
                     if name not in (datetime_type_list + bool_type_list + str_type_list)]

    for id,data in enumerate(datas_dict_of_list):

        # 将str格数据式转换为float格式
        for float_name in int_type_list:

            data.update(
                {
                    float_name: float(data[float_name]),
                }
            )
            # print(11111)

        # # 将str格数据式转换为list格式
        # for list_name in list_type_list:
        #
        #     data.update(
        #         {
        #             list_name: json.loads(data[list_name]),
        #         }
        #     )

        # 将str格数据式转换为datetime格式
        for datetime_name in datetime_type_list:

            data.update(
                {
                    datetime_name: pd.to_datetime(data[datetime_name]),
                }
            )

        # 将str格数据式转换为bool格式
        for bool_name in bool_type_list:
            data.update(
                {
                    bool_name: bool(data[bool_name]),
                }
            )

    return datas_dict_of_list

def route_pre_api(datas_dict_of_list, arr_model, dpt_model):
    '''
    滚动预测加入后处理逻辑并返回预测结果
    :param datas_dict_of_list:  数据字典的列表
    :type datas_dict_of_list:  list
    :param arr_model:  到晚模型
    :type arr_model:
    :param dpt_model:  发晚模型
    :type dpt_model:
    :param reg_xs_rule_table:  区间吸收后处理逻辑表
    :type reg_xs_rule_table:
    :param stn_xs_rule_table:  站点吸收后处理逻辑表
    :type stn_xs_rule_table:
    :return: 预测结果
    :rtype: array
    '''
    # 计时
    pred_time = 0
    data_start_time = time.time()
    # 将json里的str转换为所需要的格式
    datas_dict_of_list = type_transform(datas_dict_of_list)
    # print('datas_dict_of_list',datas_dict_of_list)

    curr_stn_name_list = [i['stn_name'] for i in datas_dict_of_list[1:]]
    result = []
    p = Pool()
    for i, (dict, curr_stn_name) in enumerate(
            zip(
                datas_dict_of_list, curr_stn_name_list
            )
    ):
        result.append(p.apply_async(rout_prep_data, args=(i, dict, curr_stn_name)))
    p.close()
    p.join()
    result = [res.get() for res in result]
    result = sorted(result, key=lambda k:k[0], reverse=False)
    rule_dict_of_list = [i[1] for i in result]
    # print('rule_dict_of_list',rule_dict_of_list)

    data_deal_time = time.time() - data_start_time

    prep_start_time = time.time()
    # 接受预测结果
    pred = np.array([[]], dtype=np.int64).reshape(0, 4)
    # 到晚模型使用的上一站发晚值
    pred_former_dpt_delay = 0


    # 滚动预测
    for i in range(len(datas_dict_of_list)-1):

        arrival_time = datas_dict_of_list[i + 1]['arrival_time'] # 要预测站的站名为当前站
        depart_time = datas_dict_of_list[i + 1]['depart_time'] # 要预测站的站名为当前站
        former_depart_time = datas_dict_of_list[i]['depart_time']

        DPT_MODEL_FLAG = datas_dict_of_list[i + 1]['dpt_model_flag'] # 要预测站的flag
        PRED_FLAG = datas_dict_of_list[i]['pred_flag']

        max_reg_xs_pct = rule_dict_of_list[i]['max_reg_xs_pct']
        max_stn_xs_pct = rule_dict_of_list[i]['max_stn_xs_pct']

        # 得到处理好的数据
        x_arr_data, x_dpt_data = get_data_api(datas_dict_of_list[i])
        # print('x_arr_data',x_arr_data)
        # print('x_dpt_data',x_dpt_data)
        # print(x_arr_data)
        # print(x_arr_data[0, 0])
        # 不是初始晚点站用上一站发晚修改数据
        if PRED_FLAG:
            x_arr_data[0, 0] = pred_former_dpt_delay
            # print('x_arr_data[0, 0]')
            # print('pred_former_dpt_delay',pred_former_dpt_delay)
        # 是初始晚点站则直接预测
        arr_pred = arr_model.predict(x_arr_data)
        # print('arr_pred',arr_pred)
        # 是否使用发晚模型（非行调站使用）
        if DPT_MODEL_FLAG:
            x_dpt_data[0, 0] = arr_pred
            # print('x_dpt_data[0, 0]',x_dpt_data[0, 0])
            dpt_pred = dpt_model.predict(x_dpt_data)

        else:
            dpt_pred = arr_pred

        # 统计区间吸收时间
        reg_xs_time = pred_former_dpt_delay - arr_pred # 预测的区间吸收时间, 换成秒用发晚秒数减去到晚秒数
        # 初始晚点站的区间吸收时间用真实晚点时间
        if not PRED_FLAG:
            reg_xs_time = datas_dict_of_list[0]['dpt_delay'] - arr_pred # 预测的区间吸收时间

        # 计算预测的区间吸收概率
        reg_tdtl_time = (arrival_time - former_depart_time).seconds
        if reg_xs_time != 0:
            reg_xs_pct = reg_tdtl_time / reg_xs_time
        else:
            reg_xs_pct = 0

        # 增加截断逻辑：预测吸收概率大于历史最大吸收概率则截断
        if reg_xs_pct > max_reg_xs_pct:
            reg_xs_time = reg_tdtl_time * max_reg_xs_pct
            # 通过截断后的区间吸收修改到达晚点值
            arr_pred = pred_former_dpt_delay - reg_xs_time
            # 初始晚点站到达晚点值修改
            if not PRED_FLAG:
                arr_pred = datas_dict_of_list[0]['dpt_delay'] - reg_xs_time

            # 重新预测发晚值
            if DPT_MODEL_FLAG:
                x_dpt_data[0, 0] = arr_pred
                dpt_pred = dpt_model.predict(x_dpt_data)

            else:
                dpt_pred = arr_pred

        # 计算站点吸收时间和吸收概率
        stn_xs_time = arr_pred - dpt_pred # 预测的站点吸收时间
        stn_tdtl_time = (depart_time - arrival_time).seconds
        if stn_xs_time != 0:
            stn_xs_pct = stn_tdtl_time / stn_xs_time
        else:
            stn_xs_pct = 0

        # 增加截断逻辑：预测吸收概率大于历史最大吸收概率则截断
        if stn_xs_pct > max_stn_xs_pct:
            stn_xs_time = stn_tdtl_time * max_stn_xs_pct

            # 通过截断后的区间吸收修改出发晚点值
            dpt_pred = arr_pred - stn_xs_time
        pred_former_dpt_delay = dpt_pred # 修改上一站发晚值
        # print('pred_former_dpt_delay',pred_former_dpt_delay)

        arr_pred = np.array(arr_pred).reshape(-1, 1)
        dpt_pred = np.array(dpt_pred).reshape(-1, 1)
        reg_xs_time = np.array(reg_xs_time).reshape(-1, 1)
        stn_xs_time = np.array(stn_xs_time).reshape(-1, 1)

        pred_result = np.concatenate([arr_pred, dpt_pred, reg_xs_time, stn_xs_time]).reshape(-1, 4)
        pred = np.concatenate([pred, pred_result], axis=0)

        # print('pred',pred)
        # 计算预测时间
        pred_time = time.time() - prep_start_time

    return pred, data_deal_time, pred_time


def get_json_api(pred):
    '''
    [将预测结果进行展示]
    :param pred:  模型得到的预测值
    :param former_stn_num:  列车前序站和当前站数
    :param status:  返回的状态
    :return:  json
    '''
    pred = np.around(pred / 60,  decimals=2)
    # 预测时避免修改数据造成不必要的影响只预测了当前站到后续站，剔除了第一站的数据，所以要-2
    pred_show = [(i[0], i[1], i[3], i[2]) for i in pred]
    # # 返回json数据
    result_json = {"Status": "success", 'results': pred_show}
    return result_json

