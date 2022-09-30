# -*- coding: utf-8 -*-
# @Time    : 2022/3/2 10:24
# @Author  : ZZC
# @File    : config.py

class OriginConfig(object):
    # 第七版到晚模型输入特征字段  （# todo：增删）
    # 'REG_XS_MAX_TIME',
    X_ARR_COLUMUS = [
        'FORMER_DPT_DELAY', 'Capacity', 'REG_TDTL_TIME', 'qujian_xs_max', 'arr_diff', 'STN_TRAIN_STATE',
        'qujian_xs_pct', 'qujian_ks_pct', 'DISTANCE', 'PLAN_SPEED', 'WhatTime',
    ]
    # 发晚模型输入特征字段  （#todo:增删）
    X_DPT_COLUMNS = [
        'ARR_DELAY', 'STN_TDTL_TIME', 'STN_TRAIN_STATE', 'WHAT_TIME',
                     'SC_RENSHU', 'XC_RENSHU'
    ]
    # 需要独热编码的特征  （#todo:增删）
    ONE_HOT_COLUMNS = ['STN_TRAIN_STATE', 'WHAT_TIME']

    # 目标变量  （#todo:增删）
    # Y_COLUMNS = ['ARR_DELAY', 'DPT_DELAY', 'REG_XS_TIME', 'STN_XS_TIME']
    Y_COLUMNS = ['ARR_DELAY', 'DPT_DELAY' ]
    # 因变量名称  （#todo:增删）
    TARGET_NAME_Y = [i + '_Y' for i in Y_COLUMNS]
    # 到晚模型预测目标字段
    Y_ARR_COLUMNS = [TARGET_NAME_Y[0] ]
    # 发晚模型预测目标字段
    Y_DPT_COLUMNS = [TARGET_NAME_Y[1] ]

    #用于训练得到需要的df  （#todo:增删）
    PART_DF_PREP = ['STATIS_ID', 'ARR_NAME', 'DAODA_TIME', 'CHUFA_TIME', 'STN_NAME',
                    'DEPART0','ARRIVAL0', 'front_train_dpt_delay', 'STN_TRAIN_STATE',
                     'WhatTime', 'STN_TDTL_TIME', 'SC_RENSHU','XC_RENSHU','ARR_DELAY',
                    'DPT_DELAY', 'REG_XS_TIME','STN_XS_TIME', 'DPT_MODEL_FLAG',
                    ]

    # 修改部分数据列明  （#todo:增删）
    CHANGE_DF_DIC = {'STATIS_ID': 'ID', 'front_train_dpt_delay': 'FORNT_TRAIN_DPT_DELAY',
                     'WhatTime': 'WHAT_TIME'}


    # 路径相关
    DF_PATH = './data/G-JH-HN_datas_deal.csv'
    FEATURE_DICT = './feature_dict'
    FEATURE_REG_TABLE = 'F:/Python-WorkSpace/train_delay_jh/feature_dict/reg_max_time_table.csv'
    FEATURE_STN_TABLE = 'F:/Python-WorkSpace/train_delay_jh/feature_dict/stn_xs_max_time_table.csv'
    MODEL_PATH = './model'
    MODELARR_PKL_PATH = 'F:/Python-WorkSpace/train_delay_jh/model/LGBM_ARR.model'
    MODELDPT_PKL_PATH = 'F:/Python-WorkSpace/train_delay_jh/model/LGBM_DPT.model'
    DATA_PATH = './data'

    # 统计指标
    ACC_SECONDS = 60


    # 第八版多特征
    one_hot_columns = ['STN_TRAIN_STATE', 'WhatTime', ]
    one_hot_arr_columns = ['STN_TRAIN_STATE', 'WhatTime', ]
    one_hot_dpt_columns = []
    # 第八版多特征  16
    x_arr_columns = [
        'FORMER_DPT_DELAY', 'Capacity', 'REG_TDTL_TIME', 'qujian_xs_max', 'arr_diff', 'STN_TRAIN_STATE',
        'qujian_xs_pct', 'qujian_ks_pct', 'DISTANCE', 'PLAN_SPEED', 'WhatTime',

    ]
    # 第八版多特征  10
    x_dpt_columns = [
        'ARR_DELAY', 'stn_arr_delay_ave', 'stn_arr_early_pct', 'Capacity', 'stn_xs_max',
        'STN_TDTL_TIME', 'stn_train_arr_delay_pct',
        'stn_arr_delay_pct', 'stn_arr_early_ave', 'train_arr_delay_ave',

    ]

    # 第八版
    y_columns = ['ARR_DELAY', 'DPT_DELAY']
    target_names_y = [i + '_Y' for i in y_columns]  # 目标当作因变量的名称
    # 第八版
    y_arr_model_columns = target_names_y[0]
    y_dpt_model_columns = target_names_y[1]



class MultVarCongfig(object):

    # 第八版
    # 到晚模型输入特征字段，多特征  16  （# todo：增删）
    X_ARR_COLUMNS = [
        'former_dpt_delay', 'capacity', 'reg_tdtl_time', 'qujian_xs_max', 'arr_diff', 'stn_train_state',
        'qujian_xs_pct', 'qujian_ks_pct', 'distance', 'plan_speed', 'whattime',
    ]
    # 发晚模型输入特征字段 ， 多特征  10 （#todo:增删）
    X_DPT_COLUMNS = [
        'ARR_DELAY', 'stn_arr_delay_ave', 'stn_arr_early_pct', 'capacity', 'stn_xs_max',
        'stn_tdtl_time', 'stn_train_arr_delay_pct',
        'stn_arr_delay_pct', 'stn_arr_early_ave', 'train_arr_delay_ave',
    ]

    # 需要独热编码的特征  （#todo:增删）
    ONE_HOT_COLUMNS = ['stn_train_state', 'whattime']
    ONE_HOT_ARR_COLUMNS = ['stn_train_state', 'whattime' ]
    ONE_HOT_DPT_COLUMNS = []

    # 第八版目标变量  （#todo:增删）
    Y_COLUMNS = ['ARR_DELAY', 'DPT_DELAY']
    # 因变量名称  （#todo:增删）
    TARGET_NAME_Y = [i + '_Y' for i in Y_COLUMNS]
    # 到晚模型预测目标字段
    Y_ARR_COLUMNS = [TARGET_NAME_Y[0]]
    # Y_ARR_COLUMNS = ['ARR_DELAY_Y']
    # 发晚模型预测目标字段
    Y_DPT_COLUMNS = [TARGET_NAME_Y[1]]
    # Y_DPT_COLUMNS = ['DPT_DELAY_Y']

    # 修改部分数据列明  （#todo:优化）
    CHANGE_DF_DIC = {
        'FORMER_DPT_DELAY': 'former_dpt_delay', 'STN_TDTL_TIME': 'stn_tdtl_time',
        'REG_TDTL_TIME': 'reg_tdtl_time', 'Capacity': 'capacity', 'WhatTime': 'whattime',
        'DISTANCE': 'distance', 'PLAN_SPEED': 'plan_speed', 'STN_TRAIN_STATE': 'stn_train_state',
    }

    # 路径相关
    DATA_PATH = './data'
    TRAIN_DATA_PATH = './data/train_data'
    ROUTE_PRE_DATA_PATH = './data/route_pre_data'

    # FEATURE_DICT = './feature_dict'
    # SAVE_HISTORY_FEATURE_PATH = './feature_dict/history_feature'  # 保存目录
    # SAVE_POST_PROCESS_PATH = './feature_dict/post_process'  # 保存目录
    SAVE_HISTORY_FEATURE_PATH = '../feature_dict/history_feature'  # 保存目录
    SAVE_POST_PROCESS_PATH = '../untitled/post_process'  # 保存目录

    HISTORY_FEATURE_DATA_PATH = './data/CDG-2019-init-jh-pureall_seq.csv'  # 数据文件

    # GDC_MODEL_PATH = './model/gdc_model'
    # KTZ_MODEL_PATH = './model/ktz_model'
    GDC_MODEL_PATH = './model/gdc_model'
    KTZ_MODEL_PATH = '../model/ktz_model'

    TRAIN_TYPE_LIST = ['GDC', 'KTZ']
    # 模型与数据的不同时间段名称
    GRADE_NAME_LIST = ['0-5', '5-10', '10-30', '30+']
    # 处理后的数据名称
    MARK_GDC = 'CDG-2020-init-{}_DATA_DEAL.csv'
    MARK_KTZ = 'KTZ-2020-init-{}_DATA_DEAL.csv'
    ROUTE_GDC = 'CDG-{}_{}_DATA_DEAL.csv'
    ROUTE_KTZ = 'KTZ-{}_{}_DATA_DEAL.csv'

    # 用于统计历史特征用于接口预测 #TODO plan_speed应根据车次衍生，添加qujian_train_table
    STN_COL = ['STN_NAME']
    TRAIN_COL = ['ARR_NAME']
    FORMER_STN_COL = 'FORMER_STN'
    QUJIAN_COL = STN_COL + [FORMER_STN_COL]
    STN_BASE_COLS = ['capacity', 'stn_arr_delay_pct', 'stn_arr_early_pct', 'stn_arr_delay_ave', 'stn_arr_early_ave', 'stn_xs_max']
    TRAIN_BASE_COLS = ['train_arr_delay_ave']
    QUJIAN_BASE_COLS = ['distance', 'plan_speed', 'qujian_xs_ave', 'qujian_xs_pct', 'qujian_ks_ave', 'qujian_ks_pct', 'qujian_xs_max']
    STN_TRAIN_BASE_COLS = ['stn_train_arr_delay_pct']
    STN_QUJIAN_BASE_COLS = []

    # 用于后处理逻辑
    AFTERTREATMENT_DEAL_COLUMNS = [
        'STATIS_ID', 'STN_NAME', 'ARR_NAME', 'ARR_DELAY', 'DPT_DELAY','stn_tdtl_time', 'STN_SJTL_TIME', 'STN_XS_TIME',
        'reg_tdtl_time', 'REG_SJTL_TIME', 'REG_XS_TIME'
    ]

    # 统计指标
    TIME_THRESHOUD_180 = 180
class MultVarCongfig1(object):
    # 第九版
    # 到晚模型输入特征字段，多特征  16  （# todo：增删）
    X_ARR_COLUMNS = [
        'former_dpt_delay1',
        'td_yx_time',
        'train_arr_delay_pct',
        'train_arr_early_pct',
        'train_arr_delay_ave',
        'train_arr_early_ave',
        'qujian_xs_ave',
        'train_qujian_xs_ave', 'qujian_ks_ave', 'train_qujian_ks_ave',
        'qujian_xs_pct', 'train_qujian_xs_pct', 'qujian_ks_pct',
        'train_qujian_ks_pct', 'capacity','qujian_xs_max',
        'train_qujian_xs_max',
        'distance', 'plan_speed'
    ]
    # 'Autumn', 'Spring', 'Summer', 'Winter', 'Holiday', 'Weekday', 'Weekend', 'Afternoon',
    #         'EarlyMorning', 'Morning', 'Night'
    # 发晚模型输入特征字段 ， 多特征  10 （#todo:增删）
    X_DPT_COLUMNS = [
        'arr_delay', 'stn_arr_delay_pct',
        'train_arr_delay_pct', 'stn_train_arr_delay_pct', 'stn_arr_early_pct',
        'train_arr_early_pct', 'stn_train_arr_early_pct', 'stn_arr_delay_ave',
        'train_arr_delay_ave', 'stn_train_arr_delay_ave', 'stn_arr_early_ave',
        'train_arr_early_ave', 'stn_train_arr_early_ave', 'stn_tdtl_time', 'capacity',
        'stn_xs_max', 'stn_train_xs_max','stn_train_ks_ave','stn_train_xs_pct',
        'stn_train_xs_ave','stn_train_ks_pct'
    ]
    # 'Weekday', 'Weekend', 'Afternoon',
    #         'EarlyMorning', 'Morning', 'Night','Autumn', 'Spring',  'Summer', 'Winter', 'Holiday'

    # 需要独热编码的特征  （#todo:增删）
    ONE_HOT_COLUMNS = ['qujian_yx_state', 'whattime','whatday','season']
    ONE_HOT_ARR_COLUMNS = ['qujian_yx_state', 'whattime','whatday','season']
    ONE_HOT_DPT_COLUMNS = ['whattime','whatday','season']

    # 第八版目标变量  （#todo:增删）
    Y_COLUMNS = ['arr_delay_1', 'dpt_delay_1']
    # 因变量名称  （#todo:增删）
    TARGET_NAME_Y = [i + '_Y' for i in Y_COLUMNS]
    # 到晚模型预测目标字段
    Y_ARR_COLUMNS = [Y_COLUMNS[0]]
    # Y_ARR_COLUMNS = ['ARR_DELAY_Y']
    # 发晚模型预测目标字段
    Y_DPT_COLUMNS = [Y_COLUMNS[1]]
    # Y_DPT_COLUMNS = ['DPT_DELAY_Y']

    # 修改部分数据列明  （#todo:优化）
    CHANGE_DF_DIC = {
        'FORMER_DPT_DELAY': 'former_dpt_delay', 'STN_TDTL_TIME': 'stn_tdtl_time',
        'REG_TDTL_TIME': 'reg_tdtl_time', 'Capacity': 'capacity', 'WhatTime': 'whattime',
        'DISTANCE': 'distance', 'PLAN_SPEED': 'plan_speed', 'STN_TRAIN_STATE': 'stn_train_state',
    }

    # 路径相关
    DATA_PATH = './data'
    TRAIN_DATA_PATH = './data/train_data'
    ROUTE_PRE_DATA_PATH = './data/route_pre_data'

    # FEATURE_DICT = './feature_dict'
    # SAVE_HISTORY_FEATURE_PATH = './feature_dict/history_feature'  # 保存目录
    # SAVE_POST_PROCESS_PATH = './feature_dict/post_process'  # 保存目录
    SAVE_HISTORY_FEATURE_PATH = '../feature_dict/history_feature'  # 保存目录
    SAVE_POST_PROCESS_PATH = '../feature_dict/post_process'  # 保存目录

    HISTORY_FEATURE_DATA_PATH = './data/CDG-2019-init-jh-pureall_seq.csv'  # 数据文件

    # GDC_MODEL_PATH = './model/gdc_model'
    # KTZ_MODEL_PATH = './model/ktz_model'
    GDC_MODEL_PATH = './model/gdc_model'
    KTZ_MODEL_PATH = '../model/ktz_model'

    TRAIN_TYPE_LIST = ['GDC', 'KTZ']
    # 模型与数据的不同时间段名称
    GRADE_NAME_LIST = ['0-5', '5-10', '10-30', '30+']
    # 处理后的数据名称
    MARK_GDC = 'CDG-2020-init-{}_DATA_DEAL.csv'
    MARK_KTZ = 'KTZ-2020-init-{}_DATA_DEAL.csv'
    ROUTE_GDC = 'CDG-{}_{}_DATA_DEAL.csv'
    ROUTE_KTZ = 'KTZ-{}_{}_DATA_DEAL.csv'

    # 用于统计历史特征用于接口预测 #TODO plan_speed应根据车次衍生，添加qujian_train_table
    STN_COL = ['stn_name']
    TRAIN_COL = ['arr_name']
    FORMER_STN_COL = 'former_stn'
    QUJIAN_COL = STN_COL + [FORMER_STN_COL]
    STN_BASE_COLS = ['capacity', 'stn_arr_delay_pct', 'stn_arr_early_pct', 'stn_arr_delay_ave', 'stn_arr_early_ave', 'stn_xs_max']

    TRAIN_BASE_COLS = ['train_arr_delay_pct','train_arr_early_pct','train_arr_delay_ave','train_arr_early_ave']


    QUJIAN_BASE_COLS = ['distance', 'plan_speed', 'qujian_xs_ave', 'qujian_xs_pct', 'qujian_ks_ave', 'qujian_ks_pct', 'qujian_xs_max']

    STN_TRAIN_BASE_COLS = ['stn_train_arr_delay_pct','stn_train_arr_early_pct','stn_train_arr_delay_ave','stn_train_arr_early_ave',
                           'stn_train_xs_ave','stn_train_ks_ave','stn_train_xs_pct','stn_train_ks_pct','stn_train_xs_max']

    TRAIN_QUJIAN_BASE_COLS = ['train_qujian_xs_ave','train_qujian_ks_ave','train_qujian_xs_pct','train_qujian_ks_pct','train_qujian_xs_max']

    # 用于后处理逻辑
    AFTERTREATMENT_DEAL_COLUMNS = [
        'statis_id', 'stn_name', 'arr_name', 'arr_delay', 'dpt_delay','stn_tdtl_time', 'stn_sjtl_time', 'stn_xs_time',
        'qujian_tdtl_time', 'qujian_sjtl_time', 'qujian_xs_time'
    ]

    # 统计指标
    TIME_THRESHOUD_180 = 180

cur_config = MultVarCongfig1()


