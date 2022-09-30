# -*- coding: utf-8 -*-
''' 
    @date        : 2020/7/14 21:06
    @Author      : Zyy 
    @File Name   : route_prediction.py
    @Description : 全段模型预测，并统计准确率
'''

import os
import numpy as np
import pandas as pd
import joblib
import warnings
from config import cur_config as cfg
from train import acc, mae
from sklearn.metrics import r2_score

pd.set_option('display.max_columns',None)
warnings.filterwarnings('ignore')

def route_pre(qc,elimination_rearrangement,columns_arr,columns_dpt, arr_model, dpt_model, df):
    '''
    [进行滚动预测]
    :param data:  根据行程ID得到的数据
    :param arr_model_name:  要加载的到晚模型的文件名称
    :param dpt_model_name:   要加载的发晚模型的文件
    :return:  到晚模型发晚模型预测值的矩阵
    '''


    # Partition = elimination_rearrangement.sample(frac=1).reset_index(drop=True)
    # loan_inner = pd.merge(df, Partition['statis_id'], how='inner')
    # qc = loan_inner.drop_duplicates(subset=['statis_id'], keep='first').reset_index(drop=True)
    arr = []
    dpt = []
    count = 0
    for i in qc['statis_id']:
        # print(i)
        model = loan_inner[loan_inner['statis_id'] == i]
        count = count + len(model)
        for i in range(0, len(model)):
            loaded_model_1 = dpt_model
            loaded_model = arr_model
            if i == 0:
                if model['arr_delay'].iat[i] > 0:
                    if model['arrival0'].iat[i] == model['depart0'].iat[i]:
                        arr.append(model['arr_delay'].iat[i])
                        dpt.append(model['arr_delay'].iat[i])
                    else:
                        dpt_delay_model = model[columns_dpt]
                        prediction_dpt_delay = loaded_model_1.predict(dpt_delay_model.iloc[[i]])
                        prediction_dpt_delay = prediction_dpt_delay[0]
                        #                             print(prediction_dpt_delay)
                        #                         print(prediction_dpt_delay)
                        arr.append(model['arr_delay'].iat[i])
                        dpt.append(prediction_dpt_delay)


                else:
                    arr.append(model['arr_delay'].iat[i])
                    dpt.append(model['dpt_delay'].iat[i])
            else:
            #                     loaded_model = joblib.load("arr_delay.joblib.dat")
                if model['arrival0'].iat[i] == model['depart0'].iat[i]:
                    arr_delay_model = model[columns_arr]
                #                         print(arr_delay_model['former_dpt_delay1'].iloc[[i]])
                #                         print(i)
                #                         print(dpt[i-1])
                #                         arr_delay_model['former_dpt_delay1'].iloc[[i]]=dpt[i-1]
                    arr_delay_model.iat[i, 18] = dpt[i - 1]
                    prediction_arr_delay = loaded_model.predict(arr_delay_model.iloc[[i]])
                    prediction_arr_delay = prediction_arr_delay[0]
                    arr.append(prediction_arr_delay)
                    dpt.append(prediction_arr_delay)
                else:
                    arr_delay_model = model[columns_arr]
                #                         print(arr_delay_model['former_dpt_delay1'].iloc[i])
                #                         print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                #                         print( arr_delay_model.iat[i,18])
                    arr_delay_model.iat[i, 18] = dpt[i - 1]
                #                         print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                #                         print( arr_delay_model.iat[i,18])
                    prediction_arr_delay = loaded_model.predict(arr_delay_model.iloc[[i]])
                    prediction_arr_delay = prediction_arr_delay[0]
                    arr.append(prediction_arr_delay)

                    dpt_delay_model = model[columns_dpt]
                    dpt_delay_model.iat[i, 0] = prediction_arr_delay
                    prediction_dpt_delay = loaded_model_1.predict(dpt_delay_model.iloc[[i]])
                    prediction_dpt_delay = prediction_dpt_delay[0]
                    dpt.append(prediction_dpt_delay)
    return arr,dpt


if __name__ == '__main__':
    '''
    usage: python route_prediction.py
    '''
    # 根据序列去重，通过序列得到序列数据， 需更改路径
    # route_path = os.path.join(cfg.ROUTE_PRE_DATA_PATH, cfg.ROUTE_GDC.format(cfg.GRADE_NAME_LIST[1], 'TEST'))
    df_route = pd.read_csv('test1.csv')
    df_route.sort_values(['seq_id_arr', 'daoda_time'], ascending=[True, True], inplace=True)

    df_route.reset_index(inplace=True,drop=True)
    route_seqs =df_route.drop_duplicates(subset=['statis_id'],keep='first')
    print('提取测试数据所有seq数', len(route_seqs))
    print('提取的测试数据', df_route.shape)

    Partition = route_seqs.sample(frac=1).reset_index(drop=True)
    loan_inner = pd.merge(df_route, Partition['statis_id'], how='inner')
    qc = loan_inner.drop_duplicates(subset=['statis_id'], keep='first').reset_index(drop=True)
    # 模型载入 更换模型需要修改路径
    model_arr = joblib.load(os.path.join(cfg.GDC_MODEL_PATH, 'arr_delay.joblib.dat'))
    model_dpt = joblib.load(os.path.join(cfg.GDC_MODEL_PATH, 'dpt_delay.joblib.dat'))

    #

    # 加载独热编码表（独热编码后的特征名称）
    # one_hot_arr_columns = joblib.load('./one_hot_arr_columns.pkl')
    # one_hot_dpt_columns = joblib.load('./one_hot_dpt_columns.pkl')

    print('滚动预测中......')
    # 循环调用滚动预测方法，每次传入不同的行程数据，最后将返回的预测值一次拼接
    # pred_result = np.concatenate(
    #     [route_pre(x, cfg.X_ARR_COLUMNS, cfg.X_DPT_COLUMNS, model_arr, model_dpt, df_route) for x in route_seqs]
    # )
    arr,dpt=route_pre(qc,route_seqs,cfg.X_ARR_COLUMNS, cfg.X_DPT_COLUMNS, model_arr, model_dpt, df_route)

    loan_inner['y_arr_pred'] = arr
    loan_inner['y_dpt_pred'] = dpt
    assert (loan_inner['y_arr_pred'].shape[0] == loan_inner['arr_delay'].shape[0])
    print('到晚模型全路高铁大于30分钟一分钟内准确率：',
          sum((np.abs(loan_inner['y_arr_pred'] - loan_inner['arr_delay']) / 60) <= 1) / loan_inner['arr_delay'].shape[
              0])
    print('到晚模型全路高铁大于30分钟二分钟内准确率：',
          sum((np.abs(loan_inner['y_arr_pred'] - loan_inner['arr_delay']) / 60) <= 2) / loan_inner['arr_delay'].shape[
              0])
    print('到晚模型全路高铁大于30分钟三分钟内准确率：',
          sum((np.abs(loan_inner['y_arr_pred'] - loan_inner['arr_delay']) / 60) <= 3) / loan_inner['arr_delay'].shape[
              0])
    print('到晚模型全路高铁大于30分钟五分钟内准确率：',
          sum((np.abs(loan_inner['y_arr_pred'] - loan_inner['arr_delay']) / 60) <= 5) / loan_inner['arr_delay'].shape[
              0])
    print('到晚模型全路高铁大于30分钟十分钟内准确率：',
          sum((np.abs(loan_inner['y_arr_pred'] - loan_inner['arr_delay']) / 60) <= 10) / loan_inner['arr_delay'].shape[
              0])
    print('到晚模型全路高铁大于30分钟30分钟内准确率：',
          sum((np.abs(loan_inner['y_arr_pred'] - loan_inner['arr_delay']) / 60) <= 30) / loan_inner['arr_delay'].shape[
              0])
    assert (loan_inner['y_dpt_pred'].shape[0] == loan_inner['dpt_delay'].shape[0])
    print('发晚模型全路高铁大于30分钟一分钟内准确率：',
          sum((np.abs(loan_inner['y_dpt_pred'] - loan_inner['dpt_delay']) / 60) <= 1) / loan_inner['dpt_delay'].shape[
              0])
    print('发晚模型全路高铁大于30分钟三分钟内准确率：',
          sum((np.abs(loan_inner['y_dpt_pred'] - loan_inner['dpt_delay']) / 60) <= 3) / loan_inner['dpt_delay'].shape[
              0])
    print('发晚模型全路高铁大于30分钟五分钟内准确率：',
          sum((np.abs(loan_inner['y_dpt_pred'] - loan_inner['dpt_delay']) / 60) <= 5) / loan_inner['dpt_delay'].shape[
              0])
    print('发晚模型全路高铁大于30分钟十分钟内准确率：',
          sum((np.abs(loan_inner['y_dpt_pred'] - loan_inner['dpt_delay']) / 60) <= 10) / loan_inner['dpt_delay'].shape[
              0])
    print('发晚模型全路高铁大于30分钟30分钟内准确率：',
          sum((np.abs(loan_inner['y_dpt_pred'] - loan_inner['dpt_delay']) / 60) <= 30) / loan_inner['dpt_delay'].shape[
              0])

    print('=====================================')
    print('到晚mae值为:', np.mean(np.abs(loan_inner['y_arr_pred'] - loan_inner['arr_delay'])))
    print('发晚mae值为:', np.mean(np.abs(loan_inner['y_dpt_pred'] - loan_inner['dpt_delay'])))


    #
    # print('有预测值的条数：', pred_result.shape)
    #
    # pred_indexs = pred_result[:, 0]
    # pred_values = pred_result[:, 1:3]
    # print('pred_indexs.shape', pred_indexs.shape)
    # print('pred_values.shape', pred_values.shape)
    #
    # mark = np.isin(df_route.index.tolist(), pred_indexs)
    # pre_df = df_route[mark]
    #
    # # 评价指标：统计MAE、R2和三分钟内的准确率（包含三分钟）
    # print("=======================================")
    # print("MAE:\n", mae(pred_values, pre_df[cfg.TARGET_NAME_Y]))
    # print('R2:\n', r2_score(pred_values, pre_df[cfg.TARGET_NAME_Y]))
    # print("" + str(cfg.TIME_THRESHOUD_180 / 60) + "分钟准确率:\n",acc(pred_values, pre_df[cfg.TARGET_NAME_Y], cfg.TIME_THRESHOUD_180))
    # print("=======================================")



