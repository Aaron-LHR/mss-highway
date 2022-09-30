import os
from config import cur_config as cfg
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score
pd.set_option('display.max_columns',None)
def train_arr(df, after_columns, y_columns, model, save_name):
    '''
    [传入数据和模型进行训练并持久化到model下]
    :param df: 训练数据
    :param model:  模型
    :param save_name:  保存模型的路径
    :return: 返回训练集、测试集的预测值和真实值
    '''


    # x_data = df[after_columns]
    xl=pd.DataFrame(df)
    xl.drop_duplicates(['seq_id_arr'],keep='first',inplace=True)
    print(xl.shape)
    y_data = pd.concat([df['seq_id_arr'],df['arr_delay_1'],df['dpt_delay_1']],axis=1)
    y=pd.DataFrame(y_data)
    y.drop_duplicates(['seq_id_arr'],keep='first',inplace=True)
    # 随机切分训练集、测试集
    x_train, x_test, y_train, y_test = train_test_split(xl, y, test_size=0.3, random_state=777)

    print([i.shape for i in[x_train, x_test, y_train, y_test]])
    x_train1=df[df['seq_id_arr'].isin(x_train['seq_id_arr'])]
    x_train1=x_train1[after_columns]
    x_test1 = df[df['seq_id_arr'].isin(x_test['seq_id_arr'])]
    x_test1 = x_test1[after_columns]
    test=df[df['seq_id_arr'].isin(x_test['seq_id_arr'])]
    test.to_csv('test1.csv',index=None)
    y_train1=y_data[y_data['seq_id_arr'].isin(y_train['seq_id_arr'])]['arr_delay_1']
    y_test1=y_data[y_data['seq_id_arr'].isin(y_test['seq_id_arr'])]['arr_delay_1']

    # 训练传入的模型
    lightgbm_model = model.fit(x_train1, y_train1)

    # 保存模型到model文件夹下
    print("[INFO] 模型保存中...")
    joblib.dump(lightgbm_model, os.path.join(cfg.GDC_MODEL_PATH, save_name))
    print("Finished!")
    # 分别将训练集和测试集放入模型预测
    x_train_pred = lightgbm_model.predict(x_train1)
    x_test_pred = lightgbm_model.predict(x_test1)
    print(mean_absolute_error(y_train1,x_train_pred))
    print(mean_absolute_error(y_test1,x_test_pred))
    # print(x_test_pred.shape)
    # print(x_test_pred.shape)
    print('============训练集到达晚点准确率==================')
    assert (len(x_train_pred) == y_train1.shape[0])
    print(sum((np.abs(x_train_pred - y_train1) / 60) <= 1) / y_train1.shape[0])
    print(sum((np.abs(x_train_pred - y_train1) / 60) <= 3) / y_train1.shape[0])
    print(sum((np.abs(x_train_pred - y_train1) / 60) <= 5) / y_train1.shape[0])
    print(sum((np.abs(x_train_pred - y_train1) / 60) <= 10) / y_train1.shape[0])
    print(sum((np.abs(x_train_pred - y_train1) / 60) <= 30) / y_train1.shape[0])
    print('============测试集到达晚点准确率==================')
    assert (len(x_test_pred) == y_test1.shape[0])
    print(sum((np.abs(x_test_pred - y_test1) / 60) <= 1) / y_test1.shape[0])
    print(sum((np.abs(x_test_pred - y_test1) / 60) <= 3) / y_test1.shape[0])
    print(sum((np.abs(x_test_pred - y_test1) / 60) <= 5) / y_test1.shape[0])
    print(sum((np.abs(x_test_pred - y_test1) / 60) <= 10) / y_test1.shape[0])
    print(sum((np.abs(x_test_pred - y_test1) / 60) <= 30) / y_test1.shape[0])
    # print(y_train.label())
    return x_train_pred, x_test_pred, y_train, y_test, lightgbm_model
def train_dpt(df, after_columns, y_columns, model, save_name):
    '''
    [传入数据和模型进行训练并持久化到model下]
    :param df: 训练数据
    :param model:  模型
    :param save_name:  保存模型的路径
    :return: 返回训练集、测试集的预测值和真实值
    '''


    # x_data = df[after_columns]
    xl=pd.DataFrame(df)
    xl.drop_duplicates(['seq_id_arr'],keep='first',inplace=True)
    print(xl.shape)
    y_data = pd.concat([df['seq_id_arr'],df['arr_delay_1'],df['dpt_delay_1']],axis=1)
    y=pd.DataFrame(y_data)
    y.drop_duplicates(['seq_id_arr'],keep='first',inplace=True)
    # 随机切分训练集、测试集
    x_train, x_test, y_train, y_test = train_test_split(xl, y, test_size=0.3, random_state=777)

    print([i.shape for i in[x_train, x_test, y_train, y_test]])
    x_train1=df[df['seq_id_arr'].isin(x_train['seq_id_arr'])]
    x_train1=x_train1[after_columns]
    x_test1 = df[df['seq_id_arr'].isin(x_test['seq_id_arr'])]
    x_test1 = x_test1[after_columns]
    test=df[df['seq_id_arr'].isin(x_test['seq_id_arr'])]
    test.to_csv('test1.csv')
    y_train1=y_data[y_data['seq_id_arr'].isin(y_train['seq_id_arr'])]['dpt_delay_1']
    y_test1=y_data[y_data['seq_id_arr'].isin(y_test['seq_id_arr'])]['dpt_delay_1']

    # 训练传入的模型
    lightgbm_model = model.fit(x_train1, y_train1)

    # 保存模型到model文件夹下
    print("[INFO] 模型保存中...")
    joblib.dump(lightgbm_model, os.path.join(cfg.GDC_MODEL_PATH, save_name))
    print("Finished!")
    # 分别将训练集和测试集放入模型预测
    x_train_pred = lightgbm_model.predict(x_train1)
    x_test_pred = lightgbm_model.predict(x_test1)
    print(mean_absolute_error(y_train1, x_train_pred))
    print(mean_absolute_error(y_test1,x_test_pred))
    # print(x_test_pred.shape)
    # print(x_test_pred.shape)
    print('============训练集出发晚点准确率==================')
    assert (len(x_train_pred) == y_train1.shape[0])
    print(sum((np.abs(x_train_pred - y_train1) / 60) <= 1) / y_train1.shape[0])
    print(sum((np.abs(x_train_pred - y_train1) / 60) <= 3) / y_train1.shape[0])
    print(sum((np.abs(x_train_pred - y_train1) / 60) <= 5) / y_train1.shape[0])
    print(sum((np.abs(x_train_pred - y_train1) / 60) <= 10) / y_train1.shape[0])
    print(sum((np.abs(x_train_pred - y_train1) / 60) <= 30) / y_train1.shape[0])
    print('============测试集出发晚点准确率==================')
    assert(len(x_test_pred)==y_test1.shape[0])
    print(sum((np.abs(x_test_pred-y_test1)/60)<=1)/ y_test1.shape[0])
    print(sum((np.abs(x_test_pred - y_test1) / 60) <= 3) / y_test1.shape[0])
    print(sum((np.abs(x_test_pred - y_test1) / 60) <= 5) / y_test1.shape[0])
    print(sum((np.abs(x_test_pred - y_test1) / 60) <= 10) / y_test1.shape[0])
    print(sum((np.abs(x_test_pred - y_test1) / 60) <= 30) / y_test1.shape[0])
    # print(y_train.label())
    return x_train_pred, x_test_pred, y_train, y_test, lightgbm_model
if __name__ == '__main__':
    df = pd.read_csv('2020_ql_dwandian.csv')
    df.columns=[i.lower() for i in df.columns]
    df1=pd.get_dummies(df['season'])
    df2 = pd.get_dummies(df['whatday'])
    df3 = pd.get_dummies(df['whattime'])
    df=df.join(df1)
    df = df.join(df2)
    df = df.join(df3)
    df['arr_delay_1']=0
    df['dpt_delay_1']=0
    for i in range(len(df)):
        if i == len(df) - 1:
            df['arr_delay_1'].iat[i] = df['arr_delay'].iat[i]
            df['dpt_delay_1'].iat[i] = df['dpt_delay'].iat[i]
        else:
            #         print(i)
            if df['statis_id'].iat[i] == df['statis_id'].iat[i + 1]:
                df['arr_delay_1'].iat[i] = df['arr_delay'].iat[i + 1]
                df['dpt_delay_1'].iat[i] = df['dpt_delay'].iat[i]
            else:
                df['arr_delay_1'].iat[i] = df['arr_delay'].iat[i]
                df['dpt_delay_1'].iat[i] = df['dpt_delay'].iat[i]
    df_arr_data = pd.DataFrame(df)
    df_dpt_data = pd.DataFrame(df)
    x_arr_train_pred, x_arr_test_pred, y_arr_train_true, y_arr_test_true, lightgbm_arr_model = train_arr(
        df_arr_data, cfg.X_ARR_COLUMNS, cfg.Y_ARR_COLUMNS,
        LGBMRegressor(), 'arr_delay.joblib.dat',
        # 'LGBM_ARR_GDC_'+cfg.GRADE_NAME_LIST[1]+'.pkl'
    )

    x_dpt_train_pred, x_dpt_test_pred, y_dpt_train_true, y_dpt_test_true, lightgbm_dpt_model = train_dpt(
        df_dpt_data, cfg.X_DPT_COLUMNS, cfg.Y_DPT_COLUMNS,
        LGBMRegressor(), 'dpt_delay.joblib.dat',
        # 'LGBM_DPT_GDC_' + cfg.GRADE_NAME_LIST[1] + '.pkl'
    )


