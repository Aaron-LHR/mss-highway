from flask import Flask, request, jsonify
import json
import os
import sys
import time
import joblib
from config import cur_config as cfg
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from config import cur_config as cfg
# from restful.rest_pre import data_parse, route_pre_api, get_json_api
from rest_pre import data_parse, route_pre_api, get_json_api


# 加载模型
def init_model():
    '''
    :param train_type: 列车型号
    :type train_type:  str
    :param delay_interval_time: 晚点时间范围
    :type delay_interval_time:  str
    :return:  arr_model, dpt_model
    :rtype: model
    '''

    # train_type_list = cfg.TRAIN_TYPE_LIST
    # time_range_list = cfg.GRADE_NAME_LIST
    model_map = {}

    # for train_type in train_type_list:
    #     for time_range in time_range_list:
            # 获取模型路径
            # arr_model_name = 'LGBM_ARR_{}_{}.pkl'.format(train_type, time_range)
            # dpt_model_name = 'LGBM_DPT_{}_{}.pkl'.format(train_type, time_range)
    arr_model_name ='arr_delay.joblib.dat'
    dpt_model_name ='dpt_delay.joblib.dat'

    # if train_type == 'GDC':
    model_save_path = cfg.GDC_MODEL_PATH
            # else:
            #     model_save_path = cfg.KTZ_MODEL_PATH

            # 初始化模型
    model_map.update(
        {
             arr_model_name:(joblib.load(os.path.join(model_save_path, arr_model_name))),
             dpt_model_name:(joblib.load(os.path.join(model_save_path, dpt_model_name))),
        }
            )

    return model_map

@app.route('/rest_v3/data_preprocess', methods=['POST'])
def data_preprocess():
    '''
    flask api server
    '''

    # 计时
    all_start_time = time.time()
    #获取数据
    preprocess_data = request.json
    # preprocess_data =
    print()
    #数据预处理
    datas_dict_of_list, status = data_parse(preprocess_data)
    data_preprocessed = {'datas_preprocessed':datas_dict_of_list, 'status':status}

    print("Data Preprocess Tiem used: {}".format(time.time() - all_start_time))

    return jsonify(data_preprocessed)

@app.route('/rest_v3/train_delay', methods=['POST'])
def train_delay():

    # 计时
    all_start_time = time.time()
    start_time = time.time()
    #获取数据
    json_data = request.json
    orgin_stn = json_data["datas_preprocessed"][0]
    arr_name = orgin_stn["arr_name"]
    # print(json_data["datas"])
    # 列车类型
    train_type = "KTZ" if arr_name[0] in "KTZ" else "GDC"
    # 晚点时间区间
    # delay_time = int(orgin_stn['DPT_DELAY'])
    # if delay_time > 0 and delay_time <= 5:
    #     delay_interval = cfg.GRADE_NAME_LIST[0]
    # elif delay_time > 5 and delay_time <= 10:
    #     delay_interval = cfg.GRADE_NAME_LIST[1]
    # elif delay_time > 10 and delay_time <= 30:
    #     delay_interval = cfg.GRADE_NAME_LIST[2]
    # else:
    #     delay_interval = cfg.GRADE_NAME_LIST[3]

    # arr_model_name = 'LGBM_ARR_{}_{}.pkl'.format(train_type, delay_interval)
    # dpt_model_name = 'LGBM_DPT_{}_{}.pkl'.format(train_type, delay_interval)
    # arr_model_name = 'LGBM_ARR_{}.pkl'.format(train_type)
    # dpt_model_name = 'LGBM_DPT_{}.pkl'.format(train_type)
    arr_model_name = 'arr_delay.joblib.dat'
    dpt_model_name = 'dpt_delay.joblib.dat'
    #加载模型
    start_time = time.time()
    model_map = init_model()
    print("Init_model Used time:{}".format(time.time() - start_time))

    arr_model = model_map[arr_model_name]
    dpt_model = model_map[dpt_model_name]
    print("choose model used time:{}".format(time.time() - start_time))

    pred, data_deal_time, pred_time = route_pre_api(json_data['datas_preprocessed'], arr_model, dpt_model)
    print('predicting...data_deal_time',data_deal_time)
    print('predicting...pred_time',pred_time)

    # 返回json数据
    start_time = time.time()
    result_json = get_json_api(pred)
    print("get json used time:{}".format(time.time() - start_time))

    print("Model Predict ALL Tiem used: {}".format(time.time() - all_start_time))

    return jsonify(result_json)


if __name__ == '__main__':
    # , port = 3535,  threaded = False
    app.run(host = '0.0.0.0', port = 5050,  threaded = False)
    # data_preprocess()
    # app.run(host='0.0.0.0', port=int(sys.argv[1]), debug=True,threaded=False)
