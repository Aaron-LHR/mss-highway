**1.数据预处理和特征提取**
usage: 运行./jupyter/202010_preprocess.ipynb   #需要在顶部配置相关变量和路径（在这里文件名为united）
result: 生成全特征，各个晚点时间段的序列csv文件 

**2、数据准备**

usage: statis_history_feature_table.py
result: 
① 需要传入全年历史数据统计特征，在 ./feature_dict目录下分别筛选生成：
qujian_base_feature.csv、stn_base_feature.csv、stn_qujian_base_feature.csv、stn_train_base_feature.csv、
train_base_feature.csv、train_qujian_base_feature.csv六个文件（用于接口预测时根据站点信息和到达车次信息查询对应的X特征）。

② 需要传入全年历史数据统计特征，在 ./feature_dict目录下分别筛选生成：

reg_xs_rule.csv、 stn_xs_rule.csv文件用于后处理，根据历史数据统计得到历史区间、站点最大吸收概率。

**3、训练**
usage: python train_test.py  例如: python train_test.py (训练时需要配置相关路径)
result: 在 ./model/gdc_model, 目录下生成训练好的高铁到晚模型和发晚模型，
并统计训练集和测试集的MAE与一分钟三分钟五分钟以及十分钟的Acc（准确率）。

**4、预测**
usage: python route_prediction.py  例如: python prediction.py
result: 控制台打印滚动预测结果（全段预测），并计算到达晚点以及出发晚点各自的mae与一分钟三分钟五分钟以及十分钟的Acc（准确率）。

**5、启用Restful API** (默认端口5050, 请求URL为http://ip:port/rest_v2/data_preprocess（该端口为将数据准备生成的六个文件根据站点信息和到达车次信息匹配对应的X特征）以及 http://ip:port/rest_v2/train_delay（该端口为滚动预测）)
python rest_server.py

