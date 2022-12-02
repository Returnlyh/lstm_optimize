# 深度：单变量多步预测
from numpy import array
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from pandas import read_csv

import tensorflow as tf

# 将序列数据处理成神经网络样本数据
'''
示例：
1、原样本数据：
sequence = [10, 20, 30, 40, 50, 60, 70, 80, 90]
n_steps_in = 3, n_steps_out = 2
2、生成样本：
[10 20 30] [40 50]
[20 30 40] [50 60]
[30 40 50] [60 70]
[40 50 60] [70 80]
[50 60 70] [80 90]
'''
def split_sequence(sequence, n_steps_in, n_steps_out):
    '''
    :param sequence: 原始序列数据
    :param n_steps_in: 利用过去n_steps_in个时间步的数据，
    :param n_steps_out: 来预测未来n_steps_out个时间步的值
    :return: 将喂入神经网络用于训练模型的数据
    '''
    X, y = list(), list()
    for i in range(len(sequence)):
        # 末端数据下标
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # 检查是否越界
        if out_end_ix > len(sequence):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

np.random.seed(7) # 随机种子

input_csv_path = "../data/train_use_lstm.csv"

# 读取数据
df = read_csv(input_csv_path) # 数据为某一传感器的单变量序列数据
dataset = df.values

# 归一化数据集
scaler = MinMaxScaler()
dataset = scaler.fit_transform(dataset)
print(scaler.min_, scaler.scale_)

# plt.plot(dataset)
# plt.show()
dataset = dataset.squeeze() # 压缩数据维度
# print("dataset:", dataset.shape)

# 划分训练集和测试集（8:2，可调整)
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size], dataset[train_size:len(dataset)]

# 也就是用前n_steps_in步，预测之后的n_steps_out步（一小时一个时间步），为可调整的参数
# n_steps_in, n_steps_out = 72, 24 # 过去3天的数据预测未来1天
n_steps_in, n_steps_out = 32, 16 # 过去14天的数据来预测未来7天

trainX, trainY = split_sequence(train, n_steps_in, n_steps_out)
testX, testY = split_sequence(test, n_steps_in, n_steps_out)
print("trainX:", trainX.shape)
print("trainY:", trainY.shape)
print("testX:", testX.shape)
print("testY:", testY.shape)

print(testX[0, :].tolist())
print(testY[0, :].tolist())

# out_data1 = np.array([0.220218, 0.217337, 0.217760, 0.219832, 0.218967, 0.214273, 0.220631, 0.213548, 0.214587, 0.217494, 0.216369, 0.219658, 0.217301, 0.218477, 0.217961, 0.222868])
# out_data_1 = scaler.inverse_transform(out_data1.reshape(1, -1))
# print(out_data_1)
'''
640.33555998 640.18543107 640.2074736  640.31544552 640.27037037
640.02576603 640.35708141 639.98798628 640.04212857 640.19361234
640.13498859 640.30637838 640.18355511 640.24483647 640.21794771
640.47365148  
'''

# out_data2 = np.array([0.2330654263496399, 0.2311858832836151, 0.23263661563396454, 0.23210613429546356, 0.23301264643669128, 0.231489360332489, 0.23352599143981934, 0.23187758028507233, 0.231843501329422, 0.23603099584579468, 0.23412379622459412, 0.23772099614143372, 0.23690207302570343, 0.23821064829826355, 0.2367182970046997, 0.23820297420024872])
# out_data_2 = scaler.inverse_transform(out_data2.reshape(1, -1))
# print(out_data_2)
'''
641.00503937 640.90709638 640.98269404 640.95505066 641.00228901
640.92291057 641.02903941 640.94314071 640.94136485 641.15957519
641.06019102 641.24764111 641.20496703 641.27315688 641.19539046
641.27275699
'''
# reshape from [samples, timesteps] into [samples, timesteps, features]
# n_features = 1（单变量）
# X = X.reshape((X.shape[0], n_features, X.shape[1]))
# 投入到 LSTM 的 X 需要有这样的结构： [samples, time steps, features]，所以做一下变换
# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# 定义网络
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(1, n_steps_in), name='input'),
    tf.keras.layers.LSTM(n_steps_in, activation='tanh',recurrent_activation='sigmoid', return_sequences=True, name="lstm"),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(n_steps_out, name='output')
])
model.compile(loss='mae', optimizer='adam', metrics=['mae'])
#开始训练
history = model.fit(trainX,
                    trainY,
                    epochs=50,
                    batch_size=128,
                    validation_data=(testX, testY),
                    verbose=2) # 为每个epoch输出一行记录

# 保存训练得到的模型
run_model = tf.function(lambda x: model(x))
concrete_func = run_model.get_concrete_function(
    tf.TensorSpec([1, 1, n_steps_in], model.inputs[0].dtype))
model.save("../models/lstm", save_format="tf", signatures=concrete_func)

# 绘制结果
# 分别绘制训练过程中模型在训练数据和验证数据上的损失和精度
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1) # 训练纪元（横坐标）

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

# 利用训练得到的网络模型检验训练集和测试集的拟合效果
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
print(trainPredict.shape)
# 将得到的预测值反归一化
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform(trainY)
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform(testY)

# 计算 mean squared error（值越小越好，表示预测值和真实值距离越近）
train_score = np.sqrt(mean_squared_error(trainY, trainPredict))
print("train score RMSE: %.2f"% train_score)
train_score = np.sqrt(mean_squared_error(testY, testPredict))
print("test score RMSE: %.2f"% train_score)

plt.title('The train values')
plt.xlabel('Future 24 timesteps(24h)')
plt.ylabel('value range')
plt.plot(trainY[-1, :], 'r', label='true')
plt.plot(trainPredict[-1, :], 'y', label='predictions')
plt.legend()
plt.show()

plt.title('The test values')
plt.xlabel('Future 24 timesteps(24h)')
plt.ylabel('value range')
plt.plot(testY[0, :], 'r', label='true')
plt.plot(testPredict[0, :], 'y', label='predictions')
plt.legend()
plt.show()
