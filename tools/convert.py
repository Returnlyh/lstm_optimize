'''
Descripttion: 
version: 1.0.0
Author: Gager
Date: 2022-11-16 16:36:47
LastEditors: Gager
'''
from sklearn.preprocessing import MinMaxScaler
from numpy import array

import tensorflow as tf
import pandas as pd
import numpy as np

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
    

input_csv_path = "../data/train_use_lstm.csv"
df = pd.read_csv(input_csv_path) # 数据为某一传感器的单变量序列数据
dataset = df.values

# 归一化数据集
scaler = MinMaxScaler()
dataset = scaler.fit_transform(dataset)
dataset = dataset.squeeze() # 压缩数据维度


# 划分训练集和测试集（8:2，可调整)
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size], dataset[train_size:len(dataset)]
n_steps_in, n_steps_out = 32, 16 # 过去14天的数据来预测未来7天

trainX, trainY = split_sequence(train, n_steps_in, n_steps_out)
testX, testY = split_sequence(test, n_steps_in, n_steps_out)



def representative_data_gen():
    for input_value in testX[:100]:
        input_value = input_value.reshape(1, 1, -1)
        input_value = tf.convert_to_tensor(input_value, dtype=tf.float32)
        yield[input_value]


converter = tf.lite.TFLiteConverter.from_saved_model("../models/lstm")
tflite_model = converter.convert()
open("../models/lstm.tflite", 'wb').write(tflite_model)
# tf.lite.experimental.Analyzer.analyze(model_content=tflite_model)

converter = tf.lite.TFLiteConverter.from_saved_model("../models/lstm")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
open("../models/lstm_qua.tflite", 'wb').write(tflite_model)
# tf.lite.experimental.Analyzer.analyze(model_content=tflite_model)


# converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter = tf.lite.TFLiteConverter.from_saved_model("../models/lstm")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# converter.target_spec.supported_types = [tf.int8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model_quant = converter.convert()
open("../models/lstm_full_qua.tflite", "wb").write(tflite_model_quant)
# tf.lite.experimental.Analyzer.analyze(model_content=tflite_model_quant)

