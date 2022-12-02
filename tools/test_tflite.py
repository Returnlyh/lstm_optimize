from sklearn.preprocessing import MinMaxScaler
from numpy import array

import tensorflow as tf
import pandas as pd
import numpy as np

np.random.seed(1)


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
df = pd.read_csv(input_csv_path) # 数据为某一传感器的单变量序列数据
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


# load keras model
model = tf.keras.models.load_model("../models/lstm")

# Run the model with TensorFlow Lite
interpreter = tf.lite.Interpreter(model_path='../models/lstm_qua.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details)
print(output_details)

error_num = 0

for i in range(len(testX)):

    try:
        x = testX[i, :].reshape(1, 1, -1).astype(np.float32)
        expected = model.predict(x)
        interpreter.set_tensor(input_details[0]["index"], x)
        interpreter.invoke()
        result = interpreter.get_tensor(output_details[0]["index"])

        print(expected, '\n', result)

        # Assert if the result of TFLite model is consistent with the TF model.
        np.testing.assert_almost_equal(expected, result, decimal=2)
        print("Done. The result of TensorFlow matches the result of TensorFlow Lite.")

        # Please note: TfLite fused Lstm kernel is stateful, so we need to reset
        # the states.
        # Clean up internal states.
        interpreter.reset_all_variables()

    except Exception as e:
        print(e)
        interpreter.reset_all_variables()
        error_num += 1

print(f"[INFO]>>> total:{len(testX)}, error:{error_num}")