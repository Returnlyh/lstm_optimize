# LSTM模型硬件定制化部署方案

![](https://teamblog-1254331889.cos.ap-guangzhou.myqcloud.com/gager/20190409132450618.png)

## 0.最小单元简介

<img src="https://teamblog-1254331889.cos.ap-guangzhou.myqcloud.com/gager/1.png" alt="1" style="zoom:50%;" />

<img src="https://teamblog-1254331889.cos.ap-guangzhou.myqcloud.com/gager/image-20221124180013178.png" alt="image-20221124180013178" style="zoom:50%;" />

## 1.技术方案选型

    选择了使用比较多的几个框架，并通过对比分析几个关键指标来选择最合适的技术方案。

### 1.1 方案对比

| Name         | 数据               | 加速指令                                     | 移植难度           | 代码大小   | 内存大小 | 算子支持 | 开发语言 |
| ------------ | ------------------ | -------------------------------------------- | ------------------ | ---------- | -------- | -------- | -------- |
| TinyMaix     | INT8/FP16/FP32/FP8 | ARM SIMD/NEON/MVEI `<br>`RISC-V P/V extend | Easy               | 3~10KB     | 1.3~2X   | 少       | c        |
| NNoM         | INT8/INT16         | ARM SIMD                                     | Easy               | ~10KB      | 1.3~2X   | 中       | c        |
| tinyengine   | INT8/FP32          | ARM SIMD                                     | only for ARM       | ---        | 1X       | 少       | c        |
| MicroTVM     | INT8/FP32          | ARM SIMD                                     | ---                | ---        | ~3X      | 中       | c        |
| TFlite-micro | INT8/FP32          | ARM SIMD/Xtensa                              | Medium             | 20~100KB   | ~2.5X    | 多       | c++      |
| NCNN         | INT8/FP32          | ARM SIMD/NEON                                | Need code clipping | 200~1000KB | ---      | 多       | c++      |
| libonnx      | INT8/FP16/FP32/FP8 | ---                                          | Easy               | ~10KB      | 1.3~2X   | 少       | c        |

### 1.2 方案选择

深度学习模型的边缘端部署，目前工业界大多是走的Tensorflow➕TFlite-micro路线，即使用Tensorflow搭建训练模型，使用tflite-micro做边缘硬件端推理，tflite-micro分别使用ARM的CMSIM_NN库和Xtensa的esp_nn来支持SIMD加速，使用了c++11/17来进行开发，相较于c，c++的特性导致其程序体积要大于c且在移植时需要考虑是否有c++编译器支持，复杂的框架结构也导致当训练模型中有tflite-micro框架中没有实现的自定义算子时，整个模型将无法正常导出运行。需要深入了解tflite-micro框架结构，并手动在tflite-micro源码中使用c++实现自定义算子才能跑通整个运算流程。

NCNN也是业界较早开源的优秀推理框架，有着比tflite-micro更加丰富的算子支持，同样使用c++进行开发，有着tflite-micro同样的庞大结构的同时，NCNN定位的方向也不是边缘端，而是ARM移动端设备（如手机等），所以NCNN支持的加速架构也是定位更高的ARM-V7/8s系列

NNom的结构设计相对于NCNN和tflite-micro要简洁许多，使用c进行框架开发，基本的算子也都有实现也有ARM的CMSIS_NN库加速支持，自带权重量化工具，当网络中使用的算子比较简单时算是一个比较合适的选择方案。

tinyengine由MIT韩松团队研发，主要是用于解决大卷积导致的内存需求瓶颈，通过将大卷积进行切割分块运算以降低极限内存需求来实现小内存运行大模型的效果。其中降低极限内存消耗的设计值得借鉴。

TinyMaix和libonnx算是上述方案中程序体量能够做到最小的方案了，libonnx采用标准的onnx模型格式，任何框架训练得到的模型都可以转换为onnx，再由libonnx直接解析onnx模型，是一种兼容所有框架的设计方案，但是目前实现的onnx算子数量还比较少，且暂时没有实现额外的加速方案，而TingMaix则是只支持tflite模型，复用tflite的模型量化方案，通过解析tflite量化后的模型来获取内部权重参数，在加速优化方面，不同于NNom等框架通过调用现成的CMSIS_NN来加速，而是针对不同架构硬件直接手写汇编加速算子，所以有着更好的加速的同时，代码的体积也更小，但同样支持的算子比较少。

MicroTVM以及MegCC不同于以上所有的方案，采用深度学习编译器技术，将不同框架描述的深度学习模型转换为某个硬件平台生成优化的代码，不需要手动算子优化过程也能产生与之相当的优化代码，目前了解较少。

由于不确定自研硬件的性能，为了尽可能降低对硬件的性能需求，减少不必要的额外库依赖，最终选择通过简化模型结构➕手动算子实现的方案，借鉴上述框架中的加速优化方案，对于不同的硬件可以选择库函数加速或者手写汇编加速，这种定制化的设计优点很明显，简洁易维护，对于一个固定结构的简单网络可以说是最佳的方案。

## 3.方案执行流程

下述介绍一下只包含lstm➕dense两种算子的简单序列预测模型从模型搭建训练到优化部署的全流程。

### 3.0 数据需求

在一切准备工作之前，需根据实际任务分类选择好相应的模型以及数据，在时间序列预测任务中，通过对输入数据的**类型个数**分类可以分为**单序列预测**和**多序列预测**，这里的单和多就是表示的数据类型数。单序列预测表示输入数据只有一个类型，所以输出也只有一个类型数据，进而多序列预测就是表示有多个不同类型的输入数据，可以输出多个类型数据。

本次流程演示采用的**lstm模型为单序列预测模型**，所以本例是用来进行单序列预测任务，当让通过适当的调整可以使lstm模型具有多序列预测的能力，但具体的预测性能代测试。

### 3.1 模型定义+训练

定义了一个单层的lstm➕Dense(全连接层)组成的单序列预测网络，其输入步长为32，输出步长为16。

```python
n_steps_in, n_steps_out = 32, 16

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(1, n_steps_in), name='input'),
    tf.keras.layers.LSTM(n_steps_in, activation='tanh',recurrent_activation='sigmoid', return_sequences=True, name="lstm"),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(n_steps_out, name='output')
])
```

### 3.2 模型量化+反量化

模型量化一般指的是模型权重参数的量化，权重weights数据一般是float32类型的，量化即将他们转换为int8类型，量化的优点很明显，int8占用内存更少运算更快，量化后的模型可以更好地跑在低功耗嵌入式设备上。缺点自然也很明显，量化后的模型损失了精度，造成模型准确率下降。量化的本质是：**找到一个映射关系，使得float32与int8能够一一对应**。同理模型的反量化就是指**将模型的权重参数从量化后的int8转换回原始的float32**，反量化一般是当模型输入为浮点数据（如float32），而权重参数为定点数据（如int8）时，需要将权重反量化回浮点型进行计算，当输入与权重都为定点（如int8）数据时，则不需要反量化。具体的量化与反量化公式如下：

<img src="https://teamblog-1254331889.cos.ap-guangzhou.myqcloud.com/gager/image-20221125151914613.png" alt="image-20221125151914613" style="zoom:80%;" />

### 3.3 模型转换+量化

- tf转tflite

```python
converter = tf.lite.TFLiteConverter.from_saved_model("../models/lstm")
tflite_model = converter.convert()
open("../models/lstm.tflite", 'wb').write(tflite_model)
tf.lite.experimental.Analyzer.analyze(model_content=tflite_model)
```

![](https://teamblog-1254331889.cos.ap-guangzhou.myqcloud.com/gager/image-20221124172304811.png)

![](https://teamblog-1254331889.cos.ap-guangzhou.myqcloud.com/gager/image-20221124172403606.png)

- tf转tflite+默认量化

```python
converter = tf.lite.TFLiteConverter.from_saved_model("../models/lstm")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
open("../models/lstm_qua.tflite", 'wb').write(tflite_model)
tf.lite.experimental.Analyzer.analyze(model_content=tflite_model)
```

![](https://teamblog-1254331889.cos.ap-guangzhou.myqcloud.com/gager/image-20221124172005312.png)

![](https://teamblog-1254331889.cos.ap-guangzhou.myqcloud.com/gager/image-20221124172131625.png)

- tf转tflite+全量化(输入+权重参数)

```python
converter = tf.lite.TFLiteConverter.from_saved_model("../models/lstm")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model_quant = converter.convert()
open("../models/lstm_full_qua.tflite", "wb").write(tflite_model_quant)
tf.lite.experimental.Analyzer.analyze(model_content=tflite_model_quant)
```

 ![image-20221124171825229](https://teamblog-1254331889.cos.ap-guangzhou.myqcloud.com/gager/image-20221124171825229.png)

![](https://teamblog-1254331889.cos.ap-guangzhou.myqcloud.com/gager/image-20221124171715276.png)

- 量化后模型尺寸对比

![](https://teamblog-1254331889.cos.ap-guangzhou.myqcloud.com/gager/image-20221124170900781.png)

### 3.4 模型读取+参数保存

    通过读取tflite模型可以获取lstm内部参数，下图中通过netron展示的参数名称与实际lstm模型内部参数对应关系如下：

<img src="https://teamblog-1254331889.cos.ap-guangzhou.myqcloud.com/gager/image-20221125152312941.png" alt="image-20221125152312941" style="zoom:50%;" />




![](https://teamblog-1254331889.cos.ap-guangzhou.myqcloud.com/gager/image-20221124173101172.png)

通过脚本读取量化之后的.tflite模型文件，从中提取上述公式中相对应的权重数据以一维数据形式存储，并最终生成.h文件保存。

![](https://teamblog-1254331889.cos.ap-guangzhou.myqcloud.com/gager/image-20221125112839466.png)

### 3.5 模型权重读取

![](https://teamblog-1254331889.cos.ap-guangzhou.myqcloud.com/gager/image-20221125113335258.png)

### 3.6 模型运行

![](https://teamblog-1254331889.cos.ap-guangzhou.myqcloud.com/gager/image-20221125113522416.png)

## 4.优化

库函数加速以及汇编加速（to do）
