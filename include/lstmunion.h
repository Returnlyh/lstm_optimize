/*
 * @Descripttion: 
 * @version: 1.0.0
 * @Author: Gager
 * @Date: 2022-11-18 09:05:19
 * @LastEditors: Gager
 */
#ifndef _LSTMUNION_H

#define _LSTMUNION_H

#include "lstmcell.h"


// LSTM 多单元的链表
struct lstms
{
    int input; // 输入特征长度
    int output; // 输出特征长度
    int lstm_num; // LSTM单元 数量
    struct lstmcell *first; // 第一个 LSTM 单元
    struct lstmcell *end; // 最后一个 LSTM 单元
};

struct lstms* lstm_create(int input, int output, int lstm_num); // 创建 LSTM 多单元对象
char lstm_run(struct lstms *lstms, float *inputs, float *outputs, int return_sequences);
char lstm_run_unit(struct lstms *lstms); // 运行 LSTMs
char lstm_release(struct lstms *lstms); // 释放LSTMs

#endif