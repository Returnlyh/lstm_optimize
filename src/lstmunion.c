/*
 * @Descripttion: 
 * @version: 1.0.0
 * @Author: Gager
 * @Date: 2022-11-18 09:04:53
 * @LastEditors: Gager
 */
#include "lstmunion.h"

struct lstms* lstm_create(int input, int output, int lstm_num) // TODO az13js 创建 LSTM 多单元对象
{
    int i;
    struct lstms *lstms;
    struct lstmcell *cell;
    struct lstmcell *cell_last = NULL;
    if (input < 1 || output < 1 || lstm_num < 1) {
        return NULL;
    }
    lstms = (struct lstms*)malloc(sizeof (struct lstms));
    if (NULL == lstms) {
        return lstms;
    }
    (*lstms).input = input;
    (*lstms).output = output;
    (*lstms).lstm_num = lstm_num;
    for (i = 0; i < lstm_num; i++) {
        cell = lstmcell_create(input, output);
        if (NULL == cell) {
            return NULL;
        }
        (*cell).before = cell_last;
        (*cell).after = NULL;
        if (NULL != cell_last) {
            (*cell_last).after = cell;
            (*cell).before = cell_last;
        }
        cell_last = cell;
        if (0 == i) {
            (*lstms).first = cell;
        }
        if (lstm_num - 1 == i) {
            (*lstms).end = cell;
        }
    }
    return lstms;
}

char lstm_run(struct lstms *lstms, float *inputs, float *outputs, int return_sequences){

    int cell_num = 0;
    float *input, *output;
    struct lstmcell *cell;

    if (NULL == lstms) {
        return 0;
    }
    if (NULL == inputs) {
        return 0;
    }
    if (NULL == outputs) {
        return 0;
    }

    cell = (*lstms).first;
    
    while (cell) {
        input = inputs + lstms->input * cell_num;
        if(return_sequences){
            output = outputs + lstms->output * cell_num;
        }
        else{
            output = outputs;
        }
        (*cell).x = input;
        (*cell).h = output;
        lstmcell_run_unit(cell);
        
        cell = (*cell).after;
        cell_num++;
    }
    return 1;
    return 1;
}

char lstm_run_unit(struct lstms *lstms)
{
    struct lstmcell *cell;
    if (NULL == lstms) {
        return 0;
    }
    cell = (*lstms).first;
    while (cell) {
        lstmcell_run_unit(cell);
        cell = (*cell).after;
    }
    return 1;
}

char lstm_release(struct lstms *lstms){

    struct lstmcell *cell, *temp;
    if (NULL == lstms) {
        return 0;
    }

    cell = (*lstms).end;
    while (cell) {
        temp = (*cell).before;
        lstmcell_release(cell);
        cell = temp;
    }
    return 1;
}