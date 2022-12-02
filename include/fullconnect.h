/*
 * @Descripttion: 
 * @version: 1.0.0
 * @Author: Gager
 * @Date: 2022-11-17 10:28:22
 * @LastEditors: Gager
 */
#ifndef _FULLCONNECT_H
#define _FULLCONNECT_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

char full_connecte(float *input, float *output, float *fc_w, float *fc_b, int in, int out);

// int main(){

//     float in = 2;
//     clock_t start, finish;
//     start = clock();
//     // float out = _exp(in);
//     // float out = exp(in);
//     float out = _tanh(in);
//     finish = clock();
//     float total_time = (float)(finish - start) / CLOCKS_PER_SEC; //单位换算成秒
//     printf("[INFO]>>> out:%f, %f/s \n", out, total_time);

// }

#ifdef __cplusplus
}
#endif

#endif