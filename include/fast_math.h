/*
 * @Descripttion: 
 * @version: 1.0.0
 * @Author: Gager
 * @Date: 2022-11-17 10:28:22
 * @LastEditors: Gager
 */
#ifndef _FASTMATH_H
#define _FASTMATH_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>


static inline float _exp(float x) {
    float p = 1.442695040f * x;
    uint32_t i = 0;
    uint32_t sign = (i >> 31);
    int w = (int) p;
    float z = p - (float) w + (float) sign;
    union {
        uint32_t i;
        float f;
    } v = {.i = (uint32_t) ((1 << 23) * (p + 121.2740838f + 27.7280233f / (4.84252568f - z) - 1.49012907f * z))};
    return v.f;
}

/* Schraudolph's published algorithm with John's constants */
/* 1065353216 - 486411 = 1064866805 */
static inline float expf_fast(float a) {
  union { float f; int x; } u;
  u.x = (int) (12102203 * a + 1064866805);
  return u.f;
}

//  1056478197 
static inline double better_expf_fast(float a) {
  union { float f; int x; } u, v;
  u.x = (long long)(6051102 * a + 1056478197);
  v.x = (long long)(1056478197 - 6051102 * a);
  return u.f / v.f;
}

/* 1065353216 - 722019 */
static inline float expf_fast_lb(float a) {
  union { float f; int x; } u;
  u.x = (int) (12102203 * a + 1064631197);
  return u.f;
}


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