/*
 * @Descripttion: 
 * @version: 1.0.0
 * @Author: Gager
 * @Date: 2022-11-17 09:09:25
 * @LastEditors: Gager
 */
#ifndef _LSTMCELL_H

#define _LSTMCELL_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>
#include <math.h>
#include <stdio.h>

#if FAST_MATH
    #include "fast_math.h"
    #define EXP expf_fast
    #define SIGMOD(x) (1.0 / (1.0 + EXP(-(x))))
    #define TANH(x) (2*SIGMOD(2*(x)) - 1)
#else
    #define SIGMOD(x) (1.0 / (1.0 + exp(-(x))))
    #define TANH(x) tanh((x))
#endif

#if W_TYPE_INT8
    #define DEQUANT(i8,s,zp) (((float)(i8)-(zp))*(s))
    #define QUANT(fp32,s,zp) ((int)((fp32)/(s)+zp))
#else   //FP32,FP16
    #define DEQUANT(x,s,zp)  (x)
    #define QUANT(x,s,zp)    (x)
#endif

#if W_TYPE_INT8
    typedef int8_t mytype;
#else
    typedef float_t mytype;
#endif

#define ALIGN(x, n) (((x) + ((n) - 1)) & ~((n) - 1))
#define ALIGN8(x) ALIGN((x), 7)


struct lstmcell
{
    int input;
    int output;
    float *x;
    float *h;
    float *f;
    float *i;
    float *c_hat;
    float *c;
    float *o;

    mytype *W_fh;
    float w_fh_scale, w_fh_zeropoint;
    mytype *W_fx;
    float w_fx_scale, w_fx_zeropoint;
    float *b_f;
    float b_f_scale, b_f_zeropoint;

    mytype *W_ih;
    float w_ih_scale, w_ih_zeropoint;
    mytype *W_ix; 
    float w_ix_scale, w_ix_zeropoint;
    float *b_i; 
    float b_i_scale, b_i_zeropoint;

    mytype *W_ch; 
    float w_ch_scale, w_ch_zeropoint;
    mytype *W_cx;
    float w_cx_scale, w_cx_zeropoint;
    float *b_c; 
    float b_c_scale, b_c_zeropoint;

    mytype *W_oh;
    float w_oh_scale, w_oh_zeropoint;
    mytype *W_ox; 
    float w_ox_scale, w_ox_zeropoint;
    float *b_o; 
    float b_o_scale, b_o_zeropoint;

    // float *f_b;
    // float *i_b;
    // float *o_b;
    // float *c_b;

    int error_no;
    char *error_msg;

    struct lstmcell *before;
    struct lstmcell *after;
};

struct lstmcell* lstmcell_create(int input, int output);
char lstmcell_set_params(
    struct lstmcell *unit, void *wx, void *wh, float *bias, 
    float *w_x_s, float *w_x_z, float *w_h_s, float *w_h_z,
    float *b_s, float *b_z
);
char lstmcell_random_params(struct lstmcell *unit, float min, float max);
char lstmcell_run(struct lstmcell *unit, float *input, float *output);
char lstmcell_run_unit(struct lstmcell *unit);
char lstmcell_release(struct lstmcell *unit);

#ifdef __cplusplus
}
#endif

#endif