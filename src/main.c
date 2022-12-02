/*
 * @Descripttion: 
 * @version: 1.0.0
 * @Author: Gager
 * @Date: 2022-11-17 11:49:37
 * @LastEditors: Gager
 */
# include "lstmcell.h"
// #include "lstmunion.h"

#if W_TYPE_INT8
    #include "lstm_qua_params.h"
#else
    #include "lstm_params.h"
#endif

#include "fullconnect.h"

int main(int argc, char *argv[])
{
    // int in=8, out=4;
    int in=32, out=16;
    float *lstm_output, *fc_output;
    float total_time;

    struct lstmcell *unit;
    unit = lstmcell_create(in, in);
    lstmcell_set_params(unit, lstm_w_x, lstm_w_h, lstm_bias, lstm_w_x_s, lstm_w_x_z, lstm_w_h_s, lstm_w_h_z, lstm_bias_s, lstm_bias_z);
    lstm_output = (float*)calloc(in, sizeof(float));
    // output = (float*)calloc(16, sizeof(float));

    // struct lstms *lstm;
    // lstm = lstm_create(8, 4, 1);
    // lstm_set_params(lstm, w_x, w_h, bias);
    // output = (float*)calloc(4, sizeof(float));

    clock_t start, finish;
    start = clock();
    lstmcell_run(unit, input, lstm_output);
    // lstm_run(lstm, input, output, 0);
    finish = clock();
    total_time = (float)(finish - start) / CLOCKS_PER_SEC;
    printf("[INFO]>>> lstm use time:%lf/s \n", total_time);
    // printf("id,x,h,f,i,c_hat,C,o\n");
    for (int i = 0; i < out; i++) {
        // printf("%d, %f, %f, %f, %f, %f, %f, %f\n", i, input[i], output[i], (*lstm).end->f[i], (*lstm).end->i[i], (*lstm).end->c_hat[i], (*lstm).end->c[i], (*lstm).end->o[i]);
        printf("%d, %f, %f, %f, %f, %f, %f, %f\n", i, input[i], lstm_output[i], (*unit).f[i], (*unit).i[i], (*unit).c_hat[i], (*unit).c[i], (*unit).o[i]);
    }

    start = clock();
    fc_output = (float*)calloc(out, sizeof(float));
    full_connecte(lstm_output, fc_output, fc_w, fc_b, in, out);
    finish = clock();
    total_time = (float)(finish - start) / CLOCKS_PER_SEC;
    printf("[INFO]>>> lstm use time:%lf/s \n", total_time);
    for(int i=0; i<out; i++){
        printf("%f, ", fc_output[i]);
    }
    printf("\n");

    return 0;
}