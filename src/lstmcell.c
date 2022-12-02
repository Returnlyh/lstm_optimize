#include "lstmcell.h"


struct lstmcell* lstmcell_create(int input, int output)
{
    struct lstmcell* unit;
    if (input < 1) {
        return NULL;
    }
    unit = (struct lstmcell*)malloc(sizeof (struct lstmcell));
    if (!unit) {
        return NULL;
    }
    (*unit).error_no = 0;
    (*unit).error_msg = "\0";

    (*unit).input = input;
    (*unit).output = output;
    printf("in:%d, out:%d\n", (*unit).input, (*unit).output);

    int align8_out_len = ALIGN8(output);

    (*unit).h = (float*)calloc(align8_out_len, sizeof (float));
    if (NULL == (*unit).h) {
        free((*unit).x);
        free(unit);
        return NULL;
    }
    (*unit).f = (float*)calloc(align8_out_len, sizeof (float));
    if (NULL == (*unit).f) {
        free((*unit).h);
        free((*unit).x);
        free(unit);
        return NULL;
    }
    (*unit).i = (float*)calloc(align8_out_len, sizeof (float));
    if (NULL == (*unit).i) {
        free((*unit).f);
        free((*unit).h);
        free((*unit).x);
        free(unit);
        return NULL;
    }
    (*unit).c_hat = (float*)calloc(align8_out_len, sizeof (float));
    if (NULL == (*unit).c_hat) {
        free((*unit).i);
        free((*unit).f);
        free((*unit).h);
        free((*unit).x);
        free(unit);
        return NULL;
    }
    (*unit).c = (float*)calloc(align8_out_len, sizeof (float));
    if (NULL == (*unit).c) {
        free((*unit).c_hat);
        free((*unit).i);
        free((*unit).f);
        free((*unit).h);
        free((*unit).x);
        free(unit);
        return NULL;
    }
    (*unit).o = (float*)calloc(align8_out_len, sizeof (float));
    if (NULL == (*unit).o) {
        free((*unit).c);
        free((*unit).c_hat);
        free((*unit).i);
        free((*unit).f);
        free((*unit).h);
        free((*unit).x);
        free(unit);
        return NULL;
    }

    // (*unit).f_b = (float*)calloc(output, sizeof (float));
    // (*unit).i_b = (float*)calloc(output, sizeof (float));
    // (*unit).o_b = (float*)calloc(output, sizeof (float));
    // (*unit).c_b = (float*)calloc(output, sizeof (float));
    // if (
    //     NULL == (*unit).f_b || NULL == (*unit).i_b || NULL == (*unit).o_b || NULL == (*unit).c_b
    
    // ) {
    //     free((*unit).c);
    //     free((*unit).c_hat);
    //     free((*unit).i);
    //     free((*unit).f);
    //     free((*unit).h);
    //     free((*unit).x);
    //     free((*unit).o);
    //     free(unit);
    //     return NULL;
    // }

    // lstmcell_random_params(unit, -1, 1);
    return unit;
}

char lstmcell_random_params(struct lstmcell *unit, float min, float max)
{
    int p_h_l, p_x_l, p_b_l;
    float diff;
    if (NULL == unit) {
        return 0;
    }
    if (max < min) {
        return 0;
    }
    diff = max - min;
    p_h_l = (*unit).input * (*unit).output;
    p_x_l = (*unit).output * (*unit).output;
    p_b_l = (*unit).output;

    for(int i=0; i<p_h_l; ){
        (*unit).W_fh[i] = (float)rand() / RAND_MAX * diff + min;
        (*unit).W_ih[i] = (float)rand() / RAND_MAX * diff + min;
        (*unit).W_oh[i] = (float)rand() / RAND_MAX * diff + min;
        (*unit).W_ch[i] = (float)rand() / RAND_MAX * diff + min;
    }

    for(int i=0; i<p_x_l; i++){
        (*unit).W_fx[i] = (float)rand() / RAND_MAX * diff + min;
        (*unit).W_ix[i] = (float)rand() / RAND_MAX * diff + min;
        (*unit).W_ox[i] = (float)rand() / RAND_MAX * diff + min;
        (*unit).W_cx[i] = (float)rand() / RAND_MAX * diff + min;
    }

    for(int i=0; i<p_b_l; i++){
        (*unit).b_f[i] = (float)rand() / RAND_MAX * diff + min;
        (*unit).b_i[i] = (float)rand() / RAND_MAX * diff + min;
        (*unit).b_o[i] = (float)rand() / RAND_MAX * diff + min;
        (*unit).b_c[i] = (float)rand() / RAND_MAX * diff + min;
    }

    return 1;
}

char lstmcell_set_params(
    struct lstmcell *unit, void *wx, void *wh, float *bias, 
    float *w_x_s, float *w_x_z, float *w_h_s, float *w_h_z,
    float *b_s, float *b_z
    ){

    int i=0;
    int p_h_l, p_x_l, p_b_l;
    p_x_l = (*unit).input * (*unit).output;
    p_h_l = (*unit).output * (*unit).output;
    p_b_l = (*unit).output;
    
    (*unit).W_ix = (mytype*)wx; (*unit).W_ih = (mytype*)wh; (*unit).b_i = (float*)bias;
    (*unit).W_fx = (mytype*)wx + 1*p_x_l; (*unit).W_fh = (mytype*)wh + 1*p_h_l; (*unit).b_f = bias + 1*p_b_l;
    (*unit).W_cx = (mytype*)wx + 2*p_x_l; (*unit).W_ch = (mytype*)wh + 2*p_h_l; (*unit).b_c = bias + 2*p_b_l;
    (*unit).W_ox = (mytype*)wx + 3*p_x_l; (*unit).W_oh = (mytype*)wh + 3*p_h_l; (*unit).b_o = bias + 3*p_b_l;

    (*unit).w_ix_scale=w_x_s[0]; (*unit).w_ih_scale=w_h_s[0]; 
    (*unit).w_fx_scale=w_x_s[1]; (*unit).w_fh_scale=w_h_s[1]; 
    (*unit).w_cx_scale=w_x_s[2]; (*unit).w_ch_scale=w_h_s[2]; 
    (*unit).w_ox_scale=w_x_s[3]; (*unit).w_oh_scale=w_h_s[3]; 

    (*unit).w_ix_zeropoint=w_x_z[0]; (*unit).w_ih_zeropoint=w_h_z[0]; 
    (*unit).w_fx_zeropoint=w_x_z[1]; (*unit).w_fh_zeropoint=w_h_z[1]; 
    (*unit).w_cx_zeropoint=w_x_z[2]; (*unit).w_ch_zeropoint=w_h_z[2]; 
    (*unit).w_ox_zeropoint=w_x_z[3]; (*unit).w_oh_zeropoint=w_h_z[3]; 

    (*unit).b_i_scale = b_s[0]; (*unit).b_i_zeropoint = b_z[0];
    (*unit).b_f_scale = b_s[1]; (*unit).b_f_zeropoint = b_z[1];
    (*unit).b_c_scale = b_s[2]; (*unit).b_c_zeropoint = b_z[2];
    (*unit).b_o_scale = b_s[3]; (*unit).b_o_zeropoint = b_z[3];

    // printf("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n");
    // printf("W_ix:");
    // for (int i=0; i<p_x_l; i++){

    //     printf("%d, ", ((int8_t*)wx)[i]);
    // }
    // printf("\n\n");

    return 1;

}

char lstmcell_run(struct lstmcell *unit, float *input, float *output)
{
    float *input_back;
    float *output_back;
    if (NULL == unit) {
        return 0;
    }
    if (NULL == input) {
        return 0;
    }
    if (NULL == output) {
        return 0;
    }

    (*unit).x = input;
    (*unit).h = output;
    lstmcell_run_unit(unit);

    return 1;
}

char lstmcell_release(struct lstmcell *unit){

    free((*unit).c);
    free((*unit).c_hat);
    free((*unit).i);
    free((*unit).f);
    free((*unit).h);
    free((*unit).x);

    free((*unit).W_oh);
    free((*unit).W_ox);
    free((*unit).b_o);

    free((*unit).W_fh);
    free((*unit).W_fx);
    free((*unit).b_f);

    free((*unit).W_ih);
    free((*unit).W_ix);
    free((*unit).b_i);

    free((*unit).W_ch);
    free((*unit).W_cx);
    free((*unit).b_c);
    free(unit);

    return 1;

}

char lstmcell_run_unit(struct lstmcell *unit)
{
    int i, j, k, input, output;
    double r_f, r_i, r_o, r_c;
    if (NULL == unit) {
        return 0;
    }
    input = (*unit).input;
    output = (*unit).output;
    int num = (input>>2)<<2;
    // printf("input:%d, num:%d\n", input, num);
    // printf("#########################\n");
    if ((*unit).before == NULL) {

        for (i = 0; i < output; i++) {
            for(j=0; j<num; ){
                for(k=0; k<4; k++){
                    r_f += DEQUANT((*unit).W_fx[i*input+j], (*unit).w_fx_scale, (*unit).w_fx_zeropoint) * (*unit).x[j];
                    r_i += DEQUANT((*unit).W_ix[i*input+j], (*unit).w_ix_scale, (*unit).w_ix_zeropoint) * (*unit).x[j];
                    r_o += DEQUANT((*unit).W_ox[i*input+j], (*unit).w_ox_scale, (*unit).w_ox_zeropoint) * (*unit).x[j];
                    r_c += DEQUANT((*unit).W_cx[i*input+j], (*unit).w_cx_scale, (*unit).w_cx_zeropoint) * (*unit).x[j];
                    j++;
                }
            }
            for(; j< input; j++){
                r_f += DEQUANT((*unit).W_fx[i*input+j], (*unit).w_fx_scale, (*unit).w_fx_zeropoint) * (*unit).x[j];
                r_i += DEQUANT((*unit).W_ix[i*input+j], (*unit).w_ix_scale, (*unit).w_ix_zeropoint) * (*unit).x[j];
                r_o += DEQUANT((*unit).W_ox[i*input+j], (*unit).w_ox_scale, (*unit).w_ox_zeropoint) * (*unit).x[j];
                r_c += DEQUANT((*unit).W_cx[i*input+j], (*unit).w_cx_scale, (*unit).w_cx_zeropoint) * (*unit).x[j];
            }

            printf("[INFO]>>> i:%d, r_f:%f, r_i:%f, r_o:%f, r_c:%f, ", i, r_f, r_i, r_o, r_c);
            printf("b_f:%f, b_i:%f, b_o:%f, b_c:%f \n", (*unit).b_f[i], (*unit).b_i[i], (*unit).b_o[i], (*unit).b_c[i]);
            (*unit).f[i] = SIGMOD(r_f + (*unit).b_f[i]);
            (*unit).i[i] = SIGMOD(r_i + (*unit).b_i[i]);
            (*unit).o[i] = SIGMOD(r_o + (*unit).b_o[i]);
            (*unit).c_hat[i] = TANH(r_c + (*unit).b_c[i]);
            (*unit).c[i] = (*unit).i[i] * (*unit).c_hat[i];
            (*unit).h[i] = (*unit).o[i] * TANH((*unit).c[i]);
            printf("[INFO]>>> f_i:%f, i_i:%f, o_i:%f, c_hat_i:%f, c_i:%f, h_i:%f\n\n", 
             (*unit).f[i], (*unit).i[i], (*unit).o[i], (*unit).c_hat[i], (*unit).c[i], (*unit).h[i]);
            r_f = r_i = r_o = r_c = 0;
        }
        
    } else {

        float *ht_1 = (*unit).before->h;
        float *ct_1 = (*unit).before->c;
        for (i=0; i<output; i++) {
            for(j=0; j<num; ){
                for(k=0; k<4; k++){
                    r_f += DEQUANT((*unit).W_fx[i*input+j], (*unit).w_fx_scale, (*unit).w_fx_zeropoint) * (*unit).x[j];
                    r_i += DEQUANT((*unit).W_ix[i*input+j], (*unit).w_ix_scale, (*unit).w_ix_zeropoint) * (*unit).x[j];
                    r_o += DEQUANT((*unit).W_ox[i*input+j], (*unit).w_ox_scale, (*unit).w_ox_zeropoint) * (*unit).x[j];
                    r_c += DEQUANT((*unit).W_cx[i*input+j], (*unit).w_cx_scale, (*unit).w_cx_zeropoint) * (*unit).x[j];
                    j++;
                }
            }
            for(; j<input; j++){
                    r_f += DEQUANT((*unit).W_fx[i*input+j], (*unit).w_fx_scale, (*unit).w_fx_zeropoint) * (*unit).x[j];
                    r_i += DEQUANT((*unit).W_ix[i*input+j], (*unit).w_ix_scale, (*unit).w_ix_zeropoint) * (*unit).x[j];
                    r_o += DEQUANT((*unit).W_ox[i*input+j], (*unit).w_ox_scale, (*unit).w_ox_zeropoint) * (*unit).x[j];
                    r_c += DEQUANT((*unit).W_cx[i*input+j], (*unit).w_cx_scale, (*unit).w_cx_zeropoint) * (*unit).x[j];
            }

            for (j=0; j<output;){
                r_f += DEQUANT((*unit).W_fh[i*output+j], (*unit).w_fh_scale, (*unit).w_fh_zeropoint) * ht_1[j];
                r_i += DEQUANT((*unit).W_ih[i*output+j], (*unit).w_ih_scale, (*unit).w_ih_zeropoint) * ht_1[j];
                r_o += DEQUANT((*unit).W_oh[i*output+j], (*unit).w_oh_scale, (*unit).w_oh_zeropoint) * ht_1[j];
                r_c += DEQUANT((*unit).W_ch[i*output+j], (*unit).w_ch_scale, (*unit).w_ch_zeropoint) * ht_1[j];
            }

            (*unit).f[i] = SIGMOD(r_f + (*unit).b_f[i]);
            (*unit).i[i] = SIGMOD(r_i + (*unit).b_i[i]);
            (*unit).o[i] = SIGMOD(r_o + (*unit).b_o[i]);
            (*unit).c_hat[i] = TANH(r_c + (*unit).b_c[i]);
            (*unit).c[i] = (*unit).f[i] * ct_1[i] + (*unit).i[i] * (*unit).c_hat[i];
            (*unit).h[i] = (*unit).o[i] * TANH((*unit).c[i]);
            r_f = r_i = r_o = r_c = 0;
        }
    }
    return 1;
}
