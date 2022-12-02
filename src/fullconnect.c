/*
 * @Descripttion: 
 * @version: 1.0.0
 * @Author: Gager
 * @Date: 2022-11-22 12:54:54
 * @LastEditors: Gager
 */
#include "fullconnect.h"

#define A(i,j) a[ (i)*lda + (j) ]
#define B(i,j) b[ (i)*ldb + (j) ]
#define C(i,j) c[ (i)*ldc + (j) ]

// char full_connecte(float *input, float *output, float *fc_w, float *fc_b, int in, int out){

//   float *fc_w_r; 
//   for (int i=0; i<out; i++){
//     fc_w_r = fc_w + i * in;
//     output[i] += fc_b[i];
//     for (int j=0; j<in; j++){
//       output[i] += input[j] * fc_w_r[j];
//     }
//     printf("%f, ", output[i]);
//   }

//   return 1;
// }

char full_connecte(float *input, float *output, float *fc_w, float *fc_b, int in, int out){

  float *fc_w_r; 
  int i, j, num;
  for (i=0; i<out; i++){
    fc_w_r = fc_w + i * in;
    output[i] += fc_b[i];

    num = (in>>3)<<3;
    for(j=0; j<num; ){
      output[i] += input[j] * fc_w_r[j]; j++;
      output[i] += input[j] * fc_w_r[j]; j++;
      output[i] += input[j] * fc_w_r[j]; j++;
      output[i] += input[j] * fc_w_r[j]; j++;
      output[i] += input[j] * fc_w_r[j]; j++;
      output[i] += input[j] * fc_w_r[j]; j++;
      output[i] += input[j] * fc_w_r[j]; j++;
      output[i] += input[j] * fc_w_r[j]; j++;
    }
    for(; j<in; ){
      output[i] += input[j] * fc_w_r[j]; j++; 
    }

  }

  return 1;
}

void AddDot1x4( int k, float *a, int lda,  float *b, int ldb, float *c, int ldc ){

  int p;
  register float 
    c_r0_c0_reg, c_r0_c1_reg, c_r0_c2_reg, c_r0_c3_reg;

  float *a_r0_p; 
  float a_r0_c0, a_r0_c1, a_r0_c2, a_r0_c3;
  float b_r_c0, b_r_c1, b_r_c2, b_r_c3;

  a_r0_p = &A(0, 0);

  c_r0_c0_reg = 0.0; 
  c_r0_c1_reg = 0.0; 
  c_r0_c2_reg = 0.0; 
  c_r0_c3_reg = 0.0;
 
  for ( p=0; p<k; p++ ){

    a_r0_c0 = *a_r0_p++;
    a_r0_c1 = *a_r0_p++;
    a_r0_c2 = *a_r0_p++;
    a_r0_c3 = *a_r0_p++;

    b_r_c0 = B(p, 0);
    b_r_c1 = B(p, 1);
    b_r_c2 = B(p, 2);
    b_r_c3 = B(p, 3);

    c_r0_c0_reg += a_r0_c0 * b_r_c0;
    c_r0_c1_reg += a_r0_c1 * b_r_c1;
    c_r0_c2_reg += a_r0_c2 * b_r_c2;
    c_r0_c3_reg += a_r0_c3 * b_r_c3;
  }

  C(0, 0) += c_r0_c0_reg; 
  C(1, 0) += c_r0_c1_reg; 
  C(2, 0) += c_r0_c2_reg; 
  C(3, 0) += c_r0_c3_reg;
}

void MY_MMult_1x4_7( int m, int k, float *a, int lda, 
                                    float *b, int ldb,
                                    float *c, int ldc )
{
  int i;

  for ( i=0; i<m; i+=4 ){
    AddDot1x4(k, &A(0, 0), lda, &B(i, 0), ldb, &C(0, i), ldc);
  }
}