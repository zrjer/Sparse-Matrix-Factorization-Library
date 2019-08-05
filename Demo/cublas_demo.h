#ifndef DEMO_CUBLAS_DEMO_H
#define DEMO_CUBLAS_DEMO_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <cblas.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#define BATCH 16384
#define DIM_M 64
#define DIM_N 64
#define DIM_K 32
#define PAD 1

struct syrk_meta
{
    cublasHandle_t cublasHandle;
    int n;
    int k;
    double alpha;
    double *A;
    int lda;
    double beta;
    double *C;
    int ldc;
};

struct gemm_meta
{
    cublasHandle_t cublasHandle;
    int m;
    int n;
    int k;
    double alpha;
    double *A;
    int lda;
    double *B;
    int ldb;
    double beta;
    double *C;
    int ldc;
};

extern int dsyrk_(char *uplo, char *trans, int *n, int *k, const double *alpha, double *a, int *lda, const double *beta, double *c, int *ldc);

extern int dgemm_(char *transa, char *transb, int *m, int *n, int *k, const double *alpha, double *a, int *lda, double *b, int *ldb, const double *beta, double *c, int *ldc);

#endif
