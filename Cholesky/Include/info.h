#ifndef INCLUDE_INFO_H
#define INCLUDE_INFO_H

#include <omp.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include "parameter.h"

struct common_info_struct
{
    int numCPU;
    int numGPU;
    int numGPU_physical;
    size_t minDevMemSize;
    size_t minHostMemSize;

    int matrixThreadNum;

    int numSparseMatrix;

    size_t devSlotSize;

    double allocateTime;
    double computeTime;
    double freeTime;
};

struct gpu_info_struct
{
    int gpuIndex_physical;

    omp_lock_t gpuLock;

    void *devMem;
    size_t devMemSize;
    void *hostMem;
    size_t hostMemSize;

    size_t sharedMemSize;

    void *h_A[A_MULTIPLE];
    void *h_B[B_MULTIPLE];
    void *h_Lsi;

    void *d_A[A_MULTIPLE+1];
    void *d_B[B_MULTIPLE];
    void *d_C[C_MULTIPLE];
    void *d_Lsi;

    cudaEvent_t s_cudaEvent_factorized;
    cudaEvent_t d_cudaEvent_onDevice[MAX_D_STREAM];
    cudaEvent_t d_cudaEvent_applied[MAX_D_STREAM];

    cudaStream_t s_cudaStream;
    cudaStream_t s_cudaStream_copyback;
    cudaStream_t d_cudaStream[MAX_D_STREAM];
    cudaStream_t d_cudaStream_copy[MAX_D_STREAM];

    cublasHandle_t s_cublasHandle;
    cublasHandle_t d_cublasHandle[MAX_D_STREAM];

    cusolverDnHandle_t s_cusolverDnHandle;

    int lastMatrix;
};

struct matrix_info_struct
{
    int serial;

    const char *path;

    FILE *file;

    enum FactorizeType factorizeType;

    int isSymmetric;
    int isComplex;

    Long ncol;
    Long nrow;
    Long nzmax;

    Long *Tj;
    Long *Ti;
    Float *Tx;

    Long *Cp;
    Long *Ci;
    Float *Cx;

    Long *Lp;
    Long *Li;
    Float *Lx;

    Long *LTp;
    Long *LTi;
    Float *LTx;

    enum PermMethod permMethod;

    Long *Perm;

    Long *Parent;
    Long *Post;
    Long *ColCount;
    Long *RowCount;

    Long nsuper;
    Long *Super;
    Long *SuperMap;
    Long *Sparent;

    Long nsleaf;
    Long *LeafQueue;

    Long isize;
    Long xsize;
    Long *Lsip;
    Long *Lsxp;
    Long *Lsi;
    Float *Lsx;

    Long csize;

    Long nstage;
    Long *ST_Map;
    Long *ST_Pointer;
    Long *ST_Index;
    Long *ST_Parent;

    size_t *Aoffset;
    size_t *Moffset;

    void *workspace;
    size_t workSize;

    Float *Bx;
    Float *Xx;
    Float *Rx;

    Float residual;

    double readTime;
    double analyzeTime;
    double factorizeTime;
    double solveTime;
};

#endif
