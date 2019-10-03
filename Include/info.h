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

    int AMultiple;
    int BMultiple;
    int BCMultiple;
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

    cudaEvent_t s_cudaEvent_onDevice;
    cudaEvent_t d_cudaEvent_onDevice[MAX_D_STREAM];

    cudaStream_t s_cudaStream;
    cudaStream_t d_cudaStream[MAX_D_STREAM];

    cublasHandle_t s_cublasHandle;
    cublasHandle_t d_cublasHandle[MAX_D_STREAM];

    cusolverDnHandle_t s_cusolverDnHandle;

    int h_lastMatrix;
    Long h_lastNode;
    int d_lastMatrix;
    Long d_lastNode;
};

struct matrix_info_struct
{
    int serial;

    const char *path;

    FILE *file;

    enum FactorizeType factorizeType;

    int isComplex;

    int isSymmetric;

    enum MatrixState state;

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

    Long *Up;
    Long *Ui;
    Float *Ux;

    Float *Bx;
    Float *Xx;
    Float *Rx;

    enum PermMethod permMethod;

    Long *Perm;
    Long *Pinv;

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

    Long nsubtree;
    Long *ST_Map;
    Long *ST_Pointer;
    Long *ST_Index;
    Long *ST_Parent;

    size_t *Aoffset;
    size_t *Moffset;

    void *workspace;
    size_t workSize;

    Float residual;

    double readTime;
    double analyzeTime;
    double factorizeTime;
    double solveTime;
};

#endif
