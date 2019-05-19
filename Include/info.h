#ifndef INCLUDE_INFO_H
#define INCLUDE_INFO_H

#include <omp.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

struct common_info_struct
{
    int numGPU;
    int numThread;
    int numSparseMatrix;

    size_t minDevMemSize;

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

    cudaEvent_t s_cudaEvent_reset;
    cudaEvent_t s_cudaEvent_onDevice;
    cudaEvent_t s_cudaEvent_assembled;
    cudaEvent_t d_cudaEvent_onDevice[B_SLOT_NUM];
    cudaEvent_t d_cudaEvent_updated[B_SLOT_NUM];
    cudaEvent_t d_cudaEvent_assembled[C_SLOT_NUM];

    cudaStream_t s_cudaStream;
    cudaStream_t d_cudaStream[B_SLOT_NUM];

    cublasHandle_t s_cublasHandle;
    cublasHandle_t d_cublasHandle[B_SLOT_NUM];

    cusolverDnHandle_t s_cusolverDnHandle;
    cusolverDnHandle_t d_cusolverDnHandle[B_SLOT_NUM];
};

struct matrix_info_struct
{
    const char *path;

    FILE *file;

    enum FactorizeType factorizeType;

    int isComplex;

    int isSymmetric;

    enum MatrixState state;

    Long ncol;
    Long nrow;
    Long nzmax;

    size_t devSlotSize;

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

    Long nsubtree;
    Long *ST_Map;
    Long *ST_Pointer;
    Long *ST_Index;
    Long *ST_Parent;

    Long nstleaf;
    Long *ST_LeafQueue;

    Long isize;
    Long xsize;
    Long *Lsip;
    Long *Lsxp;
    Long *Lsi;
    Float *Lsx;

    Long csize;

    void *workspace;
    size_t workSize;

    Float residual;

    double readTime;
    double analyzeTime;
    double factorizeTime;
    double solveTime;
};

#endif
