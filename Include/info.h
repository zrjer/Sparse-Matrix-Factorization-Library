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
    cudaEvent_t d_cudaEvent_updated;

    cudaStream_t s_cudaStream;
    cudaStream_t d_cudaStream[4];

    cublasHandle_t s_cublasHandle;
    cublasHandle_t d_cublasHandle[4];

    cusolverDnHandle_t s_cusolverDnHandle;
    cusolverDnHandle_t d_cusolverDnHandle[4];
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

    Long *Head;
    Long *Next;
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

    Long ST_Num;
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
