#ifndef INCLUDE_INFO_H
#define INCLUDE_INFO_H

#include <omp.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

struct common_info_struct
{
    int numCPU;
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

    cudaEvent_t s_cudaEvent_onDevice;

    cudaStream_t s_cudaStream;
    cudaStream_t d_cudaStream[MAX_D_STREAM];

    cublasHandle_t s_cublasHandle;
    cublasHandle_t d_cublasHandle[MAX_D_STREAM];

    cusolverDnHandle_t s_cusolverDnHandle;
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

    int AMultiple;
    int BCMultiple;
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

struct node_size_struct
{
    Long node;
    size_t col;
    size_t row;
    size_t size;
};

struct cholesky_apply_task_struct
{
    int dn;
    int dm;
    int dk;
    Long slda;
    Long dlda;
    Long dldc;
    Float *d_A;
    Float *d_B;
    Float *d_C;
};

struct cholesky_solve_task_struct
{
    int dn;
    int dm;
    int dk;
    Long slda;
    Float *d_A;
    Float *d_B;
    Float *d_C;
};

#endif
