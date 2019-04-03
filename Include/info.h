#ifndef INCLUDE_INFO_H
#define INCLUDE_INFO_H

#include <omp.h>

struct common_info_struct
{
    int numGPU;
    int numThreads;
    int numSparseMatrix;

    double allocateTime;
    double computeTime;
    double freeTime;
};

struct gpu_info_struct
{
    omp_lock_t gpu_lock;

    void *devMem;
    size_t devMemSize;
    void *hostMem;
    size_t hostMemSize;

    size_t sharedMemSize;
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

    Long isize;
    Long xsize;
    Long *Lsip;
    Long *Lsxp;
    Long *Lsi;
    Float *Lsx;

    void *workspace;
    size_t workSize;

    Float residual;

    double readTime;
    double analyzeTime;
    double factorizeTime;
    double solveTime;
};

#endif
