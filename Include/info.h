#ifndef INCLUDE_INFO_H
#define INCLUDE_INFO_H

struct common_info_struct
{
    int numGPU;
    int numSparseMatrix;
};

struct gpu_info_struct
{
    int busy;

    void *devMem;
    size_t devMemSize;
    void *hostMem;
    size_t hostMemSize;
};

struct matrix_info_struct
{
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
    Float *Ty;

    Long *Cp;
    Long *Ci;
    Float *Cx;
    Float *Cy;

    Long *Lp;
    Long *Li;
    Float *Lx;
    Float *Ly;

    enum PermMethod permMethod;

    Long *Head;
    Long *Next;
    Long *Perm;
    Long *Pinv;

    Long *ColCount;

    void *workspace;
    size_t workSize;

    double readTime;
    double analyzeTime;
    double factorizeTime;
    double solveTime;
};

#endif
