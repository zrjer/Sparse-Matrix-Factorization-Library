#ifndef INFO_H
#define INFO_H

struct common_info_struct
{
    int numGPU;
    int numSparseMatrix;
};

struct gpu_info_struct
{
    int busy;

    void *dev_mem;
    size_t dev_memsize;
    void *host_mem;
    size_t host_memsize;
};

#define MAXMMLINE 1024
size_t max_mm_line_size = MAXMMLINE + 1;
enum FactorizeType {TYPE_CHOLESKY, TYPE_QR, TYPE_LU};
enum MatrixState {MATRIX_STATE_IDLE, MATRIX_STATE_TRIPLET, MATRIX_STATE_COMPRESSED};

struct matrix_info_struct
{
    FILE *file;

    enum FactorizeType factorize_type;

    int isComplex;

    enum MatrixState state;

    uLong ncol;
    uLong nrow;
    uLong nzmax;

    uLong *Tj;
    uLong *Ti;
    Float *Tx;
    Float *Ty;

    uLong *Cp;
    uLong *Ci;
    Float *Cx;
    Float *Cy;

    uLong *Len;
    uLong *Nv;
    uLong *Next;
    uLong *Perm;
    uLong *Head;
    uLong *Elen;
    uLong *Degree;
    uLong *Wi;

    double read_time;
    double analyze_time;
    double factorize_time;
    double solve_time;
};

#endif
