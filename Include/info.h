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

struct matrix_info_struct
{
    FILE *file;

    enum FactorizeType factorize_type;

    int isComplex;

    int state;

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

    double read_time;
    double analyze_time;
    double factorize_time;
    double solve_time;
};

#endif
