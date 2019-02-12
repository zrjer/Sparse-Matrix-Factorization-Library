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

    Long anz;
    Long *Ap;
    Long *Ai;

    Long *Len;
    Long *Nv;
    Long *Next;
    Long *Perm;
    Long *Head;
    Long *Elen;
    Long *Degree;
    Long *Wi;

    double read_time;
    double analyze_time;
    double factorize_time;
    double solve_time;
};

#endif
