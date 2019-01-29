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

struct matrix_info_struct
{
    FILE *file;

    int isComplex;

    int state;

    uLong *Ti;
    uLong *Tj;
    Float *Tx;
    Float *Ty;
};

#endif
