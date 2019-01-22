#ifndef INFO_H
#define INFO_H

struct common_info_struct
{
    int numGPU;
};

struct gpu_info_struct
{
    int busy;

    void *host_mem;
    size_t host_memsize;
    void *dev_mem;
    size_t dev_memsize;
};

struct matrix_info_struct
{
    int state;
};

#endif
