#ifndef INFO_H
#define INFO_H

struct common_info_struct
{
    int numGPU = 0;
};

struct gpu_info_struct
{
    int busy = 0;

    void *host_mem = NULL;
    size_t host_memsize = 0;
    void *dev_mem = NULL;
    size_t dev_memsize = 0;
};

struct matrix_info_struct
{
    int state;
};

#endif
