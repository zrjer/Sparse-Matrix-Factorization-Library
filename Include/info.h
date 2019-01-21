#ifndef INFO_H
#define INFO_H

typedef
struct common_info_struct
{
    void *mem;
    size_t memsize;
}
common_info;

typedef
struct gpu_info_struct
{
    void *host_mem;
    size_t host_memsize;
    void *dev_mem;
    size_t dev_memsize;
}
gpu_info;

typedef
struct matrix_info_struct
{
}
matrix_info;

#endif
