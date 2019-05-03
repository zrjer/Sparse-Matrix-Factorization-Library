#ifndef INCLUDE_PARAMETER_H
#define INCLUDE_PARAMETER_H

//#define PRINT_CALLS
//#define PRINT_DEBUG
#define PRINT_INFO

#define MAX_NUM_THREAD 32
#define CP_NUM_THREAD 8
#define CP_THREAD_THRESHOLD 64

#define MAX_NUM_GPU (1)
#define MAX_GPU_SPLIT (2)

#define MEM_LIMITS

#ifdef MEM_LIMITS
    #define MAX_MEM 4
    #define MAX_HOST_MEM 4
    #define MAX_DEV_MEM 4
#else
#endif

#undef MEM_LIMITS

#define MATRIX_THREAD_NUM 2

#define MAXMMLINE 1024

const enum PermMethod perm_method = PERM_METIS;

const double prune_dense = 10.0;
const double aggressive = 1;

#define RELAX_RATE (0.2)

#endif
