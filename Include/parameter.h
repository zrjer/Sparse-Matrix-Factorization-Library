#ifndef INCLUDE_PARAMETER_H
#define INCLUDE_PARAMETER_H

//#define PRINT_CALLS
//#define PRINT_DEBUG
#define PRINT_INFO

#define MATRIX_THREAD_NUM 2

#define MAX_NUM_THREAD 32
#define CP_NUM_THREAD 8
#define CP_THREAD_THRESHOLD 64

#define MAX_NUM_CPU (0)

#define MAX_NUM_GPU (-1)
#define MAX_GPU_SPLIT (4)

#define MEM_LIMITS

#ifdef MEM_LIMITS
    #define MAX_MEM 4
    #define MAX_HOST_MEM 4
    #define MAX_DEV_MEM 4
#else
#endif

#undef MEM_LIMITS

#define MAXMMLINE 1024

#define BC_MULTIPLE 5

#define MAX_D_STREAM 4

#define MAX_BATCHSIZE 4096

#define CUDA_BLOCKDIM_X 16
#define CUDA_BLOCKDIM_Y 32

const enum PermMethod perm_method = PERM_METIS;

const double prune_dense = 10.0;
const double aggressive = 1;

#define RELAX_RATE (0.2)
#define MIN_SUPERNODE_COLUMN (32)

#endif
