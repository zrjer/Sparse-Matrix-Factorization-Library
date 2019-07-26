#ifndef INCLUDE_PARAMETER_H
#define INCLUDE_PARAMETER_H

#include "macro.h"

//#define PRINT_CALLS
//#define PRINT_DEBUG
#define PRINT_INFO

#define MATRIX_THREAD_NUM (2)

#define MAX_NUM_THREAD (32)
#define CP_NUM_THREAD (8)
#define CP_THREAD_THRESHOLD (32)

#define MAX_NUM_CPU (0)

#define MAX_NUM_GPU (-1)
#define MAX_GPU_SPLIT (1)

#define MEM_LIMITS

#ifdef MEM_LIMITS
    #define MAX_MEM (4)
    #define MAX_HOST_MEM (4)
    #define MAX_DEV_MEM (4)
#else
#endif

#undef MEM_LIMITS

#define MAXMMLINE (1024)

#define A_MULTIPLE (2)
#define BC_MULTIPLE (4)

#define MAX_D_EVENT (2)
#define MAX_D_STREAM (4)

#define CUDA_BLOCKDIM_X (16)
#define CUDA_BLOCKDIM_Y (32)

#define MAX_BATCH (1024)

#define BLAS_THRESHOLD_N (128)
#define BLAS_THRESHOLD_M (128)
#define BLAS_THRESHOLD_K (128)

const enum PermMethod perm_method = PERM_METIS;

const double prune_dense = 10.0;
const double aggressive = 1;

#define RELAXED_SUPERNODE

#ifdef RELAXED_SUPERNODE
int should_relax ( Long col, double rate )
{
    const int n_checks = 3;
    const Long relax_threshold_col[] = { 4, 16, 64 };
    const double relax_threshold_rate[] = { 0.8, 0.1, 0.05 };

    for ( int k = n_checks - 1; k >= 0; k-- )
    {
        if ( col > relax_threshold_col[k] && rate > relax_threshold_rate[k] )
            return FALSE;
    }

    return TRUE;
}
#endif

#endif
