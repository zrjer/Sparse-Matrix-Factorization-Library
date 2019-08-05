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

#define MAXMMLINE (1024)

const enum PermMethod perm_method = PERM_METIS;

const double prune_dense = 10.0;
const double aggressive = 1;

#define RELAXED_SUPERNODE

#ifdef RELAXED_SUPERNODE
int should_relax ( Long col, double rate )
{
    const int n_checks = 3;
    const Long relax_threshold_col[] = { 4, 16, 48 };
    const double relax_threshold_rate[] = { 0.8, 0.1, 0.05 };

    for ( int k = n_checks - 1; k >= 0; k-- )
    {
        if ( col > relax_threshold_col[k] && rate > relax_threshold_rate[k] )
        {
            return FALSE;
        }
    }

    return TRUE;
}
#endif

struct node_size_struct
{
    Long node;
    Long n;
    Long m;
    Long k;
};

const int dimension_n_checks = 3;
const Long dimension_threshold_x[] = { 24, 48, 96 };
const Long dimension_threshold_y[] = { 16, 32, 64 };
const Long dimension_threshold_mn = 256;
const Long dimension_threshold_k = 32;

Long node_score ( const struct node_size_struct *node )
{
    Long n, m, k;

    n = node->n;
    m = node->m;
    k = node->k;

    for ( int idx = 0; idx < dimension_n_checks; idx++ )
    {
        if ( k <= dimension_threshold_x[idx] && m <= dimension_threshold_y[idx] && n <= dimension_threshold_y[idx] )
        {
            return dimension_threshold_x[idx];
        }
    }

    if ( m + n < dimension_threshold_mn || k < dimension_threshold_k ) return - ( m + n ) * k;

    return ( m + n ) * k;
}

#define MAX_BATCH (16384)

#define MAX_D_STREAM (4)

#define A_MULTIPLE (2)
#define BC_MULTIPLE ( 2 * MAX_D_STREAM )

#define CUDA_BLOCKDIM_X (16)
#define CUDA_BLOCKDIM_Y (24)

#endif
