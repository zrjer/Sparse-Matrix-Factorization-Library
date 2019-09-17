#ifndef INCLUDE_PARAMETER_H
#define INCLUDE_PARAMETER_H

#include "macro.h"

//#define PRINT_CALLS
//#define PRINT_DEBUG
#define PRINT_INFO

#define MATRIX_THREAD_NUM (2)

#define CP_NUM_THREAD (4)
#define CP_THREAD_THRESHOLD (256)

#define MAX_NUM_CPU (0)

#define MAX_NUM_GPU (-1)
#define GPU_SPLIT_LIMIT (4)

#define MAXMMLINE (1024)

const enum PermMethod perm_method = PERM_METIS;

const double prune_dense = 10.0;
const double aggressive = 1;

#define RELAXED_SUPERNODE

#ifdef RELAXED_SUPERNODE
int should_relax ( Long col, double rate )
{
    const int n_checks = 3;
    const Long relax_threshold_col[] = { 16, 64, 256 };
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
    Long score;
};

#define MAX_BATCH (256)

#if ( defined ( MAX_BATCH ) && ( MAX_BATCH != 0 ) )
const int dimension_n_checks = 3;
const Long dimension_threshold_x[] = { 16, 32, 64 };
const Long dimension_threshold_y[] = { 24, 48, 96 };
#endif
const Long dimension_threshold_mn = 64;
const Long dimension_threshold_k = 16;

Long set_node_score ( struct node_size_struct *node, int useBatch )
{
    Long n, m, k, score;

    n = node->n;
    m = node->m;
    k = node->k;

#if ( defined ( MAX_BATCH ) && ( MAX_BATCH != 0 ) )
    if ( useBatch )
    {
        for ( int gpu_blas_batch_loop_idx = 0; gpu_blas_batch_loop_idx < dimension_n_checks; gpu_blas_batch_loop_idx++ )
        {
            if ( k <= dimension_threshold_x[gpu_blas_batch_loop_idx] && m <= dimension_threshold_y[gpu_blas_batch_loop_idx] && n <= dimension_threshold_y[gpu_blas_batch_loop_idx] )
            {
                score = dimension_threshold_x[gpu_blas_batch_loop_idx];
                node->score = score;
                return score;
            }
        }
    }
#endif

    if ( m + n >= dimension_threshold_mn && k >= dimension_threshold_k ) 
        score = ( m + n ) * k;
    else
        score = - ( m + n ) * k;

    node->score = score;
    return score;
}

int set_factorize_location ( Long nscol, Long nsrow )
{
    const Long potrf_dimension_threshold_n = 256;
    const Long trsm_dimension_threshold_m = 64;

    return ( nscol < potrf_dimension_threshold_n && ( nsrow - nscol ) < trsm_dimension_threshold_m );
}

const Long potrf_split_threshold = 1024;
const Long potrf_split_block = 256;

Long get_node_score ( const struct node_size_struct *node )
{
    return node->score;
}

#define MAX_D_STREAM (2) // must be at least 2

#define A_MULTIPLE (2)
#define BC_MULTIPLE ( 2 * MAX_D_STREAM )

#define CUDA_BLOCKDIM_X (16)
#define CUDA_BLOCKDIM_Y (24)

#endif
