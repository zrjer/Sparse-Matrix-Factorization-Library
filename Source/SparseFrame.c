#include <stdio.h>
#include <stdlib.h>
#include "SparseFrame.h"

int SparseFrame_allocate_gpu ( struct gpu_info_struct **gpu_info_ptr, struct common_info_struct *common_info )
{
    int numGPU;
    int gpu_index;

    cudaGetDeviceCount ( &numGPU );
    common_info->numGPU = numGPU;

#ifdef PRINT_INFO
    printf ( "Num of GPUs = %d\n", numGPU );
#endif

    *gpu_info_ptr = malloc ( sizeof ( struct gpu_info_struct ) * numGPU );

    if ( *gpu_info_ptr == NULL ) return 0;

    for ( gpu_index = 0; gpu_index < numGPU; gpu_index++ )
    {
        struct cudaDeviceProp prop;
        size_t dev_memsize;
        size_t host_memsize;

        cudaError_t cudaStatus;

        cudaSetDevice ( gpu_index );
        cudaGetDeviceProperties ( &prop, gpu_index );

        dev_memsize = prop.totalGlobalMem;
        dev_memsize = ( size_t ) ( ( double ) dev_memsize * 0.9 );
        dev_memsize = dev_memsize - dev_memsize % ( 0x400 * 0x400 ); // align to 1 MB
        cudaStatus = cudaMalloc ( &( (*gpu_info_ptr)[gpu_index].dev_mem ), dev_memsize );

        if ( cudaStatus == cudaSuccess )
        {
            host_memsize = dev_memsize;
            cudaStatus = cudaMallocHost ( &( (*gpu_info_ptr)[gpu_index].host_mem ), host_memsize );

            if ( cudaStatus == cudaSuccess )
            {
                (*gpu_info_ptr)[gpu_index].busy = 0;
                (*gpu_info_ptr)[gpu_index].dev_memsize = dev_memsize;
                (*gpu_info_ptr)[gpu_index].host_memsize = host_memsize;
#ifdef PRINT_INFO
                printf ( "GPU %d device memory size = %lf GiB host memory size = %lf GiB\n", gpu_index, ( double ) dev_memsize / ( 0x400 * 0x400 * 0x400 ), ( double ) host_memsize / ( 0x400 * 0x400 * 0x400 ) );
#endif
            }
            else
            {
                (*gpu_info_ptr)[gpu_index].busy = 1;
                (*gpu_info_ptr)[gpu_index].dev_mem = NULL;
                (*gpu_info_ptr)[gpu_index].dev_memsize = 0;
                (*gpu_info_ptr)[gpu_index].host_mem = NULL;
                (*gpu_info_ptr)[gpu_index].host_memsize = 0;
#ifdef PRINT_INFO
                printf ( "GPU %d cudaMalloc fail\n", gpu_index );
#endif
            }
        }
        else
        {
            (*gpu_info_ptr)[gpu_index].busy = 1;
            (*gpu_info_ptr)[gpu_index].dev_mem = NULL;
            (*gpu_info_ptr)[gpu_index].dev_memsize = 0;
            (*gpu_info_ptr)[gpu_index].host_mem = NULL;
            (*gpu_info_ptr)[gpu_index].host_memsize = 0;
#ifdef PRINT_INFO
            printf ( "GPU %d cudaMalloc fail\n", gpu_index );
#endif
        }
    }

    return 1;
}

int SparseFrame_free_gpu ( struct gpu_info_struct **gpu_info_ptr, struct common_info_struct *common_info )
{
    int numGPU;
    int gpu_index;

    if ( *gpu_info_ptr == NULL ) return 0;

    numGPU = common_info->numGPU;

    for ( gpu_index = 0; gpu_index < numGPU; gpu_index++ )
    {
        cudaSetDevice ( gpu_index );

        if ( (*gpu_info_ptr)[gpu_index].dev_mem != NULL )
            cudaFree ( (*gpu_info_ptr)[gpu_index].dev_mem );
        if ( (*gpu_info_ptr)[gpu_index].host_mem != NULL )
            cudaFreeHost ( (*gpu_info_ptr)[gpu_index].host_mem );

        (*gpu_info_ptr)[gpu_index].busy = 1;
        (*gpu_info_ptr)[gpu_index].dev_mem = NULL;
        (*gpu_info_ptr)[gpu_index].dev_memsize = 0;
        (*gpu_info_ptr)[gpu_index].host_mem = NULL;
        (*gpu_info_ptr)[gpu_index].host_memsize = 0;
    }

    free ( (*gpu_info_ptr) );
    *gpu_info_ptr = NULL;

    common_info->numGPU = 0;

    return 1;
}

int SparseFrame_allocate_matrix ( struct matrix_info_struct **matrix_info_ptr, struct common_info_struct *common_info )
{
    int numSparseMatrix;

    numSparseMatrix = common_info->numSparseMatrix;

    *matrix_info_ptr = malloc ( sizeof ( struct matrix_info_struct ) * numSparseMatrix );

    if ( *matrix_info_ptr == NULL ) return 0;

    return 1;
}

int SparseFrame_free_matrix ( struct matrix_info_struct **matrix_info_ptr, struct common_info_struct *common_info )
{
    if ( *matrix_info_ptr == NULL ) return 0;

    free ( *matrix_info_ptr );

    common_info->numSparseMatrix = 0;

    return 1;
}

int SparseFrame_read_matrix ( char *path, struct matrix_info_struct *matrix_info, struct common_info_struct *common_info )
{
    matrix_info->file = fopen ( path, "r" );

    if ( matrix_info->file == NULL ) return 0;

    fclose ( matrix_info->file );

    return 1;
}

int SparseFrame ( int argc, char **argv )
{
    int numSparseMatrix, matrixIndex;

    struct common_info_struct common_info_object;
    struct common_info_struct *common_info = &common_info_object;
    struct gpu_info_struct *gpu_info;
    struct matrix_info_struct *matrix_info;

    // Allocate resources

    SparseFrame_allocate_gpu (&gpu_info, common_info);

    numSparseMatrix = argc - 1;
    common_info->numSparseMatrix = numSparseMatrix;

    SparseFrame_allocate_matrix ( &matrix_info, common_info );

    for ( matrixIndex = 0; matrixIndex < numSparseMatrix; matrixIndex++ )
    {
        // Read matrices

        SparseFrame_read_matrix ( argv [ 1 + matrixIndex ], matrix_info + matrixIndex, common_info );

        // Factorize

        // Solve

    }

    // Free resources

    SparseFrame_free_gpu (&gpu_info, common_info);

    SparseFrame_free_matrix ( &matrix_info, common_info );

    return 0;
}
