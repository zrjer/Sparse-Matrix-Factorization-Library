#include <stdio.h>
#include <stdlib.h>
#include "SparseFrame.h"

int SparseFrame ( int argc, char **argv )
{
    int numSparseMatrix;
    FILE **files;

    int numGPU;
    int gpu_index;

    struct common_info_struct common_info_object;
    struct common_info_struct *common_info = &common_info_object;
    struct gpu_info_struct *gpu_info;
    struct matrix_info_struct *matrix_info;

    // Allocate resources

    cudaGetDeviceCount ( &numGPU );
    common_info->numGPU = numGPU;

#ifdef PRINT_INFO
    printf ( "Num of GPUs = %d\n", numGPU );
#endif

    gpu_info = malloc ( sizeof ( struct gpu_info_struct ) * numGPU );

    for ( gpu_index = 0; gpu_index < numGPU; gpu_index++ )
    {
        struct cudaDeviceProp prop;
        size_t dev_memsize;

        cudaError_t cudaStatus;

        cudaSetDevice ( gpu_index );
        cudaGetDeviceProperties ( &prop, gpu_index );
        dev_memsize = prop.totalGlobalMem;
        dev_memsize = ( size_t ) ( ( double ) dev_memsize * 0.9 );
        dev_memsize = dev_memsize - dev_memsize % ( 0x400 * 0x400 ); // align to 1 MB
        gpu_info[gpu_index].dev_memsize = dev_memsize;
        cudaStatus = cudaMalloc ( &( gpu_info[gpu_index].dev_mem ), dev_memsize );
#ifdef PRINT_INFO
        if ( cudaStatus == cudaSuccess )
            printf ( "GPU %d device memory size = %lf GiB\n", gpu_index, ( double ) dev_memsize / ( 0x400 * 0x400 * 0x400 ) );
        else
            printf ( "GPU %d cudaMalloc fail\n", gpu_index );
#endif
    }

    numSparseMatrix = argc - 1;
    files = ( FILE ** ) malloc ( sizeof( FILE * ) * numSparseMatrix );

    matrix_info = malloc ( sizeof ( struct matrix_info_struct ) * numSparseMatrix );

    // Free resources

    for ( gpu_index = 0; gpu_index < numGPU; gpu_index++ )
    {
        cudaSetDevice ( gpu_index );
        cudaFree ( gpu_info[gpu_index].dev_mem );
    }

    free ( gpu_info );
    free ( matrix_info );
    free ( files );

    return 0;
}
