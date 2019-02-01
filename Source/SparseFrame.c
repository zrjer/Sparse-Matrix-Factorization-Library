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
#ifdef PRINT_INFO
                printf ( "\n" );
#endif

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

    free ( *gpu_info_ptr );
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

int SparseFrame_read_matrix_triplet ( char **buf_ptr, struct matrix_info_struct *matrix_info )
{
    char s0[max_mm_line_size];
    char s1[max_mm_line_size];
    char s2[max_mm_line_size];
    char s3[max_mm_line_size];
    char s4[max_mm_line_size];

    int n_scanned;
    uLong ncol, nrow, nzmax;
    uLong Tj, Ti;
    Float Tx, Ty;

    uLong nz;

    while ( ( getline ( buf_ptr, &max_mm_line_size, matrix_info->file ) != -1 ) && ( strcmp (*buf_ptr, "") == 0 ) );

    if ( strncmp ( *buf_ptr, "%%MatrixMarket", 14 ) != 0 )
    {
        printf ("Matrix format error\n\n");
        return 0;
    }

    sscanf ( *buf_ptr, "%s %s %s %s %s\n", s0, s1, s2, s3, s4 );

    if ( strcmp ( s3, "real" ) == 0 )
        matrix_info->isComplex = 0;
    else
        matrix_info->isComplex = 1;

    while ( ( getline ( buf_ptr, &max_mm_line_size, matrix_info->file ) != -1 ) && ( ( strcmp (*buf_ptr, "") == 0 ) || ( strncmp (*buf_ptr, "%", 1) == 0 ) ) );

    n_scanned = sscanf ( *buf_ptr, "%ld %ld %ld\n", &ncol, &nrow, &nzmax );

    if (n_scanned != 3)
    {
        printf ("Matrix format error\n\n");
        return 0;
    }

#ifdef PRINT_INFO
    if ( matrix_info->isComplex == 0 )
        printf ("matrix is real, ncol = %ld nrow = %ld nzmax = %ld\n", ncol, nrow, nzmax);
    else
        printf ("matrix is complex, ncol = %ld nrow = %ld nzmax = %ld\n", ncol, nrow, nzmax);
#endif

    matrix_info->Tj = malloc ( sizeof(uLong) * nzmax );
    matrix_info->Ti = malloc ( sizeof(uLong) * nzmax );
    matrix_info->Tx = malloc ( sizeof(Float) * nzmax );
    if ( matrix_info->isComplex )
        matrix_info->Ty = malloc ( sizeof(Float) * nzmax );
    else
        matrix_info->Ty = NULL;

    nz = 0;

    while ( ( getline ( buf_ptr, &max_mm_line_size, matrix_info->file ) != -1 ) )
    {
        if ( strcmp (*buf_ptr, "") != 0 )
        {
            if ( nz >= nzmax )
            {
                printf ( "Error: nzmax exceeded\n" );
                return 0;
            }
            n_scanned = sscanf ( *buf_ptr, "%ld %ld %lg %lg\n", &Tj, &Ti, &Tx, &Ty );
            if ( matrix_info->isComplex && n_scanned < 4 )
            {
                printf ( "Error: imaginary part not present\n" );
                return 0;
            }
            matrix_info->Tj[nz] = Tj;
            matrix_info->Ti[nz] = Ti;
            matrix_info->Tx[nz] = Tx;
            if ( matrix_info->isComplex )
                matrix_info->Ty[nz] = Ty;
            nz++;
        }
    }

    return 1;
}

int SparseFrame_compress ( struct matrix_info_struct *matrix_info )
{
    uLong nz, nzmax;
    uLong p;
    uLong *workspace;

    nzmax = matrix_info->nzmax;

    matrix_info->Cp = calloc ( 0, sizeof(uLong) * ( nzmax + 1 ) );
    matrix_info->Ci = malloc ( sizeof(uLong) * nzmax );
    matrix_info->Cx = malloc ( sizeof(Float) * nzmax );
    if ( matrix_info->isComplex )
        matrix_info->Cy = malloc ( sizeof(Float) * nzmax );
    else
        matrix_info->Cy = NULL;

    for ( nz = 0; nz < nzmax; nz++ )
        matrix_info->Cp[ matrix_info->Tj[nz] + 1 ] ++;

    for ( nz = 0; nz < nzmax; nz++ )
        matrix_info->Cp[nz+1] += matrix_info->Cp[nz];

    workspace = malloc ( sizeof(uLong) * nzmax );
    memcpy ( workspace, matrix_info->Cp, sizeof(uLong) * nzmax );

    for ( nz = 0; nz < nzmax; nz++ )
    {
        p = workspace [ matrix_info->Tj [ nz ] ] ++;
        matrix_info->Ci [p] = matrix_info->Ti[nz];
        matrix_info->Cx [p] = matrix_info->Tx[nz];
        if ( matrix_info->isComplex )
            matrix_info->Cy [p] = matrix_info->Ty[nz];
    }

    if ( matrix_info->Tj != NULL) free ( matrix_info->Tj );
    if ( matrix_info->Ti != NULL) free ( matrix_info->Ti );
    if ( matrix_info->Tx != NULL) free ( matrix_info->Tx );
    if ( matrix_info->Ty != NULL) free ( matrix_info->Ty );

    matrix_info->Tj = NULL;
    matrix_info->Ti = NULL;
    matrix_info->Tx = NULL;
    matrix_info->Ty = NULL;

    return 1;
}

int SparseFrame_cleanup_matrix ( struct matrix_info_struct *matrix_info )
{
    if ( matrix_info->Tj != NULL) free ( matrix_info->Tj );
    if ( matrix_info->Ti != NULL) free ( matrix_info->Ti );
    if ( matrix_info->Tx != NULL) free ( matrix_info->Tx );
    if ( matrix_info->Ty != NULL) free ( matrix_info->Ty );

    if ( matrix_info->Cp != NULL) free ( matrix_info->Cp );
    if ( matrix_info->Ci != NULL) free ( matrix_info->Ci );
    if ( matrix_info->Cx != NULL) free ( matrix_info->Cx );
    if ( matrix_info->Cy != NULL) free ( matrix_info->Cy );

    matrix_info->ncol = 0;
    matrix_info->nrow = 0;
    matrix_info->nzmax = 0;
    matrix_info->Tj = NULL;
    matrix_info->Ti = NULL;
    matrix_info->Tx = NULL;
    matrix_info->Ty = NULL;
    matrix_info->Cp = NULL;
    matrix_info->Ci = NULL;
    matrix_info->Cx = NULL;
    matrix_info->Cy = NULL;

    return 1;
}

int SparseFrame_read_matrix ( char *path, struct matrix_info_struct *matrix_info )
{
    char *buf;

    matrix_info->file = fopen ( path, "r" );

    if ( matrix_info->file == NULL ) return 0;

    buf = malloc ( sizeof(char) * max_mm_line_size );

    SparseFrame_read_matrix_triplet ( &buf, matrix_info );

    fclose ( matrix_info->file );

    free ( buf );

    SparseFrame_cleanup_matrix ( matrix_info );

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

        SparseFrame_read_matrix ( argv [ 1 + matrixIndex ], matrix_info + matrixIndex );

        // Factorize

        // Solve

    }

    // Free resources

    SparseFrame_free_gpu (&gpu_info, common_info);

    SparseFrame_free_matrix ( &matrix_info, common_info );

    return 0;
}
