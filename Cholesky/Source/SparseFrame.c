#include "SparseFrame.h"

double SparseFrame_time ()
{
    struct timespec tp;

#ifdef PRINT_CALLS
    printf ("\n================SparseFrame_time================\n\n");
#endif

    clock_gettime ( CLOCK_REALTIME, &tp );

    return ( tp.tv_sec + ( double ) ( tp.tv_nsec ) / 1.0e9 );
}

int SparseFrame_allocate_gpu ( struct common_info_struct *common_info, struct gpu_info_struct **gpu_info_list_ptr )
{
    int numCPU, numGPU, numGPU_physical, numSplit;

    size_t minDevMemSize, minHostMemSize;

    size_t devSlotSize;

#ifdef PRINT_CALLS
    printf ("\n================SparseFrame_allocate_gpu================\n\n");
#endif

    numGPU_physical = 0;
    cudaGetDeviceCount ( &numGPU_physical );
#if ( defined ( MAX_NUM_GPU ) && ( MAX_NUM_GPU >= 0 ) )
    numGPU_physical = MIN ( numGPU_physical, MAX_NUM_GPU );
#endif

    common_info->numGPU_physical = numGPU_physical;

    numSplit = 1;
#if ( defined ( GPU_SPLIT_LIMIT ) && ( GPU_SPLIT_LIMIT > 1 ) )
    numSplit = MAX ( ( GPU_SPLIT_LIMIT / numGPU_physical ), 1 );
#endif

    numGPU = numGPU_physical * numSplit;

    common_info->numGPU = numGPU;

    numCPU = sysconf(_SC_NPROCESSORS_ONLN);
#if ( defined ( MAX_NUM_CPU ) && ( MAX_NUM_CPU >= 0 ) )
    numCPU = MIN ( MAX ( numCPU - numGPU, 0 ), MAX_NUM_CPU );
#endif

    if ( numCPU <= 0 && numGPU_physical <= 0 )
        numCPU = 1;

    common_info->numCPU = numCPU;

#ifdef PRINT_INFO
    printf ( "Num of CPU = %d\n", numCPU );
    printf ( "Num of GPU = %d\n", numGPU_physical );
#endif

    *gpu_info_list_ptr = malloc ( ( numGPU + numCPU ) * sizeof ( struct gpu_info_struct ) );

    if ( *gpu_info_list_ptr == NULL ) return 1;

    minDevMemSize = ( numGPU_physical > 0 ) ? SIZE_MAX : 0;
    minHostMemSize = ( numGPU_physical > 0 ) ? SIZE_MAX : 0;

    for ( int gpuIndex_physical = 0; gpuIndex_physical < numGPU_physical; gpuIndex_physical++ )
    {
        struct cudaDeviceProp prop;
        size_t devMemSize;
        size_t hostMemSize;
        size_t sharedMemSize;

        cudaError_t cudaStatus;

        cudaSetDevice ( gpuIndex_physical );
        cudaGetDeviceProperties ( &prop, gpuIndex_physical );

        sharedMemSize = prop.sharedMemPerBlock;

        devMemSize = prop.totalGlobalMem;
        devMemSize = ( size_t ) ( ( ( double ) devMemSize - 64 * ( 0x400 * 0x400 ) ) * 0.9 );
        devMemSize /= numSplit;
        devMemSize /= ( A_MULTIPLE + 1 + B_MULTIPLE + C_MULTIPLE + 1 );
        devMemSize = devMemSize - devMemSize % ( 0x400 * 0x400 ); // align to 1 MB
        hostMemSize = devMemSize * ( A_MULTIPLE + B_MULTIPLE + 1 );
        devMemSize = devMemSize * ( A_MULTIPLE + 1 + B_MULTIPLE + C_MULTIPLE + 1 );

        for ( int gpuIndex = gpuIndex_physical; gpuIndex < numGPU; gpuIndex += numGPU_physical )
        {
            (*gpu_info_list_ptr)[gpuIndex].gpuIndex_physical = gpuIndex_physical;
            cudaStatus = cudaMalloc ( &( (*gpu_info_list_ptr)[gpuIndex].devMem ), devMemSize );

            if ( cudaStatus == cudaSuccess )
            {
                cudaStatus = cudaMallocHost ( &( (*gpu_info_list_ptr)[gpuIndex].hostMem ), hostMemSize );

                if ( cudaStatus == cudaSuccess )
                {
                    omp_init_lock ( &( (*gpu_info_list_ptr)[gpuIndex].gpuLock ) );
                    (*gpu_info_list_ptr)[gpuIndex].devMemSize = devMemSize;
                    (*gpu_info_list_ptr)[gpuIndex].hostMemSize = hostMemSize;
                    (*gpu_info_list_ptr)[gpuIndex].sharedMemSize = sharedMemSize;

                    cudaEventCreateWithFlags ( &( (*gpu_info_list_ptr)[gpuIndex].s_cudaEvent_factorized ), cudaEventDisableTiming );
                    for ( int k = 0; k < MAX_D_STREAM; k++ )
                    {
                        cudaEventCreateWithFlags ( &( (*gpu_info_list_ptr)[gpuIndex].d_cudaEvent_onDevice[k] ), cudaEventDisableTiming );
                        cudaEventCreateWithFlags ( &( (*gpu_info_list_ptr)[gpuIndex].d_cudaEvent_applied[k] ), cudaEventDisableTiming );
                    }
                    cudaStreamCreate ( &( (*gpu_info_list_ptr)[gpuIndex].s_cudaStream ) );
                    cublasCreate ( &( (*gpu_info_list_ptr)[gpuIndex].s_cublasHandle ) );
                    cublasSetStream ( (*gpu_info_list_ptr)[gpuIndex].s_cublasHandle, (*gpu_info_list_ptr)[gpuIndex].s_cudaStream );
                    cusolverDnCreate ( &( (*gpu_info_list_ptr)[gpuIndex].s_cusolverDnHandle ) );
                    cusolverDnSetStream ( (*gpu_info_list_ptr)[gpuIndex].s_cusolverDnHandle, (*gpu_info_list_ptr)[gpuIndex].s_cudaStream );
                    cudaStreamCreate ( &( (*gpu_info_list_ptr)[gpuIndex].s_cudaStream_copyback ) );
                    for ( int k = 0; k < MAX_D_STREAM; k++ )
                    {
                        cudaStreamCreate ( &( (*gpu_info_list_ptr)[gpuIndex].d_cudaStream[k] ) );
                        cublasCreate ( &( (*gpu_info_list_ptr)[gpuIndex].d_cublasHandle[k] ) );
                        cublasSetStream ( (*gpu_info_list_ptr)[gpuIndex].d_cublasHandle[k], (*gpu_info_list_ptr)[gpuIndex].d_cudaStream[k] );
                        cudaStreamCreate ( &( (*gpu_info_list_ptr)[gpuIndex].d_cudaStream_copy[k] ) );
                    }

                    if ( minDevMemSize > devMemSize )
                        minDevMemSize = devMemSize;

                    if ( minHostMemSize > hostMemSize )
                        minHostMemSize = hostMemSize;

#ifdef PRINT_INFO
                    printf ( "GPU %d device handler %d device memory size = %lf GiB host memory size = %lf GiB shared memory size per block = %ld KiB\n",
                            gpuIndex_physical, gpuIndex, ( double ) devMemSize / ( 0x400 * 0x400 * 0x400 ), ( double ) hostMemSize / ( 0x400 * 0x400 * 0x400 ), sharedMemSize / 1024 );
#endif
                }
                else
                {
                    (*gpu_info_list_ptr)[gpuIndex].devMem = NULL;
                    (*gpu_info_list_ptr)[gpuIndex].devMemSize = 0;
                    (*gpu_info_list_ptr)[gpuIndex].hostMem = NULL;
                    (*gpu_info_list_ptr)[gpuIndex].hostMemSize = 0;
                    (*gpu_info_list_ptr)[gpuIndex].sharedMemSize = 0;
                    (*gpu_info_list_ptr)[gpuIndex].s_cudaEvent_factorized = 0;
                    for ( int k = 0; k < MAX_D_STREAM; k++ )
                    {
                        (*gpu_info_list_ptr)[gpuIndex].d_cudaEvent_onDevice[k] = 0;
                        (*gpu_info_list_ptr)[gpuIndex].d_cudaEvent_applied[k] = 0;
                    }
                    (*gpu_info_list_ptr)[gpuIndex].s_cudaStream = 0;
                    (*gpu_info_list_ptr)[gpuIndex].s_cublasHandle = 0;
                    (*gpu_info_list_ptr)[gpuIndex].s_cusolverDnHandle = 0;
                    (*gpu_info_list_ptr)[gpuIndex].s_cudaStream_copyback = 0;
                    for ( int k = 0; k < MAX_D_STREAM; k++ )
                    {
                        (*gpu_info_list_ptr)[gpuIndex].d_cudaStream[k] = 0;
                        (*gpu_info_list_ptr)[gpuIndex].d_cublasHandle[k] = 0;
                        (*gpu_info_list_ptr)[gpuIndex].d_cudaStream_copy[k] = 0;
                    }
#ifdef PRINT_INFO
                    printf ( "GPU %d device handler %d cudaMallocHost fail\n", gpuIndex_physical, gpuIndex );
#endif
                }
            }
            else
            {
                (*gpu_info_list_ptr)[gpuIndex].devMem = NULL;
                (*gpu_info_list_ptr)[gpuIndex].devMemSize = 0;
                (*gpu_info_list_ptr)[gpuIndex].hostMem = NULL;
                (*gpu_info_list_ptr)[gpuIndex].hostMemSize = 0;
                (*gpu_info_list_ptr)[gpuIndex].sharedMemSize = 0;
                (*gpu_info_list_ptr)[gpuIndex].s_cudaEvent_factorized = 0;
                for ( int k = 0; k < MAX_D_STREAM; k++ )
                {
                    (*gpu_info_list_ptr)[gpuIndex].d_cudaEvent_onDevice[k] = 0;
                    (*gpu_info_list_ptr)[gpuIndex].d_cudaEvent_applied[k] = 0;
                }
                (*gpu_info_list_ptr)[gpuIndex].s_cudaStream = 0;
                (*gpu_info_list_ptr)[gpuIndex].s_cublasHandle = 0;
                (*gpu_info_list_ptr)[gpuIndex].s_cusolverDnHandle = 0;
                (*gpu_info_list_ptr)[gpuIndex].s_cudaStream_copyback = 0;
                for ( int k = 0; k < MAX_D_STREAM; k++ )
                {
                    (*gpu_info_list_ptr)[gpuIndex].d_cudaStream[k] = 0;
                    (*gpu_info_list_ptr)[gpuIndex].d_cublasHandle[k] = 0;
                    (*gpu_info_list_ptr)[gpuIndex].d_cudaStream_copy[k] = 0;
                }
#ifdef PRINT_INFO
                printf ( "GPU %d device handler %d cudaMalloc fail\n", gpuIndex_physical, gpuIndex );
#endif
            }

            (*gpu_info_list_ptr)[gpuIndex].lastMatrix = -1;
        }
    }

    common_info->minDevMemSize = minDevMemSize;
    common_info->minHostMemSize = minHostMemSize;

    devSlotSize = ( common_info->minDevMemSize ) / ( A_MULTIPLE + 1 + B_MULTIPLE + C_MULTIPLE + 1 );
    common_info->devSlotSize = devSlotSize;

    for ( int gpuIndex = 0; gpuIndex < numGPU; gpuIndex++ )
    {
        for ( int k = 0; k < A_MULTIPLE; k++ )
            (*gpu_info_list_ptr)[gpuIndex].h_A[k] = (*gpu_info_list_ptr)[gpuIndex].hostMem + k * devSlotSize;

        for ( int k = 0; k < B_MULTIPLE; k++ )
            (*gpu_info_list_ptr)[gpuIndex].h_B[k] = (*gpu_info_list_ptr)[gpuIndex].hostMem + ( A_MULTIPLE + k ) * devSlotSize;

        (*gpu_info_list_ptr)[gpuIndex].h_Lsi = (*gpu_info_list_ptr)[gpuIndex].hostMem + ( A_MULTIPLE + B_MULTIPLE ) * devSlotSize;

        for ( int k = 0; k < A_MULTIPLE + 1; k++ )
            (*gpu_info_list_ptr)[gpuIndex].d_A[k] = (*gpu_info_list_ptr)[gpuIndex].devMem + k * devSlotSize;

        for ( int k = 0; k < B_MULTIPLE; k++ )
            (*gpu_info_list_ptr)[gpuIndex].d_B[k] = (*gpu_info_list_ptr)[gpuIndex].devMem + ( A_MULTIPLE + 1 + k ) * devSlotSize;

        for ( int k = 0; k < C_MULTIPLE; k++ )
            (*gpu_info_list_ptr)[gpuIndex].d_C[k] = (*gpu_info_list_ptr)[gpuIndex].devMem + ( A_MULTIPLE + 1 + B_MULTIPLE + k ) * devSlotSize;

        (*gpu_info_list_ptr)[gpuIndex].d_Lsi = (*gpu_info_list_ptr)[gpuIndex].devMem + ( A_MULTIPLE + 1 + B_MULTIPLE + C_MULTIPLE ) * devSlotSize;
    }

    for ( int gpuIndex = numGPU; gpuIndex < numGPU + numCPU; gpuIndex++ )
    {
        for ( int k = 0; k < A_MULTIPLE; k++ )
            (*gpu_info_list_ptr)[gpuIndex].h_A[k] = NULL;

        for ( int k = 0; k < B_MULTIPLE; k++ )
            (*gpu_info_list_ptr)[gpuIndex].h_B[k] = NULL;

        (*gpu_info_list_ptr)[gpuIndex].h_Lsi = NULL;

        for ( int k = 0; k < A_MULTIPLE + 1; k++ )
            (*gpu_info_list_ptr)[gpuIndex].d_A[k] = NULL;

        for ( int k = 0; k < B_MULTIPLE; k++ )
            (*gpu_info_list_ptr)[gpuIndex].d_B[k] = NULL;

        for ( int k = 0; k < C_MULTIPLE; k++ )
            (*gpu_info_list_ptr)[gpuIndex].d_C[k] = NULL;

        (*gpu_info_list_ptr)[gpuIndex].d_Lsi = NULL;
    }

    for ( int gpuIndex = numGPU; gpuIndex < numGPU + numCPU; gpuIndex++ )
    {
        (*gpu_info_list_ptr)[gpuIndex].gpuIndex_physical = -1;
        omp_init_lock ( &( (*gpu_info_list_ptr)[gpuIndex].gpuLock ) );
        (*gpu_info_list_ptr)[gpuIndex].devMem = NULL;
        (*gpu_info_list_ptr)[gpuIndex].devMemSize = 0;
        (*gpu_info_list_ptr)[gpuIndex].hostMem = NULL;
        (*gpu_info_list_ptr)[gpuIndex].hostMemSize = 0;
        (*gpu_info_list_ptr)[gpuIndex].sharedMemSize = 0;
        (*gpu_info_list_ptr)[gpuIndex].s_cudaEvent_factorized = 0;
        for ( int k = 0; k < MAX_D_STREAM; k++ )
        {
            (*gpu_info_list_ptr)[gpuIndex].d_cudaEvent_onDevice[k] = 0;
            (*gpu_info_list_ptr)[gpuIndex].d_cudaEvent_applied[k] = 0;
        }
        (*gpu_info_list_ptr)[gpuIndex].s_cudaStream = 0;
        (*gpu_info_list_ptr)[gpuIndex].s_cublasHandle = 0;
        (*gpu_info_list_ptr)[gpuIndex].s_cusolverDnHandle = 0;
        (*gpu_info_list_ptr)[gpuIndex].s_cudaStream_copyback = 0;
        for ( int k = 0; k < MAX_D_STREAM; k++ )
        {
            (*gpu_info_list_ptr)[gpuIndex].d_cudaStream[k] = 0;
            (*gpu_info_list_ptr)[gpuIndex].d_cublasHandle[k] = 0;
            (*gpu_info_list_ptr)[gpuIndex].d_cudaStream_copy[k] = 0;
        }
#ifdef PRINT_INFO
        printf ( "CPU %d device handler %d ( pretended GPU )\n", gpuIndex - numGPU, gpuIndex);
#endif
    }

#ifdef PRINT_INFO
    printf ( "Minimum device memory size = %lf GiB, pinned host memory size = %lf\n", ( double ) minDevMemSize / ( 0x400 * 0x400 * 0x400 ), ( double ) minHostMemSize / ( 0x400 * 0x400 * 0x400 ) );
#endif

#ifdef PRINT_INFO
    printf ("\n");
#endif

    return 0;
}

int SparseFrame_free_gpu ( struct common_info_struct *common_info, struct gpu_info_struct **gpu_info_list_ptr )
{
    int numGPU;

#ifdef PRINT_CALLS
    printf ("\n================SparseFrame_free_gpu================\n\n");
#endif

    if ( *gpu_info_list_ptr == NULL ) return 1;

    numGPU = common_info->numGPU;

    for ( int gpuIndex = 0; gpuIndex < numGPU; gpuIndex++ )
    {
        int gpuIndex_physical = (*gpu_info_list_ptr)[gpuIndex].gpuIndex_physical;

        if ( gpuIndex_physical >= 0 )
            cudaSetDevice ( gpuIndex_physical );

        if ( (*gpu_info_list_ptr)[gpuIndex].devMem != NULL )
            cudaFree ( (*gpu_info_list_ptr)[gpuIndex].devMem );
        if ( (*gpu_info_list_ptr)[gpuIndex].hostMem != NULL )
            cudaFreeHost ( (*gpu_info_list_ptr)[gpuIndex].hostMem );
        if ( (*gpu_info_list_ptr)[gpuIndex].s_cudaEvent_factorized != 0 )
            cudaEventDestroy ( (*gpu_info_list_ptr)[gpuIndex].s_cudaEvent_factorized );
        for ( int k = 0; k < MAX_D_STREAM; k++ )
        {
            if ( (*gpu_info_list_ptr)[gpuIndex].d_cudaEvent_onDevice[k] != 0 )
                cudaEventDestroy ( (*gpu_info_list_ptr)[gpuIndex].d_cudaEvent_onDevice[k] );
            if ( (*gpu_info_list_ptr)[gpuIndex].d_cudaEvent_applied[k] != 0 )
                cudaEventDestroy ( (*gpu_info_list_ptr)[gpuIndex].d_cudaEvent_applied[k] );
        }
        if ( (*gpu_info_list_ptr)[gpuIndex].s_cusolverDnHandle != 0 )
            cusolverDnDestroy ( (*gpu_info_list_ptr)[gpuIndex].s_cusolverDnHandle );
        if ( (*gpu_info_list_ptr)[gpuIndex].s_cublasHandle != 0 )
            cublasDestroy ( (*gpu_info_list_ptr)[gpuIndex].s_cublasHandle );
        if ( (*gpu_info_list_ptr)[gpuIndex].s_cudaStream != 0 )
            cudaStreamDestroy ( (*gpu_info_list_ptr)[gpuIndex].s_cudaStream );
        if ( (*gpu_info_list_ptr)[gpuIndex].s_cudaStream_copyback != 0 )
            cudaStreamDestroy ( (*gpu_info_list_ptr)[gpuIndex].s_cudaStream_copyback );
        for ( int k = 0; k < MAX_D_STREAM; k++ )
        {
            if ( (*gpu_info_list_ptr)[gpuIndex].d_cublasHandle[k] != 0 )
                cublasDestroy ( (*gpu_info_list_ptr)[gpuIndex].d_cublasHandle[k] );
            if ( (*gpu_info_list_ptr)[gpuIndex].d_cudaStream[k] != 0 )
                cudaStreamDestroy ( (*gpu_info_list_ptr)[gpuIndex].d_cudaStream[k] );
            if ( (*gpu_info_list_ptr)[gpuIndex].d_cudaStream_copy[k] != 0 )
                cudaStreamDestroy ( (*gpu_info_list_ptr)[gpuIndex].d_cudaStream_copy[k] );
        }

        omp_destroy_lock ( &( (*gpu_info_list_ptr)[gpuIndex].gpuLock ) );
        (*gpu_info_list_ptr)[gpuIndex].devMem = NULL;
        (*gpu_info_list_ptr)[gpuIndex].devMemSize = 0;
        (*gpu_info_list_ptr)[gpuIndex].hostMem = NULL;
        (*gpu_info_list_ptr)[gpuIndex].hostMemSize = 0;
        (*gpu_info_list_ptr)[gpuIndex].s_cudaEvent_factorized = 0;
        for ( int k = 0; k < MAX_D_STREAM; k++ )
        {
            (*gpu_info_list_ptr)[gpuIndex].d_cudaEvent_onDevice[k] = 0;
            (*gpu_info_list_ptr)[gpuIndex].d_cudaEvent_applied[k] = 0;
        }
        (*gpu_info_list_ptr)[gpuIndex].s_cudaStream = 0;
        (*gpu_info_list_ptr)[gpuIndex].s_cublasHandle = 0;
        (*gpu_info_list_ptr)[gpuIndex].s_cusolverDnHandle = 0;
        (*gpu_info_list_ptr)[gpuIndex].s_cudaStream_copyback = 0;
        for ( int k = 0; k < MAX_D_STREAM; k++ )
        {
            (*gpu_info_list_ptr)[gpuIndex].d_cudaStream[k] = 0;
            (*gpu_info_list_ptr)[gpuIndex].d_cublasHandle[k] = 0;
            (*gpu_info_list_ptr)[gpuIndex].d_cudaStream_copy[k] = 0;
        }
    }

    free ( *gpu_info_list_ptr );
    *gpu_info_list_ptr = NULL;

    common_info->numGPU = 0;

    return 0;
}

int SparseFrame_allocate_matrix ( struct common_info_struct *common_info, struct matrix_info_struct **matrix_info_list_ptr )
{
    int matrixThreadNum;

#ifdef PRINT_CALLS
    printf ("\n================SparseFrame_allocate_matrix================\n\n");
#endif

    matrixThreadNum = common_info->matrixThreadNum;

    *matrix_info_list_ptr = malloc ( matrixThreadNum * sizeof ( struct matrix_info_struct ) );

    if ( *matrix_info_list_ptr == NULL ) return 1;

    return 0;
}

int SparseFrame_free_matrix ( struct common_info_struct *common_info, struct matrix_info_struct **matrix_info_list_ptr )
{
#ifdef PRINT_CALLS
    printf ("\n================SparseFrame_free_matrix================\n\n");
#endif

    if ( *matrix_info_list_ptr == NULL ) return 1;

    free ( *matrix_info_list_ptr );

    common_info->numSparseMatrix = 0;

    return 0;
}

int SparseFrame_read_matrix_triplet ( char **buf_ptr, struct matrix_info_struct *matrix_info )
{
    size_t max_mm_line_size = MAXMMLINE + 1;

    char s0[ MAXMMLINE + 1 ];
    char s1[ MAXMMLINE + 1 ];
    char s2[ MAXMMLINE + 1 ];
    char s3[ MAXMMLINE + 1 ];
    char s4[ MAXMMLINE + 1 ];

    int n_scanned;

    int isSymmetric, isComplex;
    Long ncol, nrow, nzmax;
    Long Tj, Ti;
    Float Tx, Ty;

    Long nz;

#ifdef PRINT_CALLS
    printf ("\n================SparseFrame_read_matrix_triplet================\n\n");
#endif

    while ( ( getline ( buf_ptr, &max_mm_line_size, matrix_info->file ) != -1 ) && ( strcmp (*buf_ptr, "") == 0 ) );

    if ( strncmp ( *buf_ptr, "%%MatrixMarket", 14 ) != 0 )
    {
        printf ("Matrix format error\n\n");
        return 1;
    }

    sscanf ( *buf_ptr, "%s %s %s %s %s\n", s0, s1, s2, s3, s4 );

    if ( strcmp ( s4, "symmetric" ) == 0 )
        isSymmetric = TRUE;
    else
        isSymmetric = FALSE;

    matrix_info->isSymmetric = isSymmetric;

    if ( strcmp ( s3, "real" ) == 0 )
        isComplex = FALSE;
    else
        isComplex = TRUE;

    matrix_info->isComplex = isComplex;

    while ( ( getline ( buf_ptr, &max_mm_line_size, matrix_info->file ) != -1 ) && ( ( strcmp (*buf_ptr, "") == 0 ) || ( strncmp (*buf_ptr, "%", 1) == 0 ) ) );

    n_scanned = sscanf ( *buf_ptr, "%ld %ld %ld\n", &nrow, &ncol, &nzmax );

    if ( n_scanned != 3 )
    {
        printf ("Matrix format error\n\n");
        matrix_info->ncol = 0;
        matrix_info->nrow = 0;
        matrix_info->nzmax = 0;
        return 1;
    }

#ifdef PRINT_INFO
    printf ( "matrix %s is ", basename( (char*) ( matrix_info->path ) ) );
    if ( isSymmetric && !isComplex )
        printf ("real symmetric, ncol = %ld nrow = %ld nzmax = %ld\n\n", ncol, nrow, nzmax);
    else if ( isSymmetric && isComplex )
        printf ("complex symmetric, ncol = %ld nrow = %ld nzmax = %ld\n\n", ncol, nrow, nzmax);
    else if ( !isSymmetric && !isComplex )
        printf ("real unsymmetric, ncol = %ld nrow = %ld nzmax = %ld\n\n", ncol, nrow, nzmax);
    else
        printf ("complex unsymmetric, ncol = %ld nrow = %ld nzmax = %ld\n\n", ncol, nrow, nzmax);
#endif

    matrix_info->Tj = malloc ( nzmax * sizeof(Long) );
    matrix_info->Ti = malloc ( nzmax * sizeof(Long) );
    if ( !isComplex )
        matrix_info->Tx = malloc ( nzmax * sizeof(Float) );
    else
        matrix_info->Tx = malloc ( nzmax * sizeof(Complex) );

    nz = 0;

    while ( ( getline ( buf_ptr, &max_mm_line_size, matrix_info->file ) != -1 ) )
    {
        if ( strcmp (*buf_ptr, "") != 0 )
        {
            n_scanned = sscanf ( *buf_ptr, "%ld %ld %lg %lg\n", &Ti, &Tj, &Tx, &Ty );
            if ( n_scanned < 3 )
            {
                printf ( "Error: invalid matrix entry\n" );
                return 1;
            }
            if ( isComplex && n_scanned < 4 )
            {
                printf ( "Error: imaginary part not present\n" );
                return 1;
            }
            if ( Tx != 0 || ( n_scanned >= 4 && Ty != 0 ) )
            {
                if ( nz >= nzmax )
                {
                    printf ( "Error: nzmax exceeded\n" );
                    return 1;
                }
                matrix_info->Tj[nz] = Tj - 1;
                matrix_info->Ti[nz] = Ti - 1;
                if ( !isComplex )
                    matrix_info->Tx[nz] = Tx;
                else
                {
                    ( (Complex*) (matrix_info->Tx) )[nz].x = Tx;
                    ( (Complex*) (matrix_info->Tx) )[nz].y = Ty;
                }
                nz++;
            }
        }
    }

    nzmax = nz;

    matrix_info->ncol = ncol;
    matrix_info->nrow = nrow;
    matrix_info->nzmax = nzmax;

    return 0;
}

int SparseFrame_compress ( struct matrix_info_struct *matrix_info )
{
    int isComplex;
    Long ncol, nzmax;
    Long *Tj, *Ti, *Cp, *Ci;
    Float *Tx, *Cx;

    Long *workspace;

#ifdef PRINT_CALLS
    printf ("\n================SparseFrame_compress================\n\n");
#endif

    isComplex = matrix_info->isComplex;
    ncol = matrix_info->ncol;
    nzmax = matrix_info->nzmax;

    Tj = matrix_info->Tj;
    Ti = matrix_info->Ti;
    Tx = matrix_info->Tx;

    Cp = calloc ( ( ncol + 1 ), sizeof(Long) );
    Ci = malloc ( nzmax * sizeof(Long) );
    if ( !isComplex )
        Cx = malloc ( nzmax * sizeof(Float) );
    else
        Cx = malloc ( nzmax * sizeof(Complex) );

    matrix_info->Cp = Cp;
    matrix_info->Ci = Ci;
    matrix_info->Cx = Cx;

    workspace = matrix_info->workspace;

    for ( Long nz = 0; nz < nzmax; nz++ )
        Cp[ Tj[nz] + 1 ] ++;

    for ( Long j = 0; j < ncol; j++ )
        Cp[j+1] += Cp[j];

    memcpy ( workspace, Cp, ncol * sizeof(Long) );

    for ( Long nz = 0; nz < nzmax; nz++ )
    {
        Long p = workspace [ Tj [ nz ] ] ++;
        Ci [p] = Ti[nz];
        if ( !isComplex )
            Cx [p] = Tx[nz];
        else
            ( (Complex*) Cx ) [p] = ( (Complex*) Tx ) [nz];
    }

    if ( Tj != NULL ) free ( Tj );
    if ( Ti != NULL ) free ( Ti );
    if ( Tx != NULL ) free ( Tx );

    Tj = matrix_info->Tj = NULL;
    Ti = matrix_info->Ti = NULL;
    Tx = matrix_info->Tx = NULL;

    return 0;
}

int SparseFrame_initialize_matrix ( struct matrix_info_struct *matrix_info )
{
#ifdef PRINT_CALLS
    printf ("\n================SparseFrame_initialize_matrix================\n\n");
#endif

    matrix_info->ncol = 0;
    matrix_info->nrow = 0;
    matrix_info->nzmax = 0;
    matrix_info->Tj = NULL;
    matrix_info->Ti = NULL;
    matrix_info->Tx = NULL;
    matrix_info->Cp = NULL;
    matrix_info->Ci = NULL;
    matrix_info->Cx = NULL;
    matrix_info->Lp = NULL;
    matrix_info->Li = NULL;
    matrix_info->Lx = NULL;
    matrix_info->LTp = NULL;
    matrix_info->LTi = NULL;
    matrix_info->LTx = NULL;

    matrix_info->Perm = NULL;
    matrix_info->Parent = NULL;
    matrix_info->Post = NULL;
    matrix_info->ColCount = NULL;
    matrix_info->RowCount = NULL;

    matrix_info->nsuper = 0;
    matrix_info->Super = NULL;
    matrix_info->SuperMap = NULL;
    matrix_info->Sparent = NULL;

    matrix_info->nsleaf = 0;
    matrix_info->LeafQueue = NULL;

    matrix_info->isize = 0;
    matrix_info->xsize = 0;
    matrix_info->Lsip = NULL;
    matrix_info->Lsxp = NULL;
    matrix_info->Lsi = NULL;
    matrix_info->Lsx = NULL;

    matrix_info->csize = 0;

    matrix_info->nstage = 0;
    matrix_info->ST_Map = NULL;
    matrix_info->ST_Pointer = NULL;
    matrix_info->ST_Index = NULL;
    matrix_info->ST_Parent = NULL;

    matrix_info->Aoffset = NULL;
    matrix_info->Moffset = NULL;

    matrix_info->workSize = 0;
    matrix_info->workspace = NULL;

    matrix_info->Bx = NULL;
    matrix_info->Xx = NULL;
    matrix_info->Rx = NULL;

    return 0;
}

int SparseFrame_read_matrix ( struct matrix_info_struct *matrix_info )
{
    double timestamp;

    const char *path;

    char *buf;

#ifdef PRINT_CALLS
    printf ("\n================SparseFrame_read_matrix================\n\n");
#endif

    timestamp = SparseFrame_time ();

    path = matrix_info->path;
    matrix_info->file = fopen ( path, "r" );

    if ( matrix_info->file == NULL ) return 1;

    buf = malloc ( ( MAXMMLINE + 1 ) * sizeof(char) );

    SparseFrame_read_matrix_triplet ( &buf, matrix_info );

    fclose ( matrix_info->file );

    free ( buf );

    matrix_info->workSize
        = MAX (
                ( 10 * matrix_info->nrow + ( 2 * matrix_info->nzmax - matrix_info->nrow ) + 1 ) * sizeof(Long),
                ( 3 * matrix_info->nrow + ( 2 * matrix_info->nzmax - matrix_info->nrow ) + 1 ) * sizeof(idx_t)
              );
    matrix_info->workspace = malloc ( matrix_info->workSize );

    SparseFrame_compress ( matrix_info );

    matrix_info->readTime = SparseFrame_time () - timestamp;

    return 0;
}

int SparseFrame_amd ( struct matrix_info_struct *matrix_info )
{
    Long ncol, nrow;
    Long *Cp, *Ci;
    Long *Head, *Next, *Perm;

    Long *workspace;
    Long anz;
    Long *Ap, *Ai, *Len, *Nv, *Elen, *Degree, *Wi;

    double Control[AMD_CONTROL], Info[AMD_INFO];

#ifdef PRINT_CALLS
    printf ("\n================SparseFrame_amd================\n\n");
#endif

    ncol = matrix_info->ncol;
    nrow = matrix_info->nrow;

    Cp = matrix_info->Cp;
    Ci = matrix_info->Ci;

    Perm = matrix_info->Perm;

    workspace = matrix_info->workspace;

    Head      = workspace +  0 * nrow;
    Next      = workspace +  1 * nrow;
    Len       = workspace +  2 * nrow;
    Nv        = workspace +  3 * nrow;
    Elen      = workspace +  4 * nrow;
    Degree    = workspace +  5 * nrow;
    Wi        = workspace +  8 * nrow;
    Ap        = workspace +  9 * nrow;
    Ai        = workspace + 10 * nrow + 1;

    memset ( Ap, 0, ( nrow + 1 ) * sizeof(Long) );

    for ( Long j = 0; j < ncol; j++ )
    {
        for ( Long p = Cp[j]; p < Cp[j+1]; p++ )
        {
            Long i = Ci[p];
            if (i > j)
            {
                Ap[i+1]++;
                Ap[j+1]++;
            }
        }
    }

    for ( Long j = 0; j < nrow; j++)
    {
        Ap[j+1] += Ap[j];
    }

    anz = Ap[nrow];

    memcpy ( workspace, Ap, nrow * sizeof(Long) ); // Be careful of overwriting Ap

    for ( Long j = 0; j < ncol; j++ )
    {
        for ( Long p = Cp[j]; p < Cp[j+1]; p++ )
        {
            Long i = Ci[p];
            if (i > j)
            {
                Ai [ workspace[i]++ ] = j;
                Ai [ workspace[j]++ ] = i;
            }
        }
    }

    for ( Long j = 0; j < matrix_info->nrow; j++ )
        Len[j] = Ap[j+1] - Ap[j];

    Control[AMD_DENSE] = prune_dense;
    Control[AMD_AGGRESSIVE] = aggressive;

    amd_l2 ( nrow, Ap, Ai, Len, anz, Ap[nrow], Nv, Next, Perm, Head, Elen, Degree, Wi, Control, Info );

    return 0;
}

int SparseFrame_camd ( struct matrix_info_struct *matrix_info )
{
    Long ncol, nrow;
    Long *Cp, *Ci;
    Long *Head, *Next, *Perm;

    Long *workspace;
    Long anz;
    Long *Ap, *Ai, *Len, *Nv, *Elen, *Degree, *Wi;
    Long *Cmember, *BucketSet;

    double Control[AMD_CONTROL], Info[AMD_INFO];

#ifdef PRINT_CALLS
    printf ("\n================SparseFrame_camd================\n\n");
#endif

    ncol = matrix_info->ncol;
    nrow = matrix_info->nrow;

    Cp = matrix_info->Cp;
    Ci = matrix_info->Ci;

    Perm = matrix_info->Perm;

    workspace = matrix_info->workspace;

    Head      = workspace +  0 * nrow;
    Next      = workspace +  1 * nrow;
    Len       = workspace +  2 * nrow;
    Nv        = workspace +  3 * nrow;
    Elen      = workspace +  4 * nrow;
    Degree    = workspace +  5 * nrow;
    Cmember   = workspace +  6 * nrow;
    BucketSet = workspace +  7 * nrow;
    Wi        = workspace +  8 * nrow;
    Ap        = workspace +  9 * nrow;
    Ai        = workspace + 10 * nrow + 1;

    memset ( Ap, 0, ( nrow + 1 ) * sizeof(Long) );

    for ( Long j = 0; j < ncol; j++ )
    {
        for ( Long p = Cp[j]; p < Cp[j+1]; p++ )
        {
            Long i = Ci[p];
            if (i > j)
            {
                Ap[i+1]++;
                Ap[j+1]++;
            }
        }
    }

    for ( Long j = 0; j < nrow; j++)
    {
        Ap[j+1] += Ap[j];
    }

    anz = Ap[nrow];

    memcpy ( workspace, Ap, nrow * sizeof(Long) ); // Be careful of overwriting Ap

    for ( Long j = 0; j < ncol; j++ )
    {
        for ( Long p = Cp[j]; p < Cp[j+1]; p++ )
        {
            Long i = Ci[p];
            if (i > j)
            {
                Ai [ workspace[i]++ ] = j;
                Ai [ workspace[j]++ ] = i;
            }
        }
    }

    for ( Long j = 0; j < matrix_info->nrow; j++ )
        Len[j] = Ap[j+1] - Ap[j];

    Control[AMD_DENSE] = prune_dense;
    Control[AMD_AGGRESSIVE] = aggressive;

    camd_l2 ( nrow, Ap, Ai, Len, anz, Ap[nrow], Nv, Next, Perm, Head, Elen, Degree, Wi, Control, Info, Cmember, BucketSet );

    return 0;
}

int SparseFrame_metis ( struct matrix_info_struct *matrix_info )
{
    Long ncol, nrow;
    Long *Cp, *Ci;
    Long *Perm;

    idx_t *Mworkspace;
    Long mnz;
    idx_t *Mp, *Mi, *Mperm, *Miperm;

#ifdef PRINT_CALLS
    printf ("\n================SparseFrame_metis================\n\n");
#endif

    ncol = matrix_info->ncol;
    nrow = matrix_info->nrow;

    Cp = matrix_info->Cp;
    Ci = matrix_info->Ci;

    Perm = matrix_info->Perm;

    Mworkspace = matrix_info->workspace;

    if ( sizeof(idx_t) == sizeof(Long) )
        Mperm = (idx_t*) Perm;
    else
        Mperm  = Mworkspace;
    Miperm = Mworkspace + 1 * nrow;
    Mp     = Mworkspace + 2 * nrow;
    Mi     = Mworkspace + 3 * nrow + 1;

    memset ( Mp, 0, ( nrow + 1 ) * sizeof(idx_t) );

    for ( Long j = 0; j < ncol; j++ )
    {
        for ( Long p = Cp[j]; p < Cp[j+1]; p++ )
        {
            Long i = Ci[p];
            if ( j < i )
            {
                Mp[j+1]++;
                Mp[i+1]++;
            }
        }
    }

    for ( Long j = 0; j < nrow; j++)
    {
        Mp[j+1] += Mp[j];
    }

    mnz = Mp[nrow];

    memcpy ( Mworkspace, Mp, nrow * sizeof(idx_t) ); // Be careful of overwriting Mp

    for ( Long j = 0; j < ncol; j++ )
    {
        for ( Long p = Cp[j]; p < Cp[j+1]; p++ )
        {
            Long i = Ci[p];
            if ( j < i )
            {
                Mi [ Mworkspace[j]++ ] = i;
                Mi [ Mworkspace[i]++ ] = j;
            }
        }
    }

    if ( mnz == 0 )
    {
        for ( Long i = 0; i < nrow; i++ )
        {
            Mperm[i] = i;
        }
    }
    else
    {
        METIS_NodeND ( (idx_t*) &nrow, Mp, Mi, NULL, NULL, Mperm, Miperm );
    }

    if ( sizeof(idx_t) != sizeof(Long) )
    {
        for ( Long i = 0; i < nrow; i++ )
        {
            Perm[i] = Mperm[i];
        }
    }

    return 0;
}

int SparseFrame_perm ( struct matrix_info_struct *matrix_info )
{
    int isComplex;
    Long nrow;
    Long *Cp, *Ci;
    Float *Cx;
    Long *Lp, *Li;
    Float *Lx;
    Long *LTp, *LTi;
    Float *LTx;
    Long *Perm, *Pinv;

    Long *Lworkspace, *LTworkspace;

#ifdef PRINT_CALLS
    printf ("\n================SparseFrame_perm================\n\n");
#endif

    isComplex = matrix_info->isComplex;

    nrow = matrix_info->nrow;

    Cp = matrix_info->Cp;
    Ci = matrix_info->Ci;
    Cx = matrix_info->Cx;

    Lp = matrix_info->Lp;
    Li = matrix_info->Li;
    Lx = matrix_info->Lx;

    LTp = matrix_info->LTp;
    LTi = matrix_info->LTi;
    LTx = matrix_info->LTx;

    Perm = matrix_info->Perm;
    Pinv = matrix_info->workspace;

    Lworkspace = matrix_info->workspace + 2 * nrow * sizeof(Long);
    LTworkspace = matrix_info->workspace + 3 * nrow * sizeof(Long);

    memset ( Lp, 0, ( nrow + 1 ) * sizeof(Long) );
    memset ( LTp, 0, ( nrow + 1 ) * sizeof(Long) );

    for ( Long j = 0; j < nrow; j++ )
    {
        Pinv[j] = -1;
    }

    for ( Long j = 0; j < nrow; j++ )
    {
        Long jold = Perm[j];
        if ( jold >= 0 )
            Pinv[ jold ] = j;
    }

    for ( Long j = 0; j < nrow; j++ )
    {
        Long jold = Perm[j];
        if ( jold >= 0 )
        {
            for ( Long pold = Cp[jold]; pold < Cp[jold+1]; pold++ )
            {
                Long iold = Ci[pold];
                Long i = Pinv[iold];

                Lp [ MIN (i, j) + 1 ] ++;
                LTp [ MAX (i, j) + 1 ] ++;
            }
        }
    }

    for ( Long j = 0; j < nrow; j++ )
    {
        Lp[j+1] += Lp[j];
        LTp[j+1] += LTp[j];
    }

    memcpy ( Lworkspace, Lp, nrow * sizeof(Long) );
    memcpy ( LTworkspace, LTp, nrow * sizeof(Long) );

    for ( Long j = 0; j < nrow; j++ )
    {
        Long jold = Perm[j];
        if ( jold >= 0 )
        {
            for ( Long pold = Cp[jold]; pold < Cp[jold+1]; pold++ )
            {
                Long lp, ltp;

                Long iold = Ci[pold];
                Long i = Pinv[iold];

                lp = Lworkspace [ MIN(i, j) ] ++;
                Li[lp] = MAX(i, j);
                if ( !isComplex )
                    Lx[lp] = Cx[pold];
                else
                    ( (Complex*) Lx ) [lp] = ( (Complex*) Cx ) [pold];

                ltp = LTworkspace [ MAX(i, j) ] ++;
                LTi[ltp] = MIN(i, j);
                if ( !isComplex )
                    LTx[ltp] = Cx[pold];
                else
                    ( (Complex*) LTx ) [ltp] = ( (Complex*) Cx ) [pold];
            }
        }
    }

    return 0;
}

int SparseFrame_etree ( struct matrix_info_struct *matrix_info )
{
    Long nrow;
    Long *LTp, *LTi;
    Long *Parent;

    Long *workspace;
    Long *Ancestor;

#ifdef PRINT_CALLS
    printf ("\n================SparseFrame_etree================\n\n");
#endif

    nrow = matrix_info->nrow;

    LTp = matrix_info->LTp;
    LTi = matrix_info->LTi;

    Parent = matrix_info->Parent;

    workspace = matrix_info->workspace;

    Ancestor = workspace;

    for ( Long j = 0; j < nrow; j++ )
    {
        Parent[j] = -1;
        Ancestor[j] = -1;
    }

    for ( Long j = 0; j < nrow; j++ )
    {
        for ( Long p = LTp[j]; p < LTp[j+1]; p++ )
        {
            Long i = LTi[p];
            if ( i < j )
            {
                Long ancestor;
                do
                {
                    ancestor = Ancestor[i];
                    if ( ancestor < 0 )
                    {
                        Parent[i] = j;
                        Ancestor[i] = j;
                    }
                    else if ( ancestor != j )
                    {
                        Ancestor[i] = j;
                        i = ancestor;
                    }
                    else
                        ancestor = -1;
                } while ( ancestor >= 0 );
            }
        }
    }

    return 0;
}

int SparseFrame_postorder ( struct matrix_info_struct *matrix_info )
{
    Long nrow;
    Long *Head, *Next;
    Long *Post, *Parent, *ColCount;

    Long *workspace;
    Long top, k;
    Long *Whead, *Stack;

#ifdef PRINT_CALLS
    printf ("\n================SparseFrame_postorder================\n\n");
#endif

    nrow = matrix_info->nrow;

    Post = matrix_info->Post;
    Parent = matrix_info->Parent;

    ColCount = matrix_info->ColCount;

    workspace = matrix_info->workspace;

    Head = workspace + 0 * nrow;
    Next = workspace + 1 * nrow;

    for ( Long j = 0; j < nrow; j++ )
    {
        Head[j] = -1;
        Next[j] = -1;
    }

    if ( ColCount == NULL )
    {
        for ( Long j = nrow - 1; j >= 0; j-- )
        {
            Long p = Parent[j];
            if ( p >= 0 && p < nrow )
            {
                Next[j] = Head[p];
                Head[p] = j;
            }
        }
    }
    else
    {
        Whead = workspace + 2 * nrow;
        for ( Long j = 0; j < nrow; j++ )
            Whead[j] = -1;
        for ( Long j = 0; j < nrow; j++ )
        {
            Long p = Parent[j];
            if ( p >= 0 )
            {
                Long w = ColCount[j];
                Next[j] = Whead[w];
                Whead[w] = j;
            }
        }
        for ( Long w = nrow - 1; w >= 0; w-- )
        {
            Long j = Whead[w];
            while ( j >= 0 )
            {
                Long jnext = Next[j];
                Long p = Parent[j];
                Next[j] = Head[p];
                Head[p] = j;
                j = jnext;
            }
        }
    }

    Stack = workspace + 2 * nrow;
    top = -1;

    for ( Long j = nrow - 1; j >= 0; j-- )
    {
        Long p = Parent[j];
        if ( p < 0 )
        {
            top++;
            Stack[top] = j;
        }
    }

    k = 0;

    while ( top >= 0 )
    {
        Long j = Stack[top];
        Long child = Head[j];
        if ( child >= 0 && child < nrow )
        {
            top++;
            Stack[top] = child;
            Head[j] = Next[child];
        }
        else
        {
            top--;
            Post[k] = j;
            k++;
        }
    }

    return 0;
}

int SparseFrame_colcount ( struct matrix_info_struct *matrix_info )
{
    Long nrow;
    Long *Lp, *Li;
    Long *Post, *Parent, *ColCount, *RowCount;

    Long *workspace;
    Long *Level, *First, *SetParent, *PrevLeaf, *PrevNbr;

#ifdef PRINT_CALLS
    printf ("\n================SparseFrame_colcount================\n\n");
#endif

    nrow = matrix_info->nrow;

    Lp = matrix_info->Lp;
    Li = matrix_info->Li;

    Post = matrix_info->Post;
    Parent = matrix_info->Parent;

    ColCount = matrix_info->ColCount;
    RowCount = matrix_info->RowCount;

    workspace = matrix_info->workspace;

    Level = workspace;
    First = workspace + 1 * nrow;
    SetParent = workspace + 2 * nrow;
    PrevLeaf = workspace + 3 * nrow;
    PrevNbr = workspace + 4 * nrow;

    for ( Long k = nrow - 1; k >= 0; k-- )
    {
        Long j = Post[k];
        Long p = Parent[j];
        if ( p < 0 )
            Level[j] = 0;
        else
            Level[j] = Level[p] + 1;
    }

    for ( Long j = 0; j < nrow; j++ )
    {
        First[j] = -1;
    }

    for ( Long k = 0; k < nrow; k++ )
    {
        Long j = Post[k];
        for ( Long p = j; p >= 0 && First[p] < 0; p = Parent[p] )
        {
            First[p] = k;
        }
    }

    for ( Long k = 0; k < nrow; k++ )
    {
        Long j = Post[k];
        ColCount[j] = 0;
        RowCount[j] = 1;
    }

    for ( Long k = 0; k < nrow; k++ )
    {
        Long j = Post[k];
        SetParent[j] = j;
        PrevLeaf[j] = j;
        PrevNbr[j] = -1;
    }

    for ( Long k = 0; k < nrow; k++ )
    {
        Long j = Post[k];
        PrevNbr[j] = k;
        for ( Long p = Lp[j]; p < Lp[j+1]; p++ )
        {
            Long i = Li[p];
            if ( i > j )
            {
                if ( First[j] > PrevNbr[i] )
                {
                    Long q;
                    Long prevleaf = PrevLeaf[i];
                    for ( q = prevleaf; q != SetParent[q]; q = SetParent[q] );
                    for ( Long s = prevleaf; s != q; s = SetParent[s] )
                    {
                        SetParent[s] = q;
                    }
                    ColCount[j]++;
                    ColCount[q]--;
                    RowCount[i] += ( Level[j] - Level[q] );
                    PrevLeaf[i] = j;
                }
                PrevNbr[i] = k;
            }
        }
        SetParent[j] = Parent[j];
    }

    for ( Long k = 0; k < nrow; k++ )
    {
        Long j = Post[k];
        Long p = Parent[j];
        if ( p >= 0 )
            ColCount[p] += ColCount[j];
    }

    for ( Long k = 0; k < nrow; k++ )
    {
        Long j = Post[k];
        ColCount[j]++;
    }

    return 0;
}

int SparseFrame_analyze_supernodal ( struct common_info_struct *common_info, struct matrix_info_struct *matrix_info )
{
    size_t devSlotSize;

    int isComplex;

    Long nrow;

    Long *LTp, *LTi;

    Long *Perm, *Post, *Parent, *ColCount;

    Long *workspace;

    Long *InvPost;
    Long *Bperm, *Bparent, *Bcolcount;

    Long nfsuper, nsuper;
    Long *Super, *SuperMap, *Sparent;
    Long *Nchild, *Nscol, *Scolcount;
#ifdef RELAXED_SUPERNODE
    Long *Nschild, *Nsz, *Merge;
#endif
    Long *Lsip_copy, *Marker;

    Long isize, xsize;
    Long *Lsip, *Lsxp, *Lsi;
    Float *Lsx;

    Long csize;

    Long nsleaf;
    Long *LeafQueue;

    Long *Head, *Next;

    Long nstage;
    Long *ST_Map, *ST_Pointer, *ST_Index;

    Long *ST_Head, *ST_Next;
    Long *ST_Asize, *ST_Msize;

    size_t *Aoffset, *Moffset;

#ifdef PRINT_CALLS
    printf ("\n================SparseFrame_analyze_supernodal================\n\n");
#endif

    devSlotSize = common_info->devSlotSize;

    isComplex = matrix_info->isComplex;

    nrow = matrix_info->nrow;

    LTp = matrix_info->LTp;
    LTi = matrix_info->LTi;

    Perm = matrix_info->Perm;
    Post = matrix_info->Post;
    Parent = matrix_info->Parent;
    ColCount = matrix_info->ColCount;

    Super = matrix_info->Super;
    SuperMap = matrix_info->SuperMap;
    Sparent = matrix_info->Sparent;

    workspace = matrix_info->workspace;

    Head = workspace + 0 * nrow;
    Next = workspace + 1 * nrow;
    InvPost = workspace + 2 * nrow;
    Bperm = workspace + 3 * nrow;
    Bparent = workspace + 4 * nrow;
    Bcolcount = workspace + 5 * nrow;

    for ( Long k = 0; k < nrow; k++ )
    {
        InvPost [ Post [k] ] = k;
    }

    for ( Long k = 0; k < nrow; k++ )
    {
        Long parent = Parent [ Post [ k ] ];

        Bperm[k] = Perm [ Post [ k ] ];
        Bparent[k] = ( parent < 0 ) ? -1 : InvPost[parent];
        Bcolcount[k] = ColCount [ Post [ k ] ];
    }

    memcpy ( Perm, Bperm, nrow * sizeof(Long) );
    memcpy ( Parent, Bparent, nrow * sizeof(Long) );
    memcpy ( ColCount, Bcolcount, nrow * sizeof(Long) );

    SparseFrame_perm ( matrix_info );

    Nchild = workspace + 2 * nrow;
    Nscol = workspace + 3 * nrow;
    Scolcount = workspace + 4 * nrow;
    Nschild = Nchild; // use Nchild
#ifdef RELAXED_SUPERNODE
    Nsz = workspace + 5 * nrow;
    Merge = workspace + 6 * nrow;
#endif
    ST_Head = Head;
    ST_Next = Next;
    ST_Asize = workspace + 2 * nrow;
    ST_Msize = workspace + 4 * nrow;

    memset ( Nchild, 0, nrow * sizeof(Long) );

    for ( Long j = 0; j < nrow; j++ )
    {
        Long parent = Parent[j];
        if ( parent >= 0 && parent < nrow )
            Nchild[parent]++;
    }

    nfsuper = ( nrow > 0 ) ? 1 : 0;
    Super[0] = 0;

    for ( Long j = 1; j < nrow; j++ )
    {
        if (
                ( Parent[j-1] != j || ColCount[j-1] != ColCount[j] + 1 || Nchild[j] > 1 )
                ||
                (
                 !isComplex &&
                 (
                  ( j - Super[nfsuper-1] + 1 ) * ColCount [ Super[nfsuper-1] ] * sizeof(Float)
                  + ColCount [ Super[nfsuper-1] ] * sizeof(Long)
                  > devSlotSize
                 )
                )
                ||
                (
                 isComplex &&
                 (
                  ( j - Super[nfsuper-1] + 1 ) * ColCount [ Super[nfsuper-1] ] * sizeof(Complex)
                  + ColCount [ Super[nfsuper-1] ] * sizeof(Long)
                  > devSlotSize
                 )
                )
           )
           {
               Super[nfsuper] = j;
               nfsuper++;
           }
    }
    Super[nfsuper] = nrow;

    for ( Long s = 0; s < nfsuper; s++ )
    {
        Nscol[s] = Super[s+1] - Super[s];
        Scolcount[s] = ColCount [ Super [ s ] ];
    }

    for ( Long s = 0; s < nfsuper; s++ )
    {
        for ( Long j = Super[s]; j < Super[s+1]; j++ )
            SuperMap[j] = s;
    }

    for ( Long s = 0; s < nfsuper; s++ )
    {
        Long j = Super[s+1] - 1;
        Long parent = Parent[j];

        Sparent[s] = ( parent < 0 ) ? -1 : SuperMap[parent];
    }

#ifdef RELAXED_SUPERNODE
    {
        memset ( Nschild, 0, nfsuper * sizeof(Long) );

        for ( Long s = 0; s < nfsuper; s++ )
        {
            Long sparent = Sparent[s];
            if ( sparent >= 0 && sparent < nfsuper )
                Nschild[sparent]++;
        }

        for ( Long s = 0; s < nfsuper; s++ )
        {
            Merge[s] = s;
        }

        for ( Long s = 0; s < nfsuper; s++ )
        {
            Nsz[s] = 0;
        }

        for ( Long s = nfsuper - 2; s >= 0; s-- )
        {
            Long sparent = Sparent[s];
            if ( sparent >= 0 && sparent < nfsuper && Merge[s+1] == Merge[sparent] )
            {
                Long smerge = Merge[sparent];
                Long s_ncol = Nscol[s];
                Long p_ncol = Nscol[smerge];
                Long s_colcount = Scolcount[s];
                Long p_colcount = Scolcount[smerge];

                if (
                        (
                         !isComplex &&
                         (
                          ( s_ncol + p_ncol ) * ( s_ncol + p_colcount ) * sizeof(Float)
                          + ( s_ncol + p_colcount ) * sizeof(Long)
                          <= devSlotSize
                         )
                        )
                        ||
                        (
                         isComplex &&
                         (
                          ( s_ncol + p_ncol ) * ( s_ncol + p_colcount ) * sizeof(Complex)
                          + ( s_ncol + p_colcount ) * sizeof(Long)
                          <= devSlotSize
                         )
                        )
                   )
                {
                    Long s_zero = Nsz[s];
                    Long p_zero = Nsz[smerge];
                    Long new_zero = s_ncol * ( s_ncol + p_colcount - s_colcount );
                    Long total_zero = s_zero + p_zero + new_zero;

                    if ( should_relax ( s_ncol + p_ncol, (double)total_zero / ( ( s_ncol + p_ncol ) * ( s_ncol + p_ncol + 1 ) / 2 + ( s_ncol + p_ncol ) * ( p_colcount - p_ncol ) ) ) )
                    {
                        Nscol[smerge] = s_ncol + p_ncol;
                        Scolcount[smerge] = s_ncol + p_colcount;
                        Nsz[smerge] = total_zero;
                        Merge[s] = smerge;
                    }
                }
            }
        }
    }

    nsuper = 0;

    Super[0] = 0;

    for ( Long s = 0; s < nfsuper; s++ )
    {
        if ( Merge[s] == s )
        {
            Super[nsuper+1] = Super[s+1];
            Nscol[nsuper] = Nscol[s];
            Scolcount[nsuper] = Scolcount[s];
            nsuper++;
        }
    }

    Super[nsuper] = nrow;

    for ( Long s = 0; s < nsuper; s++ )
    {
        for ( Long j = Super[s]; j < Super[s+1]; j++ )
            SuperMap[j] = s;
    }

    for ( Long s = 0; s < nsuper; s++ )
    {
        Long j = Super[s+1] - 1;
        Long parent = Parent[j];

        Sparent[s] = ( parent < 0 ) ? -1 : SuperMap[parent];
    }
#else
    nsuper = nfsuper;
#endif

    matrix_info->nsuper = nsuper;

    isize = 0;
    xsize = 0;

    Lsip = malloc ( ( nsuper + 1 ) * sizeof(Long) );
    Lsxp = malloc ( ( nsuper + 1 ) * sizeof(Long) );

    Lsip[0] = 0;
    Lsxp[0] = 0;

    for ( Long s = 0; s < nsuper; s++ )
    {
        Lsip[s+1] = Lsip[s] + Scolcount[s];
        Lsxp[s+1] = Lsxp[s] + Nscol[s] * Scolcount[s];
    }

    isize = Lsip[nsuper];
    xsize = Lsxp[nsuper];

    Lsi = malloc ( isize * sizeof(Long) );
    if ( !isComplex )
        Lsx = malloc ( xsize * sizeof(Float) );
    else
        Lsx = malloc ( xsize * sizeof(Complex) );

    matrix_info->isize = isize;
    matrix_info->xsize = xsize;
    matrix_info->Lsip = Lsip;
    matrix_info->Lsxp = Lsxp;
    matrix_info->Lsi = Lsi;
    matrix_info->Lsx = Lsx;

    Lsip_copy = workspace + 2 * nrow; // don't overwrite Super
    Marker = workspace + 3 * nrow;

    memcpy ( Lsip_copy, Lsip, nsuper * sizeof(Long) );

    for ( Long s = 0; s < nsuper; s++ )
    {
        Marker[s] = Super[s+1];
    }

    for ( Long s = 0; s < nsuper; s++ )
    {
        for ( Long k = Super[s]; k < Super[s+1]; k++ )
        {
            Lsi [ Lsip_copy[s]++ ] = k;
        }
    }

    for ( Long s = 0; s < nsuper; s++ )
    {
        for ( Long j = Super[s]; j < Super[s+1]; j++ )
        {
            for ( Long p = LTp[j]; p < LTp[j+1]; p++ )
            {
                Long i = LTi[p];
                for ( Long sdescendant = SuperMap[i]; sdescendant >= 0 && Marker[sdescendant] <= j; sdescendant = Sparent[sdescendant] )
                {
                    Lsi [ Lsip_copy[sdescendant]++ ] = j;
                    Marker[sdescendant] = j+1;
                }
            }
        }
    }

    csize = 0;
    for ( Long s = 0; s < nsuper; s++ )
    {
        Long nscol = Super[s+1] - Super[s];
        Long nsrow = Lsip[s+1] - Lsip[s];

        if ( nscol < nsrow )
        {
            Long si_last, sparent_last;

            si_last = nscol;
            sparent_last = SuperMap [ Lsi [ Lsip[s] + nscol ] ];
            for ( Long si = nscol; si < nsrow; si++ )
            {
                Long sparent = SuperMap [ Lsi [ Lsip[s] + si ] ];
                if ( sparent != sparent_last )
                {
                    csize = MAX ( csize, ( si - si_last ) * ( nsrow - si_last ) );
                    si_last = si;
                    sparent_last = sparent;
                }
            }
            csize = MAX ( csize, ( nsrow - si_last ) * ( nsrow - si_last ) );
        }
    }
    matrix_info->csize = csize;

    ST_Map = malloc ( nsuper * sizeof(Long) );

    for ( Long s = 0; s < nsuper; s++ )
    {
        ST_Head[s] = -1;
        ST_Next[s] = -1;
        ST_Asize[s] = 0;
        ST_Msize[s] = 0;
        ST_Map[s] = -1;
    }

    nstage = ( nsuper > 0 ) ? 1 : 0;

    for ( Long s = nsuper - 1; s >= 0; s-- )
    {
        Long st;

        if ( Sparent[s] >= 0 )
        {
            st = ST_Map [ Sparent[s] ];
            if (
                    (
                     !isComplex
                     && ( ST_Asize[st] + ( Super[s+1] - Super[s] ) * ( Lsip[s+1] - Lsip[s] ) ) * sizeof(Float) + ( ST_Msize[st] + ( Lsip[s+1] - Lsip[s] ) ) * sizeof(Long) <= devSlotSize
                    )
                    ||
                    (
                     isComplex
                     && ( ST_Asize[st] + ( Super[s+1] - Super[s] ) * ( Lsip[s+1] - Lsip[s] ) ) * sizeof(Complex) + ( ST_Msize[st] + ( Lsip[s+1] - Lsip[s] ) ) * sizeof(Long) <= devSlotSize
                    )
               )
            {
                ST_Map[s] = st;
                ST_Asize[st] += ( ( Super[s+1] - Super[s] ) * ( Lsip[s+1] - Lsip[s] ) );
                ST_Msize[st] += ( Lsip[s+1] - Lsip[s] );
                continue;
            }
            else
                st = ST_Head [ ST_Map [ Sparent[s] ] ];
        }
        else
            st = 0;

        while ( st >= 0 )
        {
            if (
                    (
                     !isComplex
                     && ( ST_Asize[st] + ( Super[s+1] - Super[s] ) * ( Lsip[s+1] - Lsip[s] ) ) * sizeof(Float) + ( ST_Msize[st] + ( Lsip[s+1] - Lsip[s] ) ) * sizeof(Long) <= devSlotSize
                    )
                    ||
                    (
                     isComplex
                     && ( ST_Asize[st] + ( Super[s+1] - Super[s] ) * ( Lsip[s+1] - Lsip[s] ) ) * sizeof(Complex) + ( ST_Msize[st] + ( Lsip[s+1] - Lsip[s] ) ) * sizeof(Long) <= devSlotSize
                    )
               )
            {
                ST_Map[s] = st;
                ST_Asize[st] += ( ( Super[s+1] - Super[s] ) * ( Lsip[s+1] - Lsip[s] ) );
                ST_Msize[st] += ( Lsip[s+1] - Lsip[s] );
                break;
            }
            else
                st = ST_Next[st];
        }

        if ( st < 0 )
        {
            ST_Map[s] = nstage;
            ST_Asize[nstage] = ( ( Super[s+1] - Super[s] ) * ( Lsip[s+1] - Lsip[s] ) );
            ST_Msize[nstage] = ( Lsip[s+1] - Lsip[s] );
            if ( Sparent[s] >= 0 )
            {
                ST_Next[nstage] = ST_Head [ ST_Map [ Sparent[s] ] ];
                ST_Head [ ST_Map [ Sparent[s] ] ] = nstage;
            }
            else
            {
                ST_Next[nstage] = ST_Next[0];
                ST_Next[0] = nstage;
            }
            nstage++;
        }
    }

    for ( Long s = 0; s < nsuper; s++ )
    {
        ST_Map[s] = nstage - 1 - ST_Map[s];
    }

    ST_Pointer = calloc ( nstage + 1, sizeof(Long) );
    ST_Index = malloc ( nsuper * sizeof(Long) );

    for ( Long s = 0; s < nsuper; s++ )
        ST_Pointer [ ST_Map[s] + 1 ] ++;

    for ( Long st = 0; st < nstage; st++ )
        ST_Pointer[st+1] += ST_Pointer[st];

    memcpy ( workspace, ST_Pointer, nstage * sizeof(Long) );

    for ( Long s = 0; s < nsuper; s++ )
    {
        ST_Index [ workspace [ ST_Map[s] ] ++ ] = s;
    }

    matrix_info->nstage = nstage;
    matrix_info->ST_Map = ST_Map;
    matrix_info->ST_Pointer = ST_Pointer;
    matrix_info->ST_Index = ST_Index;

    memset ( Nschild, 0, nsuper * sizeof(Long) );
    for ( Long s = 0; s < nsuper; s++ )
    {
        Long nscol = Super[s+1] - Super[s];
        Long nsrow = Lsip[s+1] - Lsip[s];

        if ( nscol < nsrow )
            Nschild [ SuperMap [ Lsi [ Lsip[s] + nscol ] ] ] = 1;
    }

    LeafQueue = malloc ( nsuper * sizeof(Long) );

    nsleaf = 0;
    for ( Long sp = 0; sp < nsuper; sp++ )
    {
        Long s = ST_Index[sp];

        if ( Nschild[s] == 0 )
            LeafQueue[nsleaf++] = s;
    }

    for ( Long s = nsleaf; s < nsuper; s++ )
        LeafQueue[s] = -1;

    matrix_info->nsleaf = nsleaf;
    matrix_info->LeafQueue = LeafQueue;

    Aoffset = malloc ( nsuper * sizeof(size_t) );
    Moffset = malloc ( nsuper * sizeof(size_t) );

    for ( Long st = 0; st < nstage; st++ )
    {
        size_t Asize, Msize;

        Asize = 0;
        Msize = 0;

        for ( Long pt = ST_Pointer[st]; pt < ST_Pointer[st+1]; pt++ )
        {
            Long s = ST_Index[pt];

            Long nscol = Super[s+1] - Super[s];
            Long nsrow = Lsip[s+1] - Lsip[s];

            Aoffset[s] = Asize;
            if ( !isComplex )
                Asize += nscol * nsrow * sizeof(Float);
            else
                Asize += nscol * nsrow * sizeof(Complex);

            Moffset[s] = Msize;
            Msize += nsrow * sizeof(Long);
        }

        for ( Long pt = ST_Pointer[st]; pt < ST_Pointer[st+1]; pt++ )
            Moffset [ ST_Index[pt] ] += Asize;
    }

    matrix_info->Aoffset = Aoffset;
    matrix_info->Moffset = Moffset;

#ifdef PRINT_INFO
    printf ("nrow = %ld nfsuper = %ld nsuper = %ld nstage = %ld\n", nrow, nfsuper, nsuper, nstage);
#endif

    return 0;
}

int SparseFrame_analyze ( struct common_info_struct *common_info, struct matrix_info_struct *matrix_info )
{
    double timestamp;

    int isComplex;
    Long nrow, nzmax;

#ifdef PRINT_CALLS
    printf ("\n================SparseFrame_analyze================\n\n");
#endif

    timestamp = SparseFrame_time ();

    isComplex = matrix_info->isComplex;
    nrow = matrix_info->nrow;
    nzmax = matrix_info->nzmax;

    matrix_info->Perm = malloc ( nrow * sizeof(Long) );

    //SparseFrame_amd ( matrix_info );
    //SparseFrame_camd ( matrix_info );
    SparseFrame_metis ( matrix_info );

    matrix_info->Lp = malloc ( ( nrow + 1 ) * sizeof(Long) );
    matrix_info->Li = malloc ( nzmax * sizeof(Long) );
    if ( !isComplex )
        matrix_info->Lx = malloc ( nzmax * sizeof(Float) );
    else
        matrix_info->Lx = malloc ( nzmax * sizeof(Complex) );

    matrix_info->LTp = malloc ( ( nrow + 1 ) * sizeof(Long) );
    matrix_info->LTi = malloc ( nzmax * sizeof(Long) );
    if ( !isComplex )
        matrix_info->LTx = malloc ( nzmax * sizeof(Float) );
    else
        matrix_info->LTx = malloc ( nzmax * sizeof(Complex) );

    SparseFrame_perm ( matrix_info );

    matrix_info->Parent = malloc ( nrow * sizeof(Long) );

    SparseFrame_etree ( matrix_info );

    matrix_info->Post = malloc ( nrow * sizeof(Long) );

    SparseFrame_postorder ( matrix_info );

    matrix_info->ColCount = malloc ( nrow * sizeof(Long) );
    matrix_info->RowCount = malloc ( nrow * sizeof(Long) );

    SparseFrame_colcount ( matrix_info );

    SparseFrame_postorder ( matrix_info );

    matrix_info->Super = malloc ( ( nrow + 1 ) * sizeof(Long) );
    matrix_info->SuperMap = malloc ( nrow * sizeof(Long) );
    matrix_info->Sparent = malloc ( nrow * sizeof(Long) );

    SparseFrame_analyze_supernodal ( common_info, matrix_info );

    matrix_info->analyzeTime = SparseFrame_time () - timestamp;

    return 0;
}

int SparseFrame_node_size_cmp ( const void *l, const void *r )
{
#ifdef PRINT_CALLS
    printf ("\n================SparseFrame_node_size_cmp================\n\n");
#endif

    return ( get_node_score(r) - get_node_score(l) );
}

int SparseFrame_node_size_cmp_reverse ( const void *l, const void *r )
{
#ifdef PRINT_CALLS
    printf ("\n================SparseFrame_node_size_cmp_reverse================\n\n");
#endif

    return -SparseFrame_node_size_cmp ( l, r );
}

int SparseFrame_cpuApply ( int isComplex, Long *SuperMap, Long *Super, Long *Lsip, Long *Lsi, Long *Lsxp, void *Lsx, Long *Head, Long *Next, Long *Lpos, Long *Map, Long *RelativeMap, Long s, Long nsrow, void *A, Long d, void *C )
{
    Long ndcol, ndrow, lpos_next;
    Long dn, dm, dk, dlda, dldc;

#ifdef PRINT_CALLS
    printf ("\n================SparseFrame_cpuApply================\n\n");
#endif

    ndcol = Super[d+1] - Super[d];
    ndrow = Lsip[d+1] - Lsip[d];

    for ( lpos_next = Lpos[d]; lpos_next < ndrow && Lsi [ Lsip[d] + lpos_next ] < Super[s+1]; lpos_next++ );

    dn = lpos_next - Lpos[d];
    dm = ndrow - Lpos[d] - dn;
    dk = ndcol;
    dlda = ndrow;
    dldc = ndrow - Lpos[d];

    for ( Long di = 0; di < ndrow - Lpos[d]; di++ )
    {
        RelativeMap [ di ] = Map [ Lsi [ Lsip[d] + Lpos[d] + di ] ];
    }

    if (!isComplex)
        dsyrk_ ( "L", "N", &dn, &dk, one, (Float*) Lsx + Lsxp[d] + Lpos[d], &dlda, zero, (Float*) C, &dldc );
    else
        zherk_ ( "L", "N", &dn, &dk, (Complex*) one, (Complex*) Lsx + Lsxp[d] + Lpos[d], &dlda, (Complex*) zero, (Complex*) C, &dldc );

    if ( dm > 0 )
    {
        if (!isComplex)
            dgemm_ ( "N", "C", &dm, &dn, &dk, one, (Float*) Lsx + Lsxp[d] + lpos_next, &dlda, (Float*) Lsx + Lsxp[d] + Lpos[d], &dlda, zero, (Float*) C + dn, &dldc );
        else
            zgemm_ ( "N", "C", &dm, &dn, &dk, (Complex*) one, (Complex*) Lsx + Lsxp[d] + lpos_next, &dlda, (Complex*) Lsx + Lsxp[d] + Lpos[d], &dlda, (Complex*) zero, (Complex*) C + dn, &dldc );
    }

#pragma omp parallel for schedule(auto) num_threads(CP_NUM_THREAD) if(dn>=CP_THREAD_THRESHOLD)
    for ( Long cj = 0; cj < dn; cj++ )
    {
        for ( Long ci = cj; ci < dn + dm; ci++ )
        {
            if (!isComplex)
                ( (Float*) A ) [ RelativeMap [cj] * nsrow + RelativeMap[ci] ] -= ( (Float*) C ) [ cj * dldc + ci ];
            else
            {
                ( (Complex*) A ) [ RelativeMap [cj] * nsrow + RelativeMap[ci] ].x -= ( (Complex*) C ) [ cj * dldc + ci ].x;
                ( (Complex*) A ) [ RelativeMap [cj] * nsrow + RelativeMap[ci] ].y -= ( (Complex*) C ) [ cj * dldc + ci ].y;
            }
        }
    }

    if ( lpos_next < ndrow )
    {
        Long dancestor;

        dancestor = SuperMap [ Lsi [ Lsip[d] + lpos_next ] ];
#pragma omp critical (HeadNext)
        {
            Next[d] = Head[dancestor];
            Head[dancestor] = d;
        }
    }
    Lpos[d] = lpos_next;

    return 0;
}

int SparseFrame_cpuApplyFactorize ( int isComplex, Long *Lp, Long *Li, void *Lx, Long *SuperMap, Long *Super, Long *Lsip, Long *Lsi, Long *Lsxp, void *Lsx, Long *Head, Long *Next, Long *Lpos, Long *Map, Long *RelativeMap, Long s, Long nscol, Long nsrow, Long sn, Long sm, Long slda, void *C )
{
    int info;

#ifdef PRINT_CALLS
    printf ("\n================SparseFrame_cpuApplyFactorize================\n\n");
#endif

    for ( Long si = 0; si < Lsip[s+1] - Lsip[s]; si++ )
        Map [ Lsi [ Lsip[s] + si ] ] = si;

    if ( !isComplex )
        memset ( (Float*) Lsx + Lsxp[s], 0, nscol * nsrow * sizeof(Float) );
    else
        memset ( (Complex*) Lsx + Lsxp[s], 0, nscol * nsrow * sizeof(Complex) );

    for ( Long j = Super[s]; j < Super[s+1]; j++ )
    {
        Long sj = j - Super[s];
        for ( Long p = Lp[j]; p < Lp[j+1]; p++ )
        {
            Long i = Li[p];
            Long si = Map[i];
            if ( !isComplex )
                ( (Float*) Lsx ) [ Lsxp[s] + sj * nsrow + si ] = ( (Float*) Lx )[p];
            else
            {
                ( (Complex*) Lsx + Lsxp[s] ) [ sj * nsrow + si ].x = ( (Complex*) Lx )[p].x;
                ( (Complex*) Lsx + Lsxp[s] ) [ sj * nsrow + si ].y = ( (Complex*) Lx )[p].y;
            }
        }
    }

    while ( Head[s] >= 0 )
    {
        Long d;
        void *A;

        d = Head[s];

        Head[s] = Next[d];

        if ( !isComplex )
            A = (Float*) Lsx + Lsxp[s];
        else
            A = (Complex*) Lsx + Lsxp[s];

        SparseFrame_cpuApply ( isComplex, SuperMap, Super, Lsip, Lsi, Lsxp, Lsx, Head, Next, Lpos, Map, RelativeMap, s, nsrow, A, d, C );
    }

    if (!isComplex)
        dpotrf_ ( "L", &sn, (Float*) Lsx + Lsxp[s], &slda, &info );
    else
        zpotrf_ ( "L", &sn, (Complex*) Lsx + Lsxp[s], &slda, &info );

    if ( nscol < nsrow )
    {
        if (!isComplex)
            dtrsm_ ( "R", "L", "C", "N", &sm, &sn, one, (Float*) Lsx + Lsxp[s], &slda, (Float*) Lsx + Lsxp[s] + sn, &slda );
        else
            ztrsm_ ( "R", "L", "C", "N", &sm, &sn, (Complex*) one, ( (Complex*) Lsx ) + Lsxp[s], &slda, ( (Complex*) Lsx ) + Lsxp[s] + sn, &slda );
    }

    return 0;
}

int SparseFrame_factorize_supernodal ( struct common_info_struct *common_info, struct gpu_info_struct *gpu_info_list, struct matrix_info_struct *matrix_info )
{
    int numCPU, numGPU;

    int isComplex;
    Long nrow;
    Long *Lp, *Li;
    Float *Lx;

    Long nsuper, csize, isize, nsleaf, leafQueueHead, leafQueueTail;
    Long *Super, *SuperMap;
    Long *Lsip, *Lsxp, *Lsi;
    Float *Lsx;
    Long *Head, *Next, *Lpos, *Lpos_next, *Nschild, *LeafQueue;

    int d_Lsi_valid;

    int *GPUSerial;
    Long *NodeSTPass;

    enum NodeLocationType { NODE_LOCATION_NULL, NODE_LOCATION_MAIN, NODE_LOCATION_GPU };

    enum NodeLocationType *NodeLocation;

    size_t *Aoffset, *Moffset;

    Long *ST_Map;

    Long *workspace;

#ifdef PRINT_CALLS
    printf ("\n================SparseFrame_factorize_supernodal================\n\n");
#endif

    numCPU = common_info->numCPU;
    numGPU = common_info->numGPU;

    isComplex = matrix_info->isComplex;
    nrow = matrix_info->nrow;

    Lp = matrix_info->Lp;
    Li = matrix_info->Li;
    Lx = matrix_info->Lx;

    nsuper = matrix_info->nsuper;
    Super = matrix_info->Super;
    SuperMap = matrix_info->SuperMap;

    Lsip = matrix_info->Lsip;
    Lsxp = matrix_info->Lsxp;
    Lsi = matrix_info->Lsi;
    Lsx = matrix_info->Lsx;

    csize = matrix_info->csize;
    isize = matrix_info->isize;
    d_Lsi_valid = ( isize  * sizeof(Long) <= common_info->devSlotSize );

    nsleaf = matrix_info->nsleaf;
    LeafQueue = matrix_info->LeafQueue;

    ST_Map = matrix_info->ST_Map;

    Aoffset = matrix_info->Aoffset;
    Moffset = matrix_info->Moffset;

    workspace = matrix_info->workspace;

    Lpos = workspace + 0 * nsuper;
    Lpos_next = workspace + 1 * nsuper;
    Head = workspace + 2 * nsuper;
    Next = workspace + 3 * nsuper;
    Nschild = workspace + 4 * nsuper;
    GPUSerial = (void*) ( workspace + 5 * nsuper );
    NodeSTPass = workspace + 6 * nsuper;
    NodeLocation = (void*) ( workspace + 7 * nsuper );

    for ( Long s = 0; s < nsuper; s++ )
    {
        Head[s] = -1;
        Next[s] = -1;
    }

    memset ( (void*) Nschild, 0, nsuper * sizeof(Long) );

    for ( Long s = 0; s < nsuper; s++ )
    {
        Long nscol, nsrow, sparent;

        nscol = Super[s+1] - Super[s];
        nsrow = Lsip[s+1] - Lsip[s];

        if ( nscol < nsrow )
        {
            sparent = SuperMap [ Lsi [ Lsip[s] + nscol ] ];
            Nschild[sparent]++;
        }
    }

    for ( Long s = 0; s < nsuper; s++ )
    {
        GPUSerial[s] = -1;
        NodeSTPass[s] = -1;
        NodeLocation[s] = NODE_LOCATION_NULL;
    }

    for ( Long s = 0; s < nsuper; s++ )
    {
        Lpos[s] = 0;
    }

    leafQueueHead = 0;
    leafQueueTail = nsleaf;

#pragma omp parallel num_threads( numGPU + numCPU )
    {
        int gpuIndex;
        struct gpu_info_struct *gpu_info;

        Long leafQueueIndex;
        Long *Map, *RelativeMap;
        Float *C;
        struct node_size_struct *node_size_queue;
        Long *h_Lsi, *d_Lsi, *d_Map;

        Long st_last, stPass;

        gpuIndex = omp_get_thread_num();
        gpu_info = gpu_info_list + gpuIndex;

        omp_set_lock ( &( gpu_info->gpuLock ) );

        if ( gpuIndex < numGPU )
            cudaSetDevice ( gpu_info->gpuIndex_physical );

        Map = NULL;
        RelativeMap = NULL;
        C = NULL;
        node_size_queue = NULL;
        h_Lsi = NULL;
        d_Lsi = NULL;
        d_Map = NULL;

#pragma omp critical (leafQueue)
        {
            if ( leafQueueHead >= leafQueueTail )
                leafQueueIndex = nsuper;
            else
                leafQueueIndex = leafQueueHead++;
        }

        if ( leafQueueIndex < nsuper )
        {
            Map = malloc ( nrow * sizeof(Long) );
            RelativeMap = malloc ( nrow * sizeof(Long) );
            if (!isComplex)
                C = malloc ( csize * sizeof(Float) );
            else
                C = malloc ( csize * sizeof(Complex) );
            node_size_queue = malloc ( nsuper * sizeof(struct node_size_struct) );
            if ( gpuIndex < numGPU && d_Lsi_valid )
            {
                h_Lsi = gpu_info->h_Lsi;
                d_Lsi = gpu_info->d_Lsi;
                for ( int i = 0; i < isize; i++ )
                    h_Lsi[i] = Lsi[i];
                cudaMemcpyAsync ( d_Lsi, h_Lsi, isize * sizeof(Long), cudaMemcpyHostToDevice, gpu_info->s_cudaStream );
                d_Map = d_Lsi + isize;
            }
        }

        st_last = -1;
        stPass = 0;

        while ( leafQueueIndex < nsuper )
        {
            Long s, nscol, nsrow;
            Long sn, sm, slda;

            s = LeafQueue[leafQueueIndex];

            nscol = Super[s+1] - Super[s];
            nsrow = Lsip[s+1] - Lsip[s];

            sn = nscol;
            sm = nsrow - nscol;
            slda = sn + sm;

            if ( gpuIndex < numGPU )
            {
                int useCpuPotrf;

                Long d_count, gpu_blas_count, gpu_blas_single_count;

                useCpuPotrf = set_factorize_location ( nscol, nsrow );

                d_count = 0;
                gpu_blas_count = 0;
                gpu_blas_single_count = 0;

                for ( Long d = Head[s]; d >= 0; d = Next[d] )
                {
                    Long ndcol, ndrow;
                    Long lpos, lpos_next;

                    Long dn, dm, dk;

                    Long score;

                    ndcol = Super[d+1] - Super[d];
                    ndrow = Lsip[d+1] - Lsip[d];

                    lpos = Lpos[d];
                    for ( lpos_next = lpos; lpos_next < ndrow && ( Lsi + Lsip[d] ) [ lpos_next ] < Super[s+1]; lpos_next++ );
                    Lpos_next[d] = lpos_next;

                    dn = lpos_next - lpos;
                    dm = ndrow - lpos_next;
                    dk = ndcol;

                    node_size_queue[d_count].node = d;
                    node_size_queue[d_count].n = dn;
                    node_size_queue[d_count].m = dm;
                    node_size_queue[d_count].k = dk;
                    score = set_node_score ( node_size_queue + d_count );

                    if ( score >= 0 )
                    {
                        gpu_blas_count++;
                        if ( score > 0 )
                            gpu_blas_single_count++;
                    }

                    d_count++;
                }

                if ( gpu_blas_count <= 0 && useCpuPotrf )
                {
                    SparseFrame_cpuApplyFactorize ( isComplex, Lp, Li, Lx, SuperMap, Super, Lsip, Lsi, Lsxp, Lsx, Head, Next, Lpos, Map, RelativeMap, s, nscol, nsrow, sn, sm, slda, C );

                    NodeLocation[s] = NODE_LOCATION_NULL;
                }
                else
                {
                    Long st;

                    void *A, *h_A, *h_A_reserve, *d_A, *d_A_, *d_A_reserve;

                    int devWorkSize;
                    Float *d_workspace;
                    int *d_info;

                    if ( !isComplex )
                        A = (Float*) Lsx + Lsxp[s];
                    else
                        A = (Complex*) Lsx + Lsxp[s];
                    h_A = gpu_info->h_A[0] + Aoffset[s];
                    h_A_reserve = gpu_info->h_A[1] + Aoffset[s];
                    d_A = gpu_info->d_A[0] + Aoffset[s];
                    d_A_ = gpu_info->d_A[1] + Aoffset[s];
                    d_A_reserve = gpu_info->d_A[2] + Aoffset[s];

                    d_info = gpu_info->d_B[0];
                    d_workspace = gpu_info->d_B[0] + MAX ( sizeof(int), MAX ( sizeof(Float), sizeof(Complex) ) );

                    GPUSerial[s] = gpuIndex;

                    st = ST_Map[s];

                    for ( Long d_index = 0; d_index < d_count; d_index++ )
                    {
                        Long d;
                        Long ndcol, ndrow;
                        Long lpos;

                        d = node_size_queue[d_index].node;

                        ndcol = Super[d+1] - Super[d];
                        ndrow = Lsip[d+1] - Lsip[d];

                        lpos = Lpos[d];

                        if ( GPUSerial[d] == gpuIndex )
                        {
                            if ( gpu_info->lastMatrix == matrix_info->serial && ST_Map[d] == st_last && NodeSTPass[d] == stPass && NodeLocation[d] != NODE_LOCATION_NULL )
                            {
                                if ( NodeLocation[d] == NODE_LOCATION_MAIN )
                                {
                                    void *h_R, *d_R;

                                    if (!isComplex)
                                    {
                                        h_R = (Float*) ( gpu_info->h_A[0] + Aoffset[d] ) + lpos;
                                        d_R = (Float*) ( gpu_info->d_A[0] + Aoffset[d] ) + lpos;
                                    }
                                    else
                                    {
                                        h_R = (Complex*) ( gpu_info->h_A[0] + Aoffset[d] ) + lpos;
                                        d_R = (Complex*) ( gpu_info->d_A[0] + Aoffset[d] ) + lpos;
                                    }

                                    if ( !isComplex )
                                        cudaMemcpy2DAsync ( (Float*) d_R, ndrow * sizeof(Float), (Float*) h_R, ndrow * sizeof(Float), ( ndrow - lpos ) * sizeof(Float), ndcol, cudaMemcpyHostToDevice, gpu_info->s_cudaStream );
                                    else
                                        cudaMemcpy2DAsync ( (Complex*) d_R, ndrow * sizeof(Complex), (Complex*) h_R, ndrow * sizeof(Complex), ( ndrow - lpos ) * sizeof(Complex), ndcol, cudaMemcpyHostToDevice, gpu_info->s_cudaStream );
                                }

                                NodeLocation[d] = NODE_LOCATION_GPU;
                            }
                            else
                                NodeLocation[d] = NODE_LOCATION_NULL;
                        }
                    }

                    for ( Long si = 0; si < nsrow; si++ )
                    {
                        Map [ Lsi [ Lsip[s] + si ] ] = si;
                    }

                    if ( d_Lsi_valid && gpu_blas_count > 0 )
                        createMap ( d_Map, d_Lsi, Lsip[s], nsrow, gpu_info->s_cudaStream );

                    if ( !isComplex )
                        memset ( h_A_reserve, 0, nscol * nsrow * sizeof(Float) );
                    else
                        memset ( h_A_reserve, 0, nscol * nsrow * sizeof(Complex) );

#pragma omp parallel for schedule(auto) num_threads(CP_NUM_THREAD) if(nscol>=CP_THREAD_THRESHOLD)
                    for ( Long j = Super[s]; j < Super[s+1]; j++ )
                    {
                        Long sj = j - Super[s];
                        for ( Long p = Lp[j]; p < Lp[j+1]; p++ )
                        {
                            Long i = Li[p];
                            Long si = Map[i];
                            if ( !isComplex )
                                ( (Float*) h_A_reserve ) [ sj * nsrow + si ] = Lx[p];
                            else
                            {
                                ( (Complex*) h_A_reserve ) [ sj * nsrow + si ].x = ( (Complex*) Lx )[p].x;
                                ( (Complex*) h_A_reserve ) [ sj * nsrow + si ].y = ( (Complex*) Lx )[p].y;
                            }
                        }
                    }

                    if ( d_count > 0 )
                        qsort ( node_size_queue, d_count, sizeof(struct node_size_struct), SparseFrame_node_size_cmp );

                    if ( gpu_blas_count <= 0 )
                    {
                        while ( Head[s] >= 0 )
                        {
                            Long d;

                            d = Head[s];
                            Head[s] = Next[d];

                            SparseFrame_cpuApply ( isComplex, SuperMap, Super, Lsip, Lsi, Lsxp, Lsx, Head, Next, Lpos, Map, RelativeMap, s, nsrow, h_A_reserve, d, C );
                        }

                        if ( !isComplex )
                            cudaMemcpyAsync ( d_A, h_A_reserve, nscol * nsrow * sizeof(Float), cudaMemcpyHostToDevice, gpu_info->s_cudaStream );
                        else
                            cudaMemcpyAsync ( d_A, h_A_reserve, nscol * nsrow * sizeof(Complex), cudaMemcpyHostToDevice, gpu_info->s_cudaStream );
                    }
                    else
                    {
                        Long d_index_small;

                        d_index_small = d_count;

                        if ( !isComplex )
                            cudaMemsetAsync ( d_A_, 0, nscol * nsrow * sizeof(Float), gpu_info->s_cudaStream );
                        else
                            cudaMemsetAsync ( d_A_, 0, nscol * nsrow * sizeof(Complex), gpu_info->s_cudaStream );

                        qsort ( node_size_queue, MIN ( gpu_blas_count, MAX_D_STREAM ), sizeof(struct node_size_struct), SparseFrame_node_size_cmp_reverse );

                        cudaStreamSynchronize ( gpu_info->s_cudaStream );

                        for ( Long d_index = 0; d_index < gpu_blas_count; d_index++ )
                        {
                            int stream_offset;
                            cudaError_t cudaError;

                            if ( d_index_small >= gpu_blas_count )
                                for ( stream_offset = 0; stream_offset < MAX_D_STREAM && ( cudaError = cudaEventQuery ( gpu_info->d_cudaEvent_onDevice[ ( d_index + stream_offset ) % MAX_D_STREAM ] ) ) != cudaSuccess; stream_offset++ );
                            else
                                for ( stream_offset = 0; ( cudaError = cudaEventQuery ( gpu_info->d_cudaEvent_onDevice[ ( d_index + stream_offset ) % MAX_D_STREAM ] ) ) != cudaSuccess; stream_offset = ( stream_offset + 1 ) % MAX_D_STREAM );

                            if ( cudaError == cudaSuccess )
                            {
                                int stream_index;

                                Long d, ndcol, ndrow ,lpos, lpos_next;
                                Long dn, dm, dk, dlda, dldc;

                                void *d_C;

                                stream_index = ( d_index + stream_offset ) % MAX_D_STREAM;

                                d_C = gpu_info->d_C[stream_index];

                                d = node_size_queue[d_index].node;

                                ndcol = Super[d+1] - Super[d];
                                ndrow = Lsip[d+1] - Lsip[d];
                                lpos = Lpos[d];
                                lpos_next = Lpos_next[d];

                                dn = lpos_next - lpos;
                                dm = ndrow - lpos_next;
                                dk = ndcol;
                                dlda = dn + dm;
                                dldc = dn + dm;

                                if ( GPUSerial[d] == gpuIndex && NodeLocation[d] == NODE_LOCATION_GPU )
                                {
                                    void *d_R;
                                    Long *h_RelativeMap, *d_RelativeMap;

                                    if (!isComplex)
                                        d_R = (Float*) ( gpu_info->d_A[0] + Aoffset[d] ) + lpos;
                                    else
                                        d_R = (Complex*) ( gpu_info->d_A[0] + Aoffset[d] ) + lpos;

                                    h_RelativeMap = gpu_info->h_A[0] + Moffset[d];
                                    d_RelativeMap = gpu_info->d_A[0] + Moffset[d];

                                    if (!isComplex)
                                        cublasDsyrk ( gpu_info->d_cublasHandle[stream_index], CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, dn, dk, one, d_R, ndrow, zero, d_C, dldc);
                                    else
                                        cublasZherk ( gpu_info->d_cublasHandle[stream_index], CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, dn, dk, one, d_R, ndrow, zero, d_C, dldc);

                                    if ( dm > 0 )
                                    {
                                        if (!isComplex)
                                            cublasDgemm ( gpu_info->d_cublasHandle[stream_index], CUBLAS_OP_N, CUBLAS_OP_T, dm, dn, dk, one, (Float*) d_R + dn, ndrow, d_R, ndrow, zero, (Float*) d_C + dn, dldc );
                                        else
                                            cublasZgemm ( gpu_info->d_cublasHandle[stream_index], CUBLAS_OP_N, CUBLAS_OP_C, dm, dn, dk, (Complex*) one, (Complex*) d_R + dn, ndrow, d_R, ndrow, (Complex*) zero, (Complex*) d_C + dn, dldc );
                                    }

                                    if ( !d_Lsi_valid )
                                    {
                                        for ( Long di = 0; di < ndrow - lpos; di++ )
                                        {
                                            h_RelativeMap[di] = Map [ Lsi [ Lsip[d] + lpos + di ] ];
                                        }

                                        cudaMemcpyAsync ( d_RelativeMap, h_RelativeMap, ( ndrow - lpos ) * sizeof(Long), cudaMemcpyHostToDevice, gpu_info->d_cudaStream[stream_index] );
                                    }
                                    else
                                        createRelativeMap ( d_RelativeMap, d_Map, d_Lsi, Lsip[d] + lpos, ndrow - lpos, gpu_info->d_cudaStream[stream_index] );

                                    mappedSubtract ( TRUE, isComplex, d_A_, slda, d_C, 0, 0, dn, dn + dm, dldc, d_RelativeMap, gpu_info->d_cudaStream[stream_index] );
                                }
                                else
                                {
                                    void *h_B, *d_B;
                                    Long *h_RelativeMap, *d_RelativeMap;

                                    h_B = gpu_info->h_B[stream_index];
                                    d_B = gpu_info->d_B[stream_index];

                                    if ( !isComplex )
                                    {
                                        h_RelativeMap = (void*) h_B + dk * dlda * sizeof(Float);
                                        d_RelativeMap = (void*) d_B + dk * dlda * sizeof(Float);
                                    }
                                    else
                                    {
                                        h_RelativeMap = (void*) h_B + dk * dlda * sizeof(Complex);
                                        d_RelativeMap = (void*) d_B + dk * dlda * sizeof(Complex);
                                    }

#pragma omp parallel for schedule(auto) num_threads(CP_NUM_THREAD) if(ndcol>=CP_THREAD_THRESHOLD)
                                    for ( Long dj = 0; dj < ndcol; dj++ )
                                    {
                                        for ( Long di = 0; di < ndrow - lpos; di++ )
                                        {
                                            if (!isComplex)
                                                ( (Float*) h_B ) [ dj * dlda + di ] = ( Lsx + Lsxp[d] + lpos ) [ dj * ndrow + di ];
                                            else
                                            {
                                                ( (Complex*) h_B ) [ dj * dlda + di ].x = ( (Complex*) Lsx + Lsxp[d] + lpos ) [ dj * ndrow + di ].x;
                                                ( (Complex*) h_B ) [ dj * dlda + di ].y = ( (Complex*) Lsx + Lsxp[d] + lpos ) [ dj * ndrow + di ].y;
                                            }
                                        }
                                    }

                                    if ( !d_Lsi_valid )
                                    {
                                        for ( Long di = 0; di < ndrow - lpos; di++ )
                                        {
                                            h_RelativeMap[di] = Map [ Lsi [ Lsip[d] + lpos + di ] ];
                                        }
                                    }

                                    cudaStreamWaitEvent ( gpu_info->d_cudaStream_copy[stream_index], gpu_info->d_cudaEvent_applied[stream_index], 0 );

                                    if ( !d_Lsi_valid )
                                    {
                                        if (!isComplex)
                                            cudaMemcpyAsync ( d_B, h_B, dk * dlda * sizeof(Float) + ( ndrow - lpos ) * sizeof(Long), cudaMemcpyHostToDevice, gpu_info->d_cudaStream_copy[stream_index] );
                                        else
                                            cudaMemcpyAsync ( d_B, h_B, dk * dlda * sizeof(Complex) + ( ndrow - lpos ) * sizeof(Long), cudaMemcpyHostToDevice, gpu_info->d_cudaStream_copy[stream_index] );
                                    }
                                    else
                                    {
                                        if (!isComplex)
                                            cudaMemcpyAsync ( d_B, h_B, dk * dlda * sizeof(Float), cudaMemcpyHostToDevice, gpu_info->d_cudaStream_copy[stream_index] );
                                        else
                                            cudaMemcpyAsync ( d_B, h_B, dk * dlda * sizeof(Complex), cudaMemcpyHostToDevice, gpu_info->d_cudaStream_copy[stream_index] );
                                    }

                                    cudaEventRecord ( gpu_info->d_cudaEvent_onDevice[stream_index], gpu_info->d_cudaStream_copy[stream_index] );

                                    if ( d_Lsi_valid )
                                        createRelativeMap ( d_RelativeMap, d_Map, d_Lsi, Lsip[d] + lpos, ndrow - lpos, gpu_info->d_cudaStream[stream_index] );

                                    cudaStreamWaitEvent ( gpu_info->d_cudaStream[stream_index], gpu_info->d_cudaEvent_onDevice[stream_index], 0 );

                                    if (!isComplex)
                                        cublasDsyrk ( gpu_info->d_cublasHandle[stream_index], CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, dn, dk, one, d_B, dlda, zero, d_C, dldc);
                                    else
                                        cublasZherk ( gpu_info->d_cublasHandle[stream_index], CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, dn, dk, one, d_B, dlda, zero, d_C, dldc);

                                    if ( dm > 0 )
                                    {
                                        if (!isComplex)
                                            cublasDgemm ( gpu_info->d_cublasHandle[stream_index], CUBLAS_OP_N, CUBLAS_OP_T, dm, dn, dk, one, (Float*) d_B + dn, dlda, d_B, dlda, zero, (Float*) d_C + dn, dldc );
                                        else
                                            cublasZgemm ( gpu_info->d_cublasHandle[stream_index], CUBLAS_OP_N, CUBLAS_OP_C, dm, dn, dk, (Complex*) one, (Complex*) d_B + dn, dlda, d_B, dlda, (Complex*) zero, (Complex*) d_C + dn, dldc );
                                    }

                                    mappedSubtract ( TRUE, isComplex, d_A_, slda, d_C, 0, 0, dn, dn + dm, dldc, d_RelativeMap, gpu_info->d_cudaStream[stream_index] );

                                    cudaEventRecord ( gpu_info->d_cudaEvent_applied[stream_index], gpu_info->d_cudaStream[stream_index] );
                                }

                                if ( lpos_next < ndrow )
                                {
                                    Long dancestor;

                                    dancestor = SuperMap [ Lsi [ Lsip[d] + lpos_next ] ];
#pragma omp critical (HeadNext)
                                    {
                                        Next[d] = Head[dancestor];
                                        Head[dancestor] = d;
                                    }
                                }
                                Lpos[d] = lpos_next;
                            }
                            else
                            {
                                d_index--;

                                if ( d_index_small > gpu_blas_count )
                                {
                                    Long d;

                                    d_index_small--;

                                    d = node_size_queue[d_index_small].node;

                                    SparseFrame_cpuApply ( isComplex, SuperMap, Super, Lsip, Lsi, Lsxp, Lsx, Head, Next, Lpos, Map, RelativeMap, s, nsrow, h_A_reserve, d, C );
                                }
                                else if ( d_index_small == gpu_blas_count )
                                {
                                    d_index_small--;

                                    if ( !isComplex )
                                        cudaMemcpyAsync ( d_A_reserve, h_A_reserve, nscol * nsrow * sizeof(Float), cudaMemcpyHostToDevice, gpu_info->s_cudaStream );
                                    else
                                        cudaMemcpyAsync ( d_A_reserve, h_A_reserve, nscol * nsrow * sizeof(Complex), cudaMemcpyHostToDevice, gpu_info->s_cudaStream );
                                }
                            }
                        }

                        while ( d_index_small > gpu_blas_count )
                        {
                            Long d;

                            d_index_small--;

                            d = node_size_queue[d_index_small].node;

                            SparseFrame_cpuApply ( isComplex, SuperMap, Super, Lsip, Lsi, Lsxp, Lsx, Head, Next, Lpos, Map, RelativeMap, s, nsrow, h_A_reserve, d, C );
                        }

                        if ( d_index_small == gpu_blas_count )
                        {
                            if ( !isComplex )
                                cudaMemcpyAsync ( d_A_reserve, h_A_reserve, nscol * nsrow * sizeof(Float), cudaMemcpyHostToDevice, gpu_info->s_cudaStream );
                            else
                                cudaMemcpyAsync ( d_A_reserve, h_A_reserve, nscol * nsrow * sizeof(Complex), cudaMemcpyHostToDevice, gpu_info->s_cudaStream );
                        }

                        for ( int index = 0; index < MAX_D_STREAM; index++ )
                            cudaStreamWaitEvent ( gpu_info->s_cudaStream, gpu_info->d_cudaEvent_applied[index], 0 );

                        deviceSum ( isComplex, d_A, d_A_reserve, d_A_, nscol, nsrow, gpu_info->s_cudaStream );

                        if ( useCpuPotrf )
                        {
                            if ( !isComplex )
                                cudaMemcpyAsync ( h_A, d_A, nscol * nsrow * sizeof(Float), cudaMemcpyDeviceToHost, gpu_info->s_cudaStream );
                            else
                                cudaMemcpyAsync ( h_A, d_A, nscol * nsrow * sizeof(Complex), cudaMemcpyDeviceToHost, gpu_info->s_cudaStream );

                            cudaStreamSynchronize ( gpu_info->s_cudaStream );
                        }
                    }

                    if ( useCpuPotrf )
                    {
                        int info;

                        if (!isComplex)
                            dpotrf_ ( "L", &sn, h_A, &slda, &info );
                        else
                            zpotrf_ ( "L", &sn, h_A, &slda, &info );

                        if ( nscol < nsrow )
                        {
                            if (!isComplex)
                                dtrsm_ ( "R", "L", "C", "N", &sm, &sn, one, h_A, &slda, (Float*) h_A + sn, &slda );
                            else
                                ztrsm_ ( "R", "L", "C", "N", &sm, &sn, (Complex*) one, h_A, &slda, (Complex*) h_A + sn, &slda );
                        }

#pragma omp parallel for schedule(auto) num_threads(CP_NUM_THREAD) if(nscol>=CP_THREAD_THRESHOLD)
                        for ( Long sj = 0; sj < nscol; sj++ )
                        {
                            for ( Long si = sj; si < nsrow; si++ )
                            {
                                if ( !isComplex )
                                    ( (Float*) A ) [ sj * nsrow + si ] = ( (Float*) h_A ) [ sj * nsrow + si ];
                                else
                                {
                                    ( (Complex*) A ) [ sj * nsrow + si ].x = ( (Complex*) h_A ) [ sj * nsrow + si ].x;
                                    ( (Complex*) A ) [ sj * nsrow + si ].y = ( (Complex*) h_A ) [ sj * nsrow + si ].y;
                                }
                            }
                        }
                    }
                    else
                    {
                        if ( nscol < potrf_split_threshold )
                        {
                            if (!isComplex)
                            {
                                cusolverDnDpotrf_bufferSize ( gpu_info->s_cusolverDnHandle, CUBLAS_FILL_MODE_LOWER, sn, d_A, slda, &devWorkSize );
                                cusolverDnDpotrf ( gpu_info->s_cusolverDnHandle, CUBLAS_FILL_MODE_LOWER, sn, d_A, slda, d_workspace, devWorkSize, d_info );
                            }
                            else
                            {
                                cusolverDnZpotrf_bufferSize ( gpu_info->s_cusolverDnHandle, CUBLAS_FILL_MODE_LOWER, sn, d_A, slda, &devWorkSize );
                                cusolverDnZpotrf ( gpu_info->s_cusolverDnHandle, CUBLAS_FILL_MODE_LOWER, sn, d_A, slda, (Complex*) d_workspace, devWorkSize, d_info );
                            }

                            if ( nscol < nsrow )
                            {
                                if (!isComplex)
                                    cublasDtrsm ( gpu_info->s_cublasHandle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, sm, sn, one, d_A, slda, (Float*) d_A + sn, slda );
                                else
                                    cublasZtrsm ( gpu_info->s_cublasHandle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_C, CUBLAS_DIAG_NON_UNIT, sm, sn, (Complex*) one, d_A, slda, (Complex*) d_A + sn, slda );
                            }

                            if ( !isComplex )
                                cudaMemcpyAsync ( h_A, d_A, nscol * nsrow * sizeof(Float), cudaMemcpyDeviceToHost, gpu_info->s_cudaStream );
                            else
                                cudaMemcpyAsync ( h_A, d_A, nscol * nsrow * sizeof(Complex), cudaMemcpyDeviceToHost, gpu_info->s_cudaStream );

                            cudaStreamSynchronize ( gpu_info->s_cudaStream );

#pragma omp parallel for schedule(auto) num_threads(CP_NUM_THREAD) if(nscol>=CP_THREAD_THRESHOLD)
                            for ( Long sj = 0; sj < nscol; sj++ )
                            {
                                for ( Long si = sj; si < nsrow; si++ )
                                {
                                    if ( !isComplex )
                                        ( (Float*) A ) [ sj * nsrow + si ] = ( (Float*) h_A ) [ sj * nsrow + si ];
                                    else
                                    {
                                        ( (Complex*) A ) [ sj * nsrow + si ].x = ( (Complex*) h_A ) [ sj * nsrow + si ].x;
                                        ( (Complex*) A ) [ sj * nsrow + si ].y = ( (Complex*) h_A ) [ sj * nsrow + si ].y;
                                    }
                                }
                            }
                        }
                        else
                        {
                            const Long sjblock =
                                potrf_split_block;
                            Long sjl, sjr, sjl_last, sjr_last;

                            sjl_last = 0;
                            sjr_last = 0;

                            for ( sjl = 0; sjl < nscol; sjl += sjblock )
                            {
                                sjr = MIN ( sjl + sjblock, nscol );

                                if ( sjl > 0 )
                                {
                                    if (!isComplex)
                                        cublasDsyrk ( gpu_info->s_cublasHandle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, sjr - sjl, sjl, minus_one, (Float*) d_A + sjl, slda, one, (Float*) d_A + sjl * ( slda + 1 ), slda);
                                    else
                                        cublasZherk ( gpu_info->s_cublasHandle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, sjr - sjl, sjl, minus_one, (Complex*) d_A + sjl, slda, one, (Complex*) d_A + sjl * ( slda + 1 ), slda);

                                    if ( sjr < nsrow )
                                    {
                                        if (!isComplex)
                                            cublasDgemm ( gpu_info->s_cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, nsrow - sjr, sjr - sjl, sjl, minus_one, (Float*) d_A + sjr, slda, (Float*) d_A + sjl, slda, one, (Float*) d_A + sjl * slda + sjr, slda );
                                        else
                                            cublasZgemm ( gpu_info->s_cublasHandle, CUBLAS_OP_N, CUBLAS_OP_C, nsrow - sjr, sjr - sjl, sjl, (Complex*) minus_one, (Complex*) d_A + sjr, slda, (Complex*) d_A + sjl, slda, (Complex*) one, (Complex*) d_A + sjl * slda + sjr, slda );
                                    }
                                }

                                if (!isComplex)
                                {
                                    cusolverDnDpotrf_bufferSize ( gpu_info->s_cusolverDnHandle, CUBLAS_FILL_MODE_LOWER, sjr - sjl, (Float*) d_A + sjl * ( slda + 1 ), slda, &devWorkSize );
                                    cusolverDnDpotrf ( gpu_info->s_cusolverDnHandle, CUBLAS_FILL_MODE_LOWER, sjr - sjl, (Float*) d_A + sjl * ( slda + 1 ), slda, d_workspace, devWorkSize, d_info );
                                }
                                else
                                {
                                    cusolverDnZpotrf_bufferSize ( gpu_info->s_cusolverDnHandle, CUBLAS_FILL_MODE_LOWER, sjr - sjl, (Complex*) d_A + sjl * ( slda + 1 ), slda, &devWorkSize );
                                    cusolverDnZpotrf ( gpu_info->s_cusolverDnHandle, CUBLAS_FILL_MODE_LOWER, sjr - sjl, (Complex*) d_A + sjl * ( slda + 1 ), slda, (Complex*) d_workspace, devWorkSize, d_info );
                                }

                                if ( sjr < nsrow )
                                {
                                    if (!isComplex)
                                        cublasDtrsm ( gpu_info->s_cublasHandle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, nsrow - sjr, sjr - sjl, one, (Float*) d_A + sjl * ( slda + 1 ), slda, (Float*) d_A + sjl * slda + sjr, slda );
                                    else
                                        cublasZtrsm ( gpu_info->s_cublasHandle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_C, CUBLAS_DIAG_NON_UNIT, nsrow - sjr, sjr - sjl, (Complex*) one, (Complex*) d_A + sjl * ( slda + 1 ), slda, (Complex*) d_A + sjl * slda + sjr, slda );
                                }

                                if ( sjl > 0 )
                                {
                                    cudaStreamWaitEvent ( gpu_info->s_cudaStream_copyback, gpu_info->s_cudaEvent_factorized, 0 );

                                    if ( !isComplex )
                                        cudaMemcpy2DAsync ( (Float*) h_A + sjl_last * ( slda + 1 ), nsrow * sizeof(Float), (Float*) d_A + sjl_last * ( slda + 1 ), nsrow * sizeof(Float), ( nsrow - sjl_last ) * sizeof(Float), sjblock, cudaMemcpyDeviceToHost, gpu_info->s_cudaStream_copyback );
                                    else
                                        cudaMemcpy2DAsync ( (Complex*) h_A + sjl_last * ( slda + 1 ), nsrow * sizeof(Complex), (Complex*) d_A + sjl_last * ( slda + 1 ), nsrow * sizeof(Complex), ( nsrow - sjl_last ) * sizeof(Complex), sjblock, cudaMemcpyDeviceToHost, gpu_info->s_cudaStream_copyback );

                                    cudaStreamSynchronize ( gpu_info->s_cudaStream_copyback );

#pragma omp parallel for schedule(auto) num_threads(CP_NUM_THREAD) if(sjblock>=CP_THREAD_THRESHOLD)
                                    for ( Long sj = sjl_last; sj < sjr_last; sj++ )
                                    {
                                        for ( Long si = sj; si < nsrow; si++ )
                                        {
                                            if ( !isComplex )
                                                ( (Float*) A ) [ sj * nsrow + si ] = ( (Float*) h_A ) [ sj * nsrow + si ];
                                            else
                                            {
                                                ( (Complex*) A ) [ sj * nsrow + si ].x = ( (Complex*) h_A ) [ sj * nsrow + si ].x;
                                                ( (Complex*) A ) [ sj * nsrow + si ].y = ( (Complex*) h_A ) [ sj * nsrow + si ].y;
                                            }
                                        }
                                    }
                                }

                                cudaEventRecord ( gpu_info->s_cudaEvent_factorized, gpu_info->s_cudaStream );

                                sjl_last = sjl;
                                sjr_last = sjr;
                            }

                            if ( !isComplex )
                                cudaMemcpy2DAsync ( (Float*) h_A + sjl_last * ( slda + 1 ), nsrow * sizeof(Float), (Float*) d_A + sjl_last * ( slda + 1 ), nsrow * sizeof(Float), ( nsrow - sjl_last ) * sizeof(Float), ( sjr_last - sjl_last ) , cudaMemcpyDeviceToHost, gpu_info->s_cudaStream );
                            else
                                cudaMemcpy2DAsync ( (Complex*) h_A + sjl_last * ( slda + 1 ), nsrow * sizeof(Complex), (Complex*) d_A + sjl_last * ( slda + 1 ), nsrow * sizeof(Complex), ( nsrow - sjl_last ) * sizeof(Complex), ( sjr_last - sjl_last ) , cudaMemcpyDeviceToHost, gpu_info->s_cudaStream );

                            cudaStreamSynchronize ( gpu_info->s_cudaStream );

#pragma omp parallel for schedule(auto) num_threads(CP_NUM_THREAD) if(sjr-sjl>=CP_THREAD_THRESHOLD)
                            for ( Long sj = sjl_last; sj < sjr_last; sj++ )
                            {
                                for ( Long si = sj; si < nsrow; si++ )
                                {
                                    if ( !isComplex )
                                        ( (Float*) A ) [ sj * nsrow + si ] = ( (Float*) h_A ) [ sj * nsrow + si ];
                                    else
                                    {
                                        ( (Complex*) A ) [ sj * nsrow + si ].x = ( (Complex*) h_A ) [ sj * nsrow + si ].x;
                                        ( (Complex*) A ) [ sj * nsrow + si ].y = ( (Complex*) h_A ) [ sj * nsrow + si ].y;
                                    }
                                }
                            }
                        }
                    }

                    NodeLocation[s] = useCpuPotrf ? NODE_LOCATION_MAIN : NODE_LOCATION_GPU;

                    if ( st != st_last || gpu_info->lastMatrix != matrix_info->serial )
                        stPass++;

                    gpu_info->lastMatrix = matrix_info->serial;

                    NodeSTPass[s] = stPass;

                    st_last = st;
                }
            }
            else
            {
                SparseFrame_cpuApplyFactorize ( isComplex, Lp, Li, Lx, SuperMap, Super, Lsip, Lsi, Lsxp, Lsx, Head, Next, Lpos, Map, RelativeMap, s, nscol, nsrow, sn, sm, slda, C );
            }

            Lpos[s] = Super[s+1] - Super[s];

            if ( nscol < nsrow )
            {
                Long sparent;

                sparent = SuperMap [ Lsi [ Lsip[s] + nscol ] ];
#pragma omp critical (HeadNext)
                {
                    Next[s] = Head[sparent];
                    Head[sparent] = s;
                }
#pragma omp critical (leafQueue)
                {
                    Nschild[sparent]--;
                    if ( Nschild[sparent] <= 0 )
                        LeafQueue[leafQueueTail++] = sparent;
                }
            }

#pragma omp critical (leafQueue)
            {
                if ( leafQueueHead >= leafQueueTail )
                    leafQueueIndex = nsuper;
                else
                    leafQueueIndex = leafQueueHead++;
            }
        }

        if ( Map != NULL ) free ( Map );
        if ( RelativeMap != NULL ) free ( RelativeMap );
        if ( C != NULL ) free ( C );
        if ( node_size_queue != NULL ) free ( node_size_queue );

        Map = NULL;
        RelativeMap = NULL;
        C = NULL;
        node_size_queue = NULL;
        h_Lsi = NULL;
        d_Lsi = NULL;
        d_Map = NULL;

        omp_unset_lock ( &( gpu_info->gpuLock ) );
    }

    return 0;
}

int SparseFrame_factorize ( struct common_info_struct *common_info, struct gpu_info_struct *gpu_info_list, struct matrix_info_struct *matrix_info )
{
    double timestamp;

#ifdef PRINT_CALLS
    printf ("\n================SparseFrame_factorize================\n\n");
#endif

    timestamp = SparseFrame_time ();

    SparseFrame_factorize_supernodal ( common_info, gpu_info_list, matrix_info );

    matrix_info->factorizeTime = SparseFrame_time () - timestamp;

    return 0;
}

int SparseFrame_solve_supernodal ( struct matrix_info_struct *matrix_info )
{
    double timestamp;

    int isComplex;
    Long nrow;
    Long nsuper;
    Long *Super;
    Long *Lsip, *Lsxp, *Lsi;
    Float *Lsx;

    Float *Bx, *Xx;

#ifdef PRINT_CALLS
    printf ("\n================SparseFrame_solve_supernodal================\n\n");
#endif

    timestamp = SparseFrame_time ();

    isComplex = matrix_info->isComplex;
    nrow = matrix_info->nrow;

    nsuper = matrix_info->nsuper;

    Super = matrix_info->Super;
    Lsip = matrix_info->Lsip;
    Lsxp = matrix_info->Lsxp;
    Lsi = matrix_info->Lsi;
    Lsx = matrix_info->Lsx;

    Bx = matrix_info->Bx;
    Xx = matrix_info->Xx;

    if ( !isComplex )
        memcpy ( Xx, Bx, nrow * sizeof(Float) );
    else
        memcpy ( Xx, Bx, nrow * sizeof(Complex) );

    for ( Long s = 0; s < nsuper; s++ )
    {
        Long nscol = Super[s+1] - Super[s];
        Long nsrow = Lsip[s+1] - Lsip[s];

        for ( Long sj = 0; sj < nscol; sj++ )
        {
            Long j = Lsi [ Lsip[s] + sj ];

            if ( !isComplex )
                Xx[j] /= Lsx [ Lsxp[s] + sj * nsrow + sj ];
            else
            {
                // TODO
            }

            for ( Long si = sj + 1; si < nsrow; si++ )
            {
                Long i = Lsi [ Lsip[s] + si ];

                if ( !isComplex )
                    Xx[i] -= ( Lsx [ Lsxp[s] + sj * nsrow + si ] * Xx[j] );
                else
                {
                    // TODO
                }
            }
        }
    }

    for ( Long s = nsuper - 1; s >= 0; s-- )
    {
        Long nscol = Super[s+1] - Super[s];
        Long nsrow = Lsip[s+1] - Lsip[s];

        for ( Long sj = nscol - 1; sj >= 0; sj-- )
        {
            Long j = Lsi [ Lsip[s] + sj ];

            for ( Long si = sj + 1; si < nsrow; si++ )
            {
                Long i = Lsi [ Lsip[s] + si ];

                if ( !isComplex )
                    Xx[j] -= ( Lsx [ Lsxp[s] + sj * nsrow + si ] * Xx[i] );
                else
                {
                    // TODO
                }
            }

            if ( !isComplex )
                Xx[j] /= Lsx [ Lsxp[s] + sj * nsrow + sj ];
            else
            {
                // TODO
            }
        }
    }

    matrix_info->solveTime = SparseFrame_time () - timestamp;

    return 0;
}

int SparseFrame_validate ( struct matrix_info_struct *matrix_info )
{
    int isComplex;
    Long nrow;
    Long *Lp, *Li;
    Float *Lx;
    Float *Bx, *Xx, *Rx;
    Float anorm, bnorm, xnorm, rnorm;
    Float residual;

    Float *workspace;

#ifdef PRINT_CALLS
    printf ("\n================SparseFrame_validate================\n\n");
#endif

    isComplex = matrix_info->isComplex;

    nrow = matrix_info->nrow;

    Lp = matrix_info->Lp;
    Li = matrix_info->Li;
    Lx = matrix_info->Lx;

    if ( !isComplex )
    {
        Bx = malloc ( nrow * sizeof(Float) );
        Xx = malloc ( nrow * sizeof(Float) );
        Rx = malloc ( nrow * sizeof(Float) );
    }
    else
    {
        Bx = malloc ( nrow * sizeof(Complex) );
        Xx = malloc ( nrow * sizeof(Complex) );
        Rx = malloc ( nrow * sizeof(Complex) );
    }

    matrix_info->Bx = Bx;
    matrix_info->Xx = Xx;
    matrix_info->Rx = Rx;

    for ( Long i = 0; i < nrow; i++ )
    {
        if ( !isComplex )
        {
            Bx[i] = 1 + i / (Float)nrow;
        }
        else
        {
            ( (Complex*) Bx ) [i].x = 1 + i / (Float)nrow;
            ( (Complex*) Bx ) [i].y = ( (Float)nrow / 2 - i ) / ( 3 * nrow );
        }
    }

    SparseFrame_solve_supernodal ( matrix_info );

    for ( Long i = 0; i < nrow; i++ )
    {
        if ( !isComplex )
            Rx[i] = -Bx[i];
        else
        {
            ( (Complex*) Rx ) [i].x = - ( (Complex*) Bx) [i].x;
            ( (Complex*) Rx ) [i].y = - ( (Complex*) Bx) [i].y;
        }
    }

    for ( Long j = 0; j < nrow; j++ )
    {
        for ( Long p = Lp[j]; p < Lp[j+1]; p++ )
        {
            Long i = Li[p];
            Rx[i] += ( Lx[p] * Xx[j] );
            if ( i != j )
                Rx[j] += ( Lx[p] * Xx[i] );
        }
    }

    workspace = matrix_info->workspace;
    memset ( workspace, 0, nrow * sizeof(Float) );

    anorm = 0;
    for ( Long j = 0; j < nrow; j++ )
    {
        for ( Long p = Lp[j]; p < Lp[j+1]; p++ )
        {
            Long i = Li[p];
            if ( !isComplex )
            {
                workspace[j] += fabs ( Lx[p] );
                if ( i != j )
                    workspace[i] += fabs ( Lx[p] );
            }
            else
            {
                // TODO
            }
        }
    }
    for ( Long j = 0; j < nrow; j++ )
        if ( workspace[j] > anorm )
            anorm = workspace[j];

    bnorm = 0;
    xnorm = 0;
    rnorm = 0;
    for ( Long i = 0; i < nrow; i++ )
    {
        if ( !isComplex )
        {
            bnorm = MAX ( bnorm, fabs ( Bx[i] ) );
            xnorm = MAX ( xnorm, fabs ( Xx[i] ) );
            rnorm = MAX ( rnorm, fabs ( Rx[i] ) );
        }
        else
        {
            // TODO
        }
    }

    printf ("anorm = %le bnorm = %le xnorm = %le rnorm = %le\n", anorm, bnorm, xnorm, rnorm);
    residual = rnorm / ( anorm * xnorm + bnorm );
    matrix_info->residual = residual;

    return 0;
}

int SparseFrame_cleanup_matrix ( struct matrix_info_struct *matrix_info )
{
#ifdef PRINT_CALLS
    printf ("\n================SparseFrame_cleanup_matrix================\n\n");
#endif

    if ( matrix_info->Tj != NULL ) free ( matrix_info->Tj );
    if ( matrix_info->Ti != NULL ) free ( matrix_info->Ti );
    if ( matrix_info->Tx != NULL ) free ( matrix_info->Tx );

    if ( matrix_info->Cp != NULL ) free ( matrix_info->Cp );
    if ( matrix_info->Ci != NULL ) free ( matrix_info->Ci );
    if ( matrix_info->Cx != NULL ) free ( matrix_info->Cx );

    if ( matrix_info->Lp != NULL ) free ( matrix_info->Lp );
    if ( matrix_info->Li != NULL ) free ( matrix_info->Li );
    if ( matrix_info->Lx != NULL ) free ( matrix_info->Lx );

    if ( matrix_info->LTp != NULL ) free ( matrix_info->LTp );
    if ( matrix_info->LTi != NULL ) free ( matrix_info->LTi );
    if ( matrix_info->LTx != NULL ) free ( matrix_info->LTx );

    if ( matrix_info->Perm != NULL ) free ( matrix_info->Perm );
    if ( matrix_info->Post != NULL ) free ( matrix_info->Post );
    if ( matrix_info->Parent != NULL ) free ( matrix_info->Parent );
    if ( matrix_info->ColCount != NULL ) free ( matrix_info->ColCount );
    if ( matrix_info->RowCount != NULL ) free ( matrix_info->RowCount );

    if ( matrix_info->Super != NULL ) free ( matrix_info->Super );
    if ( matrix_info->SuperMap != NULL ) free ( matrix_info->SuperMap );
    if ( matrix_info->Sparent != NULL ) free ( matrix_info->Sparent );
    if ( matrix_info->LeafQueue != NULL ) free ( matrix_info->LeafQueue );

    if ( matrix_info->Lsip != NULL ) free ( matrix_info->Lsip );
    if ( matrix_info->Lsxp != NULL ) free ( matrix_info->Lsxp );
    if ( matrix_info->Lsi != NULL ) free ( matrix_info->Lsi );
    if ( matrix_info->Lsx != NULL ) free ( matrix_info->Lsx );

    if ( matrix_info->ST_Map != NULL ) free ( matrix_info->ST_Map );
    if ( matrix_info->ST_Pointer != NULL ) free ( matrix_info->ST_Pointer );
    if ( matrix_info->ST_Index != NULL ) free ( matrix_info->ST_Index );

    if ( matrix_info->Aoffset != NULL ) free ( matrix_info->Aoffset );
    if ( matrix_info->Moffset != NULL ) free ( matrix_info->Moffset );

    if ( matrix_info->workspace != NULL ) free ( matrix_info->workspace );

    if ( matrix_info->Bx != NULL ) free ( matrix_info->Bx );
    if ( matrix_info->Xx != NULL ) free ( matrix_info->Xx );
    if ( matrix_info->Rx != NULL ) free ( matrix_info->Rx );

    SparseFrame_initialize_matrix ( matrix_info );

    return 0;
}

int SparseFrame ( int argc, char **argv )
{
    double timestamp;

    int numSparseMatrix, nextMatrixIndex;
    int matrixThreadNum;

    int ompThreadNum;

    struct common_info_struct common_info_object;
    struct common_info_struct *common_info = &common_info_object;
    struct gpu_info_struct *gpu_info_list;
    struct matrix_info_struct *matrix_info_list;

#ifdef PRINT_CALLS
    printf ("\n================SparseFrame================\n\n");
#endif

    // Allocate resources

    timestamp = SparseFrame_time();

    SparseFrame_allocate_gpu (common_info, &gpu_info_list);

    numSparseMatrix = argc - 1;
    common_info->numSparseMatrix = numSparseMatrix;

    matrixThreadNum = MIN ( MATRIX_THREAD_NUM, numSparseMatrix );
    common_info->matrixThreadNum = matrixThreadNum;

#ifdef PRINT_INFO
    printf ("Num of matrices = %d\n\n", numSparseMatrix);
#endif

    ompThreadNum = MIN ( omp_get_max_threads(), OMP_THREAD_NUM );

    openblas_set_num_threads ( ompThreadNum );

    SparseFrame_allocate_matrix ( common_info, &matrix_info_list );

    common_info->allocateTime = SparseFrame_time () - timestamp;

#ifdef PRINT_INFO
    printf ("Allocate time:        %lf\n\n", common_info->allocateTime);
#endif

    timestamp = SparseFrame_time();

    omp_set_nested ( TRUE );

    nextMatrixIndex = 0;

#pragma omp parallel for schedule(auto) num_threads(matrixThreadNum)
    for ( int matrixThreadIndex = 0; matrixThreadIndex < matrixThreadNum; matrixThreadIndex++ )
    {
        int matrixIndex;

        const char *path;

#pragma omp critical ( nextMatrixIndex )
        {
            if ( nextMatrixIndex >= numSparseMatrix )
                matrixIndex = numSparseMatrix;
            else
                matrixIndex = nextMatrixIndex++;
        }

        while ( matrixIndex < numSparseMatrix )
        {
            struct matrix_info_struct *matrix_info = matrix_info_list + matrixThreadIndex;

            // Initialize
            matrix_info->serial = matrixIndex;
            SparseFrame_initialize_matrix ( matrix_info );

            // Read matrices

            path = argv [ 1 + matrixIndex ];
            matrix_info->path = path;

            SparseFrame_read_matrix ( matrix_info );

            // Analyze

            SparseFrame_analyze ( common_info, matrix_info );

            // Factorize

            cudaProfilerStart();

            SparseFrame_factorize ( common_info, gpu_info_list, matrix_info );

            cudaProfilerStop();

            // Validate

            SparseFrame_validate ( matrix_info );

            // Cleanup

            SparseFrame_cleanup_matrix ( matrix_info );

            // Output

#ifdef PRINT_INFO
            printf ( "Matrix name:    %s\n", basename ( (char*) path ) );
            printf ( "Read time:      %lf\n", matrix_info->readTime );
            printf ( "Analyze time:   %lf\n", matrix_info->analyzeTime );
            printf ( "Factorize time: %lf\n", matrix_info->factorizeTime );
            printf ( "Solve time:     %lf\n", matrix_info->solveTime );
            printf ( "residual (|Ax-b|)/(|A||x|+|b|): %le\n\n", matrix_info->residual );
#endif

#pragma omp critical ( nextMatrixIndex )
            {
                if ( nextMatrixIndex >= numSparseMatrix )
                    matrixIndex = numSparseMatrix;
                else
                    matrixIndex = nextMatrixIndex++;
            }
        }
    }

    common_info->computeTime = SparseFrame_time () - timestamp;

#ifdef PRINT_INFO
    printf ("Total computing time: %lf\n\n", common_info->computeTime);
#endif

    // Free resources

    timestamp = SparseFrame_time();

    SparseFrame_free_gpu (common_info, &gpu_info_list);

    SparseFrame_free_matrix ( common_info, &matrix_info_list );

    common_info->freeTime = SparseFrame_time () - timestamp;

#ifdef PRINT_INFO
    printf ("Free time:            %lf\n\n", common_info->freeTime);
#endif

    return 0;
}
