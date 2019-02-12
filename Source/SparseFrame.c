#include "SparseFrame.h"

double SparseFrame_time ()
{
    struct timespec tp;

#ifdef PRINT_CALLS
    printf ("================SparseFrame_time================\n\n");
#endif

    clock_gettime ( CLOCK_REALTIME, &tp );

    return ( tp.tv_sec + ( double ) ( tp.tv_nsec ) / 1.0e9 );
}

int SparseFrame_allocate_gpu ( struct gpu_info_struct **gpu_info_ptr, struct common_info_struct *common_info )
{
    int numGPU;
    int gpu_index;

#ifdef PRINT_CALLS
    printf ("================SparseFrame_allocate_gpu================\n\n");
#endif

    cudaGetDeviceCount ( &numGPU );
    common_info->numGPU = numGPU;

#ifdef PRINT_INFO
    printf ( "Num of GPUs = %d\n", numGPU );
#endif

    *gpu_info_ptr = malloc ( numGPU * sizeof ( struct gpu_info_struct ) );

    if ( *gpu_info_ptr == NULL ) return 1;

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

    return 0;
}

int SparseFrame_free_gpu ( struct gpu_info_struct **gpu_info_ptr, struct common_info_struct *common_info )
{
    int numGPU;
    int gpu_index;

#ifdef PRINT_CALLS
    printf ("================SparseFrame_free_gpu================\n\n");
#endif

    if ( *gpu_info_ptr == NULL ) return 1;

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

    return 0;
}

int SparseFrame_allocate_matrix ( struct matrix_info_struct **matrix_info_ptr, struct common_info_struct *common_info )
{
    int numSparseMatrix;

#ifdef PRINT_CALLS
    printf ("================SparseFrame_allocate_matrix================\n\n");
#endif

    numSparseMatrix = common_info->numSparseMatrix;

    *matrix_info_ptr = malloc ( numSparseMatrix * sizeof ( struct matrix_info_struct ) );

    if ( *matrix_info_ptr == NULL ) return 1;

    return 0;
}

int SparseFrame_free_matrix ( struct matrix_info_struct **matrix_info_ptr, struct common_info_struct *common_info )
{
#ifdef PRINT_CALLS
    printf ("================SparseFrame_free_matrix================\n\n");
#endif

    if ( *matrix_info_ptr == NULL ) return 1;

    free ( *matrix_info_ptr );

    common_info->numSparseMatrix = 0;

    return 0;
}

int SparseFrame_read_matrix_triplet ( char **buf_ptr, struct matrix_info_struct *matrix_info )
{
    char s0[max_mm_line_size];
    char s1[max_mm_line_size];
    char s2[max_mm_line_size];
    char s3[max_mm_line_size];
    char s4[max_mm_line_size];

    int n_scanned;
    Long ncol, nrow, nzmax;
    Long Tj, Ti;
    Float Tx, Ty;

    Long nz;

#ifdef PRINT_CALLS
    printf ("================SparseFrame_read_matrix_triplet================\n\n");
#endif

    while ( ( getline ( buf_ptr, &max_mm_line_size, matrix_info->file ) != -1 ) && ( strcmp (*buf_ptr, "") == 0 ) );

    if ( strncmp ( *buf_ptr, "%%MatrixMarket", 14 ) != 0 )
    {
        printf ("Matrix format error\n\n");
        return 1;
    }

    sscanf ( *buf_ptr, "%s %s %s %s %s\n", s0, s1, s2, s3, s4 );

    if ( strcmp ( s4, "symmetric" ) == 0 )
        matrix_info->isSymmetric = 1;
    else
        matrix_info->isSymmetric = 0;

    if ( strcmp ( s3, "real" ) == 0 )
        matrix_info->isComplex = 0;
    else
        matrix_info->isComplex = 1;

    while ( ( getline ( buf_ptr, &max_mm_line_size, matrix_info->file ) != -1 ) && ( ( strcmp (*buf_ptr, "") == 0 ) || ( strncmp (*buf_ptr, "%", 1) == 0 ) ) );

    n_scanned = sscanf ( *buf_ptr, "%ld %ld %ld\n", &nrow, &ncol, &nzmax );

    if (n_scanned != 3)
    {
        printf ("Matrix format error\n\n");
        matrix_info->ncol = 0;
        matrix_info->nrow = 0;
        matrix_info->nzmax = 0;
        return 1;
    }

#ifdef PRINT_INFO
    if ( matrix_info->isComplex == 0 && matrix_info->isSymmetric != 0)
        printf ("matrix is real symmetric, ncol = %ld nrow = %ld nzmax = %ld\n", ncol, nrow, nzmax);
    else if ( matrix_info->isComplex == 1 && matrix_info->isSymmetric != 0)
        printf ("matrix is complex symmetric, ncol = %ld nrow = %ld nzmax = %ld\n", ncol, nrow, nzmax);
    else if ( matrix_info->isComplex == 0 && matrix_info->isSymmetric == 0)
        printf ("matrix is real unsymmetric, ncol = %ld nrow = %ld nzmax = %ld\n", ncol, nrow, nzmax);
    else
        printf ("matrix is complex unsymmetric, ncol = %ld nrow = %ld nzmax = %ld\n", ncol, nrow, nzmax);
#endif

    matrix_info->Tj = malloc ( nzmax * sizeof(Long) );
    matrix_info->Ti = malloc ( nzmax * sizeof(Long) );
    matrix_info->Tx = malloc ( nzmax * sizeof(Float) );
    if ( matrix_info->isComplex )
        matrix_info->Ty = malloc ( nzmax * sizeof(Float) );
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
                return 1;
            }
            n_scanned = sscanf ( *buf_ptr, "%ld %ld %lg %lg\n", &Ti, &Tj, &Tx, &Ty );
            if ( matrix_info->isComplex && n_scanned < 4 )
            {
                printf ( "Error: imaginary part not present\n" );
                return 1;
            }
            matrix_info->Tj[nz] = Tj - 1;
            matrix_info->Ti[nz] = Ti - 1;
            matrix_info->Tx[nz] = Tx;
            if ( matrix_info->isComplex )
                matrix_info->Ty[nz] = Ty;
            nz++;
        }
    }
    if (nz != nzmax) { printf ("error: nz = %ld nzmax = %ld\n", nz, nzmax); exit(0); }

    matrix_info->ncol = ncol;
    matrix_info->nrow = nrow;
    matrix_info->nzmax = nzmax;

    return 0;
}

int SparseFrame_compress ( struct matrix_info_struct *matrix_info )
{
    int isComplex;
    Long j, ncol, nrow;
    Long nz, nzmax;
    Long p;
    Long *Tj, *Ti, *Tx, *Ty;
    Long *Cp, *Ci, *Cx, *Cy;

    Long *workspace;

#ifdef PRINT_CALLS
    printf ("================SparseFrame_compress================\n\n");
#endif

    isComplex = matrix_info->isComplex;
    ncol = matrix_info->ncol;
    nrow = matrix_info->nrow;
    nzmax = matrix_info->nzmax;

    Tj = matrix_info->Tj;
    Ti = matrix_info->Ti;
    Tx = matrix_info->Tx;
    Ty = matrix_info->Ty;

    Cp = calloc ( ( ncol + 1 ), sizeof(Long) );
    Ci = malloc ( nzmax * sizeof(Long) );
    Cx = malloc ( nzmax * sizeof(Float) );
    if ( isComplex )
    {
        Cy = malloc ( nzmax * sizeof(Float) );
    }
    else
        Cy = NULL;

    matrix_info->Cp = Cp;
    matrix_info->Ci = Ci;
    matrix_info->Cx = Cx;
    matrix_info->Cy = Cy;

    for ( nz = 0; nz < nzmax; nz++ )
        Cp[ Tj[nz] + 1 ] ++;

    for ( j = 0; j < ncol; j++ )
        Cp[j+1] += Cp[j];

    workspace = malloc ( ( ncol + 1 ) * sizeof(Long) );
    memcpy ( workspace, Cp, sizeof(Long) * ( ncol + 1 ) );

    for ( nz = 0; nz < nzmax; nz++ )
    {
        p = workspace [ Tj [ nz ] ] ++;
        Ci [p] = Ti[nz];
        Cx [p] = Tx[nz];
        if ( isComplex )
            Cy [p] = Ty[nz];
    }

    free (workspace);


    if ( Tj != NULL ) free ( Tj );
    if ( Ti != NULL ) free ( Ti );
    if ( Tx != NULL ) free ( Tx );
    if ( Ty != NULL ) free ( Ty );

    Tj = matrix_info->Tj = NULL;
    Ti = matrix_info->Ti = NULL;
    Tx = matrix_info->Tx = NULL;
    Ty = matrix_info->Ty = NULL;

    return 0;
}

int SparseFrame_read_matrix ( char *path, struct matrix_info_struct *matrix_info )
{
    char *buf;

    double timestamp;

#ifdef PRINT_CALLS
    printf ("================SparseFrame_read_matrix================\n\n");
#endif

    timestamp = SparseFrame_time ();

    matrix_info->file = fopen ( path, "r" );

    if ( matrix_info->file == NULL ) return 1;

    buf = malloc ( max_mm_line_size * sizeof(char) );

    SparseFrame_read_matrix_triplet ( &buf, matrix_info );

    fclose ( matrix_info->file );

    free ( buf );

    SparseFrame_compress ( matrix_info );

    matrix_info->read_time = SparseFrame_time () - timestamp;

#ifdef PRINT_INFO
    printf ( "Matrix read time: %lf seconds\n", matrix_info->read_time );
#endif

    return 0;
}

int SparseFrame_analyze ( struct matrix_info_struct *matrix_info )
{
    Long j, i, ncol, nrow, nzmax, p, anz;
    Long *Cp, *Ci, *Ap, *Ai;

    Long *workspace;

    double Control[AMD_CONTROL], Info[AMD_INFO];

    double timestamp;

#ifdef PRINT_CALLS
    printf ("================SparseFrame_analyze================\n\n");
#endif

    timestamp = SparseFrame_time ();

    ncol = matrix_info->ncol;
    nrow = matrix_info->nrow;
    nzmax = matrix_info->nzmax;

    Cp = matrix_info->Cp;
    Ci = matrix_info->Ci;

    Ap = calloc ( ( nrow + 1 ), sizeof(Long) );
    matrix_info->Ap = Ap;

    for ( j = 0; j < ncol; j++ )
    {
        for ( p = Cp[j]; p < Cp[j+1]; p++ )
        {
            i = Ci[p];
            if (i > j)
            {
                Ap[i+1]++;
                Ap[j+1]++;
            }
        }
    }

    for ( j = 0; j < nrow; j++)
    {
        Ap[j+1] += Ap[j];
    }

    anz = Ap[nrow];
    matrix_info->anz = anz;

    Ai = malloc ( anz * sizeof(Long) );
    matrix_info->Ai = Ai;

    workspace = malloc ( ( nrow + 1 ) * sizeof(Long) );
    memcpy ( workspace, Ap, ( nrow + 1 ) * sizeof(Long) );

    for ( j = 0; j < ncol; j++ )
    {
        for ( p = Cp[j]; p < Cp[j+1]; p++ )
        {
            i = Ci[p];
            if (i > j)
            {
                Ai [ workspace[i]++ ] = j;
                Ai [ workspace[j]++ ] = i;
            }
        }
    }

    free (workspace);

    matrix_info->Len = malloc ( matrix_info->nrow * sizeof(Long) );
    matrix_info->Nv = malloc ( matrix_info->nrow * sizeof(Long) );
    matrix_info->Next = malloc ( matrix_info->nrow * sizeof(Long) );
    matrix_info->Perm = malloc ( matrix_info->nrow * sizeof(Long) );
    matrix_info->Head = malloc ( matrix_info->nrow * sizeof(Long) );
    matrix_info->Elen = malloc ( matrix_info->nrow * sizeof(Long) );
    matrix_info->Degree = malloc ( matrix_info->nrow * sizeof(Long) );
    matrix_info->Wi = malloc ( matrix_info->nrow * sizeof(Long) );

    for ( j = 0; j < matrix_info->nrow; j++ )
        matrix_info->Len[j] = matrix_info->Ap[j+1] - matrix_info->Ap[j];

    Control[AMD_DENSE] = prune_dense;
    Control[AMD_AGGRESSIVE] = aggressive;

    amd_l2 ( matrix_info->nrow, matrix_info->Ap, matrix_info->Ai, matrix_info->Len,
            matrix_info->anz, matrix_info->Ap[matrix_info->nrow],
            matrix_info->Nv, matrix_info->Next, matrix_info->Perm, matrix_info->Head,
            matrix_info->Elen, matrix_info->Degree, matrix_info->Wi,
            Control, Info );

    matrix_info->analyze_time = SparseFrame_time () - timestamp;

    return 0;
}

int SparseFrame_cleanup_matrix ( struct matrix_info_struct *matrix_info )
{
#ifdef PRINT_CALLS
    printf ("================SparseFrame_cleanup_matrix================\n\n");
#endif

    if ( matrix_info->Tj != NULL ) free ( matrix_info->Tj );
    if ( matrix_info->Ti != NULL ) free ( matrix_info->Ti );
    if ( matrix_info->Tx != NULL ) free ( matrix_info->Tx );
    if ( matrix_info->Ty != NULL ) free ( matrix_info->Ty );

    if ( matrix_info->Cp != NULL ) free ( matrix_info->Cp );
    if ( matrix_info->Ci != NULL ) free ( matrix_info->Ci );
    if ( matrix_info->Cx != NULL ) free ( matrix_info->Cx );
    if ( matrix_info->Cy != NULL ) free ( matrix_info->Cy );

    if ( matrix_info->Ap != NULL ) free ( matrix_info->Ap );
    if ( matrix_info->Ai != NULL ) free ( matrix_info->Ai );

    if ( matrix_info->Len    != NULL ) free ( matrix_info->Len    );
    if ( matrix_info->Nv     != NULL ) free ( matrix_info->Nv     );
    if ( matrix_info->Next   != NULL ) free ( matrix_info->Next   );
    if ( matrix_info->Perm   != NULL ) free ( matrix_info->Perm   );
    if ( matrix_info->Head   != NULL ) free ( matrix_info->Head   );
    if ( matrix_info->Elen   != NULL ) free ( matrix_info->Elen   );
    if ( matrix_info->Degree != NULL ) free ( matrix_info->Degree );
    if ( matrix_info->Wi     != NULL ) free ( matrix_info->Wi     );

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
    matrix_info->Ap = NULL;
    matrix_info->Ai = NULL;

    matrix_info->Len    = NULL;
    matrix_info->Nv     = NULL;
    matrix_info->Next   = NULL;
    matrix_info->Perm   = NULL;
    matrix_info->Head   = NULL;
    matrix_info->Elen   = NULL;
    matrix_info->Degree = NULL;
    matrix_info->Wi     = NULL;

    return 0;
}

int SparseFrame ( int argc, char **argv )
{
    int numSparseMatrix, matrixIndex;

    struct common_info_struct common_info_object;
    struct common_info_struct *common_info = &common_info_object;
    struct gpu_info_struct *gpu_info;
    struct matrix_info_struct *matrix_info;

#ifdef PRINT_CALLS
    printf ("================SparseFrame================\n\n");
#endif

    // Allocate resources

    SparseFrame_allocate_gpu (&gpu_info, common_info);

    numSparseMatrix = argc - 1;
    common_info->numSparseMatrix = numSparseMatrix;

    SparseFrame_allocate_matrix ( &matrix_info, common_info );

    for ( matrixIndex = 0; matrixIndex < numSparseMatrix; matrixIndex++ )
    {
        // Read matrices

        SparseFrame_read_matrix ( argv [ 1 + matrixIndex ], matrix_info + matrixIndex );

        // Analyze

        SparseFrame_analyze ( matrix_info + matrixIndex );

        // Factorize

        // Solve

        // Cleanup

        SparseFrame_cleanup_matrix ( matrix_info + matrixIndex );

        // Output

#ifdef PRINT_INFO
        printf ("Read time:      %lf\n", matrix_info->read_time);
        printf ("Analyze time:   %lf\n", matrix_info->analyze_time);
        printf ("Factorize time: %lf\n", matrix_info->factorize_time);
        printf ("Solve time:     %lf\n", matrix_info->solve_time);
#endif
    }

    // Free resources

    SparseFrame_free_gpu (&gpu_info, common_info);

    SparseFrame_free_matrix ( &matrix_info, common_info );

    return 0;
}
