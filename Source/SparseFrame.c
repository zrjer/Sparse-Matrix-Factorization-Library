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

int SparseFrame_allocate_gpu ( struct gpu_info_struct **gpu_info_ptr, struct common_info_struct *common_info )
{
    int numGPU;
    int gpu_index;

#ifdef PRINT_CALLS
    printf ("\n================SparseFrame_allocate_gpu================\n\n");
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
        size_t devMemSize;
        size_t hostMemSize;

        cudaError_t cudaStatus;

        cudaSetDevice ( gpu_index );
        cudaGetDeviceProperties ( &prop, gpu_index );

        devMemSize = prop.totalGlobalMem;
        devMemSize = ( size_t ) ( ( double ) devMemSize * 0.9 );
        devMemSize = devMemSize - devMemSize % ( 0x400 * 0x400 ); // align to 1 MB
        cudaStatus = cudaMalloc ( &( (*gpu_info_ptr)[gpu_index].devMem ), devMemSize );

        if ( cudaStatus == cudaSuccess )
        {
            hostMemSize = devMemSize;
            cudaStatus = cudaMallocHost ( &( (*gpu_info_ptr)[gpu_index].hostMem ), hostMemSize );

            if ( cudaStatus == cudaSuccess )
            {
                (*gpu_info_ptr)[gpu_index].busy = 0;
                (*gpu_info_ptr)[gpu_index].devMemSize = devMemSize;
                (*gpu_info_ptr)[gpu_index].hostMemSize = hostMemSize;
#ifdef PRINT_INFO
                printf ( "GPU %d device memory size = %lf GiB host memory size = %lf GiB\n", gpu_index, ( double ) devMemSize / ( 0x400 * 0x400 * 0x400 ), ( double ) hostMemSize / ( 0x400 * 0x400 * 0x400 ) );
#endif
            }
            else
            {
                (*gpu_info_ptr)[gpu_index].busy = 1;
                (*gpu_info_ptr)[gpu_index].devMem = NULL;
                (*gpu_info_ptr)[gpu_index].devMemSize = 0;
                (*gpu_info_ptr)[gpu_index].hostMem = NULL;
                (*gpu_info_ptr)[gpu_index].hostMemSize = 0;
#ifdef PRINT_INFO
                printf ( "GPU %d cudaMalloc fail\n", gpu_index );
#endif
            }
        }
        else
        {
            (*gpu_info_ptr)[gpu_index].busy = 1;
            (*gpu_info_ptr)[gpu_index].devMem = NULL;
            (*gpu_info_ptr)[gpu_index].devMemSize = 0;
            (*gpu_info_ptr)[gpu_index].hostMem = NULL;
            (*gpu_info_ptr)[gpu_index].hostMemSize = 0;
#ifdef PRINT_INFO
            printf ( "GPU %d cudaMalloc fail\n", gpu_index );
#endif
        }
    }

    return 0;
}

int SparseFrame_free_gpu ( struct gpu_info_struct **gpu_info_ptr, struct common_info_struct *common_info )
{
    int numGPU;
    int gpu_index;

#ifdef PRINT_CALLS
    printf ("\n================SparseFrame_free_gpu================\n\n");
#endif

    if ( *gpu_info_ptr == NULL ) return 1;

    numGPU = common_info->numGPU;

    for ( gpu_index = 0; gpu_index < numGPU; gpu_index++ )
    {
        cudaSetDevice ( gpu_index );

        if ( (*gpu_info_ptr)[gpu_index].devMem != NULL )
            cudaFree ( (*gpu_info_ptr)[gpu_index].devMem );
        if ( (*gpu_info_ptr)[gpu_index].hostMem != NULL )
            cudaFreeHost ( (*gpu_info_ptr)[gpu_index].hostMem );

        (*gpu_info_ptr)[gpu_index].busy = 1;
        (*gpu_info_ptr)[gpu_index].devMem = NULL;
        (*gpu_info_ptr)[gpu_index].devMemSize = 0;
        (*gpu_info_ptr)[gpu_index].hostMem = NULL;
        (*gpu_info_ptr)[gpu_index].hostMemSize = 0;
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
    printf ("\n================SparseFrame_allocate_matrix================\n\n");
#endif

    numSparseMatrix = common_info->numSparseMatrix;

    *matrix_info_ptr = malloc ( numSparseMatrix * sizeof ( struct matrix_info_struct ) );

    if ( *matrix_info_ptr == NULL ) return 1;

    return 0;
}

int SparseFrame_free_matrix ( struct matrix_info_struct **matrix_info_ptr, struct common_info_struct *common_info )
{
#ifdef PRINT_CALLS
    printf ("\n================SparseFrame_free_matrix================\n\n");
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
    Long j, ncol;
    Long nz, nzmax;
    Long p;
    Long *Tj, *Ti, *Cp, *Ci;
    Float *Tx, *Ty, *Cx, *Cy;

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

    workspace = matrix_info->workspace;
    memcpy ( workspace, Cp, sizeof(Long) * ( ncol + 1 ) );

    for ( nz = 0; nz < nzmax; nz++ )
    {
        p = workspace [ Tj [ nz ] ] ++;
        Ci [p] = Ti[nz];
        Cx [p] = Tx[nz];
        if ( isComplex )
            Cy [p] = Ty[nz];
    }

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
    printf ("\n================SparseFrame_read_matrix================\n\n");
#endif

    timestamp = SparseFrame_time ();

    matrix_info->file = fopen ( path, "r" );

    if ( matrix_info->file == NULL ) return 1;

    buf = malloc ( max_mm_line_size * sizeof(char) );

    SparseFrame_read_matrix_triplet ( &buf, matrix_info );

    fclose ( matrix_info->file );

    free ( buf );

    matrix_info->workSize = MAX ( ( 6 * matrix_info->nrow + 2 * matrix_info->nzmax + 1 ) * sizeof(Long), ( 3 * matrix_info->nrow + 2 * matrix_info->nzmax + 1 ) * sizeof(idx_t) );
    matrix_info->workspace = malloc ( matrix_info->workSize );

    SparseFrame_compress ( matrix_info );

    matrix_info->readTime = SparseFrame_time () - timestamp;

#ifdef PRINT_INFO
    printf ( "Matrix read time: %lf seconds\n", matrix_info->readTime );
#endif

    return 0;
}

int SparseFrame_amd ( struct matrix_info_struct *matrix_info )
{
    Long j, i, ncol, nrow, p;
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

    Head = matrix_info->Head;
    Next = matrix_info->Next;
    Perm = matrix_info->Perm;

    workspace = matrix_info->workspace;

    Len    = workspace + 0 * nrow;
    Nv     = workspace + 1 * nrow;
    Elen   = workspace + 2 * nrow;
    Degree = workspace + 3 * nrow;
    Wi     = workspace + 4 * nrow;
    Ap     = workspace + 5 * nrow;
    Ai     = workspace + 6 * nrow + 1;

    memset ( Ap, 0, ( nrow + 1 ) * sizeof(Long) );

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

    memcpy ( workspace, Ap, ( nrow + 1 ) * sizeof(Long) ); // Be careful of overwriting Ap

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

    for ( j = 0; j < matrix_info->nrow; j++ )
        Len[j] = Ap[j+1] - Ap[j];

    Control[AMD_DENSE] = prune_dense;
    Control[AMD_AGGRESSIVE] = aggressive;

    amd_l2 ( nrow, Ap, Ai, Len, anz, Ap[nrow], Nv, Next, Perm, Head, Elen, Degree, Wi, Control, Info );

    return 0;
}

int SparseFrame_metis ( struct matrix_info_struct *matrix_info )
{
    Long j, i, ncol, nrow, p;
    Long *Cp, *Ci;
    Long *Head, *Next, *Perm;

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

    Head = matrix_info->Head;
    Next = matrix_info->Next;
    Perm = matrix_info->Perm;

    Mworkspace = matrix_info->workspace;

    if ( sizeof(idx_t) == sizeof(Long) )
        Mperm = Perm;
    else
        Mperm  = Mworkspace + 0 * nrow;
    Miperm = Mworkspace + 1 * nrow;
    Mp     = Mworkspace + 2 * nrow;
    Mi     = Mworkspace + 3 * nrow + 1;

    memset ( Mp, 0, ( nrow + 1 ) * sizeof(idx_t) );

    for ( j = 0; j < ncol; j++ )
    {
        for ( p = Cp[j]; p < Cp[j+1]; p++ )
        {
            i = Ci[p];
            if (i > j)
            {
                Mp[i+1]++;
                Mp[j+1]++;
            }
        }
    }

    for ( j = 0; j < nrow; j++)
    {
        Mp[j+1] += Mp[j];
    }

    mnz = Mp[nrow];

    memcpy ( Mworkspace, Mp, ( nrow + 1 ) * sizeof(idx_t) ); // Be careful of overwriting Mp

    for ( j = 0; j < ncol; j++ )
    {
        for ( p = Cp[j]; p < Cp[j+1]; p++ )
        {
            i = Ci[p];
            if (i > j)
            {
                Mi [ Mworkspace[i]++ ] = j;
                Mi [ Mworkspace[j]++ ] = i;
            }
        }
    }

    if ( mnz == 0 )
    {
        for ( i = 0; i < nrow; i++ )
        {
            Mperm[i] = i;
        }
    }
    else
    {
        METIS_NodeND (&nrow, Mp, Mi, NULL, NULL, Mperm, Miperm);
    }

    if ( sizeof(idx_t) != sizeof(Long) )
    {
        for ( i = 0; i < nrow; i++ )
        {
            Perm[i] = Mperm[i];
        }
    }

    return 0;
}

int SparseFrame_etree ( struct matrix_info_struct *matrix_info )
{
    Long nrow;
    Long *Head, *Next, *Perm, *ColCount;

    Long *workspace;
    Long *Parent, *Post, *First, *Level;

    nrow = matrix_info->nrow;

    Head = matrix_info->Head;
    Next = matrix_info->Next;
    Perm = matrix_info->Perm;

    ColCount = malloc ( nrow * sizeof(Long) );
    matrix_info->ColCount = ColCount;

    workspace = matrix_info->workspace;

    Parent = workspace + 0 * nrow;
    Post   = workspace + 1 * nrow;
    First  = workspace + 2 * nrow;
    Level  = workspace + 3 * nrow;

    return 0;
}

int SparseFrame_analyze ( struct matrix_info_struct *matrix_info )
{
    double timestamp;

#ifdef PRINT_CALLS
    printf ("\n================SparseFrame_analyze================\n\n");
#endif

    timestamp = SparseFrame_time ();

    matrix_info->Head = malloc ( matrix_info->nrow * sizeof(Long) );
    matrix_info->Next = malloc ( matrix_info->nrow * sizeof(Long) );
    matrix_info->Perm = malloc ( matrix_info->nrow * sizeof(Long) );

    SparseFrame_amd ( matrix_info );

    SparseFrame_metis ( matrix_info );

    SparseFrame_etree ( matrix_info );

    matrix_info->analyzeTime = SparseFrame_time () - timestamp;

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
    if ( matrix_info->Ty != NULL ) free ( matrix_info->Ty );

    if ( matrix_info->Cp != NULL ) free ( matrix_info->Cp );
    if ( matrix_info->Ci != NULL ) free ( matrix_info->Ci );
    if ( matrix_info->Cx != NULL ) free ( matrix_info->Cx );
    if ( matrix_info->Cy != NULL ) free ( matrix_info->Cy );

    if ( matrix_info->Head != NULL ) free ( matrix_info->Head );
    if ( matrix_info->Next != NULL ) free ( matrix_info->Next );
    if ( matrix_info->Perm != NULL ) free ( matrix_info->Perm );
    if ( matrix_info->ColCount != NULL ) free ( matrix_info->ColCount );

    if ( matrix_info->workspace != NULL ) free ( matrix_info->workspace );

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

    matrix_info->Head = NULL;
    matrix_info->Next = NULL;
    matrix_info->Perm = NULL;
    matrix_info->ColCount = NULL;

    matrix_info->workspace = NULL;

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
    printf ("\n================SparseFrame================\n\n");
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

        {
            Long k;
            for (k = 0; k < (matrix_info+matrixIndex)->ncol; k++)
            {
                printf ("Head[%8ld] = %8ld Next[%8ld] = %8ld Perm[%8ld] = %8ld\n", k, (matrix_info+matrixIndex)->Head[k], k, (matrix_info+matrixIndex)->Next[k], k, (matrix_info+matrixIndex)->Perm[k]);
                //if (k != -1 && (matrix_info+matrixIndex)->Next[k] != -1 && (matrix_info+matrixIndex)->Perm[(matrix_info+matrixIndex)->Next[k]] != k)
                //    printf ("exception k = %ld\n", k);
            }
        }

        // Factorize

        // Solve

        // Cleanup

        SparseFrame_cleanup_matrix ( matrix_info + matrixIndex );

        // Output

#ifdef PRINT_INFO
        printf ("Read time:      %lf\n", (matrix_info+matrixIndex)->readTime);
        printf ("Analyze time:   %lf\n", (matrix_info+matrixIndex)->analyzeTime);
        printf ("Factorize time: %lf\n", (matrix_info+matrixIndex)->factorizeTime);
        printf ("Solve time:     %lf\n", (matrix_info+matrixIndex)->solveTime);
#endif
    }

    // Free resources

    SparseFrame_free_gpu (&gpu_info, common_info);

    SparseFrame_free_matrix ( &matrix_info, common_info );

    return 0;
}
