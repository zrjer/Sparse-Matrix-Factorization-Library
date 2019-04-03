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
#if ( defined ( MAX_NUM_GPU ) && ( MAX_NUM_GPU >= 0 ) )
    numGPU = MIN ( numGPU, MAX_NUM_GPU );
#endif
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
        size_t sharedMemSize;

        cudaError_t cudaStatus;

        cudaSetDevice ( gpu_index );
        cudaGetDeviceProperties ( &prop, gpu_index );

        devMemSize = prop.totalGlobalMem;
        devMemSize = ( size_t ) ( ( double ) devMemSize * 0.9 );
        devMemSize = devMemSize - devMemSize % ( 0x400 * 0x400 ); // align to 1 MB
        cudaStatus = cudaMalloc ( &( (*gpu_info_ptr)[gpu_index].devMem ), devMemSize );
        sharedMemSize = prop.sharedMemPerBlock;

        if ( cudaStatus == cudaSuccess )
        {
            hostMemSize = devMemSize;
            cudaStatus = cudaMallocHost ( &( (*gpu_info_ptr)[gpu_index].hostMem ), hostMemSize );

            if ( cudaStatus == cudaSuccess )
            {
                (*gpu_info_ptr)[gpu_index].busy = 0;
                (*gpu_info_ptr)[gpu_index].devMemSize = devMemSize;
                (*gpu_info_ptr)[gpu_index].hostMemSize = hostMemSize;
                (*gpu_info_ptr)[gpu_index].sharedMemSize = sharedMemSize;
#ifdef PRINT_INFO
                printf ( "GPU %d device memory size = %lf GiB host memory size = %lf GiB shared memory size per block = %ld KiB\n",
                        gpu_index, ( double ) devMemSize / ( 0x400 * 0x400 * 0x400 ), ( double ) hostMemSize / ( 0x400 * 0x400 * 0x400 ), sharedMemSize / 1024 );
#endif
            }
            else
            {
                (*gpu_info_ptr)[gpu_index].busy = 1;
                (*gpu_info_ptr)[gpu_index].devMem = NULL;
                (*gpu_info_ptr)[gpu_index].devMemSize = 0;
                (*gpu_info_ptr)[gpu_index].hostMem = NULL;
                (*gpu_info_ptr)[gpu_index].hostMemSize = 0;
                (*gpu_info_ptr)[gpu_index].sharedMemSize = 0;
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
#ifdef PRINT_INFO
    printf ("\n");
#endif

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

    int isComplex, isSymmetric;
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
        isSymmetric = 1;
    else
        isSymmetric = 0;

    matrix_info->isSymmetric = isSymmetric;

    if ( strcmp ( s3, "real" ) == 0 )
        isComplex = 0;
    else
        isComplex = 1;

    matrix_info->isComplex = isComplex;

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
    printf ( "matrix %s is ", basename(matrix_info->path) );
    if ( isComplex == 0 && isSymmetric != 0)
        printf ("real symmetric, ncol = %ld nrow = %ld nzmax = %ld\n\n", ncol, nrow, nzmax);
    else if ( isComplex != 0 && isSymmetric != 0)
        printf ("complex symmetric, ncol = %ld nrow = %ld nzmax = %ld\n\n", ncol, nrow, nzmax);
    else if ( isComplex == 0 && isSymmetric == 0)
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
            if ( nz >= nzmax )
            {
                printf ( "Error: nzmax exceeded\n" );
                return 1;
            }
            n_scanned = sscanf ( *buf_ptr, "%ld %ld %lg %lg\n", &Ti, &Tj, &Tx, &Ty );
            if ( isComplex && n_scanned < 4 )
            {
                printf ( "Error: imaginary part not present\n" );
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

    for ( nz = 0; nz < nzmax; nz++ )
        Cp[ Tj[nz] + 1 ] ++;

    for ( j = 0; j < ncol; j++ )
        Cp[j+1] += Cp[j];

    memcpy ( workspace, Cp, sizeof(Long) * ( ncol + 1 ) );

    for ( nz = 0; nz < nzmax; nz++ )
    {
        p = workspace [ Tj [ nz ] ] ++;
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
    matrix_info->Up = NULL;
    matrix_info->Ui = NULL;
    matrix_info->Ux = NULL;

    matrix_info->Head = NULL;
    matrix_info->Next = NULL;
    matrix_info->Perm = NULL;
    matrix_info->Pinv = NULL;
    matrix_info->Post = NULL;
    matrix_info->Parent = NULL;
    matrix_info->ColCount = NULL;
    matrix_info->RowCount = NULL;

    matrix_info->nsuper = 0;
    matrix_info->Super = NULL;
    matrix_info->SuperMap = NULL;
    matrix_info->Sparent = NULL;

    matrix_info->nsleaf = 0;
    matrix_info->Leaf = NULL;

    matrix_info->isize = 0;
    matrix_info->xsize = 0;
    matrix_info->Lsip = NULL;
    matrix_info->Lsxp = NULL;
    matrix_info->Lsi = NULL;
    matrix_info->Lsx = NULL;

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

    buf = malloc ( max_mm_line_size * sizeof(char) );

    SparseFrame_read_matrix_triplet ( &buf, matrix_info );

    fclose ( matrix_info->file );

    free ( buf );

    matrix_info->workSize = MAX ( MAX ( ( 6 * matrix_info->nrow + 2 * matrix_info->nzmax + 1 ) * sizeof(Long), ( 3 * matrix_info->nrow + 2 * matrix_info->nzmax + 1 ) * sizeof(idx_t) ), ( matrix_info->nrow ) * sizeof(Float) );
    matrix_info->workspace = malloc ( matrix_info->workSize );

    SparseFrame_compress ( matrix_info );

    matrix_info->readTime = SparseFrame_time () - timestamp;

    return 0;
}

int SparseFrame_amd ( struct matrix_info_struct *matrix_info )
{
    Long j, i, p, ncol, nrow;
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

    Len    = workspace;
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
    Long j, i, p, ncol, nrow;
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
        Mperm = Perm;
    else
        Mperm  = Mworkspace;
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

int SparseFrame_perm ( struct matrix_info_struct *matrix_info )
{
    int isComplex;
    Long j, i, lp, up, nrow;
    Long jold, iold, pold;
    Long *Cp, *Ci;
    Float *Cx;
    Long *Lp, *Li;
    Float *Lx;
    Long *Up, *Ui;
    Float *Ux;
    Long *Perm, *Pinv;

    Long *Lworkspace, *Uworkspace;

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

    Up = matrix_info->Up;
    Ui = matrix_info->Ui;
    Ux = matrix_info->Ux;

    Perm = matrix_info->Perm;
    Pinv = matrix_info->Pinv;

    Lworkspace = matrix_info->workspace;
    Uworkspace = matrix_info->workspace + ( nrow + 1 ) * sizeof(Long);

    memset ( Lp, 0, ( nrow + 1 ) * sizeof(Long) );
    memset ( Up, 0, ( nrow + 1 ) * sizeof(Long) );

    for ( j = 0; j < nrow; j++ )
    {
        Pinv[j] = -1;
    }

    for ( j = 0; j < nrow; j++ )
    {
        jold = Perm[j];
        if ( jold >= 0 )
            Pinv[ jold ] = j;
    }

    for ( j = 0; j < nrow; j++ )
    {
        jold = Perm[j];
        if ( jold >= 0 )
        {
            for ( pold = Cp[jold]; pold < Cp[jold+1]; pold++ )
            {
                iold = Ci[pold];
                i = Pinv[iold];
                Lp [ MIN (i, j) + 1 ] ++;
                Up [ MAX (i, j) + 1 ] ++;
            }
        }
    }

    for ( j = 0; j < nrow; j++ )
    {
        Lp[j+1] += Lp[j];
        Up[j+1] += Up[j];
    }

    memcpy ( Lworkspace, Lp, ( nrow + 1 ) * sizeof(Long) );
    memcpy ( Uworkspace, Up, ( nrow + 1 ) * sizeof(Long) );

    for ( j = 0; j < nrow; j++ )
    {
        jold = Perm[j];
        if ( jold >= 0 )
        {
            for ( pold = Cp[jold]; pold < Cp[jold+1]; pold++ )
            {
                iold = Ci[pold];
                i = Pinv[iold];

                lp = Lworkspace [ MIN(i, j) ] ++;
                Li[lp] = MAX(i, j);
                if ( !isComplex )
                    Lx[lp] = Cx[pold];
                else
                    ( (Complex*) Lx ) [lp] = ( (Complex*) Cx ) [pold];

                Li[lp] = MAX(i, j);
                up = Uworkspace [ MAX(i, j) ] ++;
                Ui[up] = MIN(i, j);
                if ( !isComplex )
                    Ux[up] = Cx[pold];
                else
                    ( (Complex*) Ux ) [up] = ( (Complex*) Cx ) [pold];
            }
        }
    }

    return 0;
}

int SparseFrame_etree ( struct matrix_info_struct *matrix_info )
{
    Long j, i, p, nrow;
    Long *Up, *Ui;
    Long *Parent;

    Long ancestor;
    Long *workspace;
    Long *Ancestor;

#ifdef PRINT_CALLS
    printf ("\n================SparseFrame_etree================\n\n");
#endif

    nrow = matrix_info->nrow;

    Up = matrix_info->Up;
    Ui = matrix_info->Ui;

    Parent = matrix_info->Parent;

    workspace = matrix_info->workspace;

    Ancestor = workspace;

    for ( j = 0; j < nrow; j++ )
    {
        Parent[j] = -1;
        Ancestor[j] = -1;
    }

    for ( j = 0; j < nrow; j++ )
    {
        for ( p = Up[j]; p < Up[j+1]; p++ )
        {
            i = Ui[p];
            if ( i < j )
            {
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
    Long j, k, p, nrow;
    Long *Head, *Next;
    Long *Post, *Parent, *ColCount;

    Long *workspace;
    Long child;
    Long w, jnext, top;
    Long *Whead, *Stack;

#ifdef PRINT_CALLS
    printf ("\n================SparseFrame_postorder================\n\n");
#endif

    nrow = matrix_info->nrow;

    Head = matrix_info->Head;
    Next = matrix_info->Next;
    Post = matrix_info->Post;
    Parent = matrix_info->Parent;

    ColCount = matrix_info->ColCount;

    workspace = matrix_info->workspace;

    for ( j = 0; j < nrow; j++ )
    {
        Head[j] = -1;
        Next[j] = -1;
    }

    if ( ColCount == NULL )
    {
        for ( j = nrow - 1; j >= 0; j-- )
        {
            p = Parent[j];
            if ( p >= 0 && p < nrow )
            {
                Next[j] = Head[p];
                Head[p] = j;
            }
        }
    }
    else
    {
        Whead = workspace;
        for ( j = 0; j < nrow; j++ )
            Whead[j] = -1;
        for ( j = 0; j < nrow; j++ )
        {
            p = Parent[j];
            if ( p >= 0 )
            {
                w = ColCount[j];
                Next[j] = Whead[w];
                Whead[w] = j;
            }
        }
        for ( w = nrow - 1; w >= 0; w-- )
        {
            j = Whead[w];
            while ( j >= 0 )
            {
                jnext = Next[j];
                p = Parent[j];
                Next[j] = Head[p];
                Head[p] = j;
                j = jnext;
            }
        }
    }

    Stack = workspace;
    top = -1;

    for ( j = nrow - 1; j >= 0; j-- )
    {
        p = Parent[j];
        if ( p < 0 )
        {
            top++;
            Stack[top] = j;
        }
    }

    k = 0;

    while ( top >= 0 )
    {
        j = Stack[top];
        child = Head[j];
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
    Long j, i, k, p, q, nrow;
    Long *Lp, *Li;
    Long *Post, *Parent, *ColCount, *RowCount;

    Long *workspace;
    Long prevleaf;
    Long s;
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

    for ( k = nrow - 1; k >= 0; k-- )
    {
        j = Post[k];
        p = Parent[j];
        if ( p < 0 )
            Level[j] = 0;
        else
            Level[j] = Level[p] + 1;
    }

    for ( j = 0; j < nrow; j++ )
    {
        First[j] = -1;
    }

    for ( k = 0; k < nrow; k++ )
    {
        j = Post[k];
        for ( p = j; p >= 0 && First[p] < 0; p = Parent[p] )
        {
            First[p] = k;
        }
    }

    for ( k = 0; k < nrow; k++ )
    {
        j = Post[k];
        ColCount[j] = 0;
        RowCount[j] = 1;
    }

    for ( k = 0; k < nrow; k++ )
    {
        j = Post[k];
        SetParent[j] = j;
        PrevLeaf[j] = j;
        PrevNbr[j] = -1;
    }

    for ( k = 0; k < nrow; k++ )
    {
        j = Post[k];
        PrevNbr[j] = k;
        for ( p = Lp[j]; p < Lp[j+1]; p++ )
        {
            i = Li[p];
            if ( i > j )
            {
                if ( First[j] > PrevNbr[i] )
                {
                    prevleaf = PrevLeaf[i];
                    for ( q = prevleaf; q != SetParent[q]; q = SetParent[q] );
                    for ( s = prevleaf; s != q; s = SetParent[s] )
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

    for ( k = 0; k < nrow; k++ )
    {
        j = Post[k];
        p = Parent[j];
        if ( p >= 0 )
            ColCount[p] += ColCount[j];
    }

    for ( k = 0; k < nrow; k++ )
    {
        j = Post[k];
        ColCount[j]++;
    }

    return 0;
}

int SparseFrame_analyze_supernodal ( struct matrix_info_struct *matrix_info )
{
    int isComplex;

    Long j, i, k, p, nrow;

    Long *Up, *Ui;

    Long *Perm, *Post, *Parent, *ColCount;

    Long *workspace;

    Long *InvPost;
    Long *Bperm, *Bparent, *Bcolcount;

    Long parent, s, sparent, sdescendant, nfsuper, nsuper;
    Long *Super, *SuperMap, *Sparent;
    Long *Nchild, *Nscol, *Scolcount;
#ifdef RELAX_RATE
    Long smerge;
    Long *Nschild, *Nsz, *Merge;
    Long s_ncol, s_colcount, s_zero;
    Long p_ncol, p_colcount, p_zero;
    Long new_zero, total_zero;
#endif
    Long *Lsip_copy, *Marker;

    Long isize, xsize;
    Long *Lsip, *Lsxp, *Lsi;
    Float *Lsx;

    Long nsleaf;
    Long *Leaf;

#ifdef PRINT_CALLS
    printf ("\n================SparseFrame_analyze_supernodal================\n\n");
#endif

    isComplex = matrix_info->isComplex;

    nrow = matrix_info->nrow;

    Up = matrix_info->Up;
    Ui = matrix_info->Ui;

    Perm = matrix_info->Perm;
    Post = matrix_info->Post;
    Parent = matrix_info->Parent;
    ColCount = matrix_info->ColCount;

    Super = matrix_info->Super;
    SuperMap = matrix_info->SuperMap;
    Sparent = matrix_info->Sparent;

    workspace = matrix_info->workspace;

    InvPost = workspace;

    Bperm = workspace + 1 * nrow;
    Bparent = workspace + 2 * nrow;
    Bcolcount = workspace + 3 * nrow;

    for (k = 0; k < nrow; k++)
    {
        InvPost [ Post [k] ] = k;
    }

    for (k = 0; k < nrow; k++)
    {
        Bperm[k] = Perm [ Post [ k ] ];
        parent = Parent [ Post [ k ] ];
        Bparent[k] = ( parent < 0 ) ? -1 : InvPost[parent];
        Bcolcount[k] = ColCount [ Post [ k ] ];
    }

    memcpy ( Perm, Bperm, nrow * sizeof(Long) );
    memcpy ( Parent, Bparent, nrow * sizeof(Long) );
    memcpy ( ColCount, Bcolcount, nrow * sizeof(Long) );

    SparseFrame_perm ( matrix_info );

    Nchild = workspace;
    Nscol = workspace + 1 * nrow;
    Scolcount = workspace + 2 * nrow;
#ifdef RELAX_RATE
    Nschild = Nchild; // use Nchild
    Nsz = workspace + 3 * nrow;
    Merge = workspace + 4 * nrow;
#endif

    memset ( Nchild, 0, nrow * sizeof(Long) );

    for ( j = 0; j < nrow; j++ )
    {
        parent = Parent[j];
        if ( parent >= 0 && parent < nrow )
            Nchild[parent]++;
    }

    nfsuper = ( nrow > 0 ) ? 1 : 0;
    Super[0] = 0;

    for ( j = 1; j < nrow; j++ )
    {
        if ( Parent[j-1] != j || ColCount[j-1] != ColCount[j] + 1 || Nchild[j] > 1 )
        {
            Super[nfsuper] = j;
            nfsuper++;
        }
    }
    Super[nfsuper] = nrow;

    for ( s = 0; s < nfsuper; s++ )
    {
        Nscol[s] = Super[s+1] - Super[s];
        Scolcount[s] = ColCount [ Super [ s ] ];
    }

    for ( s = 0; s < nfsuper; s++ )
    {
        for ( j = Super[s]; j < Super[s+1]; j++ )
            SuperMap[j] = s;
    }

    for ( s = 0; s < nfsuper; s++ )
    {
        j = Super[s+1] - 1;
        parent = Parent[j];
        Sparent[s] = ( parent < 0 ) ? -1 : SuperMap[parent];
    }

#ifdef RELAX_RATE
    if ( RELAX_RATE > 0 )
    {
        memset ( Nschild, 0, nfsuper * sizeof(Long) );

        for ( s = 0; s < nfsuper; s++ )
        {
            sparent = Sparent[s];
            if ( sparent >= 0 && sparent < nfsuper )
                Nschild[sparent]++;
        }

        for ( s = 0; s < nfsuper; s++ )
        {
            Merge[s] = s;
        }

        if ( RELAX_RATE < 1 )
        {
            for ( s = 0; s < nfsuper; s++ )
            {
                Nsz[s] = 0;
            }
        }

        for ( s = 0; s < nfsuper; s++ )
        {
            sparent = Sparent[s];
            if ( sparent >= 0 && sparent < nfsuper && Nschild[sparent] == 1 && sparent == s+1 )
            {
                smerge = Merge[s];

                if ( RELAX_RATE < 1 )
                {
                    s_ncol = Nscol[smerge];
                    p_ncol = Nscol[sparent];
                    s_colcount = Scolcount[smerge];
                    p_colcount = Scolcount[sparent];
                    s_zero = Nsz[smerge];
                    p_zero = Nsz[sparent];
                    new_zero = s_ncol * ( s_ncol + p_colcount - s_colcount );
                    total_zero = s_zero + p_zero + new_zero;
                    if ( (double)total_zero / ( ( s_ncol + p_ncol ) * ( s_ncol + p_ncol + 1 ) / 2 + ( s_ncol + p_ncol ) * ( p_colcount - p_ncol ) ) < RELAX_RATE )
                    {
                        Nscol[smerge] = s_ncol + p_ncol;
                        Scolcount[smerge] = s_ncol + p_colcount;
                        Nsz[smerge] = total_zero;
                        Merge[sparent] = smerge;
                    }
                }
                else
                    Merge[sparent] = smerge;
            }
        }
    }

    nsuper = 0;

    for ( s = 0; s < nfsuper; s++ )
    {
        if ( Merge[s] == s )
        {
            Super[nsuper] = Super[s];
            Nscol[nsuper] = Nscol[s];
            Scolcount[nsuper] = Scolcount[s];
            nsuper++;
        }
    }

    for ( s = nsuper; s < nfsuper; s++ )
        Super[s] = -1;

    Super[nsuper] = nrow;

    for ( s = 0; s < nsuper; s++ )
    {
        for ( j = Super[s]; j < Super[s+1]; j++ )
            SuperMap[j] = s;
    }

    for ( s = 0; s < nsuper; s++ )
    {
        j = Super[s+1] - 1;
        parent = Parent[j];
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

    for ( s = 0; s < nsuper; s++ )
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

    Lsip_copy = workspace + 1 * nrow + 1; // don't overwrite Super
    Marker = workspace + 2 * nrow + 1;

    memcpy ( Lsip_copy, Lsip, ( nsuper + 1 ) * sizeof(Long) );

    for ( s = 0; s < nsuper; s++ )
    {
        Marker[s] = Super[s+1];
    }

    for ( s = 0; s < nsuper; s++ )
    {
        for ( k = Super[s]; k < Super[s+1]; k++ )
        {
            Lsi [ Lsip_copy[s]++ ] = k;
        }
    }

    for ( s = 0; s < nsuper; s++ )
    {
        for ( j = Super[s]; j < Super[s+1]; j++ )
        {
            for ( p = Up[j]; p < Up[j+1]; p++ )
            {
                i = Ui[p];
                for ( sdescendant = SuperMap[i]; sdescendant >= 0 && Marker[sdescendant] <= j; sdescendant = Sparent[sdescendant] )
                {
                    Lsi [ Lsip_copy[sdescendant]++ ] = j;
                    Marker[sdescendant] = j+1;
                }
            }
        }
    }

    Leaf = calloc ( nsuper, sizeof(Long) );
    for ( j = 0; j < nrow; j++ )
    {
        if ( Parent[j] >= 0 )
            Leaf [ SuperMap [ Parent[j] ] ] = 1;
    }

    nsleaf = 0;
    for ( s = 0; s < nsuper; s++ )
    {
        if ( Leaf[s] == 0 )
        {
            Leaf[nsleaf++] = s;
        }
    }

    for ( s = nsleaf; s < nsuper; s++ )
        Leaf[s] = -1;

    matrix_info->nsleaf = nsleaf;
    matrix_info->Leaf = Leaf;

    return 0;
}

int SparseFrame_analyze ( struct matrix_info_struct *matrix_info )
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

    matrix_info->Head = malloc ( nrow * sizeof(Long) );
    matrix_info->Next = malloc ( nrow * sizeof(Long) );
    matrix_info->Perm = malloc ( nrow * sizeof(Long) );
    matrix_info->Pinv = malloc ( nrow * sizeof(Long) );

    //SparseFrame_amd ( matrix_info );

    SparseFrame_metis ( matrix_info );

    matrix_info->Lp = malloc ( ( nrow + 1 ) * sizeof(Long) );
    matrix_info->Li = malloc ( nzmax * sizeof(Long) );
    if ( !isComplex )
        matrix_info->Lx = malloc ( nzmax * sizeof(Float) );
    else
        matrix_info->Lx = malloc ( nzmax * sizeof(Complex) );

    matrix_info->Up = malloc ( ( nrow + 1 ) * sizeof(Long) );
    matrix_info->Ui = malloc ( nzmax * sizeof(Long) );
    if ( !isComplex )
        matrix_info->Ux = malloc ( nzmax * sizeof(Float) );
    else
        matrix_info->Ux = malloc ( nzmax * sizeof(Complex) );

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

    SparseFrame_analyze_supernodal ( matrix_info );

    matrix_info->analyzeTime = SparseFrame_time () - timestamp;

    return 0;
}

int SparseFrame_factorize_supernodal ( struct common_info_struct *common_info, struct matrix_info_struct *matrix_info )
{
    int isComplex;
    Long nrow;
    Long *Lp, *Li;
    Float *Lx;

    Long nsuper, s;
    Long *Super, *SuperMap;

    Long xsize;
    Long *Lsip, *Lsxp, *Lsi;
    Float *Lsx;

    Long *Head, *Next;

    Long *Map, *RelativeMap, *Lpos;

    Long csize;
    Float *C;

#ifdef PRINT_CALLS
    printf ("\n================SparseFrame_factorize_supernodal================\n\n");
#endif

    isComplex = matrix_info->isComplex;
    nrow = matrix_info->nrow;

    Lp = matrix_info->Lp;
    Li = matrix_info->Li;
    Lx = matrix_info->Lx;

    nsuper = matrix_info->nsuper;
    Super = matrix_info->Super;
    SuperMap = matrix_info->SuperMap;

    xsize = matrix_info->xsize;
    Lsip = matrix_info->Lsip;
    Lsxp = matrix_info->Lsxp;
    Lsi = matrix_info->Lsi;
    Lsx = matrix_info->Lsx;

    if ( !isComplex )
        memset ( Lsx, 0, xsize * sizeof(Float) );
    else
        memset ( Lsx, 0, xsize * sizeof(Complex) );

    Head = matrix_info->Head;
    Next = matrix_info->Next;

    Map = malloc ( nrow * sizeof(Long) );
    RelativeMap = malloc ( nrow * sizeof(Long) );
    Lpos = malloc ( nsuper * sizeof(Long) );

    for ( s = 0; s < nsuper; s++ )
    {
        Head[s] = -1;
        Next[s] = -1;
    }

    for ( s = 0; s < nsuper; s++ )
        Lpos[s] = Super[s+1] - Super[s];

    csize = 0;
    for ( s = 0; s < nsuper; s++ )
    {
        Long sparent, sparent_last;
        Long nscol, nsrow;
        Long si, si_last;

        nscol = Super[s+1] - Super[s];
        nsrow = Lsip[s+1] - Lsip[s];

        if ( nscol < nsrow )
        {
            si_last = nscol;
            sparent_last = SuperMap [ Lsi [ Lsip[s] + nscol ] ];
            for ( si = nscol; si < nsrow; si++ )
            {
                sparent = SuperMap [ Lsi [ Lsip[s] + si ] ];
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
    if (!isComplex)
        C = malloc ( csize * sizeof(Float) );
    else
        C = malloc ( csize * sizeof(Complex) );

    for ( s = 0; s < nsuper; s++ )
    {
        Long j, i, p;

        Long sparent;
        Long nscol, nsrow;
        Long sj, si;

        Long sn, sm, slda;
        int info;

        nscol = Super[s+1] - Super[s];
        nsrow = Lsip[s+1] - Lsip[s];

        for ( si = 0; si < Lsip[s+1] - Lsip[s]; si++ )
            Map [ Lsi [ Lsip[s] + si ] ] = si;

        for ( j = Super[s]; j < Super[s+1]; j++ )
        {
            sj = j - Super[s];
            for ( p = Lp[j]; p < Lp[j+1]; p++ )
            {
                i = Li[p];
                si = Map[i];
                if ( !isComplex )
                    Lsx [ Lsxp[s] + sj * nsrow + si ] = Lx[p];
                else
                {
                    ( (Complex*) Lsx ) [ Lsxp[s] + sj * nsrow + si ].x = ( (Complex*) Lx )[p].x;
                    ( (Complex*) Lsx ) [ Lsxp[s] + sj * nsrow + si ].y = ( (Complex*) Lx )[p].y;
                }
            }
        }

        while ( Head[s] >= 0 )
        {
            Long d, di, dancestor;
            Long ndcol, ndrow;
            Long lpos_next;

            Long dn, dm, dk, dlda, dldc;

            Long cj, ci;

            d = Head[s];

            ndcol = Super[d+1] - Super[d];
            ndrow = Lsip[d+1] - Lsip[d];

            for ( lpos_next = Lpos[d]; lpos_next < ndrow && Lsi [ Lsip[d] + lpos_next ] < Super[s+1]; lpos_next++ );

            dn = lpos_next - Lpos[d];
            dm = ndrow - Lpos[d] - dn;
            dk = ndcol;
            dlda = ndrow;
            dldc = ndrow - Lpos[d];

            if (!isComplex)
                dsyrk_ ( "L", "N", &dn, &dk, one, Lsx + Lsxp[d] + Lpos[d], &dlda, zero, C, &dldc );
            else
                zherk_ ( "L", "N", &dn, &dk, one, (Complex*)Lsx + Lsxp[d] + Lpos[d], &dlda, zero, (Complex*)C, &dldc );

            if ( dm > 0 )
            {
                if (!isComplex)
                    dgemm_ ( "N", "C", &dm, &dn, &dk, one, Lsx + Lsxp[d] + lpos_next, &dlda, Lsx + Lsxp[d] + Lpos[d], &dlda, zero, C + dn, &dldc );
                else
                    zgemm_ ( "N", "C", &dm, &dn, &dk, one, (Complex*)Lsx + Lsxp[d] + lpos_next, &dlda, (Complex*)Lsx + Lsxp[d] + Lpos[d], &dlda, zero, (Complex*)C + dn, &dldc );
            }

            for ( di = Lpos[d]; di < ndrow; di++ )
            {
                RelativeMap [ di - Lpos[d] ] = Map [ Lsi[ Lsip[d] + di ] ];
            }

            for ( cj = 0; cj < dn; cj++ )
            {
                for ( ci = cj; ci < dn + dm; ci++ )
                {
                    if (!isComplex)
                        Lsx [ Lsxp[s] + RelativeMap [cj] * nsrow + RelativeMap[ci] ] -= C [ cj * dldc + ci ];
                    else
                    {
                        ( (Complex*) Lsx ) [ Lsxp[s] + RelativeMap [cj] * nsrow + RelativeMap[ci] ].x -= ( (Complex*) C ) [ cj * dldc + ci ].x;
                        ( (Complex*) Lsx ) [ Lsxp[s] + RelativeMap [cj] * nsrow + RelativeMap[ci] ].y -= ( (Complex*) C ) [ cj * dldc + ci ].y;
                    }
                }
            }

            Lpos[d] = lpos_next;
            Head[s] = Next[d];
            if ( lpos_next < ndrow )
            {
                dancestor = SuperMap [ Lsi [ Lsip[d] + lpos_next ] ];
                Next[d] = Head[dancestor];
                Head[dancestor] = d;
            }
        }

        sn = nscol;
        sm = nsrow - nscol;
        slda = nsrow;

        if (!isComplex)
            dpotrf_ ( "L", &sn, Lsx + Lsxp[s], &slda, &info );
        else
            zpotrf_ ( "L", &sn, (Complex*)Lsx + Lsxp[s], &slda, &info );

        if ( nscol < nsrow )
        {
            if (!isComplex)
                dtrsm_ ( "R", "L", "C", "N", &sm, &sn, one, Lsx + Lsxp[s], &slda, Lsx + Lsxp[s] + nscol, &slda );
            else
                ztrsm_ ( "R", "L", "C", "N", &sm, &sn, one, ( (Complex*) Lsx ) + Lsxp[s], &slda, ( (Complex*) Lsx ) + Lsxp[s] + nscol, &slda );
        }

        if ( nscol < nsrow )
        {
            sparent = SuperMap [ Lsi [ Lsip[s] + nscol ] ];
            Next[s] = Head[sparent];
            Head[sparent] = s;
        }
    }

    free ( Map );
    free ( RelativeMap );
    free ( Lpos );
    free ( C );

    return 0;
}

int SparseFrame_factorize ( struct common_info_struct *common_info, struct matrix_info_struct *matrix_info )
{
    double timestamp;

    int numThreads;

#ifdef PRINT_CALLS
    printf ("\n================SparseFrame_factorize================\n\n");
#endif

    timestamp = SparseFrame_time ();

    numThreads = common_info->numThreads;

    openblas_set_num_threads ( numThreads );

    SparseFrame_factorize_supernodal ( common_info, matrix_info );

    matrix_info->factorizeTime = SparseFrame_time () - timestamp;

    return 0;
}

int SparseFrame_solve_supernodal ( struct matrix_info_struct *matrix_info )
{
    double timestamp;

    int isComplex;
    Long nrow;

    Long s, nsuper;
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

    for ( s = 0; s < nsuper; s++ )
    {
        Long j, i;

        Long nscol, nsrow;
        Long sj, si;

        nscol = Super[s+1] - Super[s];
        nsrow = Lsip[s+1] - Lsip[s];

        for ( sj = 0; sj < nscol; sj++ )
        {
            j = Lsi [ Lsip[s] + sj ];

            if ( !isComplex )
                Xx[j] /= Lsx [ Lsxp[s] + sj * nsrow + sj ];
            else
            {
                // TODO
            }

            for ( si = sj + 1; si < nsrow; si++ )
            {
                i = Lsi [ Lsip[s] + si ];

                if ( !isComplex )
                    Xx[i] -= ( Lsx [ Lsxp[s] + sj * nsrow + si ] * Xx[j] );
                else
                {
                    // TODO
                }
            }
        }
    }

    for ( s = nsuper - 1; s >= 0; s-- )
    {
        Long j, i;

        Long nscol, nsrow;
        Long sj, si;

        nscol = Super[s+1] - Super[s];
        nsrow = Lsip[s+1] - Lsip[s];

        for ( sj = nscol - 1; sj >= 0; sj-- )
        {
            j = Lsi [ Lsip[s] + sj ];

            for ( si = sj + 1; si < nsrow; si++ )
            {
                i = Lsi [ Lsip[s] + si ];

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
    Long j, i, p, nrow;
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

    for ( i = 0; i < nrow; i++ )
    {
        if ( !isComplex )
        {
            Bx[i] = 1 + (Float)i / nrow;
        }
        else
        {
            ( (Complex*) Bx ) [i].x = 1 + i / (Float)nrow;
            ( (Complex*) Bx ) [i].y = ( (Float)nrow / 2 - i ) / ( 3 * nrow );
        }
    }

    SparseFrame_solve_supernodal ( matrix_info );

    for ( i = 0; i < nrow; i++ )
    {
        if ( !isComplex )
            Rx[i] = -Bx[i];
        else
        {
            ( (Complex*) Rx ) [i].x = - ( (Complex*) Bx) [i].x;
            ( (Complex*) Rx ) [i].y = - ( (Complex*) Bx) [i].y;
        }
    }

    for ( j = 0; j < nrow; j++ )
    {
        for ( p = Lp[j]; p < Lp[j+1]; p++ )
        {
            i = Li[p];
            Rx[i] += ( Lx[p] * Xx[j] );
            if ( i != j )
                Rx[j] += ( Lx[p] * Xx[i] );
        }
    }

    workspace = matrix_info->workspace;
    memset ( workspace, 0, nrow * sizeof(Float) );

    anorm = 0;
    for ( j = 0; j < nrow; j++ )
    {
        for ( p = Lp[j]; p < Lp[j+1]; p++ )
        {
            i = Li[p];
            if ( !isComplex )
            {
                workspace[j] += abs ( Lx[p] );
                if ( i != j )
                    workspace[i] += abs ( Lx[p] );
            }
            else
            {
                // TODO
            }
        }
    }
    for ( j = 0; j < nrow; j++ )
        if ( workspace[j] > anorm )
            anorm = workspace[j];

    bnorm = 0;
    xnorm = 0;
    rnorm = 0;
    for ( i = 0; i < nrow; i++ )
    {
        if ( !isComplex )
        {
            bnorm += ( Bx[i] * Bx[i] );
            xnorm += ( Xx[i] * Xx[i] );
            rnorm += ( Rx[i] * Rx[i] );
        }
        else
        {
            // TODO
        }
    }
    bnorm = sqrt ( bnorm );
    xnorm = sqrt ( xnorm );
    rnorm = sqrt ( rnorm );

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

    if ( matrix_info->Up != NULL ) free ( matrix_info->Up );
    if ( matrix_info->Ui != NULL ) free ( matrix_info->Ui );
    if ( matrix_info->Ux != NULL ) free ( matrix_info->Ux );

    if ( matrix_info->Head != NULL ) free ( matrix_info->Head );
    if ( matrix_info->Next != NULL ) free ( matrix_info->Next );
    if ( matrix_info->Perm != NULL ) free ( matrix_info->Perm );
    if ( matrix_info->Pinv != NULL ) free ( matrix_info->Pinv );
    if ( matrix_info->Post != NULL ) free ( matrix_info->Post );
    if ( matrix_info->Parent != NULL ) free ( matrix_info->Parent );
    if ( matrix_info->ColCount != NULL ) free ( matrix_info->ColCount );
    if ( matrix_info->RowCount != NULL ) free ( matrix_info->RowCount );

    if ( matrix_info->Super != NULL ) free ( matrix_info->Super );
    if ( matrix_info->SuperMap != NULL ) free ( matrix_info->SuperMap );
    if ( matrix_info->Sparent != NULL ) free ( matrix_info->Sparent );
    if ( matrix_info->Leaf != NULL ) free ( matrix_info->Leaf );

    if ( matrix_info->Lsip != NULL ) free ( matrix_info->Lsip );
    if ( matrix_info->Lsxp != NULL ) free ( matrix_info->Lsxp );
    if ( matrix_info->Lsi != NULL ) free ( matrix_info->Lsi );
    if ( matrix_info->Lsx != NULL ) free ( matrix_info->Lsx );

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

    int numThreads, numSparseMatrix, nextMatrixIndex;
    int matrixQueueSize, matrixQueueIndex;

    struct common_info_struct common_info_object;
    struct common_info_struct *common_info = &common_info_object;
    struct gpu_info_struct *gpu_info;
    struct matrix_info_struct *matrix_info;

#ifdef PRINT_CALLS
    printf ("\n================SparseFrame================\n\n");
#endif

    // Allocate resources

    timestamp = SparseFrame_time();

    SparseFrame_allocate_gpu (&gpu_info, common_info);

    numSparseMatrix = argc - 1;
    common_info->numSparseMatrix = numSparseMatrix;

    numThreads = omp_get_max_threads();
    common_info->numThreads = numThreads;

    matrixQueueSize = MIN ( MATRIX_QUEUE_SIZE, numSparseMatrix );

#ifdef PRINT_INFO
    printf ("Max threads = %d\n", numThreads);
    printf ("Num of matrices = %d\n\n", numSparseMatrix);
#endif

    SparseFrame_allocate_matrix ( &matrix_info, common_info );

    common_info->allocateTime = SparseFrame_time () - timestamp;

#ifdef PRINT_INFO
    printf ("Allocate time:        %lf\n\n", common_info->allocateTime);
#endif

    timestamp = SparseFrame_time();

    omp_set_nested ( TRUE );

    nextMatrixIndex = 0;

#pragma omp parallel for num_threads(matrixQueueSize) schedule(static)
    for ( matrixQueueIndex = 0; matrixQueueIndex < matrixQueueSize; matrixQueueIndex++ )
    {
        int matrixIndex;

        const char *path;

#pragma omp critical ( nextMatrixIndex )
        matrixIndex = nextMatrixIndex++;

        while ( matrixIndex < numSparseMatrix )
        {
            // Initialize
            SparseFrame_initialize_matrix ( matrix_info + matrixQueueIndex );

            // Read matrices

            path = argv [ 1 + matrixIndex ];
            ( matrix_info + matrixQueueIndex )->path = path;

            SparseFrame_read_matrix ( matrix_info + matrixQueueIndex );

            // Analyze

            SparseFrame_analyze ( matrix_info + matrixQueueIndex );

            // Factorize

            SparseFrame_factorize ( common_info, matrix_info + matrixQueueIndex );

            // Validate

            SparseFrame_validate ( matrix_info + matrixQueueIndex );

            // Cleanup

            SparseFrame_cleanup_matrix ( matrix_info + matrixQueueIndex );

            // Output

#ifdef PRINT_INFO
            printf ( "Matrix name:    %s\n", basename (path) );
            printf ( "Read time:      %lf\n", (matrix_info+matrixQueueIndex)->readTime );
            printf ( "Analyze time:   %lf\n", (matrix_info+matrixQueueIndex)->analyzeTime );
            printf ( "Factorize time: %lf\n", (matrix_info+matrixQueueIndex)->factorizeTime );
            printf ( "Solve time:     %lf\n", (matrix_info+matrixQueueIndex)->solveTime );
            printf ( "residual (|Ax-b|)/(|A||x|+|b|): %le\n\n", (matrix_info+matrixQueueIndex)->residual );
#endif

#pragma omp critical ( nextMatrixIndex )
            matrixIndex = nextMatrixIndex++;
        }
    }

    common_info->computeTime = SparseFrame_time () - timestamp;

#ifdef PRINT_INFO
    printf ("Total computing time: %lf\n\n", common_info->computeTime);
#endif

    // Free resources

    timestamp = SparseFrame_time();

    SparseFrame_free_gpu (&gpu_info, common_info);

    SparseFrame_free_matrix ( &matrix_info, common_info );

    common_info->freeTime = SparseFrame_time () - timestamp;

#ifdef PRINT_INFO
    printf ("Free time:            %lf\n\n", common_info->freeTime);
#endif

    return 0;
}
