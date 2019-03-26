#ifndef INCLUDE_PARAMETER_H
#define INCLUDE_PARAMETER_H

#define PRINT_CALLS
#define PRINT_DEBUG
#define PRINT_INFO

#define MAX_NUM_GPU (-1)

#define MEM_LIMITS

#ifdef MEM_LIMITS
    #define MAX_MEM 4
    #define MAX_HOST_MEM 4
    #define MAX_DEV_MEM 4
#else
#endif

#undef MEM_LIMITS

#define MAXMMLINE 1024
size_t max_mm_line_size = MAXMMLINE + 1;

const enum PermMethod perm_method = PERM_METIS;

const double prune_dense = 10.0;
const double aggressive = 1;

#define RELAX_RATE (0.1)

#endif
