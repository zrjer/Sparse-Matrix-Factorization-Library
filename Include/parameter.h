#ifndef PARAMETER_H
#define PARAMETER_H

#define PRINT_CALLS
#define PRINT_INFO

#define MEM_LIMITS

#ifdef MEM_LIMITS
    #define MAX_MEM 4
    #define MAX_HOST_MEM 4
    #define MAX_DEV_MEM 4
#else
#endif

#undef MEM_LIMITS

const double prune_dense = 10.0;
const double aggressive = 1;

#endif
