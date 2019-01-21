#ifndef PARAMETERS_H
#define PARAMETERS_H

#define MEM_LIMITS

#ifdef MEM_LIMITS
    #define MAX_MEM 4
    #define MAX_HOST_MEM 4
    #define MAX_DEV_MEM 4
#else
#endif

#undef MEM_LIMITS

#endif
