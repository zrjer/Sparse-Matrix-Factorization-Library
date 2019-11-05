#ifndef INCLUDE_MACRO_H
#define INCLUDE_MACRO_H

#define FALSE 0
#define TRUE  1

#define MAX(x,y)    ( ( (x) > (y) ) ? (x) : (y) )
#define MIN(x,y)    ( ( (x) < (y) ) ? (x) : (y) )
#define ALIGN(x,y)  ( ( (x) + (y) - 1 ) / (y) * (y) )
#define TRUNC(x,y)  ( (x) / (y) * (y) )
#define CEIL(x,y)   ( ( (x) + (y) - 1 ) / (y) )
#define FLOOR(x,y)  ( (x) / (y) )

#endif

