#ifndef INCLUDE_ARCH_H
#define INCLUDE_ARCH_H

#include <cuComplex.h>

#define ARCH_INT int
#define ARCH_LONG long
#define ARCH_FLOAT double
#define ARCH_COMPLEX cuDoubleComplex

#define Int ARCH_INT
#define uInt unsigned Int
#define Long ARCH_LONG
#define uLong unsigned Long
#define Float ARCH_FLOAT
#define Complex ARCH_COMPLEX

#endif
