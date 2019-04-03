#ifndef INCLUDE_SPARSEFRAME_H
#define INCLUDE_SPARSEFRAME_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include <omp.h>

#include <cuda_runtime.h>

#include "amd.h"
#include "metis.h"

#include "arch.h"
#include "macro.h"
#include "type.h"
#include "constant.h"
#include "parameter.h"
#include "info.h"

#ifdef __cplusplus
extern "C" {
#endif

    int SparseFrame (int argc, char **argv);

#ifdef __cplusplus
}
#endif

#endif
