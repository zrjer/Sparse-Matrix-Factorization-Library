#ifndef INCLUDE_SPARSEFRAME_H
#define INCLUDE_SPARSEFRAME_H

#include <libgen.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <omp.h>

#include <amd.h>
#include <metis.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include <cuda_profiler_api.h>

#include <cblas.h>

#include "extern.h"

#include "arch.h"
#include "macro.h"
#include "type.h"
#include "constant.h"
#include "parameter.h"
#include "info.h"

#include "cuda_kernel.cuh"

#ifdef __cplusplus
extern "C" {
#endif

    int SparseFrame (int argc, char **argv);

#ifdef __cplusplus
}
#endif

#endif
