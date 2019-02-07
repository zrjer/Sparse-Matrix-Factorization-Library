#ifndef SPARSEFRAME_H
#define SPARSEFRAME_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <cuda_runtime.h>

#include "amd.h"

#include "arch.h"
#include "parameter.h"
#include "macro.h"
#include "info.h"

#ifdef __cplusplus
extern "C" {
#endif

    int SparseFrame (int argc, char **argv);

#ifdef __cplusplus
}
#endif

#endif
