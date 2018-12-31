#include <stdio.h>
#include <stdlib.h>
#include "SparseFrame.h"

int SparseFrame (int argc, char **argv)
{
    size_t mainMemorySize, gpuMemorySize;
    int numSparseMatrix;
    FILE **files;

    numSparseMatrix = argc - 1;
    files = (FILE **) malloc ( sizeof(FILE *) * numSparseMatrix );

    free (files);

    return 0;
}
