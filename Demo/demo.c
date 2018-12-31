#include <stdio.h>
#include "SparseFrame.h"

int main (int argc, char **argv)
{
    printf ("The main function\n\n");

    printf ("Integer size:\n");
    printf ("Size of Int = %ld\n", sizeof(Int));
    printf ("Size of uInt = %ld\n", sizeof(uInt));
    printf ("Size of Long = %ld\n", sizeof(Long));
    printf ("Size of uLong = %ld\n\n", sizeof(uLong));

    SparseFrame (argc, argv);

    return 0;
}
