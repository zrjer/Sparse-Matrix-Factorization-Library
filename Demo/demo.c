#include <stdio.h>
#include "SparseFrame.h"

int main (int argc, char **argv)
{
    printf ("The main function\n\n");

#ifdef PRINT_INFO
    printf ("Type size:\n");
    printf ("Size of size_t = %ld\n", sizeof(size_t));
    printf ("Size of Int = %ld\n", sizeof(Int));
    printf ("Size of uInt = %ld\n", sizeof(uInt));
    printf ("Size of Long = %ld\n", sizeof(Long));
    printf ("Size of uLong = %ld\n", sizeof(uLong));
    printf ("Size of Float = %ld\n", sizeof(Float));
    printf ("Size of Complex = %ld\n", sizeof(Complex));
    printf ("Size of idx_t = %ld\n", sizeof(idx_t));
    printf ("\n");
#endif

    SparseFrame (argc, argv);

    return 0;
}
