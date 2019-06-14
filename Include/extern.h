#ifndef INCLUDE_EXTERN_H
#define INCLUDE_EXTERN_H

#include "arch.h"

extern int dsyrk_(char *uplo, char *trans, Long *n, Long *k, const Float *alpha, Float *a, Long *lda, const Float *beta, Float *c, Long *ldc);

extern int zherk_(char *uplo, char *trans, Long *n, Long *k, const Complex *alpha, Complex *a, Long *lda, const Complex *beta, Complex *c, Long *ldc);

extern int dgemm_(char *transa, char *transb, Long *m, Long *n, Long *k, const Float *alpha, Float *a, Long *lda, Float *b, Long *ldb, const Float *beta, Float *c, Long *ldc);

extern int zgemm_(char *transa, char *transb, Long *m, Long *n, Long *k, const Complex *alpha, Complex *a, Long *lda, Complex *b, Long *ldb, const Complex *beta, Complex *c, Long *ldc);

extern int dpotrf_(char *uplo, Long *n, Float *a, Long *lda, int *info);

extern int zpotrf_(char *uplo, Long *n, Complex *a, Long *lda, int *info);

extern int dtrsm_(char *side, char *uplo, char *transa, char *diag, Long *m, Long *n, const Float *alpha, Float *a, Long *lda, Float *b, Long *ldb);

extern int ztrsm_(char *side, char *uplo, char *transa, char *diag, Long *m, Long *n, const Complex *alpha, Complex *a, Long *lda, Complex *b, Long *ldb);

#endif
