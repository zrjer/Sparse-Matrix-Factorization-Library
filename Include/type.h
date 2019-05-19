#ifndef INCLUDE_TYPE_H
#define INCLUDE_TYPE_H

enum FactorizeType { TYPE_CHOLESKY, TYPE_QR, TYPE_LU };
enum PermMethod { PERM_IDENTITY, PERM_AMD, PERM_METIS };
enum MatrixState { MATRIX_STATE_IDLE, MATRIX_STATE_TRIPLET, MATRIX_STATE_COMPRESSED };
enum NodeState { NODE_STATE_INITIAL, NODE_STATE_FACTORIZED, NODE_STATE_ASSEMBLED };

struct node_size_struct
{
    Long node;
    size_t size;
};

#endif
