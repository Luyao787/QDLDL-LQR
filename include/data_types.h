#pragma once

#include <qdldl.h>
#include <Eigen/Dense>


using VectorXs = Eigen::VectorXd;
using MatrixXs = Eigen::MatrixXd;

using VectorMap = Eigen::Map<VectorXs>;
using MatrixMap = Eigen::Map<MatrixXs>;
using ConstVectorMap = Eigen::Map<const VectorXs>;
using ConstMatrixMap = Eigen::Map<const MatrixXs>;

using VectorRef = Eigen::Ref<VectorXs>;
using MatrixRef = Eigen::Ref<MatrixXs>;
using ConstVectorRef = Eigen::Ref<const VectorXs>;
using ConstMatrixRef = Eigen::Ref<const MatrixXs>;


struct CscMatrix {
    QDLDL_int    m;     // number of rows
    QDLDL_int    n;     // number of columns
    QDLDL_int*   p;     // column pointers
    QDLDL_int*   i;     // row indices
    QDLDL_float* x;     // non-zero values
    QDLDL_int    nzmax; // maximum number of non-zero entries
};

struct QDLDLData
{
    // data for L and D factors
    QDLDL_int    Ln;
    QDLDL_int*   Lp;
    QDLDL_int*   Li;
    QDLDL_float* Lx;
    QDLDL_float* D;
    QDLDL_float* Dinv;

    // data for elim tree calculation
    QDLDL_int* etree;
    QDLDL_int* Lnz;
    QDLDL_int  sumLnz;

    // working data for factorisation
    QDLDL_int*   iwork;
    QDLDL_bool*  bwork;
    QDLDL_float* fwork;

    // Data for results of A\b
    QDLDL_float* x;
};



