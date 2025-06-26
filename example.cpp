#include <iostream>
#include <Eigen/Sparse>
#include <chrono>
#include "data_types.h"
#include "qdldl.h"
#include "utils.h"
#include "qdldl_interface.h"

using namespace Eigen;

int main(int argc, char* argv[]) 
{
    // Parse command line argument for KKT system type
    bool use_banded = false;
    if (argc > 1) {
        std::string arg = argv[1];
        if (arg == "--banded") {
            use_banded = true;
        } else if (arg == "--qp") {
            use_banded = false;
        } else {
            std::cerr << "Unknown argument: " << arg << ". Use '--banded' or '--qp'." << std::endl;
            return -1;
        }
    }

    const int n = 12;  // state dimension
    const int m = 4;   // input dimension
    const int N = 100; // prediction horizon

    VectorXs x0(n);
    VectorXs x_ref(n);
    x0 << 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.;
    x_ref << 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.;

    /* LQR Data */
    MatrixXs Q(n, n);
    MatrixXs R(m, m);
    MatrixXs S(m, n);
    VectorXs q(n);
    VectorXs r(m);
    MatrixXs A(n, n);
    MatrixXs B(n, m);
    VectorXs c(n);
  
    Q.diagonal() << 1e-5, 1e-5, 10., 10., 10., 10., 1e-5, 1e-5, 1e-5, 5., 5., 5.;
    R.diagonal() << 0.1, 0.1, 0.1, 0.1;
    S.setZero();
    q = -x_ref.transpose() * Q;
    r.setZero();

    A << 
        1.,      0.,     0., 0., 0., 0., 0.1,     0.,     0.,  0.,     0.,     0.,
        0.,      1.,     0., 0., 0., 0., 0.,      0.1,    0.,  0.,     0.,     0.,
        0.,      0.,     1., 0., 0., 0., 0.,      0.,     0.1, 0.,     0.,     0.,
        0.0488,  0.,     0., 1., 0., 0., 0.0016,  0.,     0.,  0.0992, 0.,     0.,
        0.,     -0.0488, 0., 0., 1., 0., 0.,     -0.0016, 0.,  0.,     0.0992, 0.,
        0.,      0.,     0., 0., 0., 1., 0.,      0.,     0.,  0.,     0.,     0.0992,
        0.,      0.,     0., 0., 0., 0., 1.,      0.,     0.,  0.,     0.,     0.,
        0.,      0.,     0., 0., 0., 0., 0.,      1.,     0.,  0.,     0.,     0.,
        0.,      0.,     0., 0., 0., 0., 0.,      0.,     1.,  0.,     0.,     0.,
        0.9734,  0.,     0., 0., 0., 0., 0.0488,  0.,     0.,  0.9846, 0.,     0.,
        0.,     -0.9734, 0., 0., 0., 0., 0.,     -0.0488, 0.,  0.,     0.9846, 0.,
        0.,      0.,     0., 0., 0., 0., 0.,      0.,     0.,  0.,     0.,     0.9846;
    B <<
        0.,      -0.0726,  0.,     0.0726,
        -0.0726,   0.,      0.0726, 0.,
        -0.0152,   0.0152, -0.0152, 0.0152,
        -0.,      -0.0006, -0.,     0.0006,
        0.0006,   0.,     -0.0006, 0.0000,
        0.0106,   0.0106,  0.0106, 0.0106,
        0.,      -1.4512,  0.,     1.4512,
        -1.4512,   0.,      1.4512, 0.,
        -0.3049,   0.3049, -0.3049, 0.3049,
        -0.,      -0.0236,  0.,     0.0236,
        0.0236,   0.,     -0.0236, 0.,
        0.2107,   0.2107,  0.2107, 0.2107;
        
    c.setZero();
    /* --------------------------------------------------------------------------------- */

    /* Form the KKT system */
    QDLDL_float rho_inv = 1e-5;
    SparseMatrix<QDLDL_float> KKT_eigen;
    VectorXs rhs;
    CscMatrix* KKT_csc = nullptr;

    std::cout << "Using " << (use_banded ? "banded" : "QP-type") << " KKT system formation ..." << std::endl;

    if (use_banded) {
        form_banded_KKT_system(Q, R, S, q, r, 
                               A, B, c, 
                               x0, 
                               N, rho_inv, 
                               KKT_csc, rhs);
    } else {
        form_QP_KKT_system(Q, R, S, q, r, 
                           A, B, c, 
                           x0, 
                           N, rho_inv, 
                           KKT_csc, rhs);
    }


    if (!KKT_csc) {
        // Failed to form KKT system
        return -1;
    }

    /* Solve the sparse linear system */
    QDLDLData* qdldl_data = qdldl_setup(KKT_csc);
    if (!qdldl_data) {
        std::cerr << "Failed to set up QDLDL data." << std::endl;
        csc_spfree(KKT_csc);
        return -1;
    }

    auto start = std::chrono::system_clock::now();

    QDLDL_int fact_status = QDLDL_factor(KKT_csc->n, KKT_csc->p, KKT_csc->i, KKT_csc->x,
                                         qdldl_data->Lp, qdldl_data->Li, qdldl_data->Lx,
                                         qdldl_data->D, qdldl_data->Dinv,
                                         qdldl_data->Lnz, qdldl_data->etree,
                                         qdldl_data->bwork, qdldl_data->iwork, qdldl_data->fwork);
    if (fact_status < 0) {
        std::cerr << "Factorization failed with status: " << fact_status << std::endl;
        qdldl_cleanup(qdldl_data);
        csc_spfree(KKT_csc);
        return -1;
    }

    for (QDLDL_int i = 0; i < KKT_csc->n; ++i) {
        qdldl_data->x[i] = rhs[i];
    }

    QDLDL_solve(qdldl_data->Ln, qdldl_data->Lp, qdldl_data->Li, qdldl_data->Lx,
                qdldl_data->Dinv, 
                qdldl_data->x);

    auto end = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "QDLDL solve time: " << duration.count() / 1e3 << " ms" << std::endl;

    /* Output the solution */
    std::cout << "Control inputs:" << std::endl;
    for (int k = 0; k < std::min(N, 10) ; ++k) {
        int offset = use_banded ? k * (m + 2 * n) : k * (m + n);
        std::cout << "u[" << k << "] = [";
        for (int i = 0; i < m; ++i) {
            std::cout << qdldl_data->x[offset + i] << (i < m - 1 ? ", " : "");
        }
        std::cout << "]" << std::endl;
    }

    /* clean up */
    qdldl_cleanup(qdldl_data);
    csc_spfree(KKT_csc);

    return 0;
}