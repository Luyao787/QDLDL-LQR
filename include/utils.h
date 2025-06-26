#pragma once

#include <Eigen/Sparse>
#include "qdldl.h"
#include "data_types.h"

using namespace Eigen;

void csc_spfree(CscMatrix* mat)
{
    if (mat) {
        if (mat->p) free(mat->p);
        if (mat->i) free(mat->i);
        if (mat->x) free(mat->x);
        free(mat);
    }
}

CscMatrix* csc_spalloc(const QDLDL_int m, const QDLDL_int n, const QDLDL_int nzmax)
{
    CscMatrix* mat = (CscMatrix*) calloc(1, sizeof(CscMatrix));
    if (!mat) {
        return 0;
    }
    mat->m = m;
    mat->n = n;
    mat->nzmax = nzmax;
    mat->p = (QDLDL_int*) malloc((n + 1) * sizeof(QDLDL_int));
    mat->i = (QDLDL_int*) malloc(nzmax * sizeof(QDLDL_int));
    mat->x = (QDLDL_float*) malloc(nzmax * sizeof(QDLDL_float));
    if (!mat->p || !mat->i || !mat->x) {
        csc_spfree(mat);
        return 0;
    }
    return mat;
}

CscMatrix* create_csc_matrix(const Eigen::SparseMatrix<QDLDL_float>& mat_eigen)
{
    QDLDL_int m = mat_eigen.rows();
    QDLDL_int n = mat_eigen.cols();
    QDLDL_int nzmax = mat_eigen.nonZeros();

    CscMatrix* mat_csc = csc_spalloc(m, n, nzmax);
    if (!mat_csc) {
        return 0;
    }

    // Fill in the CSC matrix
    for (QDLDL_int j = 0; j < n; ++j) {
        mat_csc->p[j] = mat_eigen.outerIndexPtr()[j];
        for (QDLDL_int i = mat_eigen.outerIndexPtr()[j]; i < mat_eigen.outerIndexPtr()[j + 1]; ++i) {
            mat_csc->i[i] = mat_eigen.innerIndexPtr()[i];
            mat_csc->x[i] = mat_eigen.valuePtr()[i];
        }
    }
    mat_csc->p[n] = nzmax;

    return mat_csc;
}

void form_banded_KKT_system(const MatrixXs& Q, const MatrixXs& R, const MatrixXs& S,
                        const VectorXs& q, const VectorXs& r,         
                        const MatrixXs& A, const MatrixXs& B, const VectorXs& c,
                        const VectorXs& x0,
                        QDLDL_int N, QDLDL_float rho_inv,
                        CscMatrix*& KKT_csc, VectorXs& rhs)
{
    int n = Q.rows();
    int m = R.rows();
    
    // Matrix size: variables are [u_0, λ_1, x_1, u_1, λ_2, x_2, ..., u_{N-1}, λ_N, x_N]
    // Total size: N controls + N lagrange multipliers + N states = N*(m + 2n)
    int total_size = N * (m + 2 * n);
    SparseMatrix<QDLDL_float> KKT_mat(total_size, total_size);
    
    int row_offset = 0, col_offset = 0;
    
    /* KKT matrix */

    // First time step (k=0): [R_0, B_0^T; B_0, 0, -I]
    // u_0 block (R_0)
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            if (R(i, j) != 0.0) {
                KKT_mat.insert(i, j) = R(i, j);
            }
        }
    }
    
    // B_0^T block (upper right of first block)
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (B(j, i) != 0.0) {
                KKT_mat.insert(i, m + j) = B(j, i); // B^T
            }
        }
    }
    
    // B_0 block (lower left of first block)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            if (B(i, j) != 0.0) {
                KKT_mat.insert(m + i, j) = B(i, j);
            }
        }
    }
    
    // Regularization term
    for (int i = 0; i < n; ++i) {
       KKT_mat.insert(m + i, m + i) = -rho_inv; 
    }

    // -I block 
    for (int i = 0; i < n; ++i) {
        KKT_mat.insert(m + i, m + n + i) = -1.0;
    }
    
    // Middle time steps 
    for (int k = 1; k < N; ++k) {
        row_offset = m + n + (k - 1) * (m + 2 * n);
        col_offset = m + n + (k - 1) * (m + 2 * n);
        
        // -I block
        for (int i = 0; i < n; ++i) {
            KKT_mat.insert(row_offset + i, col_offset - n + i) = -1.0;
        }
        
        // Q_k block
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (Q(i, j) != 0.0) {
                    KKT_mat.insert(row_offset + i, col_offset + j) = Q(i, j);
                }
            }
        }
        
        // S_k^T block
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                if (S(j, i) != 0.0) {
                    KKT_mat.insert(row_offset + i, col_offset + n + j) = S(j, i); // S^T
                }
            }
        }
        
        // A_k^T block
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (A(j, i) != 0.0) {
                    KKT_mat.insert(row_offset + i, col_offset + n + m + j) = A(j, i); // A^T
                }
            }
        }
        
        // S_k block
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (S(i, j) != 0.0) {
                    KKT_mat.insert(row_offset + n + i, col_offset + j) = S(i, j);
                }
            }
        }
        
        // R_k block
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < m; ++j) {
                if (R(i, j) != 0.0) {
                    KKT_mat.insert(row_offset + n + i, col_offset + n + j) = R(i, j);
                }
            }
        }
        
        // B_k^T block
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (B(j, i) != 0.0) {
                    KKT_mat.insert(row_offset + n + i, col_offset + n + m + j) = B(j, i); // B^T
                }
            }
        }
        
        // A_k block
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (A(i, j) != 0.0) {
                    KKT_mat.insert(row_offset + n + m + i, col_offset + j) = A(i, j);
                }
            }
        }
        
        // B_k block
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                if (B(i, j) != 0.0) {
                    KKT_mat.insert(row_offset + n + m + i, col_offset + n + j) = B(i, j);
                }
            }
        }

        // Regularization term
        for (int i = 0; i < n; ++i) {
            KKT_mat.insert(row_offset + n + m + i, col_offset + n + m + i) = -rho_inv; 
        }
        
        // -I block
        for (int i = 0; i < n; ++i) {
            KKT_mat.insert(row_offset + n + m + i, col_offset + 2 * n + m + i) = -1.0;
        }
   
    }
    
    // Final time step (k=N): 
    row_offset = m + n + (N - 1) * (m + 2 * n);
    col_offset = m + n + (N - 1) * (m + 2 * n);
    
    // -I block
    for (int i = 0; i < n; ++i) {
        KKT_mat.insert(row_offset + i, col_offset - n + i) = -1.0;
    }
    
    // P_N block
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (Q(i, j) != 0.0) {
                KKT_mat.insert(row_offset + i, col_offset + j) = Q(i, j);
            }
        }
    }

    KKT_mat.makeCompressed();
    SparseMatrix<QDLDL_float> KKT_triu = KKT_mat.triangularView<Eigen::Upper>();
    KKT_csc = create_csc_matrix(KKT_triu);
    if (!KKT_csc) {
        std::cerr << "Failed to create CSC matrix for KKT system." << std::endl;
    }
    
    /* RHS vector */
    rhs.resize(total_size);

    // Fill in the first part for u_0
    rhs.head(m) = -r - S.transpose() * x0;
    // Fill in the first part for x_1
    rhs.segment(m, n) = -A * x0 - c;
    
    for (int k = 1; k < N; ++k) {
        int offset = m + n + (k - 1) * (m + 2 * n);

        rhs.segment(offset, n) = -q;
        rhs.segment(offset + n, m) = -r;
        rhs.segment(offset + m + n, n) = -c;
    }
    // Fill in the last part for x_N
    int last_offset = m + n + (N - 1) * (m + 2 * n);
    rhs.segment(last_offset, n) = -q; // x_N = -q

}

void form_P(const MatrixXs& Q, const MatrixXs& R, const MatrixXs& S, int N, SparseMatrix<QDLDL_float>& P) 
{
    int n = Q.rows();
    int m = R.rows();
    
    int total_size = N * (m + n);
    P.resize(total_size, total_size);
    
    int row_offset = 0, col_offset = 0;

    // First time step
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            if (R(i, j) != 0.0) {
                P.insert(i, j) = R(i, j);
            }
            // P.insert(i, j) = R(i, j);
        }
    }

    for (int k = 1; k < N; ++k) {
        row_offset = m + (k - 1) * (m + n);
        col_offset = m + (k - 1) * (m + n);
        
        // Q_k block
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (Q(i, j) != 0.0) {
                    P.insert(row_offset + i, col_offset + j) = Q(i, j);
                }
                // P.insert(row_offset + i, col_offset + j) = Q(i, j);
            }
        }
        // S.T_k block
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                if (S(j, i) != 0.0) {
                    P.insert(row_offset + i, col_offset + n + j) = S(j, i);
                }
                // P.insert(row_offset + i, col_offset + n + j) = S(j, i);
            }
        }
        // S_k block
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (S(i, j) != 0.0) {
                    P.insert(row_offset + n + i, col_offset + j) = S(i, j);
                }
                // P.insert(row_offset + n + i, col_offset + j) = S(i, j);
            }
        }
        // R_k block
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < m; ++j) {
                if (R(i, j) != 0.0) {
                    P.insert(row_offset + n + i, col_offset + n + j) = R(i, j);
                }
                // P.insert(row_offset + n + i, col_offset + n + j) = R(i, j);
            }
        }
    }
    // Last time step (k=N)
    row_offset = m + (N - 1) * (m + n);
    col_offset = m + (N - 1) * (m + n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (Q(i, j) != 0.0) {
                P.insert(row_offset + i, col_offset + j) = Q(i, j);
            }
            // P.insert(row_offset + i, col_offset + j) = Q(i, j);
        }
    }

    P.makeCompressed();

}

void form_A(const MatrixXs& Ad, const MatrixXs& Bd, int N, SparseMatrix<QDLDL_float>& A) 
{
    int n = Ad.rows();
    int m = Bd.cols();
    
    A.resize(N * n, N * (m + n));
    
    int row_offset = 0, col_offset = 0;

    // First time step [B_0, -I]
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            // if (Bd(i, j) != 0.0) {
            //     A.insert(i, j) = Bd(i, j);
            // }
            A.insert(i, j) = Bd(i, j);
        }
    }
    for (int i = 0; i < n; ++i) {
        A.insert(i, m + i) = -1.0; // -I
    }

    for (int k = 1; k < N; ++k) {
        row_offset = k * n;
        col_offset = m + (k - 1) * (m + n);
        
        // A_k block
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (Ad(i, j) != 0.0) {
                    A.insert(row_offset + i, col_offset + j) = Ad(i, j);
                }
                // A.insert(row_offset + i, col_offset + j) = Ad(i, j);
            }
        }

        // B_k block
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                if (Bd(i, j) != 0.0) {
                    A.insert(row_offset + i, col_offset + n + j) = Bd(i, j);
                }
                // A.insert(row_offset + i, col_offset + n + j) = Bd(i, j);
            }
        }
        
        // -I block
        for (int i = 0; i < n; ++i) {
            A.insert(row_offset + i, col_offset + n + m + i) = -1.0; // -I
        }

    }

    A.makeCompressed();
  
}

void form_QP_KKT_system(const MatrixXs& Q, const MatrixXs& R, const MatrixXs& S,
                        const VectorXs& q, const VectorXs& r,         
                        const MatrixXs& A, const MatrixXs& B, const VectorXs& c,
                        const VectorXs& x0,
                        QDLDL_int N, QDLDL_float rho_inv,
                        CscMatrix*& KKT_csc, VectorXs& rhs)
{
    SparseMatrix<QDLDL_float> P_eign, A_eign;
    form_P(Q, R, S, N, P_eign);
    form_A(A, B, N, A_eign);

    QDLDL_int n_var = P_eign.cols();
    QDLDL_int n_constr = A_eign.rows();
    QDLDL_int kkt_size = n_var + n_constr;

    SparseMatrix<QDLDL_float> KKT_mat(kkt_size, kkt_size);
    std::vector<Triplet<QDLDL_float>> triplets;

    // Add P block (upper-left)
    for (int col = 0; col < P_eign.outerSize(); ++col) {
        for (SparseMatrix<QDLDL_float>::InnerIterator it(P_eign, col); it; ++it) {
            triplets.emplace_back(it.row(), it.col(), it.value());
        }
    }

    // Add A^T block (upper-right)
    for (int col = 0; col < A_eign.outerSize(); ++col) {
        for (SparseMatrix<QDLDL_float>::InnerIterator it(A_eign, col); it; ++it) {
            triplets.emplace_back(it.col(), n_var + it.row(), it.value());
        }
    }

    // Add A block (lower-left)
    for (int col = 0; col < A_eign.outerSize(); ++col) {
        for (SparseMatrix<QDLDL_float>::InnerIterator it(A_eign, col); it; ++it) {
            triplets.emplace_back(n_var + it.row(), it.col(), it.value());
        }
    }
    
    // Add -rho_inv * I block (lower-right)
    for (int i = 0; i < n_constr; ++i) {
        triplets.emplace_back(n_var + i, n_var + i, -rho_inv);
    }
    KKT_mat.setFromTriplets(triplets.begin(), triplets.end());
    KKT_mat.makeCompressed();

    SparseMatrix<QDLDL_float> KKT_triu = KKT_mat.triangularView<Eigen::Upper>();
    KKT_csc = create_csc_matrix(KKT_triu);
    if (!KKT_csc) {
        std::cerr << "Failed to create CSC matrix for KKT system." << std::endl;
        return;
    }
    
    /* RHS vector */
    int n = q.size();
    int m = r.size();
    int total_size = N * (m + 2 * n);
    rhs.resize(total_size);

    // Fill in the first part of the RHS
    rhs.head(m) = -r - S * x0;

    int offset;
    for (int k = 1; k < N; ++k) {
        offset = m + (k - 1) * (m + n);
        rhs.segment(offset, n) = -q;
        rhs.segment(offset + n, m) = -r;
    }
    offset = m + (N - 1) * (m + n);
    rhs.segment(offset, n) = -q;

    offset = N * (m + n);
    rhs.segment(offset, n) = -A * x0 - c;
    for (int k = 1; k < N; ++k) {
        offset = N * (m + n) + k * n;
        rhs.segment(offset, n) = -c;
    }

}







