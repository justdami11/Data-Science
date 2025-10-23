# linear_algebra_toolbox/toolbox.py
import numpy as np

def add_matrices(A, B):
    return A + B

def subtract_matrices(A, B):
    return A - B

def scalar_multiply(A, scalar):
    return scalar * A

def matrix_multiply(A, B):
    return A @ B

def transpose_matrix(A):
    return A.T

def matrix_properties(A):
    return {
        'shape': A.shape,
        'rank': np.linalg.matrix_rank(A),
        'trace': np.trace(A),
        'is_symmetric': np.allclose(A, A.T),
        'is_identity': np.allclose(A, np.eye(A.shape[0])),
        'is_diagonal': np.all(A == np.diag(np.diagonal(A)))
    }

def determinant(A):
    return np.linalg.det(A)

def inverse(A):
    if np.linalg.det(A) == 0:
        raise np.linalg.LinAlgError("Matrix is singular")
    return np.linalg.inv(A)

def pseudoinverse(A):
    return np.linalg.pinv(A)

def solve_linear_system(A, b):
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, b, rcond=None)[0]

def eigen_decomposition(A):
    values, vectors = np.linalg.eig(A)
    return values, vectors

def qr_decomposition(A):
    Q, R = np.linalg.qr(A)
    return Q, R

def svd_decomposition(A):
    U, S, Vt = np.linalg.svd(A)
    return U, S, Vt

def low_rank_approximation(A, k):
    U, S, Vt = svd_decomposition(A)
    S_k = np.diag(S[:k])
    return U[:, :k] @ S_k @ Vt[:k, :]

def is_singular(A):
    return np.isclose(np.linalg.det(A), 0.0)