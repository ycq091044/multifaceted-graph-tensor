def optimize(u, A, B):

    # u = np.linalg.solve(A + np.eye(A.shape[1]) * 1e-8, B)
    L = la.cholesky(A_ + np.eye(A_.shape[1]) * 1e-8)
    y = la.solve_triangular(L.T, B, lower=True)
    u = la.solve_triangular(L, y, lower=False)
    return u
