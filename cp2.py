"""
PROJECT   : ChallengeProject1.py
PROGRAMMER: Kushika Senera
COURSE    : SFWRTECH 4MA3 - Numerical Linear Algebra and Numerical Optimization
INSTRUCTOR: Gagan Sidhu
"""

import numpy as np

# algorithm 3.1: householder qr factorization
def householder(A_matrix, b_vector):
    n = len(A_matrix[0])
    m = len(A_matrix)
    A = [row[:] for row in A_matrix]
    A = np.array(A, dtype=float)
    b = np.array(b_vector, dtype=FloatingPointError)
    
    # for k = 1 to n
    for k in range(n):  
        
        # αk = -sign(akk)√(a²kk + ... + a²mk) 
        a_k = A[k:m, k].copy()
        alpha_k = -np.sign(a_k[0]) * np.sqrt(np.sum(a_k**2))
        
        # vk = [0 ... 0 akk ... amk]^T - αk*ek
        v_k = a_k.copy()
        v_k[0] = v_k[0] - alpha_k
        
        # βk = v_k^T * v_k
        beta_k = np.sum(v_k * v_k)
        
        # if βk = 0 then continue with next k 
        if beta_k == 0:
            continue
        
        # for j = k to n 
        for j in range(k, n):
            # γj = v_k^T * aj
            a_j = A[k:m, j]
            gamma_j = np.sum(v_k * a_j)
            
            # aj = aj - (2γj/βk)vk
            A[k:m, j] = a_j - (2 * gamma_j / beta_k) * v_k
        
        # do gamma b transformation like A above
        b_sub = b[k:m]
        gamma_b = np.sum(v_k * b_sub)
        b[k:m] = b_sub - (2 * gamma_b / beta_k)
        
    return A, b


# upper triangle system Ux = b 
def back_sub(U, b):
    # declare and initialize output vector x
    n = len(b)
    x = [0.0] * n
    # for j = n to 1 {loop backwards over cols.} 
    for j in reversed(range(n)):
        # xj = bj/Ujj {compute soln. component}
        x[j] = b[j] / U[j][j]
        # for i = 1 to j - 1
        # range starts at 1 and stops at j-1
        # using range(stop) => j => j-1 
        for i in range(j):
            # bi = bi - uijxj {update RHS}
            b[i] = b[i] - U[i][j] * x[j]
    return x


if __name__ == "__main__":

    A_matrix = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, -1, 0, 0],
        [1, 0, -1, 0],
        [1, 0, 0, -1],
        [0, 1, -1, 0],
        [0, 1, 0, -1],
        [0, 0, 1, -1]
    ]

    b_vector = [2.95, 1.74, -1.45, 1.32, 1.23, 4.45, 1.61, 3.21, 0.45, -2.75]

    R, b_transformed = householder(A_matrix, b_vector)
    
    print("Matrix A after all Householder transformations:")
    for row in R:
        print(row)
    print()