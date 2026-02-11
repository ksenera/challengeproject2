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
        
    
    return A


# lower triangle system Lx = b
def forward_sub(L, b):
    # declare and initialize output vector x
    n = len(b)
    x = [0.0] * n
    # for j = 1 to n {loop over cols.} 
    # got length of b vector and put it in n to start loop
    for j in range(n):
        # xj = bj/Ljj {compute soln. component}
        x[j] = b[j] / L[j][j]

        # for i = j + 1 to n 
        # j starts at index 0 so j + 1 = 1 & n = 4 so from index 1 up to index 3 
        # range start to stop index n-1 => 4-1 = 3 
        for i in range(j+1,n):
            # bi = bi - Lijxj {update RHS}
            b[i] = b[i] - L[i][j] * x[j]
    return x

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

# Gaussian Elimination using Partial Pivoting solving Ax = b
def gauss_elim(A, b):
    # declare A system and b vector 
    n = len(A)
    # want to modify only local copies of A sys and b vec 
    A = [row[:] for row in A]
    b = b[:]
    # A2 Note asks to print transformation at each step of elim + pivot
    #print("Copy of A_matrix: ")
    #for row in A:
    #    print(row)
    #print(f"Copy of b_vector: {b}")

    # from ALGO 2.4 for pivoting steps
    # for k = 1 to n - 1 {loop over cols.}
    for k in range(n-1):
        # find index p |Apk| >= |Aik| for k <= i <= n {search for pivot in current col}
        p = k
        for i in range(k + 1, n):
            if abs(A[i][k]) > abs(A[p][k]):
                p = i
        # if p != k then interchange rows k and p {interchange rows if necessary}
        if p != k:
            #print(f"Swapping row {k} with row {p}")
            A[k], A[p] = A[p], A[k]
            b[k], b[p] = b[p], b[k]
            for row in A:
                print(row)
            #print(f"Updated b: {b}\n")
        # in Algo 2.4 this line is line 2 of Algo 2.3 
        # if akk = 0 then stop {stop if pivot is zero}
        if A[k][k] == 0:  # if a_kk = 0 then stop
            #print("Error: break if pivot is zero")
            break

        #print(f"Elimination for Column {k}")
        # for j = k + 1 to n {compute multipliers for current col}
        for i in range(k+1, n):
            # mik = aik/akk 
            m_ik = A[i][k] / A[k][k]
            #print(f"Multiplier m_{i}{k} = {m_ik:.2f}")

            A[i][k] = 0 

            # for j = k + 1 to n 
            for j in range(k+1, n):
                # for i = k + 1 to n -> nested for j in for i
                # {apply transformation to remaining submatrix}
                A[i][j] = A[i][j] - m_ik * A[k][j] 

            # solving Ly = b (Step 2 page 12 by updating b)
            b[i] = b[i] - m_ik * b[k]

        # printing as per assignment 
        #for row in A:
            #print([round(val, 2) for val in row])
        #print(f"Updated b: {[round(val, 2) for val in b]}\n")

    x = back_sub(A, b)
    return x

if __name__ == "__main__":
    n = 2
    while n <= 10:
        # generate the Hilbert matrix and vector b
        H, b = generatorHb(n)

        # add x = 1.0 again 
        x = []
        for i in range(n):
            x.append(1.0)

        # solve for approximate x Hx̂ = b
        x_hat = gauss_elim(H, b)
    
        # Find the ∞ −norm of the residual 𝒓 = 𝒃 − 𝑯x̂
        r = [] 
        for i in range(n):
            r_i = b[i]
            for j in range(n):
                r_i = r_i - H[i][j] * x_hat[j]
                
        # Find the error 𝚫𝒙 = 𝒙2 − 𝒙, where x is the true solution, 
        # i.e., the n-vector with all entries equal to 1 
        delta_x = []
        for i in range(n):
            delta_x.append(x_hat[i] - x[i])


        print(f"\nx̂ = {x_hat}")
        print(f"r = {r}")
        print(f"Δx = {delta_x}")

        error = max(abs(val) for val in delta_x)

        #  error reaching 100% end while loop 
        if error >= 1.0:
            break 

        n += 1
