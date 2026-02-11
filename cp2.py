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
