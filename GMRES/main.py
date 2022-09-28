import numpy as np
import numpy.linalg as linalg
import numpy.random as random

def arnoldi_iteration(A, Q , H_prev):
    n = len(Q[0])
    v = A@Q[:,-1]
    H = np.zeros([n+1,n])
    Q_next = np.zeros([len(A),n+1])
    Q_next[:,:-1] = Q
    H[:-1,:-1] = H_prev
    for i in range(n):
        H[i,n-1] = Q[:,i].T @ v
        v = v - H[i,n-1]*Q[:,i]
    H[n,n-1] = linalg.norm(v)
    Q_next[:,-1] = v/linalg.norm(v)
    return Q_next, H
    pass
 
def solve_triangular(R, b):
    x = np.zeros(len(b)).reshape(len(b),1)
    for i in range(len(b))[::-1]:
        x[i] = (b[i] - R[i,i:] @ x[i:])/R[i,i]
    return x

def solve_QR(A,b):
    Q,R = linalg.qr(A)
    b = Q.T@b
    x = solve_triangular(R,b)#linalg.solve(R,b)
    return x


def GMRES(A, b, epsilon = 1e-11):
    x0 = np.zeros(len(b)).reshape([len(b),1]) #b/linalg.norm(b)
    epsilon = linalg.cond(A) * 1.16*10**(-16) *10
    Q = b/linalg.norm(b)
    H = np.array([[]])
    for n in range(len(A)):
        Q_prev = Q[:,:]
        Q, H = arnoldi_iteration(A, Q, H)
        y = solve_QR(H,Q.T@b)
        x = x0 + Q_prev@y
        if linalg.norm(A@x - b)<epsilon:
            return x
        print("residual: ",linalg.norm(H@y - Q.T@b))
        print(n," : ",linalg.norm(A@x - b))
    return np.array([])
    pass



def main():
    """
    Q,R = linalg.qr(matrix)
    print(Q)
    Q = b/linalg.norm(b)
    Q_prev = Q[:,:]
    H = np.array([[]])
    Q, H = arnoldi_iteration(matrix, Q, H)
    print(Q,"\n",H)
    print("Sprawdzenie: ")
    print(Q@H@Q_prev.T)
    Q_prev = Q[:,:]
    print("=========================")
    Q, H = arnoldi_iteration(matrix, Q, H)
    print(Q,"\n",H)
    print("Sprawdzenie: ")
    print(Q@H@Q_prev.T)
    Q_prev = Q[:,:]
    print("=========================")
    Q, H = arnoldi_iteration(matrix, Q, H)
    print(Q,"\n",H)
    print("Sprawdzenie: ")
    print(Q@H@Q_prev.T)
    Q_prev = Q[:,:]
    print("=========================")
    Q, H = arnoldi_iteration(matrix, Q, H)
    print(Q,"\n",H)
    print("Sprawdzenie: ")
    print(Q@H@Q_prev.T)
    Q_prev = Q[:,:]
    """
    matrix = np.array([[5,2,3,4,1],[4,5,6,7,0],[7,8,20,10,7],[10,16,12,13,4],[213,32,45,1,23]])
    k = np.identity(5000) * random.random([5000,5000])*200
    
    matrix = random.random([5000,5000]) + k#
    b = np.array([[1],[2],[3],[4],[0]])
    b = random.random([5000,1])
    k = GMRES(matrix, b)
    print(linalg.cond(matrix))
    print("wynik: ","not found" if k.size == 0  else "found")
    #print(k)
    
    pass


if __name__ == '__main__':
    main()