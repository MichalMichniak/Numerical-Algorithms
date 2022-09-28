import numpy as np
""" #####################QR#########################"""
def dot(x,y):
    return np.transpose(x)@y

def QR(A):
    vect_v = []
    v = 0
    sum = 0
    for i in A.T:
        v = i.T
        for j in vect_v:
            sum += dot(j,v)*j
        v = v-sum
        sum = 0
        vect_v.append(v/np.sqrt(dot(v,v)))
    Q = np.array([i for i in vect_v]).T
    R = np.array([[0.0 if i<j else dot(A[:,i],Q[:,j]) for j in range(A.shape[1])]for i in range(A.shape[1])]).T
    return Q,R
def QR_householdera(A):
    I = np.array([[complex(i==j) for j in range(A.shape[0])] for i in range(A.shape[0])])
    Q = I.copy()
    R = np.array(list(A[:,:]),dtype = complex)
    H = 0
    for i in range(A.shape[1]):
        
        v = A[i:,i] - np.linalg.norm(A[i:,i])*I[i:,i]
        if 0== v.T@v:
            continue
        guwno = np.array([[n*k for n in v] for k in v])
        H = I[i:,i:] - (2*guwno/(v.T@v))
        HH = I.copy()
        HH[i:,i:] = H
        A = HH@A
        Q = HH@Q
    R = np.array([[0 if i>j else A[i,j] for j in range(A.shape[0])] for i in range(A.shape[0])])
    Q = Q.T
    return Q,R
    pass
#A = np.array([[1 , 9, 1],[2,5,6],[4,5,7],[4,6,8]])
A = np.array([[12,-57,4],[6,167,-68],[-4,24,-41]])
Q,R=QR_householdera(A)
#print(np.linalg.norm(Q@R-A))
"""
Q,R=QR(A)
print(np.linalg.norm(Q@R-A))
S,V,D = np.linalg.svd(A)
print(S,V,D)
print(1.11e-16*V[0]/V[-1])
"""
################## Jakobi ################
"""
def diag_dominant(m):
    try:
        if type(m) != int or m<1:
            raise ValueError
        arr = np.array([[np.random.randint(0,9) for i in range(m)] for j in range(m)])
        for i in range(len(arr)):
            arr[i,i] = np.sum(abs(arr[:,i]))+np.sum(abs(arr[i,:])) - 2*abs(arr[i,i])
        return arr , np.array([np.random.randint(0,9) for j in range(m)])
    except:
        return None
def Jacobi(A,b):
    x = b
    D = np.array([ A[i,i] for i in range(A.shape[0])])
    L_U = A - np.identity(len(D))*D
    D_inv = np.identity(len(D))*1/D
    for i in range(100):
        x = b - np.dot(L_U,x)
        x = np.dot(D_inv,x)
    return x

A,b = diag_dominant(6)
b = b.T
x = Jacobi(A,b)
print(np.linalg.norm(np.dot(A,x)-b))
print(np.linalg.svd(A)[1][0]/np.linalg.svd(A)[1][-1])
"""
import scipy.linalg as spla
def shur(A,iter):
    A = np.array(list(A),dtype = complex)
    I = np.identity(len(A))
    for i in range(iter):
        Q,R = QR_householdera(A - I*A[-1,-1])
        A = R@Q + I*A[-1,-1]
    return A
def shur2(A,iter):
    A = np.array(list(A),dtype = complex)
    I = np.identity(len(A))
    for i in range(iter):
        Q,R = QR_householdera(A - (I*A[-1,-1]))
        A = R@Q + I*A[-1,-1]
    return A

def shur3_deflation(A,iter):
    A_ = spla.hessenberg(A)
    A_ = np.array(A_,dtype = complex)
    for k in range((m:=len(A_))-1):
        A_[0:m-k,0:m-k] = shur2(A_[0:m-k,0:m-k],iter+200)
    return A_

def QR_deflation(INP,iter):
    totalit=0
    A_rob=spla.hessenberg(INP)
    A_rob = np.array(A_rob,dtype = complex)
    m = INP.shape[0]
    for k in range(0,m-1):
        A_rob[0:m-k,0:m-k]=shur2(A_rob[0:m-k,0:m-k],200)
    return A_rob#, totalit

def qr_shift(INP,maxit=101):
    A1=INP.copy()
    for k in range(1,maxit):
        mu=A1[-1,-1]
        Q,R=np.linalg.qr(A1-mu*np.eye(A1.shape[0]))
        A1= R @ Q+mu*np.eye(A1.shape[0])
        if np.abs(A1[-1,-2]) < np.spacing(1):
            if np.linalg.norm(A1-A1.T) < np.spacing(1):
                A1[-2,-1]=0
            A1[-1,-2]=0
            return (A1,k)
    return (A1,k)

def eigen_values(A,iter,shur):
    S = shur(A,iter)
    return [S[i,i] for i in range(A.shape[0])]

A = np.array([[2,2,3],[4,5,6],[7,8,9]])
A_ = shur3_deflation(A,100)
print(A_)
D = np.diag([*range(1,7)])
P = np.random.random(D.shape)


#print(P)
print(np.array(eigen_values(A,200,shur3_deflation)),  np.array(eigen_values(A,200,shur)),"\n",QR_deflation(A,2000))#,
print(np.linalg.eig(A)[0])
### DDDDD
import scipy.interpolate as spint
import matplotlib.pyplot as plt
f = lambda x: np.cos(x) * np.power(x,2)
n = 200
x = np.array([np.cos(i*np.pi/n) for i in range(n)])

plt.plot(x,f(x))
plt.grid()
#plt.show()
plt.figure()
#print(spint.barycentric_interpolate(x,f(x),0.7922295139420589))
#print(f(0.7922295139420589))
## 2 
import scipy.optimize as spop
f = lambda x : np.exp(9*x) + x
#print(spop.fsolve(f,0))
######## 
import numpy.linalg as linalg
A = np.array([[7,8,11,67],[99,22,29,28],[64,13,90,92]])
b = np.array([[61],[96],[6]])
Q,R = linalg.qr(A)
#print(linalg.norm(b - A*linalg.solve(R,Q.T@b)))
A = np.array([[4,-7], [-1,0]])
#print(linalg.norm(A))
##
A = 553630
b = 0.04579
#print(A+b-A-b)
## 
A = np.array([[1,2,7,6],[2,3,6,1],[7,1,5,6]])
Q,R = linalg.qr(A)
#print(R[1,1])
## 
#print(np.frexp(9.860548623284988))
## 

#print(linalg.solve(np.array([[-1,-3,9,-4],[0,-7,-8,3],[-2,4,2,2],[-3,2,8,-8]]),np.array([[73],[19],[43],[98]])))
## 
import scipy.integrate as spin
f = lambda x: np.sqrt(x**3+1)
x = np.linspace(13,17,560)
#print(spin.trapz(f(x),x))
## 
A = np.array([[50,0,58,-55],[0,93,98,-89],[58,98,300,-190],[-55,-89,-190,-167]])
#print(np.min(linalg.eigvals(A)))