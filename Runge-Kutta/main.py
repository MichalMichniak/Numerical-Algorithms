from typing import Callable
import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt

def rk4(f : Callable, x0 : float, h : float, n : int, t = 0):
    butcher_table = np.array([[0,0,0,0],[0.5,0,0,0],[0,0.5,0,0],[0,0,1,0]])
    b_weights = np.array([1/6,2/6,2/6,1/6], dtype=float)
    c_weights = np.array([np.sum(i) for i in butcher_table])
    k = np.array([0,0,0,0], dtype=float)
    x = x0
    res = [x0]
    res_x = [t]
    for i in range(n):
        for s in range(len(k)):
            k[s] = f(t+c_weights[s]*h, x + np.sum(butcher_table[s]*k)*h)
        x = x + h* np.sum(b_weights * k)
        res.append(x)
        t = t + h
        res_x.append(t)
    return np.array(res_x),np.array(res)

def rk4_3_8(f : Callable, x0 : float, h : float, n : int, t = 0):
    butcher_table = np.array([[0,0,0,0],[1/3,0,0,0],[-1/3,1,0,0],[1,-1,1,0]],dtype=float)
    b_weights = np.array([1/8,3/8,3/8,1/8], dtype=float)
    c_weights = np.array([np.sum(i) for i in butcher_table])
    k = np.array([0,0,0,0], dtype=float)
    x = x0
    res = [x0]
    res_x = [t]
    for i in range(n):
        for s in range(len(k)):
            k[s] = f(t+c_weights[s]*h, x + np.sum(butcher_table[s]*k)*h)
        x = x + h* np.sum(b_weights * k)
        res.append(x)
        t = t + h
        res_x.append(t)
    return np.array(res_x),np.array(res)

def euler(f : Callable, x0 : float, h : float, n : int, t = 0):
    x = x0
    res = [x0]
    res_x = [t]
    for i in range(n):
        x = x + h*f(t,x)
        res.append(x)
        t = t + h
        res_x.append(t)
    return np.array(res_x),np.array(res)



def rk45(f : Callable, x0 : float, h : float, n : int, t = 0, sc = 0.0000000000001 , q = 6):
    butcher_table = np.array([[0,0,0,0,0,0,0],[1/5,0,0,0,0,0,0],[3/40,9/40,0,0,0,0,0],[44/45, -56/15,32/9 , 0,0,0,0],[19372/6561,-25360/2187,64448/6561,-212/729,0,0,0],[9017/3168, -355/33, 46732/5247, 49/176, -5103/18656,0,0],[35/384,0,500/1113,125/192,-2187/6784,11/84,0]], dtype=float)
    c_weights = np.array([np.sum(i) for i in butcher_table])
    b_weights_4 = np.array([35/384,0,500/1113,125/192,-2187/6784,11/84,0], dtype=float)
    b_weights_5 = np.array([5179/57600,0,7571/16695,393/640,-92097/339200,187/2100,1/40], dtype=float)
    x = x0
    x_4 = x0
    res = [x0]
    res_x = [t]
    k = np.array([0,0,0,0,0,0,0], dtype=float)
    for i in range(n):
        for s in range(len(k)):
            k[s] = f(t+c_weights[s]*h, x + np.sum(butcher_table[s]*k)*h)
        x_4 = x + h* np.sum(b_weights_4 * k)
        x = x + h* np.sum(b_weights_5 * k)
        res.append(x)
        t = t + h
        res_x.append(t)
        err = abs(x_4 - x)/sc
        if err > sc:
            h = h*(1/err)**(1/(q+1))
    return np.array(res_x),np.array(res)

def solve_equation(f : Callable, df : Callable, k, c_weights, butcher_table, x : float, t : float, h : float):
    # jackobi matrix
    I = np.identity(len(butcher_table))
    r = np.array([f(t+c_weights[i]*h, x + np.sum(butcher_table[i]*k)*h) - k[i] for i in range(len(butcher_table))])
    while linalg.norm(r) > 5e-12:
        J = np.array([[df(t + c_weights[i]*h , x + np.sum(butcher_table[i]*k)*h)*h*butcher_table[i][j] for j in range(len(butcher_table))] for i in range(len(butcher_table))])
        J = J - I
        p = linalg.solve(J, (-1)*r)
        k = k+p
        r = np.array([f(t+c_weights[i]*h, x + np.sum(butcher_table[i]*k)*h) - k[i] for i in range(len(butcher_table))])
    return k

def implict_RK4(f : Callable, df : Callable, x0 : float, h : float, n : int, t = 0):
    butcher_table = np.array([[0.25,0.25-(np.sqrt(3)/6)],[0.25+(np.sqrt(3)/6),0.25]])
    c_weights = np.array([np.sum(i) for i in butcher_table])
    b_weights = np.array([0.5,0.5], dtype=float)
    k = np.array([0,0], dtype=float)
    res = [x0]
    res_x = [t]
    x = x0
    for i in range(n):
        k = solve_equation(f, df, k, c_weights, butcher_table, x, t, h)
        x = x + h* np.sum(b_weights * k)
        res.append(x)
        t = t + h
        res_x.append(t)
        #print(i)
    return np.array(res_x),np.array(res)
    pass

def main():
    #
    """fun = lambda t,x: -50*(x-np.cos(t))
    d_fun = lambda t,x: -50""" 
    fun = lambda t,x: np.sin(x*t)
    d_fun = lambda t,x: t*np.cos(t*x)
    h = 0.01
    n = 2000#800
    x0 = 0.5
    x , y = rk4(fun, x0, h, n)
    plt.plot(x,y,label="RK4")
    x , y = rk4_3_8(fun, x0, h, n)
    plt.plot(x,y,label="RK4 3/8")
    x , y = euler(fun, x0, h, n)
    #plt.plot(x,y,label="euler")
    x0 = 2.12
    x , y = rk4(fun, x0, h, n)
    plt.plot(x,y,label="RK4")
    x , y = rk4_3_8(fun, x0, h, n)
    plt.plot(x,y,label="RK4 3/8")
    x , y = euler(fun, x0, h, n)
    #plt.plot(x,y,label="euler")
    x0 = 2.1
    x , y = rk4(fun, x0, h, n)
    plt.plot(x,y,label="RK4")
    x , y = rk4_3_8(fun, x0, h, n)
    plt.plot(x,y,label="RK4 3/8")
    x , y = euler(fun, x0, h, n)
    #plt.plot(x,y,label="euler")
    plt.legend()
    plt.show()
    x0 = 0.25
    h = 1.974/50
    
    x , y = rk4_3_8(fun, x0, h, n)
    plt.plot(x,y,label="RK4 3/8")
    x , y = rk45(fun, x0, h, n)
    plt.plot(x,y,label="RK45")
    x , y = euler(fun, x0, h, n)
    plt.plot(x,y,label="euler")
    plt.xlim([0,2])
    plt.ylim([-2,2])
    plt.legend()
    plt.show()
    x , y = implict_RK4(fun,d_fun, x0, h, n)
    plt.plot(x,y,label="implict RK4")
    plt.xlim([0,2])
    plt.ylim([-2,2])
    plt.legend()
    plt.show()
    pass

if __name__ == '__main__':
    main()