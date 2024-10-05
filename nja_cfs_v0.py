#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy
import scipy.constants
import scipy.spatial
import scipy.special
import matplotlib.pyplot as plt
import warnings
from itertools import islice
from datetime import datetime
from pprint import pprint

def cron(func, *args, **kwargs):
    """
    Decorator function to monitor the runtime of a function.
    
    This function takes another function as input, along with any number of positional and keyword arguments.
    It then defines a new function that wraps the input function, adding functionality to measure and print its runtime.
    
    Parameters:
    func (function): The function to be decorated.
    *args: Variable length argument list for the function to be decorated.
    **kwargs: Arbitrary keyword arguments for the function to be decorated.

    Returns:
    new_func (function): The decorated function with added functionality to print its runtime.
    """
    def new_func(*args, **kwargs):
        #print(f'Function "{func.__name__}" was called.')
        start_time = datetime.now()

        return_values = func(*args, **kwargs)

        end_time = datetime.now()
        run_time = end_time - start_time
        print(f'Runtime {func.__name__}: {run_time}\n')
        return return_values
    return new_func

class Wigner_coeff():

    def fact(number):
        """
        Calculate the factorial of a given number.

        This function takes an integer as input and calculates its factorial. 
        The factorial of a number is the product of all positive integers less than or equal to the number.
        If a negative number is provided, the function prints an error message and exits.

        Parameters:
        number (int): The number to calculate the factorial of.

        Returns:
        factorial (int): The factorial of the input number.
        """
        number = int(number)
        if number < 0:
            print('negative number in factorial')
            exit()
        else:
            factorial = 1
            for i in range(1,number+1,1):
                factorial *= i
            return int(factorial)

    def threej_symbol(matrix):
        # computes racah formula for 3j-symbols (p 52 Ch 1 libro Boca)
        # ([a , b, c],[A, B, C]) or ([j1, j2, j3],[m1, m2, m3])
        """
        Compute the 3j-symbol using the Racah formula.

        This function takes a 2D array as input, representing a 3j-symbol in the form ([j1, j2, j3],[m1, m2, m3]).
        It then calculates the 3j-symbol using the Racah formula, which is used in quantum mechanics to calculate the coefficients in the transformation of a couple of angular momenta.

        Parameters:
        matrix (numpy.ndarray): A 2D array representing a 3j-symbol.

        Returns:
        result (float): The calculated 3j-symbol.
        """

        matrix = np.array(matrix)

        # print(matrix)

        # shortcut per calcolare solo quelli che sono != 0
        for i in range(3):
            if matrix[1,i]>= -matrix[0,i] and matrix[1,i]<= matrix[0,i]:
                pass
            else:
                return 0

        if np.sum(matrix[1,:])==0:
            pass
        else:
            return 0

        if matrix[0,-1] >= np.abs(matrix[0,0]-matrix[0,1]) and matrix[0,-1] <= matrix[0,0]+matrix[0,1]:
            pass
        else:
            return 0

        # if isinstance(np.sum(matrix[0,:]), 'int64')==True:
        if np.sum(matrix[0,:])%1==0:
            if matrix[1,0] == matrix[1,1] and matrix[1,1] == matrix[1,2]:
                if np.sum(matrix[0,:])%2==0:
                    pass
                else:
                    return 0
            else:
                pass
        else:
            return 0

        a = matrix[0,0]
        b = matrix[0,1]
        c = matrix[0,2]
        A = matrix[1,0]
        B = matrix[1,1]
        C = matrix[1,2]

        n_min_list = [0, -c+b-A, -c+a+B]
        n_min = max(n_min_list)
        n_max_list = [a+b-c, b+B, a-A]
        n_max = min(n_max_list)
        if a-b-C <0:
            factor = (1/(-1))**np.abs(a-b-C)
        else:
            factor = (-1)**(a-b-C)
        sect1 = factor*(fact(a+b-c)*fact(b+c-a)*fact(c+a-b)/fact(a+b+c+1))**(0.5)
        sect2 = (fact(a+A)*fact(a-A)*fact(b+B)*fact(b-B)*fact(c+C)*fact(c-C))**(0.5)
        sect3 = 0
        for n in np.arange(n_min,n_max+1,1):
            sect3 += (-1)**n*(fact(n)*fact(c-b+n+A)*fact(c-a+n-B)*fact(a+b-c-n)*fact(a-n-A)*fact(b-n+B))**(-1)

        result = sect1*sect2*sect3

        return result

    def sixj_symbol(matrix):
        # computes racah formula for 6j-symbols (p 57 Ch 1 libro Boca)
        # {[a, b, c],[A, B, C]} or {[j1, j2, j3],[j4, j5, j6]}
        """
        Compute the 6j-symbol using the Racah formula.

        This function takes a 2D array as input, representing a 6j-symbol in the form ([j1, j2, j3],[j4, j5, j6]).
        It then calculates the 6j-symbol using the Racah formula, which is used in quantum mechanics to calculate the coefficients in the transformation of a couple of angular momenta.

        Parameters:
        matrix (numpy.ndarray): A 2D array representing a 6j-symbol.

        Returns:
        result (float): The calculated 6j-symbol.
        """

        matrix = np.array(matrix)
        #print(matrix)

        # shortcut per calcolare solo quelli che sono != 0
        triads = np.array([[matrix[0,0], matrix[0,1], matrix[0,2]], [matrix[0,0], matrix[1,1], matrix[1,2]],
                  [matrix[1,0], matrix[0,1], matrix[1,2]], [matrix[1,0], matrix[1,1], matrix[0,2]]])

        for i in range(len(triads[:,0])):
            if np.sum(matrix[0,:])%1==0:
                pass
            else:
                #print('zero1')
                return 0

            if triads[i,2] >= np.abs(triads[i,0]-triads[i,1]) and triads[i,2] <= triads[i,0]+triads[i,1]:
                pass
            else:
                #print('zero2')
                return 0


        def f(aa, bb, cc):
            # triangular coefficient
            res = (fact(aa+bb-cc)*fact(aa-bb+cc)*fact(-aa+bb+cc)/fact(aa+bb+cc+1))**(0.5)
            if res.imag!=0:
                return 0
            else:
                return res

        a = matrix[0,0]
        b = matrix[0,1]
        c = matrix[0,2]
        A = matrix[1,0]
        B = matrix[1,1]
        C = matrix[1,2]

        n_min_list = [0, a+A-c-C, b+B-c-C]
        n_min = max(n_min_list)
        n_max_list = [a+b+A+B+1, a+b-c, A+B-c, a+B-C, A+b-C]
        n_max = min(n_max_list)

        sect1 = (-1)**(a+b+A+B)*f(a,b,c)*f(A,B,c)*f(A,b,C)*f(a,B,C)

        #print(sect1)

        sect2 = 0
        for n in np.arange(n_min,n_max+1,1):
            sect2 += (-1)**n*fact(a+b+A+B+1-n)/(fact(n)*fact(a+b-c-n)*fact(A+B-c-n)*fact(a+B-C-n)*fact(A+b-C-n)*fact(-a-A+c+C+n)*fact(-b-B+c+C+n))
            #print(sect2)

        result = sect1*sect2

        return result

    def Wigner_Dmatrix(l, m1, m, alpha, beta, gamma):
        #calcola un elemento della matrice di wigner D^l_m1m(alpha,beta,gamma)
        #gli angoli devono essere dati in radianti
        """
        Compute an element of the Wigner D-matrix (D_{m1,m}^{l}).

        This function calculates an element of the Wigner D-matrix, which is used in quantum mechanics to describe rotations of quantum states.
        The Wigner D-matrix is defined in terms of Euler angles.

        Parameters:
        l (int): The total angular momentum quantum number.
        m1 (int): The initial magnetic quantum number.
        m (int): The final magnetic quantum number.
        alpha (float): The first Euler angle, in radians.
        beta (float): The second Euler angle, in radians.
        gamma (float): The third Euler angle, in radians.

        Returns:
        result (complex): The calculated element of the Wigner D-matrix.
        """

        d = np.sqrt(fact(l+m1)*fact(l-m1)*fact(l+m)*fact(l-m))
        somma = 0
        smin = max(0, m-m1)
        smax = min(l+m,l-m1)
        for s in range(smin,smax+1):
            somma += (-1)**(m1-m+s)*np.cos(beta/2)**(2*l+m-m1-2*s)*np.sin(beta/2)**(m1-m+2*s)/(fact(l+m-s)*fact(s)*fact(m1-m+s)*fact(l-m1-s))
        d *= somma
        return np.exp(-1j*m1*alpha)*d*np.exp(-1j*m*gamma)

    def Wigner_Dmatrix_quat(l, m1, m, R, e=1e-8):
        #R = R1 1 + Rx x + Ry y + Rz z
        #wigner D matrix D^l_m1m(R) scritta con i quaternioni (presa da How should spin-weighted spherical functions be defined? di Michael Boyle eq 35)
        #Ra e Rs si riferiscono alle geometric projections of the quaternion R into “symmetric” and “antisymmetric” components
        Rs = R[0]+1j*R[-1]
        Ra = R[2]+1j*R[1]
        rs = abs(Rs)
        ra = abs(Ra)
        phis = cmath.phase(Rs)
        phia = cmath.phase(Ra)

        if abs(ra)<e:# and abs(rs)>e:
            if m1==m:
                return np.exp(1j*2*m*phis)
            else:
                return 0
        elif abs(rs)<e:# and abs(ra)>e:
            if -m1==m:
                return (-1)**(l+m)*np.exp(1j*2*m*phia)
            else:
                return 0
        elif abs(ra)>e and abs(rs)>e and ra<=rs:
            rho1 = max(0,m1-m)
            rho2 = min(l+m1,l-m)
            fattore = np.sqrt(fact(l+m)*fact(l-m)/(fact(l+m1)*fact(l-m1)))*rs**(2*l-m+m1-2*rho1)*ra**(m-m1+2*rho1)*np.exp(1j*(phis*(m+m1)+phia*(m-m1)))
            somma = 0
            for rho in range(rho1,rho2+1):
                somma += scipy.special.binom(l+m1,rho)*scipy.special.binom(l-m1,l-m-rho)*(-(ra**2/rs**2))**rho
            return fattore*somma
        elif abs(ra)>e and abs(rs)>e and rs<ra:
            rho3 = max(0,-m1-m)
            rho4 = min(l-m1,l-m)
            fattore = np.sqrt(fact(l+m)*fact(l-m)/(fact(l+m1)*fact(l-m1)))*ra**(2*l-m-m1-2*rho3)*rs**(m+m1+2*rho3)*(-1)**(l+m)*np.exp(1j*(phia*(m-m1)+phis*(m+m1)))
            somma = 0
            for rho in range(rho3,rho4+1):
                somma += scipy.special.binom(l+m1,l-m-rho)*scipy.special.binom(l-m1,rho)*(-(rs**2/ra**2))**rho
            return fattore*somma
        # else:
        #     return 0

    def Wigner_Dmatrix_quat_complete(l, R, bin=1e-8, dict = None, coeff=None):
        #R = R1 1 + Rx x + Ry y + Rz z
        #equations from reorientational correlation functions, quaternions and wigner rotation matrices
        # print(R)
        """
        Compute the Wigner D-matrix using quaternions: R = R1 1 + Rx x + Ry y + Rz z.

        This function calculates the Wigner D-matrix using quaternions, which are a more efficient way to represent rotations than Euler angles.
        The Wigner D-matrix is used in quantum mechanics to describe rotations of quantum states.
        The quaternion in input does not need to be normalized since it is normalized before the calculation.

        Parameters:
        l (int): The orbital angular momentum quantum number.
        R (numpy.ndarray): The quaternion representing the rotation.
        bin (float, optional): The tolerance size for the calculation. Default is 1e-8.
        dict (dict, optional): dictionary from tables. Default is None.
        coeff (numpy.ndarray, optional): The coefficients for the calculation from table. Default is None.

        Returns:
        D (numpy.ndarray): The calculated Wigner D-matrix as a complex numpy array.
        """
        R /= np.linalg.norm(R)

        A = R[0] -1j*R[3]
        B = -R[2]-1j*R[1]
        # print(R)
        # print('A',A)
        # print('B',B)

        if np.abs(A)<bin:
            A = 0
        if np.abs(B)<bin:
            B = 0

        Z = R[0]**2 - R[1]**2 - R[2]**2 + R[3]**2
        Ac = np.conj(A)
        Bc = np.conj(B)
        #print('values', Z, Ac, Bc, A, B)
        D = np.zeros((2*l+1, 2*l+1), dtype='complex128')
        if l==2:
            D[4,4] = A**4                           # D[2,2] =
            D[0,0] = np.conj(D[4,4])              #D[-2,-2]
            D[4,3] = 2*A**3*B                       #D[2,1] =
            D[0,1] = (-1)*np.conj(D[4,3])         #D[-2,-1]
            D[4,2] = np.sqrt(6)*A**2*B**2           #D[2,0] =
            D[0,2] = np.conj(D[4,2])               #D[-2,0] =
            D[3,4] = -2*A**3*Bc                     #D[1,2] =
            D[1,0] = -1*np.conj(D[3,4])           #D[-1,-2]
            D[3,3] = A**2*(2*Z-1)                   #D[1,1] =
            D[1,1] = np.conj(D[3,3])              #D[-1,-1]
            D[3,2] = np.sqrt(6)*A*B*Z               #D[1,0] =
            D[1,2] = -1*np.conj(D[3,2])            #D[-1,0] =
            D[2,4] = np.sqrt(6)*A**2*Bc**2          #D[0,2] =
            D[2,0] = np.conj(D[2,4])               #D[0,-2] =
            D[2,3] = -np.sqrt(6)*A*Bc*Z             #D[0,1] =
            D[2,1] = -1*np.conj(D[2,3])            #D[0,-1] =
            D[2,2] = 1/2 * (3*Z**2-1)               #D[0,0] =
            D[1,4] = -2*A*Bc**3                    #D[-1,2] =
            D[3,0] = -np.conj(D[1,4])             #D[1,-2] =
            D[1,3] = Bc**2*(2*Z+1)                 #D[-1,1] =
            D[3,1] = np.conj(D[1,3])              #D[1,-1] =
            D[0,4] = Bc**4                         #D[-2,2] =
            D[4,0] = np.conj(D[0,4])              #D[2,-2] =
            D[0,3] = -2*Ac*Bc**3                   #D[-2,1] =
            D[4,1] = -np.conj(D[0,3])             #D[2,-1] =
        if l==3:
            D[6,6] = A**6                                       #D[3,3] = A
            D[0,0] = Ac**6                                      #D[-3,-3] =
            D[6,5] = np.sqrt(6)*B*A**5                          #D[3,2] = n
            D[0,1] = -np.sqrt(6)*Bc*Ac**5                         #D[-3,-2] =
            D[6,4] = np.sqrt(15)*B**2*A**4                      #D[3,1] = n
            D[0,2] = np.sqrt(15)*Bc**2*Ac**4                  #D[-3,-1] =
            D[6,3] = 2*np.sqrt(5)*B**3*A**3                     #D[3,0] = 2
            D[0,3] = -2*np.sqrt(5)*Bc**3*Ac**3                #D[-3,0] =
            D[5,6] = -np.sqrt(6)*A**5*Bc                        #D[2,3] = -
            D[1,0] = np.sqrt(6)*Ac**5*B                       #D[-2,-3] =
            D[5,5] = A**4*(3*Z-2)                               #D[2,2] = A
            D[1,1] = Ac**4*(3*Z-2)                            #D[-2,-2] =
            D[5,4] = 1/2 * np.sqrt(10)*B*A**3*(3*Z-1)           #D[2,1] = 1
            D[1,2] = -1/2*np.sqrt(10)*Bc*Ac**3*(3*Z-1)        #D[-2,-1] =
            D[5,3] = np.sqrt(30)*B**2*A**2*Z                    #D[2,0] = n
            D[1,3] = np.sqrt(30)*Bc**2*Ac**2*Z                 #D[-2,0] =
            D[4,6] = np.sqrt(15)*A**4*Bc**2                     #D[1,3] = n
            D[2,0] = np.sqrt(15)*Ac**4*B**2                   #D[-1,-3] =
            D[4,5] = 1/2 * np.sqrt(10)*A**3*Bc*(1-3*Z)          #D[1,2] = 1
            D[2,1] = -1/2*np.sqrt(10)*Ac**3*B*(1-3*Z)         #D[-1,-2] =
            D[4,4] = 1/4*A**2*(15*Z**2-10*Z-1)                  #D[1,1] = 1
            D[2,2] = 1/4*Ac**2*(15*Z**2-10*Z-1)               #D[-1,-1] =
            D[4,3] = 1/2 * np.sqrt(3)*A*B*(5*Z**2-1)            #D[1,0] = 1
            D[2,3] = -1/2 * np.sqrt(3)*Ac*Bc*(5*Z**2-1)        #D[-1,0] =
            D[3,6] = -2*np.sqrt(5)*A**3*Bc**3                   #D[0,3] = -
            D[3,0] = 2*np.sqrt(5)*Ac**3*B**3                   #D[0,-3] =
            D[3,5] = np.sqrt(30)*A**2*Bc**2*Z                   #D[0,2] = n
            D[3,1] = np.sqrt(30)*Ac**2*B**2*Z                  #D[0,-2] =
            D[3,4] = 1/2 * np.sqrt(3)*A*Bc*(1-5*Z**2)          #D[0,1] = -
            D[3,2] = -1/2*np.sqrt(3)*Ac*B*(1-5*Z**2)            #D[0,-1] =
            D[3,3] = 1/2*(5*Z**3-3*Z)                           #D[0,0] = 1
            D[2,6] = np.sqrt(15)*A**2*Bc**4                    #D[-1,3] =
            D[4,0] = np.sqrt(15)*Ac**2*B**4                    #D[1,-3] =
            D[2,5] = -1/2 * np.sqrt(10)*A*Bc**3*(1+3*Z)        #D[-1,2] =
            D[4,1] = 1/2*np.sqrt(10)*Ac*B**3*(1+3*Z)           #D[1,-2] =
            D[2,4] = 1/4 * Bc**2*(15*Z**2+10*Z-1)              #D[-1,1] =
            D[4,2] = 1/4 * B**2*(15*Z**2+10*Z-1)               #D[1,-1] =
            D[1,6] = -np.sqrt(6)*A*Bc**5                       #D[-2,3] =
            D[5,0] = np.sqrt(6)*Ac*B**5                        #D[2,-3] =
            D[1,5] = Bc**4*(3*Z+2)                             #D[-2,2] =
            D[5,1] = B**4*(3*Z+2)                              #D[2,-2] =
            D[1,4] = -1/2 * np.sqrt(10)*Ac*Bc**3*(3*Z+1)       #D[-2,1] =
            D[5,2] = 1/2*np.sqrt(10)*A*B**3*(3*Z+1)            #D[2,-1] =
            D[0,6] = Bc**6                                     #D[-3,3] =
            D[6,0] = B**6                                      #D[3,-3] =
            D[0,5] = -np.sqrt(6)*Ac*Bc**5                      #D[-3,2] =
            D[6,1] = np.sqrt(6)*A*B**5                         #D[3,-2] =
            D[0,4] = np.sqrt(15)*Bc**4*Ac**2                   #D[-3,1] =
            D[6,2] = np.sqrt(15)*B**4*A**2                     #D[3,-1] =
        elif (l==4 or l==6) and dict is not None and coeff is not None:
            part = dict[l]
            for key,value in part.items():
                idx = [int(ii) for ii in key.split(':')]
                i = np.abs(idx[0]+l)
                j = np.abs(idx[1]+l)
                #print(idx)
                D[i,j] = eval(value)
                D[i,j] *= coeff[l][i,j]

                #print(i,j, value, eval(value), coeff[l][i,j])
            D_rep = np.zeros_like(D, dtype='complex128')
            for i,ii in enumerate(range(l,-1,-1)):
                for j,jj in enumerate(range(l,-l-1,-1)):
                    D_rep[i,j] = D[i,j] #str(ii)+','+str(jj)# D[i,j]
                    D_rep[-i-1,-j-1] = (-1)**(np.abs(i-j))*np.conj(D[i,j]) #str(ii)+','+str(jj) #(-1)**(np.abs(i-j))*np.conj(D[i,j])
            # print('pass',l, D_rep)
            # exit()
            D=D_rep

        return D


class CFP():
    """
    A class to represent a configuration interaction (CI) calculation for a given electron configuration.

    This class is used to perform a CI calculation for a given electron configuration. The electron configuration is represented by a string of the form 'nl^x', where 'n' is the principal quantum number, 'l' is the azimuthal quantum number (represented as 'd' for l=2 and 'f' for l=3), and 'x' is the number of electrons in the configuration.

    Attributes:
    l (int): orbital momentum quantum number.
    n (int): The number of electrons in the almost-closed-shell configuration.
    N (int): The total number of electrons in the configuration.
    dic_cfp (dict): A dictionary containing the coefficients of fractional parentage (CFP) for the configuration.
    closed (bool): A flag indicating whether the configuration is almost-closed-shell or not.
    dic_LS_inv_almost (dict, optional): An inverse dictionary for the LS-coupling scheme for the almost-closed-shell configuration.
    """

    def __init__(self, conf, dic_cfp, dic_LS_inv_almost=None):
        #conf è del tipo l^(numero di elettroni)
        #ATTENZIONE: il dic_LS che è qui non è della configurazione su cui sto facendo il calcolo, ma su quella corrispondente per i almost_closed_shell + 1
        #ad esempio se sto facendo un d8, quindi con conf corrispondnte d2, il dic_LS è di d3
        """
        Initializes the CFP object with the given configuration, CFP dictionary, and optional LS-coupling scheme dictionary.

        The configuration is represented by a string of the form 'l^x', where 'l' is orbital momentum quantum number (represented as 'd' for l=2 and 'f' for l=3), and 'x' is the number of electrons in the configuration.

        Parameters:
        conf (str): The electron configuration.
        dic_cfp (dict): A dictionary containing the coefficients of fractional parentage (CFP) for the configuration.
        dic_LS_inv_almost (dict, optional): An inverse dictionary for the LS-coupling scheme for the almost-closed-shell configuration. Default is None.
        """

        if conf[0]=='d':
            self.l = 2
        else:
            self.l = 3

        self.n = int(conf[1:])  #elettroni closed conf
        self.N = int(conf[1:])  #numero di elettroni vero

        self.dic_cfp = dic_cfp

        self.closed = False
        if self.l == 2 and self.n>5:
            self.closed = True
            self.n = almost_closed_shells(conf)
        elif self.l == 2 and self.n<=5:
            pass
        elif self.l == 3 and self.n>7:
            self.closed = True
            self.n = almost_closed_shells(conf)
        elif self.l == 3 and self.n<=7:
            pass

        self.dic_LS_inv_almost = dic_LS_inv_almost


    def cfp(self, v, L, S, name):
        #ritorna i cfp sulla base del termine selezionato
        #dic è un dizionario di dizionari

        dic = self.dic_cfp  #qui per i closed shell ci deve andare quella del complementare+1

        # name = str(int(2*S+1))+state_legend(str(L), inv=True)+str(v)
        if self.closed==True:  #se si questo è vero allora i cfp li devo cercare al contrario
            cfp_list = []
            for keys in dic.keys():
                values_list = [[key, float(val)] for key, val in dic[keys].items()]
                values = sum(values_list, [])
                for i in range(0,len(values),2):
                    if values[i]==name:
                        term = self.dic_LS_inv_almost[keys][0] #mi basta prendere il primo perchè sono interessata solo a L,S,v
                        Sk = term[0]/2.
                        Lk = term[1]
                        vk = term[-1]
                        N = self.N-1
                        cfp_value = values[i+1]/((-1)**(Sk+S+Lk+L-self.l-0.5)*np.sqrt((N+1)*(2*S+1)*(2*L+1)/((4*self.l+2-N)*(2*Sk+1)*(2*Lk+1))))  #non vale per self.N = 2*self.l
                        cfp_list.append([keys,cfp_value])
        else:
            cfp_list = [[key, float(val)] for key, val in dic[name].items()]

        cfp_list = np.array(cfp_list, dtype='object')

        return cfp_list


class RME():  #RME(CFP)
    #the equations are taken from BonnMag sup info (eq 10,11)
    #or from E. Konig & S. Kremer "Ligand Field Energy Diagrams" (eq 2.85,2.87)
    #or from Boca, "theoretical fundations of molecular magnetism" (Ch 8, p 516)
    """
    A class to represent a reduced matrix element (RME) calculation for a given electron configuration.

    This class is used to perform a RME calculation for a given electron configuration. The electron configuration is represented by a string of the form 'l^x', where 'l' is the azimuthal quantum number (represented as 'd' for l=2 and 'f' for l=3), and 'x' is the number of electrons in the configuration.

    References:
    1. E. Konig & S. Kremer "Ligand Field Energy Diagrams" (eq 2.85,2.87)
    2. R. Boca, "theoretical fundations of molecular magnetism" (Ch 8, p 516)

    Attributes:
    v, L, S, v1, L1, S1 (float): The quantum numbers for the state.
    label1, label2 (str): The labels for the states.
    dic_LS (dict): A dictionary for the LS-coupling scheme.
    dic_LS_inv_almost (dict): An inverse dictionary for the LS-coupling scheme for the almost-closed-shell configuration.
    s (float): The spin quantum number.
    l (int): The azimuthal quantum number.
    n (int): The number of electrons in the closed-shell configuration.
    N (int): The total number of electrons in the configuration.
    cfp (CFP): A CFP object for the configuration.
    closed (bool): A flag indicating whether the configuration is closed-shell or not.
    """

    def __init__(self, state, conf, dic_cfp, labels, dic_LS, dic_LS_inv_almost):#, CFP_obj):
        #state contiene [v, L, S, v1, L1, S1]
        #conf è del tipo nl^(numero di elettroni)
        #ATTENZIONE: il dic_LS che è qui non è della configurazione su cui sto facendo il calcolo, ma su quella corrispondente per i almost_closed_shell + 1
        """
        Initializes the RME object with the given state, configuration, CFP dictionary, labels, and LS-coupling scheme dictionaries.

        The state is represented by a list of the form [v, L, S, v1, L1, S1], where 'v', 'L', and 'S' are the quantum numbers for the initial state and 'v1', 'L1', and 'S1' are the quantum numbers for the final state. The configuration is represented by a string of the form 'nl^x', where 'n' is the principal quantum number, 'l' is the azimuthal quantum number (represented as 'd' for l=2 and 'f' for l=3), and 'x' is the number of electrons in the configuration.

        Parameters:
        state (list): The quantum numbers for the state.
        conf (str): The electron configuration.
        dic_cfp (dict): A dictionary containing the coefficients of fractional parentage (CFP) for the configuration.
        labels (list): The labels for the states.
        dic_LS (dict): A dictionary for the LS-coupling scheme.
        dic_LS_inv_almost (dict): An inverse dictionary for the LS-coupling scheme for the almost-closed-shell configuration.
        """

        self.v = state[0]
        self.L = state[1]
        self.S = state[2]
        self.v1 = state[3]
        self.L1 = state[4]
        self.S1 = state[5]
        self.label1 = labels[0]
        self.label2 = labels[1]
        self.dic_LS = dic_LS
        self.dic_LS_inv_almost = dic_LS_inv_almost

        #super().__init__(self, conf)
        self.s = 0.5
        if conf[0]=='d':
            self.l = 2
        else:
            self.l = 3

        self.n = int(conf[1:])  #elettroni closed conf
        self.N = int(conf[1:])  #numero di elettroni vero
        self.cfp = CFP(conf, dic_cfp, dic_LS_inv_almost) #cfp corretti per i closed shells

        self.closed = False
        if self.l == 2 and self.n>5:
            self.closed = True
            self.n = almost_closed_shells(conf)
        elif self.l == 2 and self.n<=5:
            pass
        elif self.l == 3 and self.n>7:
            self.closed = True
            self.n = almost_closed_shells(conf)
        elif self.l == 3 and self.n<=7:
            pass


    def Uk(self, k):#, n=None, v=None, L=None, S=None, v1=None, L1=None, S1=None, label1=None, label2=None, closed=True):   #perché qui ho dovuto ripassare tutto?
        """
        Calculate the Uk coefficient for a given set of quantum numbers and labels.

        This method calculates the Uk coefficient, which is used in the calculation of reduced matrix elements (RMEs) in atomic physics. The quantum numbers and labels for the initial and final states are either provided as arguments or taken from the RME object itself. The calculation involves summing over the coefficients of fractional parentage (CFP) for the initial and final states and the 6-j symbol of the associated angular momenta.

        Parameters:
        k (int): The rank of the tensor operator.

        Returns:
        Uk (float): The calculated Uk coefficient.
        """

        # if n is None:
        n, v, L, S, v1, L1, S1, label1, label2 = self.N, self.v, self.L, self.S, self.v1, self.L1, self.S1, self.label1, self.label2

        pref = n*(2*L+1)**0.5*(2*L1+1)**0.5

        cfp_list = self.cfp.cfp(v, L, S, label1)
        cfp_list1 = self.cfp.cfp(v1, L1, S1, label2)

        somma = 0
        for i in range(len(cfp_list)):
            for j in range(len(cfp_list1)):
                if cfp_list[i,0] == cfp_list1[j,0]:
                    L_parent = state_legend(cfp_list[i,0][1])
                    matrix = [[L, L1, k],[self.l, self.l, L_parent]]
                    somma +=  cfp_list[i,-1]*cfp_list1[j,-1]*(-1)**(L_parent+L+self.l+k)*Wigner_coeff.sixj_symbol(matrix)

        if self.closed == True:# and closed==True:
            return (-(-1)**k)*pref*somma
        else:
            return pref*somma


    def V1k(self, k=1): #[h^2]
        """
        Calculate the V1k coefficient for a given rank of the tensor operator.

        This method calculates the V1k coefficient, which is used in the calculation of reduced matrix elements (RMEs) in atomic physics. The rank of the tensor operator is provided as an argument. The calculation involves summing over the coefficients of fractional parentage (CFP) for the initial and final states and the 6-j symbols of the associated angular momenta.

        Parameters:
        k (int, optional): The rank of the tensor operator. Default is 1.

        Returns:
        V1k (float): The calculated V1k coefficient.
        """

        pref = self.N*((self.s*(self.s+1)*(2*self.s+1)))**0.5*((2*self.S+1)*(2*self.L+1)*(2*self.S1+1)*(2*self.L1+1))**0.5

        cfp_list = self.cfp.cfp(self.v, self.L, self.S, self.label1)  #i cfp sono già corretti per closed_shells
        # print(self.v, self.L, self.S, cfp_list)
        cfp_list1 = self.cfp.cfp(self.v1, self.L1, self.S1, self.label2)
        # print(self.v1, self.L1, self.S1, cfp_list1)
        # exit()

        somma = 0
        for i in range(len(cfp_list)):
            for j in range(len(cfp_list1)):
                if cfp_list[i,0] == cfp_list1[j,0]:
                    L_parent = state_legend(cfp_list[i,0][1])
                    S_parent = (int(cfp_list[i,0][0])-1)/2
                    matrix1 = [[self.S, self.S1, 1],[self.s, self.s, S_parent]]
                    matrix2 = [[self.L, self.L1, k],[self.l, self.l, L_parent]]
                    somma += cfp_list[i,-1]*cfp_list1[j,-1]*(-1)**(L_parent+S_parent+self.S+self.L+self.s+self.l+k+1)*Wigner_coeff.sixj_symbol(matrix1)*Wigner_coeff.sixj_symbol(matrix2)
        # print(self.v, self.L, self.S)
        # print(self.v1, self.L1, self.S1)
        # print(pref*somma)
        # exit()
        if self.closed == True:
            return (-1)**k*pref*somma
        else:
            return pref*somma

class Hamiltonian():

    def __init__(self, state, labels, conf, dic_cfp=None, tables=None, dic_LS=None, dic_LS_almost=None):
        #state contiene [v, L, S, v1, L1, S1, J, M, J1, M1]
        #labels = [label1, label2]
        #conf è del tipo nl^(numero di elettroni)
        #ATTENZIONE: il dic_LS qui è della configurazione sulla quale sto facendo il calcolo

        self.v = state[0]
        self.L = state[1]
        self.S = state[2]
        self.v1 = state[3]
        self.L1 = state[4]
        self.S1 = state[5]
        self.J = state[6]
        self.M = state[7]
        self.J1 = state[8]
        self.M1 = state[9]
        self.label1 = labels[0]
        self.label2 = labels[1]

        # print('state ', state)
        # self.dic_LS = dic_LS
        self.conf = conf
        self.s = 0.5
        if conf[0]=='d':
            self.l = 2
        else:
            self.l = 3

        self.n = int(conf[1:])
        self.N = int(conf[1:])  #quelli veri
        self.closed = False
        if self.l == 2 and self.n>5:
            self.closed = True
            self.n = almost_closed_shells(conf)
        elif self.l == 2 and self.n<=5:
            pass
        elif self.l == 3 and self.n>7:
            self.closed = True
            self.n = almost_closed_shells(conf)
        elif self.l == 3 and self.n<=7:
            pass

        #self.cfp = CFP(conf)  #le conf che devo dare qui sono quelle del main
        if dic_cfp is not None:
            self.TAB = False
            if self.closed==True:
                self.dic_LS_inv_almost = {}
                for k1,v1 in dic_LS_almost.items():
                    if v1 in self.dic_LS_inv_almost.keys():
                        self.dic_LS_inv_almost[v1] += [eval('['+k1.replace(':',',')+']')]
                    else:
                        self.dic_LS_inv_almost.update({v1:[eval('['+k1.replace(':',',')+']')]})
            else:
                self.dic_LS_inv_almost = None
            self.rme = RME(state, conf, dic_cfp, labels, dic_LS, self.dic_LS_inv_almost)
            self.dic_LS = dic_LS
        elif tables is not None and dic_cfp is None:
            self.TAB = True
            self.rme = tables


    def electrostatic_int(self, basis, F0=0, F2=0, F4=0, F6=0, eval_bool=True, tab_ee=None):
        #equations are taken from Boca 1999, "theoretical fundations of molecular magnetism" (Ch 8, p 518) (valid only for l2 conf)
        #for the other configurations the equations are reported in Boca 2012 (p 145 eq 4.66-4.69)

        def l_Ck_l1(l, k, l1):
            return (-1)**l*np.sqrt((2*l+1)*(2*l1+1))*Wigner_coeff.threej_symbol([[l, k, l1],[0, 0, 0]])

        def Vee(label1v, label1v1):

            idxv = term_label.index(label1v)
            Sv, Lv, vv = term_basis[idxv]
            Sv /= 2
            #print('\n',label1v, Sv, Lv, vv)

            idxv1 = term_label.index(label1v1)
            Sv1, Lv1, vv1 = term_basis[idxv1]
            Sv1 /= 2
            #print(label1v1, Sv1, Lv1, vv1)
            integral = 0
            ck_list = np.zeros(len(range(0,2*self.l+1,2)))
            for i,k in enumerate(range(0,2*self.l+1,2)):
                ck = 0
                if k!=0:
                    ck += 0.5*l_Ck_l1(self.l, k, self.l)**2
                    somma = 0
                    for ii, term in enumerate(term_basis):
                        if self.TAB == False:

                            label2 = term_label[ii]
                            S1, L1, v1 = term_basis[ii]  #2S, L, v
                            S1 /= 2

                            if label2[0]==label1v[0]:
                                #print(label2, label1v, label1v1)
                                somma += self.rme.Uk(k, n=self.N, v=vv, L=Lv, S=Sv, v1=v1, L1=L1, S1=S1, label1=label1v, label2=label2)*self.rme.Uk(k, n=self.N, v=vv1, L=Lv1, S=Sv1, v1=v1, L1=L1, S1=S1, label1=label1v1, label2=label2)  #questi non sono quelli di closed shell
                                #print(self.rme.Uk(k, n=self.N, v=vv, L=Lv, S=Sv, v1=v1, L1=L1, S1=S1, label1=label1v, label2=label2)*self.rme.Uk(k, n=self.N, v=vv1, L=Lv1, S=Sv1, v1=v1, L1=L1, S1=S1, label1=label1v1, label2=label2))
                        else:

                            label2 = term_label[ii]
                            if label2[0]==label1v[0]:
                                try:
                                    somma += self.rme['U'+str(k)][label1v][label2]*self.rme['U'+str(k)][label1v1][label2]
                                except:
                                    somma += 0
                                # print(somma)
                    #exit()
                    somma *= 1/(2*Lv+1)
                    if self.v==self.v1:
                        somma -= self.n/(2*self.l+1)
                    ck *= somma
                else:
                    if self.v==self.v1:
                        ck = self.n*(self.n-1)/2
                if self.closed==True:
                    #ck *= (-1)**((self.v1-self.v)/2.)
                    if self.v==self.v1:
                        ck -= (2*self.l+1-self.n)/(2*self.l+1)*l_Ck_l1(self.l, k, self.l)**2
                else:
                    pass
                ck_list[i] = ck

            return ck_list

        #=======================================================================

        if eval_bool==False:
            F0, F2, F4, F6 = symbols("F0, F2, F4, F6")
            coeff = [F0, F2, F4, F6]
        else:
            coeff = [F0, F2, F4, F6]

        if tab_ee is not None:
            # pprint(tab_ee)
            coeff_ee = {}
            for key1 in tab_ee.keys():
                try:
                    var = coeff_ee[key1]
                except:
                    coeff_ee[key1] = {}
                for key2 in tab_ee[key1].keys():
                    numero = tab_ee[key1][key2]

                    if len(numero)!=1:
                        try:
                            numero = [numero[ii].replace(stringa,'') for ii,stringa in enumerate(['F0', 'F2', 'F4', 'F6'])]
                        except:
                            numero = [numero[ii].replace(stringa,'') for ii,stringa in enumerate(['F2', 'F4', 'F6'])]
                    else:
                        numero = 0
                    
                    if numero!=0:
                        for ii in range(len(numero)):
                            numero[ii]=numero[ii].replace('(', '*np.sqrt(')

                            if '+/' in numero[ii] or '-/' in numero[ii]:
                                numero[ii] = numero[ii].replace('+','+1')
                                numero[ii] = numero[ii].replace('-','-1')

                            numero[ii] = eval(numero[ii])

                    if numero!=0:
                        if len(numero)<4:
                            while len(numero)<4:
                                numero.insert(0,0)
                    else:
                        numero = [0,0,0,0]


                    coeff_ee[key1].update({key2:numero})
                    if key1!=key2:
                        #print(key1,key2)
                        try:
                            #print(key2,key1)
                            coeff_ee[key2].update({key1:numero})
                        except:
                            #print(key2,key1)
                            coeff_ee[key2]={}
                            coeff_ee[key2].update({key1:numero})
        # pprint(coeff_ee)
        # exit()

        term_basis = np.array(terms_basis(self.conf[0]+str(self.n)))
        term_label = terms_labels(self.conf[0]+str(self.n))
        ck_coeff = np.zeros(len(range(0,2*self.l+1,2)))
        for i, term in enumerate(term_basis):
            if tab_ee is not None:
                ck_coeff += np.array(coeff_ee[term_label[i]][term_label[i]])*(term[0]+1)*(2*term[1]+1)
            else:
                ck_coeff += np.array(Vee(term_label[i], term_label[i]), dtype='float64')*(term[0]+1)*(2*term[1]+1)
            # for k in range(1,4):
            #     vee = Vee(term_label[i], term_label[i])
            #     print(k, term_label[i],f'{vee[1]:.18f}',f'{vee[2]:.18f}',f'{vee[3]:.18f}', ck_coeff[k])
            #     print(k, term_label[i], ck_coeff[k])
        ck_coeff /= basis.shape[0]

        # exit()

        integral = 0
        label1v = self.label1
        label1v1 = self.label2
        if tab_ee is not None:
            try:
                ck = np.array(coeff_ee[label1v][label1v1])
            except:
                ck = np.zeros(4)
        else:
            ck = Vee(label1v, label1v1)
        for i,_ in enumerate(range(0,2*self.l+1,2)):
            if self.v==self.v1:
                integral += (ck[i] - ck_coeff[i])*coeff[i]
                #print(ck[i]-ck_coeff[i],ck[i],ck_coeff[i],coeff[i])
                # print(f'{ck[i]-ck_coeff[i]:.18f}',f'{ck[i]:.18f}',f'{ck_coeff[i]:.18f}')
            else:
                integral += ck[i]*coeff[i]

        # print(integral)
        #
        # print(label1v, label1v1, ck, coeff, ck - ck_coeff, integral)
        # print(ck_coeff*basis.shape[0])
        # exit()

        return integral

    def SO_coupling(self, zeta, k=1, eval=True):
        #eq from SI of BonnMag
        #or Konig & Kremer (Ch 2, p 11, eq 2.82)

        if eval==False:
            zeta, k = symbols("zeta, k")
        else:
            pass

        pref = zeta*k*(-1)**(self.J+self.L+self.S1)*np.sqrt(self.l*(self.l+1)*(2*self.l+1))
        coeff = Wigner_coeff.sixj_symbol([[self.L, self.L1,1],[self.S1,self.S, self.J]])
        if self.TAB == False:
            rme_V1k = self.rme.V1k()
            #print('V1k',rme_V1k,self.termn2,self.termn1 )
        else:
            try:
                rme_V1k = self.rme['V11'][self.label1][self.label2]
            except:
                rme_V1k = 0

        return pref*coeff*rme_V1k

    def LF_contribution(self, dic_kq, eval=True):
        # taken from C. Goerller-Walrand, K. Binnemans, Handbook of Physics & Chemistry of Rare Earths, Vol 23, Ch 155, (1996)

        if eval==False:
            for k in range(2,2*self.l+1,2):
                for q in range(-k,k+1,1):
                    if f'{k}' in dic_kq.keys() and f'{q}' in dic_kq[f'{k}'].keys():
                        dic_kq[f'{k}'][f'{q}'] = symbols(f"B{k}{q}")
                    else:
                        dic_kq[f'{k}'][f'{q}'] = 0.0

        else:
            for k in range(2,2*self.l+1,2):
                for q in range(-k,k+1,1):
                    if f'{k}' in dic_kq.keys() and f'{q}' in dic_kq[f'{k}'].keys():
                        pass
                    else:
                        dic_kq[f'{k}'][f'{q}'] = 0.0

        result = 0
        for k in range(2,2*self.l+1,2):
            ckk = (-1)**self.l*(2*self.l+1)*Wigner_coeff.threej_symbol([[self.l, k, self.l],[0,0,0]])
            if self.TAB == False:
                Uk = self.rme.Uk(k)
            else:
                try:
                    Uk = self.rme['U'+str(k)][self.label1][self.label2]
                except:
                    Uk = 0
            #print(k,self.label1,self.label2,f'{Uk:.16f}')
            coeff2 = Wigner_coeff.sixj_symbol([[self.J, self.J1, k],[self.L1, self.L, self.S]])
            #print(coeff2, [[self.J, self.J1, k],[self.L1, self.L, self.S]])
            int = 0
            for q in range(0,k+1):
                if q==0:
                    coeff0 = Wigner_coeff.threej_symbol([[self.J, k, self.J1],[-self.M,0,self.M1]])
                    Yk0 = (-1)**(2*self.J + self.L1 + self.S - self.M + k)*np.sqrt((2*self.J+1)*(2*self.J1+1))*ckk*coeff0*coeff2*Uk
                    int += dic_kq[f'{k}'][f'{0}']*Yk0
                else:
                    coeffp = Wigner_coeff.threej_symbol([[self.J, k, self.J1],[-self.M,q,self.M1]])
                    coeffm = Wigner_coeff.threej_symbol([[self.J, k, self.J1],[-self.M,-q,self.M1]])
                    Ckqp = (-1)**(2*self.J + self.L1 + self.S - self.M + k)*np.sqrt((2*self.J+1)*(2*self.J1+1))*ckk*coeffp*coeff2*Uk
                    Ckqm = (-1)**(2*self.J + self.L1 + self.S - self.M + k)*np.sqrt((2*self.J+1)*(2*self.J1+1))*ckk*coeffm*coeff2*Uk
                    int += dic_kq[f'{k}'][f'{q}']*(Ckqm+(-1)**q*Ckqp) + 1j*dic_kq[f'{k}'][f'{-q}']*(Ckqm - (-1)**q*Ckqp)
                    dic1 = dic_kq[f'{k}'][f'{q}']
                    dic2 = dic_kq[f'{k}'][f'{-q}']
                    # print('Bkq',k,q,f'{dic1:.16f}',f'{dic2:.16f}')
                    # print('Ckq',k,q,f'{Ckqm:.16f}',f'{Ckqp:.16f}')
            result += int

        return result

    def Zeeman(self, field=np.array([0.,0.,0.]), k=1, eval=True, MM=False, print_out=False):
        # eq from Boca 2012 p 588

        if eval ==False:
            Bx, By, Bz = symbols("Bx, By, Bz")
        else:
            Bx, By, Bz = np.array(field)

        Bohr = 0.4668604  #conv from BM to cm-1
        ge = 2.0023

        Bq = [-np.sqrt(0.5)*(Bx+1j*By), Bz, np.sqrt(0.5)*(Bx-1j*By)]  # +1, 0, -1
        L1 = []
        S1 = []

        pre = np.sqrt((2*self.J+1)*(2*self.J1+1))

        L1q = k*np.sqrt(self.L*(self.L+1)*(2*self.L+1))*(-1)**(self.L+self.S+self.J+1)*Wigner_coeff.sixj_symbol([[self.L, self.J, self.S],[self.J1, self.L, 1]])#Wigner_coeff.sixj_symbol([[self.J, 1, self.J1],[self.L, self.S, self.L]])#Wigner_coeff.sixj_symbol([[self.J1, self.J, 1],[self.L, self.L, self.S]])

        S1q = ge*np.sqrt(self.S*(self.S+1)*(2*self.S+1))*(-1)**(self.L+self.L1+self.S1*2+self.J+self.J1)*(-1)**(self.L+self.S+self.J+1)*Wigner_coeff.sixj_symbol([[self.S, self.J, self.L],[self.J1, self.S, 1]])#Wigner_coeff.sixj_symbol([[self.J, 1, self.J1],[self.S, self.L, self.S]])#Wigner_coeff.sixj_symbol([[self.J1, self.J, 1],[self.S, self.S, self.L]])

        rme = pre*(L1q + S1q)

        int = 0
        for i,q in enumerate(range(-1,2,1)):
            preq = (-1)**(self.J-self.M)*Wigner_coeff.threej_symbol([[self.J, 1, self.J1],[-self.M, q, self.M1]])
            int += (-1)**q*Bq[i]*preq*rme*Bohr

            L1.append(preq*pre*L1q)
            S1.append(preq*pre*S1q)

        if not MM:
            return int
        else:
            if print_out:
                print(L1)
                print(S1)
            return L1, S1


def Full_basis(conf):
    #basis_l = label in |SLJ>
    #n_el = numero di elettroni configurazione vera
    #conf = configurazione vera
    #per vera intendo non quelle corrispondenti nel caso di cose almost open shell

    n_el = int(conf[1:])

    if conf[0]=='d' and n_el>5:
        n_el = 10-n_el
    elif conf[0]=='f' and n_el>7:
        n_el = 14-n_el
    else:
        pass

    if conf[0]=='d':
        n_freeion_SL = [1,5,8,16,16]
        TwoJp1 = [6,9,12,13,14]
    else:
        n_freeion_SL = [1,7,17,47,73,119,119]
        TwoJp1 = [8,13,18,21,24,25,26]

    basis = []
    basis_l = []
    basis_l_JM = []
    dic_LS = {}
    count=0

    for st in range(0,n_freeion_SL[n_el-1]):  #run over free ion states
        term_base = terms_basis(conf)[st]
        TwoS = term_base[0]
        L = term_base[1]
        sen = term_base[2]

        if n_el==1:
            sen_str=''
        else:
            if st!=n_freeion_SL[n_el-1]-1 and st!=0:
                termm1 = terms_basis(conf)[st-1]
                termp1 = terms_basis(conf)[st+1]
                if sen!=n_el and termp1[0]==term_base[0] and termp1[1]==term_base[1] and (termm1[0]!=term_base[0] or termm1[1]!=term_base[1]):  #forse si può ridurre
                    count=1
                    sen_str=str(count)
                elif sen!=n_el and (termp1[0]!=term_base[0] or termp1[1]!=term_base[1]) and (termm1[0]!=term_base[0] or termm1[1]!=term_base[1]):
                    sen_str=''
                    count=0
                elif sen!=n_el and termm1[0]==term_base[0] and termm1[1]==term_base[1]:
                    count+=1
                    sen_str=str(count)
                elif sen==n_el and termm1[0]==term_base[0] and termm1[1]==term_base[1]:
                    count+=1
                    sen_str=str(count)
                elif sen==n_el and (termp1[0]==term_base[0] and termp1[1]==term_base[1]) and (termm1[0]!=term_base[0] or termm1[1]!=term_base[1]):
                    count=1
                    sen_str=str(count)
                elif sen==n_el and (termp1[0]==term_base[0] or termp1[1]==term_base[1]):
                    sen_str=''
                    count=0
            else:
                if st==0:
                    termp1 = terms_basis(conf)[st+1]
                    if (sen==n_el or sen!=n_el) and (termp1[0]!=term_base[0] or termp1[1]!=term_base[1]):
                        sen_str=''
                        count=0
                    else:
                        count+=1
                        sen_str=str(count)
                else:
                    termm1 = terms_basis(conf)[st-1]
                    if (sen==n_el or sen!=n_el) and (termm1[0]!=term_base[0] or termm1[1]!=term_base[1]):
                        sen_str=''
                        count=0
                    else:
                        count+=1
                        sen_str=str(count)
        if sen_str=='10':
            sen_str = '0'
        name_LS = str(TwoS+1)+state_legend(str(L), inv=True)+sen_str
        #print(name_LS, term_base, count)
        J1 = np.abs(2*L-TwoS)
        J2 = 2*L + TwoS
        #print(TwoS, L, J1, J2)
        for TwoJ in range(J1,J2+2,2):
            #print('TwoJ', TwoJ)
            if TwoJ%2==0:
                J_str = str(int(TwoJ/2))
            else:
                J_str = str(int(TwoJ))+'/2'
            for TwoMJ in range(-TwoJ,TwoJ+2,2):
                #print(TwoMJ)
                if conf[0]=='f':
                    lista = [TwoS, L, TwoJ, TwoMJ, sen, count]
                else:
                    lista = [TwoS, L, TwoJ, TwoMJ, sen]
                basis.append(lista)
                dic_LS[':'.join(f'{n}' for n in lista)] = name_LS
                if TwoMJ%2==0:
                    MJ_str = str(int(TwoMJ/2))
                else:
                    MJ_str = str(int(TwoMJ))+'/2'
                basis_l.append(str(TwoS+1)+state_legend(str(L), inv=True)+sen_str+' ('+J_str+')')  #questo forse ci potrei aggiungere MJ
                basis_l_JM.append(str(TwoS+1)+state_legend(str(L), inv=True)+sen_str+' ('+J_str+') '+MJ_str)

    basis = np.array(basis)
    basis_l = np.array(basis_l)
    basis_l_JM = np.array(basis_l_JM)

    #[2S, L, 2J, 2M, sen]
    if (conf[0]=='f' and n_el>7) or (conf[0]=='d' and n_el>5):
        indices = np.lexsort((basis[:, 4], basis[:, 3], -basis[:, 2], -basis[:, 1], -basis[:, 0]))
    else:
        indices = np.lexsort((basis[:, 4], basis[:, 3], basis[:, 2], -basis[:, 1], -basis[:, 0]))  #the last one is the first criteria
    basis = basis[indices]
    basis_l = basis_l[indices]
    basis_l_JM = basis_l_JM[indices]

    sorted_keys = [list(dic_LS.keys())[i] for i in indices]
    dic_LS_sorted = {k: dic_LS[k] for k in sorted_keys}

    return basis, dic_LS_sorted, basis_l, basis_l_JM

#@cron
def diagonalisation(matrix, wordy=False):
    matrix = np.round(np.copy(matrix),16)
    w,v = np.linalg.eigh(matrix)
    if round(np.linalg.norm(v[:,0]),8) != 1:
        warnings.warn('Not normalized eigenvectors!')
        print('Performing normalization...\n' if wordy else "", end = "")
        for ixx in range(v.shape[1]):
            v[:,ixx] /= np.linalg.norm(v[:,ixx])
        print('...done\n' if wordy else "", end = "")
    w = np.round(w,16)
    v = np.round(v,16)
    return w,v


class calculation():  #classe principale

    def __init__(self, conf, ground_only=False, TAB=False, wordy=True):

        if conf[0]=='d':
            self.l = 2
        else:
            self.l = 3

        self.conf = conf  #questa deve rimanere sempre quella del main
        self.n = int(conf[1:])  #elettroni closed conf
        self.N = int(conf[1:])  #numero di elettroni vero
        self.closed = False
        if self.l == 2 and self.n>5:
            self.closed = True
            self.n = almost_closed_shells(conf)
            stringa = 'd'+str(self.n+1)
        elif self.l == 2 and self.n<=5:
            stringa = conf
        elif self.l == 3 and self.n>7:
            self.closed = True
            self.n = almost_closed_shells(conf)
            stringa = 'f'+str(self.n+1)
        elif self.l == 3 and self.n<=7:
            stringa = conf

        self.basis, self.dic_LS, self.basis_l, self.basis_l_JM = Full_basis(conf)

        if ground_only==True:
            self.ground = True
            self.basis, self.dic_LS, self.basis_l, self.basis_l_JM = self.ground_state_calc()
        else:
            self.ground = False

        print('\nConfiguration: '+conf+'\n' if wordy else "", end = "")
        self.microst = int(scipy.special.binom(2*(self.l*2+1), self.N))
        print('Number of microstates: '+str(self.microst)+'\n' if wordy else "", end = "")
        if self.closed == True:
            print('Almost closed Shell, corresponding configuration: '+conf[0]+str(self.n)+'\n' if wordy else "", end = "")
        if ground_only==True:
            print('Ground state only calculation\nBasis set reduced to: '+str(self.basis.shape[0])+'\n' if wordy else "", end = "")

        self.dic_cfp, self.tables, self.dic_LS_almost, self.dic_ee = self.Tables(TAB, stringa, conf)


    def ground_state_calc(self, ground=None):
        #basis --> array: n. microstates x [2S, L, 2J, 2M, sen]
        #dic_LS --> dict: '[2S, L, 2J, 2M, sen (,count)]': label as N. and K.
        #basis_l --> list: 2S+1 L (J)
        #basis_l_JM --> list: 2S+1 L (J) MJ
        if ground is None:
            ground = ground_term_legend(self.conf) #configurazione NOT almost closed
        term_num = [(int(ground[0])-1), state_legend(ground[1]), eval(ground[ground.index('(')+1:ground.index(')')])*2]
        basis_red = []
        dic_LS_red = {}
        for i in range(self.basis.shape[0]):
            if list(self.basis[i,:3])==term_num:
                basis_red.append(list(self.basis[i]))
                lista = list(self.basis[i])
                dic_LS_red[':'.join(f'{n}' for n in lista)]=self.dic_LS[':'.join(f'{n}' for n in lista)]
        basis_red = np.array(basis_red)
        basis_l_red = [item for item in self.basis_l if item==ground]
        basis_l_JM_red = [item for item in self.basis_l_JM if item[:(item.index(')')+1)]==ground]

        return basis_red, dic_LS_red, basis_l_red, basis_l_JM_red

    def Tables(self, TAB, stringa, conf):

        if TAB==False:
            dic_cfp = cfp_from_file(stringa)
            tables = None
            if self.closed==True:
                dic_LS_almost = Full_basis(conf[0]+str(self.n+1), self.n+1)[1]  #questa è la dic_LS della configurazione corrispondente per i cfp
            else:
                dic_LS_almost = None
            dic_ee = None
        else:
            dic_cfp = None
            tables = read_matrix_from_file(self.conf, self.closed)
            dic_LS_almost = None
            dic_ee = None
            if self.l==3:# and (self.n>=4 and self.n<=7):
                dic_ee = read_ee_int(self.conf, self.closed)

        return dic_cfp, tables, dic_LS_almost, dic_ee

    def reduce_basis(self, conf, dic=None, contributes=None, roots = None, wordy=False):
        # reduce the basis set according to spin multiplicity
        # for dN configurations the Hund's rules are applied
        # for fN configurations the weak field - High Spin situation is considered and 
        # default parameters are taken from Ma, C. G., Brik, M. G., Li, Q. X., & Tian, Y. (2014) Journal of alloys and compounds, 599, 93-101.

        # basis --> array: n. microstates x [2S, L, 2J, 2M, sen]
        # dic_LS --> dict: '[2S, L, 2J, 2M, sen (,count)]': label as N. and K.
        # basis_l --> list: 2S+1 L (J)
        # basis_l_JM --> list: 2S+1 L (J) MJ

        print('Performing basis reduction... \n' if wordy else "", end = "")

        def state_select(ground, basis):

            term_num = [(int(ground[0])-1), state_legend(ground[1]), eval(ground[ground.index('(')+1:ground.index(')')])*2]
            basis_new = []
            basis_l_red = []
            basis_l_JM_red = []
            dic_LS_red = {}
            for ii in range(basis.shape[0]):
                if np.equal(basis[ii,:3],term_num).all():
                    if list(basis[ii]) not in basis_new:
                        basis_new.append(list(basis[ii]))
                        lista = list(basis[ii])
                        dic_LS_red[':'.join(f'{n}' for n in lista)]=self.dic_LS[':'.join(f'{n}' for n in lista)]
                        basis_l_red.append(self.basis_l[ii])
                        basis_l_JM_red.append(self.basis_l_JM[ii])
            basis_new = np.array(basis_new)
            
            basis_update = []
            for ii in range(basis.shape[0]):
                if not any((basis[ii] == x).all() for x in basis_new):
                    basis_update.append(list(basis[ii]))
            basis_update = np.array(basis_update)

            return basis_update, basis_new, dic_LS_red, basis_l_red, basis_l_JM_red

        def gram_schmidt(vectors):
            n = vectors.shape[1]  # Number of vectors
            orthonormal_basis = np.zeros_like(vectors)
            
            for ii in range(n):
                # Start with the original vector
                w = vectors[:, ii]
                
                # Subtract the projection on the basis formed so far
                for j in range(ii):
                    uj = orthonormal_basis[:, j]
                    projection = np.dot(w, uj) / np.dot(uj, uj) * uj
                    w = w - projection
                
                # Normalize the vector
                w = w / np.linalg.norm(w)
                orthonormal_basis[:, ii] = w
            
            return orthonormal_basis

        # the basis must be complete
        if self.ground:
            raise ValueError("Basis set reduction not allowed in ground-only calculation on "+conf+" configuration")

        if conf[0]=='d':  # dN configurations follows Hund's rule
            basis_states = self.basis[:, :3]
            max_proj = np.unique(basis_states, axis=0)
            if int(conf[1:])>5:
                indices = np.lexsort((-max_proj[:, 2], -max_proj[:, 1], -max_proj[:, 0]))
            else:
                indices = np.lexsort((max_proj[:, 2], -max_proj[:, 1], -max_proj[:, 0]))  #the last one is the first criteria
            max_proj = max_proj[indices]
        else:   # fN configurations are in the weak field - High Spin situation
            #copia i parametri e fai l'hamiltoniano con solo i contributi di Hee e Hso
            #(l'implementazione sotto dovrebbe andare bene)

            if dic is None:
                dic = free_ion_param_f_HF(conf)
            if contributes is None:
                matrix = self.build_matrix(['Hee', 'Hso'], **dic)
            else:
                matrix = self.build_matrix(contributes, **dic)

            w,v = diagonalisation(matrix)
            v = np.round(v, 16)  #numbers are saved as complex128 data type
            result = np.vstack((w,v))
            result = np.copy(result[:, result[0,:].argsort()])
            #Gram-Schmidt orthogonalization of the eigenvectors
            result[1:,:] = gram_schmidt(result[1:,:])
            projected = projection_basis(result[1:,:], self.basis_l, J_label=True)  #this gives just the order of spectroscopic terms of the free ion

            max_proj = []
            for i in projected.keys():
                s_proj = projected[i]
                dlabel = list(s_proj.keys())
                dvalue = np.array(list(s_proj.values()))
                # max_proj.append(the_highest_L(s_proj, conf[0]+str(self.n))[0])   #corresponding conf for almost-closed
                max_proj.append(dlabel[dvalue.argmax()])

            unique_max_proj = []
            for item in max_proj:
                if item not in unique_max_proj:
                    unique_max_proj.append(item)
            max_proj = unique_max_proj

        #print(max_proj)

        basis_update = np.copy(self.basis)
        nroots = np.zeros(len(roots))
        for i in range(len(roots)):
            for j in range(len(max_proj)):
                if nroots[i] < roots[i][0]*roots[i][1] and int(max_proj[j][0])==roots[i][1]:
                    nroots[i] += (eval(max_proj[j][max_proj[j].index('(')+1:max_proj[j].index(')')]))*2 +1 #eval(max_proj[j][0])*(2*state_legend(max_proj[j][1])+1) #
                    basis_update, basis_proj, dic_LS_proj, basis_l_proj, basis_l_JM_proj = state_select(max_proj[j], basis_update)
                    if basis_update.size > 0:
                        if i==0 and 'basis_red' not in locals():
                            basis_red = np.copy(basis_proj)
                            dic_LS_red = dic_LS_proj.copy()
                            basis_l_red = basis_l_proj.copy()
                            basis_l_JM_red = basis_l_JM_proj.copy()
                        else:
                            basis_red = np.vstack((basis_red, basis_proj))
                            dic_LS_red.update(dic_LS_proj)
                            basis_l_red += basis_l_proj
                            basis_l_JM_red += basis_l_JM_proj
        #print(nroots)

        self.basis, self.dic_LS, self.basis_l, self.basis_l_JM = basis_red, dic_LS_red, basis_l_red, basis_l_JM_red

        print('Calculation on reduced set\nBasis set reduced to: '+str(self.basis.shape[0])+'\n' if wordy else "", end = "")

        
    #@njit
    #@cron
    def MatrixH(self, elem, F0=0, F2=0, F4=0, F6=0, zeta=0, k=1, dic_V=None, dic_bkq = None,dic_AOM = None, PCM = None, field = [0.,0.,0.], cfp_angles = None, evaluation=True,wordy=False,
                      Orth=False, Norm=False, eig_opt=False, ground_proj=False, return_proj=False, save_label=False, save_LF=False):
        #PCM is the old Stev

        print('\nPerforming calculation with the following contributions: \n' if wordy else "", end = "")
        print(str(elem)+'\n' if wordy else "", end = "")
        if 'Hz' in elem:
            print('Magnetic field: \n'+f'{field[0]:.4e}'+' '+f'{field[1]:.4e}'+' '+f'{field[2]:.4e}'+' T\n' if wordy else "", end = "")

        #choice of conventions
        if 'Hcf' in elem:
            if dic_bkq is not None:
                pass
            elif dic_bkq is None and dic_V is not None:
                dic_bkq = from_Vint_to_Bkq(dic_V, self.conf)
            elif dic_bkq is None and dic_V is None and dic_AOM is not None:
                dic_V = from_AOM_to_Vint(dic_AOM, self.conf)
                dic_bkq = from_Vint_to_Bkq(dic_V, self.conf)
            elif dic_bkq is None and dic_V is None and dic_AOM is None and PCM is not None:
                # dic_Aqkrk = calc_Aqkrk(Stev[0], self.conf, Stev[1], Stev[2]) # first is the data, the second is the sph_flag, the third are the Sternheimer shielding parameters
                # dic_bkq = from_Aqkrk_to_Bkq(dic_Aqkrk)
                dic_bkq = calc_Bkq(PCM[0], self.conf, PCM[1], PCM[2])
            else:
                print('ERROR: BKQ dict is missing in MatrixH')
                exit()

        if eig_opt==True:
            print('\nWavefunction optimization...')
            rot_angles, dic_bkq = self.eigenfunc_opt(elem, dic_CF = dic_bkq, F0=F0, F2=F2, F4=F4, F6=F6, zeta=zeta, k=k, field=field)
            print('...done')
            print('CFP rotation angles: ', rot_angles)
        elif eig_opt==False and cfp_angles is not None:
            dic_bkq = rota_LF(self.l, dic_bkq, *cfp_angles)
        else:
            pass

        self.par_dict = {'F0':F0, 'F2':F2, 'F4':F4, 'F6':F6, 'zeta':zeta, 'k':k, 'dic_bkq':dic_bkq, 'field':field}  # it's for the magnetic properties calculation
        matrix = self.build_matrix(elem, F0, F2, F4, F6, zeta, k, dic_bkq, field, evaluation, save_label, save_LF)
        # matrix = np.conj(matrix)  #se metto questo add_Zeeman va cambiato
        # print(matrix)
        # exit()

        w,v = diagonalisation(matrix)
        v = np.round(v, 16)  #numbers are saved as complex128 data type
        result = np.vstack((w,v))

        result = np.copy(result[:, result[0,:].argsort()])

        E = np.copy(result[0,:]).real

        print('Calculation result: \n' if wordy else "", end = "")
        print('E0: '+f'{min(w):.3f}'+'\n' if wordy else "", end = "")

        ####just pretty-print
        energy_print = np.around(E-min(E),8)
        energy_list, energy_count = np.unique(energy_print, return_counts=True)
        for i in range(len(energy_list)):
            if energy_count[i]!=1:
                deg_str = f' ({energy_count[i]})'
            else:
                deg_str = ''
            print(f' {energy_list[i]:10.3f}'+deg_str+'\n' if wordy else "", end = "")
        ####

        if ground_proj == True:
            print('\nGround state projection: ' if wordy else "", end = "")
            if self.ground==True:
                projected = projection_basis(result[1:,:], self.basis_l_JM, J_label=True)
                if wordy:
                    pprint(projected[1])  #il primo stato si chiama 1
            else:
                projected = projection_basis(result[1:,:], self.basis_l, J_label=True), projection_basis(result[1:,:], self.basis_l_JM, J_label=True)
                if wordy:
                    pprint(projected[0][1]) 
                    pprint(projected[1][1])

        if Orth==True:
            print('Orthogonality check... \n' if wordy else "", end = "")
            for i in range(len(w)):
                for j in range(len(w)):
                    if i != j:
                        check = np.abs(np.dot(np.conj(result[1:,i]).T, result[1:,j]))
                        if round(check, 5) != 0:
                            warnings.warn('Non-orthognal eigenvectors found')
                            print(i, j, check)
            print('...done\n' if wordy else "", end = "")
        if Norm==True:
            print('Normalization check... \n' if wordy else "", end = "")
            for i in range(len(w)):
                check = np.abs(np.dot(np.conj(result[1:,i]).T, result[1:,i]))
                if round(check, 5) != 1:
                    warnings.warn('Non-normalized eigenvectors found')
                    print(i, check)
            print('...done\n' if wordy else "", end = "")

        if return_proj:
            return result, projected
        else:
            return result

    # @cron
    def build_matrix(self, elem, F0=0, F2=0, F4=0, F6=0, zeta=0, k=1, dic_bkq = None, field = [0.,0.,0.], evaluation=True, save_label=False, save_LF=False):

        F = F0, F2, F4, F6

        basis = self.basis
        dic_LS = self.dic_LS

        if self.closed==True:
            fac = -1.
        else:
            fac = 1.

        compare = []
        # matrix = np.zeros((basis.shape[0],basis.shape[0]),dtype='object')
        matrix = np.zeros((basis.shape[0],basis.shape[0]),dtype='complex128')
        if save_label:
            label_matrix = []
        if save_LF:
            LF_matrixs = np.zeros_like(matrix)
        list_labels = []
        for i in range(basis.shape[0]):
            statei = basis[i]
            Si = statei[0]/2.
            Li = statei[1]
            Ji = statei[2]/2.
            MJi = statei[3]/2.
            seni = statei[-1]
            labeli = dic_LS[':'.join([f'{qq}' for qq in statei])]

            if save_label:
                label_matrix.append([Si*2,Li,Ji*2,MJi*2,seni])

            # print(i)
            # print('\ni',i, Li,2*Si,2*Ji,2*MJi)
            for j in range(0,i+1):
                statej = basis[j]
                Sj = statej[0]/2.
                Lj = statej[1]
                Jj = statej[2]/2.
                MJj = statej[3]/2.
                senj = statej[-1]
                labelj = dic_LS[':'.join([f'{qq}' for qq in statej])]
                # print('j',j, Lj,2*Sj,2*Jj,2*MJj)
                H = Hamiltonian([seni,Li,Si,senj,Lj,Sj,Ji,MJi,Jj,MJj], [labeli,labelj], self.conf, self.dic_cfp, self.tables, dic_LS, self.dic_LS_almost)  #self.conf è quella del main

                if 'Hee' in elem:
                    if Ji==Jj and MJi==MJj:
                        if Li == Lj and Si == Sj:
                            #print(labeli, labelj)
                            if self.l==3:
                                Hee = H.electrostatic_int(basis, *F, eval_bool=evaluation, tab_ee = self.dic_ee)
                            else:
                                Hee = H.electrostatic_int(basis, *F, eval_bool=evaluation)
                            matrix[i,j] += Hee
                            if save_LF:
                                LF_matrixs[i,j] += Hee
                                if i!=j:
                                    LF_matrixs[j,i] += np.conj(Hee)
                            if i != j:
                                matrix[j,i] += Hee

                if 'Hso' in elem:
                    if Ji==Jj and MJi==MJj:
                        Hso = fac*H.SO_coupling(zeta, k, eval=evaluation)
                        matrix[i,j] += Hso
                        if save_LF:
                            LF_matrixs[i,j] += Hso
                            if i!=j:
                                LF_matrixs[j,i] += np.conj(Hso)
                        if i != j:
                            matrix[j,i] += Hso

                if 'Hcf' in elem:
                    if Si==Sj:
                        Hcf = fac*H.LF_contribution(dic_bkq, eval=evaluation)
                        matrix[i,j] += Hcf
                        if save_LF:
                            LF_matrixs[i,j] += Hcf
                            if i!=j:
                                LF_matrixs[j,i] += np.conj(Hcf)
                        if i != j:
                            matrix[j,i] += np.conj(Hcf)

                if 'Hz' in elem:
                    if Li==Lj and Si==Sj and seni==senj:
                        Hz = H.Zeeman(field, k, eval=evaluation)
                        # print(Hz)
                        matrix[i,j] += Hz
                        if i != j:
                            matrix[j,i] += np.conj(Hz)

                else:
                    pass

                #print(matrix[i,j].real)

        if save_label:
            np.savetxt('matrix_label.txt', np.array(label_matrix))
        if save_LF:
            np.save('matrix_LF', LF_matrixs, allow_pickle=True, fix_imports=False)

        return matrix

#======================= NEW FUNCTIONS for MAGNETIC PROPERTIES ==============================

from numba import jit, njit
from numba import complex128, boolean, float64, int32, int64
import numba
from itertools import product, permutations

@njit(float64(float64))
def fact(number):
    number = int(number)
    if number < 0:
        raise ValueError('Negative number in factorial')
    else:
        factorial = 1.0
        for i in range(1, number + 1, 1):
            factorial *= i
        #print('fact', factorial)
        return factorial

@njit(complex128[:, :](complex128[:, :]))
def from_matrix_to_result_copy(matrix):
    w, v = np.linalg.eig(matrix)
    result = np.zeros((matrix.shape[0] + 1, matrix.shape[0]), dtype=complex128)
    result[0, :] = w
    result[1:, :] = v
    result = result[:, result[0, :].real.argsort()]
    return result

@jit(float64(float64[:,:]))
def sixj_symbol(matrix):
    # computes racah formula for 6j-symbols (p 57 Ch 1 libro Boca)
    # {[a, b, c],[A, B, C]} or {[j1, j2, j3],[j4, j5, j6]}

    # shortcut per calcolare solo quelli che sono != 0
    triads = np.array([[matrix[0,0], matrix[0,1], matrix[0,2]], [matrix[0,0], matrix[1,1], matrix[1,2]],
              [matrix[1,0], matrix[0,1], matrix[1,2]], [matrix[1,0], matrix[1,1], matrix[0,2]]])

    for i in range(len(triads[:,0])):
        if np.sum(matrix[0,:])%1==0:
            pass
        else:
            #print('zero1')
            return 0

        if triads[i,2] >= np.abs(triads[i,0]-triads[i,1]) and triads[i,2] <= triads[i,0]+triads[i,1]:
            pass
        else:
            #print('zero2')
            return 0


    def f(aa, bb, cc):
        # triangular coefficient
        return (fact(aa+bb-cc)*fact(aa-bb+cc)*fact(-aa+bb+cc)/fact(aa+bb+cc+1))**(0.5)

    a = matrix[0,0]
    b = matrix[0,1]
    c = matrix[0,2]
    A = matrix[1,0]
    B = matrix[1,1]
    C = matrix[1,2]

    n_min_list = np.array([0, a+A-c-C, b+B-c-C])
    n_min = max(n_min_list)
    n_max_list = np.array([a+b+A+B+1, a+b-c, A+B-c, a+B-C, A+b-C])
    n_max = min(n_max_list)

    sect1 = (-1)**(a+b+A+B)*f(a,b,c)*f(A,B,c)*f(A,b,C)*f(a,B,C)

    sect2 = 0
    for n in np.arange(n_min,n_max+1,1):
        sect2 += (-1)**n*fact(a+b+A+B+1-n)/(fact(n)*fact(a+b-c-n)*fact(A+B-c-n)*fact(a+B-C-n)*fact(A+b-C-n)*fact(-a-A+c+C+n)*fact(-b-B+c+C+n))

    result = sect1*sect2

    return result

@jit(float64(float64[:,:]))
def threej_symbol(matrix):
    # computes racah formula for 3j-symbols (p 52 Ch 1 libro Boca)
    # ([a , b, c],[A, B, C]) or ([j1, j2, j3],[m1, m2, m3])

    # shortcut per calcolare solo quelli che sono != 0
    for i in range(3):
        if matrix[1,i]>= -matrix[0,i] and matrix[1,i]<= matrix[0,i]:
            pass
        else:
            return 0

    if np.sum(matrix[1,:])==0:
        pass
    else:
        return 0

    if matrix[0,-1] >= np.abs(matrix[0,0]-matrix[0,1]) and matrix[0,-1] <= matrix[0,0]+matrix[0,1]:
        pass
    else:
        return 0

    # if isinstance(np.sum(matrix[0,:]), 'int64')==True:
    if np.sum(matrix[0,:])%1==0:
        if matrix[1,0] == matrix[1,1] and matrix[1,1] == matrix[1,2]:
            if np.sum(matrix[0,:])%2==0:
                pass
            else:
                return 0
        else:
            pass
    else:
        return 0

    a = matrix[0,0]
    b = matrix[0,1]
    c = matrix[0,2]
    A = matrix[1,0]
    B = matrix[1,1]
    C = matrix[1,2]

    n_min_list = np.array([0, -c+b-A, -c+a+B])
    n_min = max(n_min_list)
    n_max_list = np.array([a+b-c, b+B, a-A])
    n_max = min(n_max_list)
    if a-b-C <0:
        factor = (1/(-1))**np.abs(a-b-C)
    else:
        factor = (-1)**(a-b-C)
    sect1 = factor*(fact(a+b-c)*fact(b+c-a)*fact(c+a-b)/fact(a+b+c+1))**(0.5)
    sect2 = (fact(a+A)*fact(a-A)*fact(b+B)*fact(b-B)*fact(c+C)*fact(c-C))**(0.5)
    sect3 = 0
    for n in np.arange(n_min,n_max+1,1):
        sect3 += (-1)**n*(fact(n)*fact(c-b+n+A)*fact(c-a+n-B)*fact(a+b-c-n)*fact(a-n-A)*fact(b-n+B))**(-1)

    result = sect1*sect2*sect3

    # print('dentro',sect1,sect2,sect3,sect1*sect2*sect3)

    return result

@jit(numba.types.UniTuple(complex128[:], 3)(float64,float64,float64,float64,float64,float64,float64,float64,float64[:]))   #capire bene come restituire le tuple
def Zeeman(L,S,J,M,L1,S1,J1,M1,field=np.array([0.,0.,0.])):
    # eq from Boca 2012 p 588

    k=1
    Bx = field[0]
    By = field[1]
    Bz = field[2]

    Bohr = 0.4668604  #conv from BM to cm-1
    ge = 2.0023

    Bq = np.array([-np.sqrt(0.5)*(Bx+1j*By), Bz, np.sqrt(0.5)*(Bx-1j*By)])  # +1, 0, -1

    pre = np.sqrt((2*J+1)*(2*J1+1))

    L1q = k*np.sqrt(L*(L+1)*(2*L+1))*(-1)**(L+S+J+1)*sixj_symbol(np.array([[L, J, S],[J1, L, 1]]))#sixj_symbol(np.array([[J, 1, J1],[L, S, L]]))#sixj_symbol(np.array([[J1, J, 1],[L, L, S]]))

    S1q = ge*np.sqrt(S*(S+1)*(2*S+1))*(-1)**(L+L1+S1*2+J+J1)*(-1)**(L+S+J+1)*sixj_symbol(np.array([[S, J, L],[J1, S, 1]]))#sixj_symbol(np.array([[J, 1, J1],[S, L, S]]))#sixj_symbol(np.array([[J1, J, 1],[S, S, L]]))

    rme = pre*(L1q + S1q)

    int_L1 = np.zeros(3, dtype=complex128)
    int_S1 = np.zeros(3, dtype=complex128)

    integral = 0 + 0 * 1j
    for i, q in enumerate(range(-1, 2, 1)):
        preq = (-1) ** (J - M) * threej_symbol(np.array([[J, 1, J1], [-M, q, M1]]))
        int_L1[i] = preq * pre * L1q
        int_S1[i] = preq * pre * S1q

        integral_Re = (-1) ** q * preq * rme * Bohr * Bq[i].real
        integral_Im = (-1) ** q * preq * rme * Bohr * Bq[i].imag
        integral += integral_Re +1j*integral_Im

    fake_array = np.zeros(3, dtype=complex128)  #this is just because I need to return things of the same type
    fake_array[0] = integral

    return (fake_array, int_L1, int_S1)

@jit(complex128[:,:,:](float64[:,:]))
def mag_moment(basis):
    #costruction of magnetic moment matrix as -kL-geS
    #y component is divided by i (imaginary unit)

    matrix = np.zeros((3, basis.shape[0],basis.shape[0]),dtype=complex128)
    # L_matrix = np.zeros_like(matrix)
    for i in range(basis.shape[0]):
        statei = basis[i]
        Si = statei[0]/2.
        Li = statei[1]
        Ji = statei[2]/2.
        Mi = statei[3]/2.
        seni = statei[-1]

        for j in range(0,i+1):
            statej = basis[j]
            Sj = statej[0]/2.
            Lj = statej[1]
            Jj = statej[2]/2.
            Mj = statej[3]/2.
            senj = statej[-1]

            if Li==Lj and Si==Sj and seni==senj:
                integral, kL, gS = Zeeman(Li,Si,Ji,Mi,Lj,Sj,Jj,Mj,np.array([0.,0.,0.]))

                # x,y,z  -->  -1,0,+1
                matrix[0,i,j] += (kL[0]+gS[0] - (kL[2]+gS[2]))*1/(np.sqrt(2))
                matrix[1,i,j] += (kL[0]+gS[0] + kL[2]+gS[2])*1j/(np.sqrt(2))
                matrix[2,i,j] += kL[1]+gS[1]

                # L_matrix[0,i,j] += (kL[0] - kL[2])*1/(np.sqrt(2))
                # L_matrix[1,i,j] += (kL[0] + kL[2])*1j/(np.sqrt(2))
                # L_matrix[2,i,j] += kL[1]

                if i!=j:
                    for kk in range(3):
                        matrix[kk,j,i] += np.conj(matrix[kk,i,j])

    matrix = -matrix   #the minus sign is because mu = - kL - 2S
    # print('matrix_mu',matrix[0,...])
    
    return matrix

@jit
def norm(tensor):
    return np.sqrt(np.sum(np.abs(tensor)**2))

@jit
def dfridr(func, x, h, idxi, shape, fargs):

    # print(idxi)

    CON = h*2 #* 2  #10  #if this is too high the error at the end will be higher, but if it's too low the result will be always 0
    CON2 = CON * CON
    NTAB = 10  #10
    SAFE = 2  #2
    a = np.zeros((NTAB, NTAB)+shape[1:])

    hh = h
    zero = 1e-16

    dx = np.copy(x)
    dx[idxi] += hh
    sx = np.copy(x)
    sx[idxi] -= hh
    if 2*hh!=0:
        a[0,0,...] = ((func(dx,*fargs)-func(sx,*fargs))/(2*hh))
    else:
        a[0,0,...] = ((func(dx,*fargs)-func(sx,*fargs))/zero)

    err = np.inf
    risultato = None

    for i in range(1, NTAB):
        hh /= CON
        dx = np.copy(x)
        dx[idxi] += hh
        sx = np.copy(x)
        sx[idxi] -= hh
        if 2*hh!=0:
            a[0,i,...] = ((func(dx,*fargs)-func(sx,*fargs))/(2*hh))
        else:
            a[0,i,...] = ((func(dx,*fargs)-func(sx,*fargs))/zero)
        fac = CON2
        for j in range(1, i):
            if (fac - 1)!=0:
                a[j, i,...] = (a[j - 1, i,...] * fac - a[j - 1, i - 1,...]) / (fac - 1)
            else:
                a[j, i,...] = (a[j - 1, i,...] * fac - a[j - 1, i - 1,...]) / zero
            fac *= CON2
            errt = max(norm(a[j, i,...] - a[j - 1, i,...]), norm(a[j, i,...] - a[j - 1, i - 1,...]))
            if errt <= err:
                err = errt
                risultato = a[j, i,...]
        if norm(a[i, i,...] - a[i - 1, i - 1,...]) >= SAFE * err:
            # print('safe exit', a[i, i], a[i - 1, i - 1])
            return risultato, err

    return risultato, err

@jit(complex128[:,:](float64[:],float64[:,:],complex128[:,:]))
def add_Zeeman(field_vec, basis, LF_matrix):

    #print('add_Zeeman')

    matrix = np.zeros((basis.shape[0],basis.shape[0]),dtype=complex128)
    for i in range(basis.shape[0]):
        statei = basis[i]
        Si = statei[0]/2.
        Li = statei[1]
        Ji = statei[2]/2.
        Mi = statei[3]/2.
        seni = statei[-1]

        for j in range(0,i+1):
            statej = basis[j]
            Sj = statej[0]/2.
            Lj = statej[1]
            Jj = statej[2]/2.
            Mj = statej[3]/2.
            senj = statej[-1]

            matrix[i,j] += LF_matrix[i,j]
            if i!=j:
                matrix[j,i] += LF_matrix[j,i]

            if Li==Lj and Si==Sj and seni==senj:
                integral, kL, gS = Zeeman(Li,Si,Ji,Mi,Lj,Sj,Jj,Mj,field_vec)
                matrix[i,j] += integral[0]
                if i!=j:
                    matrix[j,i] += np.conj(integral[0])

    #print('matrix',matrix)

    return matrix

@jit(float64[:](float64[:],complex128[:,:,:],complex128[:,:],float64[:,:],float64))
def M_vector(field_vec, mu_matrix, LF_matrix, basis, temp):

    kB = 1.380649e-23

    mu = np.zeros((basis.shape[0], 3), dtype=complex128)
    matrix = add_Zeeman(field_vec, basis, LF_matrix)
    result = from_matrix_to_result_copy(matrix)
    E = (result[0,:].real-min(result[0,:].real)) #* 1.9865e-23
    E -= min(E)

    M = np.zeros(3, dtype=float64)

    for kk in range(3):
        CI=1
        den = 0
        num = 0
        for ii in range(len(E)):

            mu_single = np.dot(np.conj(np.ascontiguousarray(result[1:, ii]).T), np.dot(CI * mu_matrix[kk, ...], np.ascontiguousarray(result[1:, ii])))

            if np.abs(mu[ii,kk].imag)<1e-9:
                mu[ii,kk] += mu_single.real
            else:
                print('complex',mu_single)

            num += np.real(mu[ii,kk]*np.exp(-E[ii]/(kB/1.9865e-23*temp)))
            den += np.real(np.exp(-E[ii]/(kB/1.9865e-23*temp)))

        M[kk] = num/den

    return M

@jit('Tuple((float64[:,:],float64[:,:]))(float64[:,:],float64,float64[:,:],complex128[:,:],float64)')
def susceptibility_B_ord1(fields, temp, basis, LF_matrix, delta=0.001):
    # returns the derivative of the function at a point x by Ridders' method of polynomial extrapolation. The value h is input as an estimated initial stepsize.
    # it need not to be small, but rather should be an increment in x over which the function changes substantially. An estimate of the error is also computed.
    # the stepsize is decreased by CON at each iteeration. Max size of tableau is set by NTAB.

    mu0 = 1.25663706212e-06
    muB = 0.4668517532494337

    #print('ord1')
    mu_matrix = mag_moment(basis)  #complex128[:,:,:]
    # print('from ord1: ', mu_matrix)
    chi = np.zeros((fields.shape[0], 3, 3), dtype=float64)
    err = np.zeros_like(chi)
    for i in range(fields.shape[0]):
        for idx in range(3):
            #print('idx',idx)
            chi_comp, err_comp = dfridr(M_vector, fields[i], delta, idx, chi.shape[1:], fargs=(mu_matrix, LF_matrix, basis, temp))
            chi[i,idx] = chi_comp * mu0*muB*1.9865e-23
            err[i,idx] = np.ones(chi_comp.shape)*err_comp * mu0*muB*1.9865e-23

    chi_tensor = np.sum(chi, axis=0)/fields.shape[0]
    err_tensor = np.sum(err, axis=0)/fields.shape[0]

    return (chi_tensor, err_tensor)

#======================= MAGNETIC PROPERTIES ==============================

class Magnetics():

    def __init__(self, calc, contributes, par):

        calc.MatrixH(contributes, **par)
        self.par_dict = calc.par_dict

        self.calc = calc
        self.basis = self.calc.basis
        self.Hterms = contributes

        self.kB = scipy.constants.k
        self.mu0 = scipy.constants.mu_0
        self.muB = scipy.constants.physical_constants['Bohr magneton'][0]/1.9865e-23

        #print('\nMAGNETIC PROPERTIES CALCULATION\n')

    def mag_moment(self, k, evaluation=True):
        #costruction of magnetic moment matrix as kL+geS
        #y component is divided by i (imaginary unit)

        matrix = np.zeros((3, self.basis.shape[0],self.basis.shape[0]),dtype='complex128')
        for i in range(self.basis.shape[0]):
            statei = self.basis[i]
            Si = statei[0]/2.
            Li = statei[1]
            Ji = statei[2]/2.
            MJi = statei[3]/2.
            seni = statei[-1]
            labeli = self.calc.dic_LS[':'.join([f'{qq}' for qq in statei])]
            for j in range(0,i+1):
                statej = self.basis[j]
                Sj = statej[0]/2.
                Lj = statej[1]
                Jj = statej[2]/2.
                MJj = statej[3]/2.
                senj = statej[-1]
                labelj = self.calc.dic_LS[':'.join([f'{qq}' for qq in statej])]
                H = Hamiltonian([seni,Li,Si,senj,Lj,Sj,Ji,MJi,Jj,MJj], [labeli,labelj], self.calc.conf, self.calc.dic_cfp, self.calc.tables, self.calc.dic_LS, self.calc.dic_LS_almost)  #self.conf è quella del main
                if Li==Lj and Si==Sj and seni==senj:

                    kL, gS = H.Zeeman(k=k, eval=evaluation, MM=True)

                    # x,y,z  -->  -1,0,+1
                    matrix[0,i,j] += (kL[0]+gS[0] - (kL[2]+gS[2]))*1/(np.sqrt(2))
                    matrix[1,i,j] += (kL[0]+gS[0] + kL[2]+gS[2])*1j/(np.sqrt(2))
                    matrix[2,i,j] += kL[1]+gS[1]

                    if i!=j:
                        for kk in range(3):
                            matrix[kk,j,i] += np.conj(matrix[kk,i,j])

        matrix = - matrix #the minus sign is because mu = - kL - 2S
        matrix = np.round(matrix, 16)

        return matrix

    #@staticmethod
    def effGval(self, levels, evaluation=True, v_matrix=None): 

        if v_matrix is None:
            par = self.par_dict
            matrix = self.calc.build_matrix(self.Hterms, evaluation = evaluation, **par)  #mi calcolo energia e autovalori ad un certo campo
            result = from_matrix_to_result(matrix)
            v_matrix = np.copy(result[1:,:])
        else:
            pass

        levels = np.array(levels)

        mu_matrix = self.mag_moment(1, evaluation=evaluation)  
                                                             
        gexs = np.zeros(10, dtype='int32')
        ngexs = int((levels[1]-levels[0]+1)/2)

        for i in range(ngexs):
            gexs[2*i]=int(levels[0])+2*i-1
            gexs[2*i+1]=int(levels[0])+2*i

        G2 = np.zeros((3,3), dtype='complex128')
        for i in range(ngexs):
            j = gexs[2*i]
            idx1 = [j,j,j+1,j+1]
            idx2 = [j,j+1,j,j+1]
            gk = np.zeros((3,4,4), dtype='complex128')
            for ii in idx1:
                for jj in idx2:
                    for kk in range(3):
                        CI=1
                        gk[kk,ii,jj] = np.dot(np.conj(v_matrix[:,ii]).T, np.dot(CI*mu_matrix[kk,...],v_matrix[:,jj]))

            gx11 = gk[0,j,j]; gx12 = gk[0,j,j+1]; gx21 = gk[0,j+1,j]; gx22 = gk[0,j+1,j+1];
            gy11 = gk[1,j,j]; gy12 = gk[1,j,j+1]; gy21 = gk[1,j+1,j]; gy22 = gk[1,j+1,j+1];
            gz11 = gk[2,j,j]; gz12 = gk[2,j,j+1]; gz21 = gk[2,j+1,j]; gz22 = gk[2,j+1,j+1];

            G2[0,0]=gx11*gx11 + gx12*gx21 + gx21*gx12 + gx22*gx22
            G2[0,1]=gx11*gy11 + gx12*gy21 + gx21*gy12 + gx22*gy22
            G2[1,0]=G2[0,1]
            G2[0,2]=gx11*gz11 + gx12*gz21 + gx21*gz12 + gx22*gz22
            G2[2,0]=G2[0,2]
            G2[1,1]=gy11*gy11 + gy12*gy21 + gy21*gy12 + gy22*gy22
            G2[1,2]=gy11*gz11 + gy12*gz21 + gy21*gz12 + gy22*gz22
            G2[2,1]=G2[1,2]
            G2[2,2]=gz11*gz11 + gz12*gz21 + gz21*gz12 + gz22*gz22

        w,v = np.linalg.eigh(G2)

        return np.sqrt(2*w),v


    def susceptibility_field(self, fields, temp, delta=0.001, evaluation=True, wordy=False):

        par = self.par_dict

        M_list = np.zeros(fields.shape[0])
        susc_list = np.zeros(fields.shape[0])
        mu = np.zeros((fields.shape[0], self.basis.shape[0], 2), dtype='complex128')

        mu_matrix = self.mag_moment(1, evaluation=evaluation)

        for i in range(fields.shape[0]):  #calcolo un tensore per campo
            print('FIELD: '+str(fields[i])+'\n' if wordy else "", end = "")

            par['field'] = fields[i]  #se ci metto abs viene come OctoYot
            weights = fields[i]/np.linalg.norm(fields[i])
            matrix = self.calc.build_matrix(self.Hterms, evaluation = evaluation, **par)  #mi calcolo energia e autovalori ad un certo campo
            result = from_matrix_to_result(matrix)
            E = (result[0,:].real-min(result[0,:].real)) * 1.9865e-23   # per convertire da cm-1 a J
            # print(E/1.9865e-23)
            # exit()

            den = 0
            num = 0
            for ii in range(len(E)):
                for kk in range(3):
                    CI=1
                    mu_single = np.dot(np.conj(result[1:,ii]).T, np.dot(CI*mu_matrix[kk,...],result[1:,ii]))
                    if np.abs(np.copy(mu_single).imag)<1e-9:
                        mu[i,ii,0] += np.copy(mu_single.real)*weights[kk]
                    else:
                        print('complex')

                num += np.real(mu[i,ii,0]*np.exp(-E[ii]/(self.kB*temp)))
                den += np.real(np.exp(-E[ii]/(self.kB*temp)))
            E_av = num/den

            B_inc = fields[i]/np.linalg.norm(fields[i])*delta
            par['field'] = fields[i]+B_inc
            matrix = self.calc.build_matrix(self.Hterms, evaluation=evaluation, **par)  #mi calcolo energia e autovalori ad un certo campo
            result = from_matrix_to_result(matrix)
            E = (result[0,:].real-min(result[0,:].real)) * 1.9865e-23   # per convertire da cm-1 a J
            den = 0
            num = 0
            for ii in range(len(E)):
                for kk in range(3):
                    CI=1
                    mu_single = np.dot(np.conj(result[1:,ii]).T, np.dot(CI*mu_matrix[kk,...],result[1:,ii]))
                    if np.abs(np.copy(mu_single).imag)<1e-9:
                        mu[i,ii,1] += np.copy(mu_single.real)*weights[kk]
                    else:
                        print('complex')

                num += np.real(mu[i,ii,1]*np.exp(-E[ii]/(self.kB*temp)))
                den += np.real(np.exp(-E[ii]/(self.kB*temp)))
            E_av_inc = num/den

            B_inc = fields[i]/np.linalg.norm(fields[i])*delta
            par['field'] = fields[i]-B_inc
            matrix = self.calc.build_matrix(self.Hterms, evaluation=evaluation, **par)  #mi calcolo energia e autovalori ad un certo campo
            result = from_matrix_to_result(matrix)
            E = (result[0,:].real-min(result[0,:].real)) * 1.9865e-23   # per convertire da cm-1 a J
            den = 0
            num = 0
            for ii in range(len(E)):
                for kk in range(3):
                    CI=1
                    mu_single = np.dot(np.conj(result[1:,ii]).T, np.dot(CI*mu_matrix[kk,...],result[1:,ii]))
                    if np.abs(np.copy(mu_single).imag)<1e-9:
                        mu[i,ii,1] += np.copy(mu_single.real)*weights[kk]
                    else:
                        print('complex')

                num += np.real(mu[i,ii,1]*np.exp(-E[ii]/(self.kB*temp)))
                den += np.real(np.exp(-E[ii]/(self.kB*temp)))
            E_av_incm = num/den

            M_list[i] = E_av*self.muB
            susc_list[i] = (E_av_inc - E_av_incm)/(2*delta)*self.mu0*self.muB**2

            print('M '+str(M_list[i])+'\n' if wordy else "", end = "")
            print('chi '+str(susc_list[i])+'\n' if wordy else "", end = "")

        idxM = np.argmax(M_list)
        idxm = np.argmin(M_list)

        M_av = np.sum(M_list)/len(M_list)
        sopra, sotto = 0,0
        for i in range(len(M_list)):
            if M_list[i]>M_av:
                sopra += 1
            else:
                sotto += 1


        return M_list, susc_list, fields[idxM], fields[idxm], idxM, idxm, (sopra, sotto)
    

    def susceptibility_B_copy(self, fields, temp, delta=0.001, evaluation=True, wordy=False):
        # computes the magnetic susceptibility tensor as the derivative of the magnetization vector in x,y,z
        # for a set of field vectors (evenly sampled on a sphere) at a certain temperature
        # delta is the differentiation step

        #inizio modifiche 

        def M_vector_in(field_vec):
            mu = np.zeros((self.basis.shape[0], 3), dtype='complex128')
            par['field'] = field_vec
            # print(field_vec)
            matrix = self.calc.build_matrix(self.Hterms, evaluation=evaluation, **par)  #hamiltonian matrix for a field vector B
            # print('old',matrix)
            result = from_matrix_to_result(matrix)
            E = (result[0,:].real-min(result[0,:].real)) #* 1.9865e-23
            E -= min(E)

            M = np.zeros(3)

            for kk in range(3):
                CI=1
                den = 0
                num = 0
                for ij in range(len(E)):
                    mu_single = np.dot(np.conj(result[1:,ij]).T, np.dot(CI*mu_matrix[kk,...],result[1:,ij]))   # <i|mu_kk|i> for i in range N and for kk=x,y,z
                    # mu_single = np.dot(np.conj(np.ascontiguousarray(result[1:, ij]).T), np.dot(CI * mu_matrix[kk, ...], np.ascontiguousarray(result[1:, ij])))
                    if np.abs(np.copy(mu[ij,kk]).imag)<1e-15:
                        mu[ij,kk] += np.copy(mu_single.real)
                    else:
                        print('complex',mu_single)  #just to check that the values are real and everything is okay

                    num += np.real(mu[ij,kk]*np.exp(-E[ij]/(self.kB/1.9865e-23*temp)))
                    den += np.real(np.exp(-E[ij]/(self.kB/1.9865e-23*temp)))

                M[kk] = num/den

            M = np.round(M, 16)
            return M

        par = self.par_dict
        try:
            k=par['k']
        except:
            k=1

        Mag_vector = np.zeros((fields.shape[0],3))
        mu = np.zeros((fields.shape[0], self.basis.shape[0], 4, 3), dtype='complex128')
        chi = np.zeros((fields.shape[0], 3, 3))
        mu_matrix = self.mag_moment(k, evaluation=evaluation)  #computes the magnetic moments matrix <i|kL + geS|j>

        for i in range(fields.shape[0]):
            print('FIELD: '+str(fields[i])+'\n' if wordy else "", end = "")

            M = M_vector_in(fields[i])

            Mag_vector[i,:] = M
            field_inc = np.zeros((3,3))    #matrix of fields increment, one incremented field vector for the three components: x,y,z
            field_dec = np.zeros_like(field_inc)
            M_inc = np.zeros_like(field_inc, dtype='float64')
            M_dec = np.zeros_like(field_inc, dtype='float64')
            for comp in range(3):
                field_inc[comp] = np.copy(fields[i])
                field_inc[comp, comp] += delta
                M_inc[comp,:] = M_vector_in(np.round(field_inc[comp],16))   #computation of the magnetization vector (see above) for every magnetic field increment
                field_dec[comp] = np.copy(fields[i])
                field_dec[comp, comp] -= delta
                M_dec[comp,:] = M_vector_in(np.round(field_dec[comp],16))   #same for decrement

           # print('M_inc', M_inc)

            for kki in range(3):
                for kkj in range(3):
                    # print(M_inc[kkj,kki],M[kki],M_inc[kkj,kki]-M[kki])
                    # chi[i,kki,kkj] = ((M_inc[kkj,kki]-M[kki])/delta)*self.mu0*self.muB*1.9865e-23     #computation of the chi tensor as the derivative of the magnetizaion vector with respect to the magnetic field vector
                                                                                                      #chi_ab = dMb(a)/dBa, where a,b = x,y,z
                    #print('check',M_inc[kkj,kki],M_dec[kkj,kki],M_inc[kkj,kki]-M_dec[kkj,kki],2*delta)
                    chi[i,kki,kkj] = ((M_inc[kkj,kki]-M_dec[kkj,kki])/(2*delta))*self.mu0*self.muB*1.9865e-23

            print('M (BM)\n'+str(M)+'\n' if wordy else "", end = "")
            print('M_inc (BM)\n'+str(M_inc)+'\n' if wordy else "", end = "")
            print('chi (m3)\n'+str(chi[i,:,:])+'\n' if wordy else "", end = "")

        chi_tensor = np.zeros((3,3))
        chi_tensor = np.sum(chi, axis=0)/fields.shape[0]   #since I've calculated the tensor for different directions of the magnetic field vector, here there is the average calculation
        Mav = np.sum(Mag_vector, axis=0)/fields.shape[0]    #same for magnetization

        return chi_tensor, Mav

    # @staticmethod
    # @cron
    ### COPY OF THE NUMBA FUNCTIONS ###
    def dfridr(self, func, x, h, idxi, shape, fargs):

        # print(idxi)

        CON = h*2 #* 2  #10  #if this is too high the error at the end will be higher, but if it's too low the result will be always 0
        CON2 = CON * CON
        NTAB = 10  #10
        SAFE = 2  #2
        a = np.zeros((NTAB, NTAB)+shape[1:])

        hh = h
        zero = 1e-16

        dx = np.copy(x)
        dx[idxi] += hh
        sx = np.copy(x)
        sx[idxi] -= hh
        if 2*hh!=0:
            a[0,0,...] = ((func(dx,*fargs)-func(sx,*fargs))/(2*hh))
        else:
            a[0,0,...] = ((func(dx,*fargs)-func(sx,*fargs))/zero)

        err = np.inf
        risultato = None

        for i in range(1, NTAB):
            hh /= CON
            dx = np.copy(x)
            dx[idxi] += hh
            sx = np.copy(x)
            sx[idxi] -= hh
            if 2*hh!=0:
                a[0,i,...] = ((func(dx,*fargs)-func(sx,*fargs))/(2*hh))
            else:
                a[0,i,...] = ((func(dx,*fargs)-func(sx,*fargs))/zero)
            fac = CON2
            for j in range(1, i):
                if (fac - 1)!=0:
                    a[j, i,...] = (a[j - 1, i,...] * fac - a[j - 1, i - 1,...]) / (fac - 1)
                else:
                    a[j, i,...] = (a[j - 1, i,...] * fac - a[j - 1, i - 1,...]) / zero
                fac *= CON2
                errt = max(norm(a[j, i,...] - a[j - 1, i,...]), norm(a[j, i,...] - a[j - 1, i - 1,...]))
                if errt <= err:
                    err = errt
                    risultato = a[j, i,...]
            if norm(a[i, i,...] - a[i - 1, i - 1,...]) >= SAFE * err:
                # print('safe exit', a[i, i], a[i - 1, i - 1])
                return risultato, err

        return risultato, err

    #@cron
    ### COPY OF THE NUMBA FUNCTIONS ###
    def susceptibility_B_ord1(self, fields, temp, basis, LF_matrix, delta=0.1):
        # returns the derivative of the function at a point x by Ridders' method of polynomial extrapolation. The value h is input as an estimated initial stepsize.
        # it need not to be small, but rather should be an increment in x over which the function changes substantially. An estimate of the error is also computed.
        # the stepsize is decreased by CON at each iteeration. Max size of tableau is set by NTAB.

        mu0 = 1.25663706212e-06
        muB = 0.4668517532494337

        #print('ord1')
        mu_matrix = mag_moment(basis)  #complex128[:,:,:]
        # print('from ord1: ', mu_matrix)
        chi = np.zeros((fields.shape[0], 3, 3), dtype='float64')
        err = np.zeros_like(chi)
        if len(fields.shape)<2:
            fields = np.array([fields])
        for i in range(fields.shape[0]):
            for idx in range(3):
                #print('idx',idx)
                chi_comp, err_comp = self.dfridr(M_vector, fields[i], delta, idx, chi.shape[1:], fargs=(mu_matrix, LF_matrix, basis, temp))
                chi[i,idx] = chi_comp * mu0*muB*1.9865e-23
                err[i,idx] = np.ones(chi_comp.shape)*err_comp * mu0*muB*1.9865e-23

        chi_tensor = np.zeros((3,3))
        chi_tensor = np.sum(chi, axis=0)/fields.shape[0]
        err_tensor = np.sum(err, axis=0)/fields.shape[0]

        return (chi_tensor, err_tensor)

    ### NOT TESTED ###
    def susceptibility_EB(self, fields, temp, evaluation=True, wordy=False, **par):

        try:
            k=par['k']
        except:
            k=1

        delta = 0.001  #in Tesla
        delta1 = delta
        E = np.zeros((fields.shape[0], self.basis.shape[0], 4), dtype='float64')
        E_inc = np.zeros((fields.shape[0], 3, self.basis.shape[0], 4), dtype='float64')
        chi = np.zeros((fields.shape[0], 3, 3))  # (campo x tensore)
        kt = self.kB*temp/1.9865e-23  #in cm-1
        for i in range(fields.shape[0]):  #calcolo un tensore per campo
            print('FIELD: '+str(fields[i])+'\n' if wordy else "", end = "")
            par['field'] = fields[i]  #se ci metto abs viene come OctoYot
            result = self.calc.MatrixH(self.Hterms, self.basis, evaluation=evaluation, **par, wordy=wordy)  #mi calcolo energia e autovalori ad un certo campo
            E[i,:,0] = result[0,:].real#*1.9865e-23  # per convertire da cm-1 a J
            field_inc = np.zeros((3,3))
            for comp in range(3):    # incr x y z
                field_inc[comp] = np.copy(fields[i])
                field_inc[comp, comp] += delta1
                #print('field: ', field_inc[comp])
                par['field'] = field_inc[comp]
                result = self.calc.MatrixH(self.Hterms, self.basis, evaluation=evaluation, **par, wordy=wordy)
                E[i,:,1+comp] = result[0,:].real#*1.9865e-23    # per convertire da cm-1 a J
            M = np.zeros(3)
            mu = np.zeros((self.basis.shape[0], 3))
            for kk in range(3):
                num = 0
                den = 0
                for j in range(self.basis.shape[0]):
                    mu[j,kk] = -(E[i,j,1+kk]-E[i,j,0])/delta1/self.muB
                    #print(mu[i,j,kk], E[i,j,0], E[i,j,1+kk])

                    num += mu[j,kk]*np.exp(-E[i,j,0]/kt)
                    den += np.exp(-E[i,j,0]/kt)
                M[kk] = num/den

            field_inc = np.zeros((3,3))
            M_inc = np.zeros((3,3))
            for comp in range(3):    # incr x y z
                field_inc[comp] = np.copy(fields[i])
                field_inc[comp, comp] += delta
                #print('FIELD: ', field_inc[comp])
                par['field'] = field_inc[comp]
                result = self.calc.MatrixH(self.Hterms, self.basis, evaluation=evaluation, **par, wordy=wordy)  #mi calcolo energia e autovalori ad un certo campo
                E_inc[i,comp,:,0] = result[0,:].real#*1.9865e-23  # per convertire da cm-1 a J
                field_inc_inc = np.zeros((3,3))
                for kk in range(3):    # incr x y z
                    field_inc_inc[kk] = np.copy(field_inc[comp])
                    field_inc_inc[kk, kk] += delta1
                    #print('field: ', field_inc_inc[kk])
                    par['field'] = field_inc_inc[kk]
                    result = self.calc.MatrixH(self.Hterms, self.basis, evaluation=evaluation, **par, wordy=wordy)
                    E_inc[i,comp,:,1+kk] = result[0,:].real#*1.9865e-23    # per convertire da cm-1 a J
                mu = np.zeros((self.basis.shape[0], 3))
                for kk in range(3):
                    num = 0
                    den = 0
                    for j in range(self.basis.shape[0]):
                        mu[j,kk] = -(E_inc[i,comp,j,1+kk]-E_inc[i,comp,j,0])/delta1/self.muB
                        #print(mu[i,j,kk], E_inc[comp,i,j,0], E_inc[comp,i,j,1+kk])
                        num += mu[j,kk]*np.exp(-E_inc[i,comp,j,0]/kt)
                        den += np.exp(-E_inc[i,comp,j,0]/kt)
                    M_inc[comp,kk] = num/den
                #print('M_inc',M_inc[comp])

            for kki in range(3): #magnetizzazione
                for kkj in range(3):   #campo
                    chi[i,kki,kkj] = ((M_inc[kkj,kki]-M[kki])/(delta))*self.mu0*self.muB*1.9865e-23  #chi_ab = dMa/dBb

            print('M (BM)\n'+str(M)+'\n' if wordy else "", end = "")
            print('M_inc (BM)\n'+str(M_inc)+'\n' if wordy else "", end = "")
            print('chi (m3)\n'+str(chi[i,:,:])+'\n' if wordy else "", end = "")

        chi_tensor = np.zeros((3,3))
        chi_tensor = np.sum(chi, axis=0)/fields.shape[0]

        return chi_tensor

    ### NOT TESTED ###
    def susceptibility_VV(self, fields, temp, evaluation=True, ret_energy=False, save_energy=(False,0), wordy=False, **par):
        #implementation Van Vleck (approximation for low field values)

        from itertools import groupby

        try:
            k=par['k']
        except:
            k=1

        #calcolo a campo = 0
        par['field'] = np.array([0.,0.,0.])
        result_0 = self.calc.MatrixH(self.Hterms, self.basis, evaluation=evaluation, **par, wordy=wordy)
        E_0_deg = result_0[0,:]# * 1.9865e-23   # per convertire da cm-1 a J
        # E_0_deg = np.round(E_0_deg.real*1e23, 3)*1e-23
        E_0_deg = np.round(E_0_deg.real, 10)
        energy_0 = min(E_0_deg)

        kt = self.kB*temp/1.9865e-23

        mu_int = np.zeros((fields.shape[0], self.basis.shape[0], 3), dtype='complex128') #for pert 1 ord  #(campo x stato x componente)
        mu_int2 =  np.zeros((fields.shape[0], self.basis.shape[0], self.basis.shape[0], 3), dtype='complex128') #for pert 2 ord (campo x i x j x comp) <i|mu|j>
        chi_approx = np.zeros((fields.shape[0], 3, 3),dtype='float64')  # (campo x tensore)
        chi_VV = np.zeros((fields.shape[0], 3, 3), dtype='float64')
        #chi_prova = np.zeros((fields.shape[0], 3, 3), dtype='float64')
        E_pert = np.zeros(self.basis.shape[0], dtype='float64')
        E_diag = np.zeros(self.basis.shape[0], dtype='float64')
        for i in range(fields.shape[0]):  #calcolo un tensore per campo
            print('\nFIELD: '+str(fields[i])+'\n' if wordy else "", end = "")
            #calcolo a campo != 0
            #fields[i] = np.array([28.,0.,0.])#np.abs(fields[i])
            par['field'] = fields[i]
            result = self.calc.MatrixH(self.Hterms, self.basis, evaluation=evaluation, **par, wordy=wordy)
            E_field = result[0,:].real #risultato vero
            E_field = np.round(E_field.real, 10)
            energy_field = min(E_field)
            E_diag += E_field

            mu_matrix = self.mag_moment(k, evaluation=evaluation)  #è indipendente dal campo
            #aggiusto la base nel caso ci fossero stati degeneri
            E_0 = [list(j) for i, j in groupby(E_0_deg)]  #lista di liste di tuple (energia, indice lvl)

            E1 = np.zeros_like(E_0_deg, dtype='float64')

            for ii in range(len(E_0)):  #cicla sui gruppi di stati quindi non sono gli indici giusti
                if len(E_0[ii])>1:
                    print('\nDEGENERATE STATE\n' if wordy else "", end = "")
                    index = [i for i,num in enumerate(E_0_deg) if num in E_0[ii]]
                    print(str(index)+'\n' if wordy else "", end = "")
                    H1 = np.zeros((len(E_0[ii]),len(E_0[ii])), dtype='complex128')
                    result_new = np.zeros((self.basis.shape[0], len(E_0[ii])), dtype='complex128')
                    for ji, ind1 in enumerate(range(index[0],index[-1]+1)):  #indici di stati
                        for jj, ind2 in enumerate(range(index[0],index[-1]+1)):
                            # print(ji, ind1)
                            # print(jj, ind2)
                            for kk in range(3):
                                if kk==1:
                                    CI=-1j
                                else:
                                    CI=1
                                mu = 0
                                for j in range(self.basis.shape[0]):
                                    for l in range(self.basis.shape[0]):
                                        mu += np.conj(result_0[j+1,ind1])*CI*mu_matrix[kk,j,l]*result_0[l+1,ind2]
                                # print(mu, kk, ind1, ind2)
                                H1[ji,jj] += self.muB*fields[i,kk]*mu

                                if ji==jj:
                                    if np.abs(mu.real)<1e-9:
                                        pass
                                    else:
                                        mu_int[i,ind1,kk] = mu.real
                    # print('matrice_perturbazione\n', H1)

                    v, w = np.linalg.eig(H1)
                    # print('v: ', v)

                    for j, ind1 in enumerate(range(index[0],index[-1]+1)):
                        E1[ind1] = v[j].real

                    for jj in range(len(E_0[ii])):
                        for j, ind1 in enumerate(range(index[0],index[-1]+1)):
                            result_new[:,jj] += w[j,jj]*result_0[1:,ind1]

                    print('Orthogonality check... \n' if wordy else "", end = "")
                    orth = True
                    for ind1 in range(len(w)):
                        for ind2 in range(len(w)):
                            if ind1 != ind2:
                                check = np.abs(np.dot(np.conj(result_new[:,ind1]).T, result_new[:,ind2]))
                                if round(check, 10) != 0:
                                    orth=False

                    if orth==False:
                        print('Orthogonalization procedure...\n' if wordy else "", end = "")

                        for j in range(1,len(w)):
                            for jj in range(0,j):
                                if j!=jj:
                                    # print(j,jj)
                                    result_new[:,j] -= np.dot(np.conj(result_new[:,j]).T, result_new[:,jj])/np.dot(np.conj(result_new[:,jj]).T, result_new[:,jj])*result_new[:,jj]
                        for j in range(len(w)):
                            result_new[:,j] /= np.sqrt(np.dot(np.conj(result_new[:,j]).T, result_new[:,j]))

                        for ind1 in range(len(w)):
                            for ind2 in range(len(w)):
                                if ind1 != ind2:
                                    check = np.abs(np.dot(np.conj(result_new[:,ind1]).T, result_new[:,ind2]))
                                    if round(check, 10) != 0:
                                        orth=False
                    print('...done\n' if wordy else "", end = "")

                    print('Normalization check... \n' if wordy else "", end = "")
                    for ind1 in range(len(w)):
                        check = np.abs(np.dot(np.conj(result_new[:,ind1]).T, result_new[:,ind1]))
                        if round(check, 10) != 1:
                            warnings.warn('Non-normalized eigenvectors found')
                            print(ind1, check)
                    print('...done\n'  if wordy else "", end = "")

                    #substitute in the old basis
                    for j,ind1 in enumerate(range(index[0],index[-1]+1)):
                        result_0[1:,ind1] = result_new[:,j]

                else:
                    index = [i for i,num in enumerate(E_0_deg) if num in E_0[ii]]
                    ind = index[0]
                    #print(index)

                    for kk in range(3):
                        if kk==1:
                            CI=-1j
                        else:
                            CI=1

                        for j in range(self.basis.shape[0]):
                            for l in range(self.basis.shape[0]):
                                mu_int[i,ind,kk] += np.conj(result_0[j+1,ind])*CI*mu_matrix[kk,j,l]*result_0[l+1,ind]

                        if np.abs(mu_int[i,ind,kk].imag)<1e-9:
                            if np.abs(mu_int[i,ind,kk].real)<1e-9:
                                mu_int[i,ind,kk] = 0
                            else:
                                mu_int[i,ind,kk] = np.copy(mu_int[i,ind,kk].real)
                        else:
                            print('complex', mu_int[i,ind,kk])
                        # print(mu[i,index[0],kk], fields[i,kk])
                        E1[ind] += self.muB*fields[i,kk]*mu_int[i,ind,kk].real

            E_01 = np.sort(E_0_deg+E1)
            # for ii in range(self.basis.shape[0]):
            #     print(f'{E_field[ii]:.5f}'+'\t'+f'{E_0_deg[ii]:.5f}'+'\t'+f'{E_01[ii]:.5f}')

            #ricalcolo mu_int con la base nuova mu_int(campo x i x comp)
            for ind in range(self.basis.shape[0]):
                for kk in range(3):
                    if kk==1:
                        CI=-1j
                    else:
                        CI=1
                    mu_int[i,ind,kk] = np.dot(np.conj(result_0[1:,ind]).T, np.dot(CI*mu_matrix[kk,...],result_0[1:,ind]))

            #calcolo i contributi al secondo ordine
            #ricalcolo tutti gli integrali misti <i|mu|j> anche tra stati degeneri (anche se so che saranno zero, visto che ho riaggiustato la base)
            # mu_int2(campo x i x j x comp)
            E2 = np.zeros_like(E_0_deg, dtype='float64')
            for ind1 in range(self.basis.shape[0]):
                for ind2 in range(self.basis.shape[0]):
                    for kk in range(3):
                        if kk==1:
                            CI=-1j
                        else:
                            CI=1
                        mu_int2[i,ind1,ind2,kk] = np.dot(np.conj(result_0[1:,ind1]).T, np.dot(CI*mu_matrix[kk,...],result_0[1:,ind2]))

            for ind1 in range(self.basis.shape[0]):
                for ind2 in range(self.basis.shape[0]):
                    if ind2!=ind1:
                        for kki in range(3):
                            for kkj in range(3):
                                if E_0_deg[ind1]-E_0_deg[ind2]!=0:
                                    E2[ind1] += np.real(mu_int2[i,ind1,ind2,kki]*mu_int2[i,ind2,ind1,kkj]*fields[i,kkj]*fields[i,kki]*self.muB**2/(E_0_deg[ind1]-E_0_deg[ind2]))
                    else:
                        pass
                #same as
                # a = 0
                # for kki in range(3):
                #     b = 0
                #     for kkj in range(3):
                #         somma = 0
                #         for ind2 in range(self.basis.shape[0]):
                #             if ind2 != ind1 and E_0_deg[ind1]-E_0_deg[ind2]!=0:
                #                 somma += mu_int2[i,ind1,ind2,kki]*mu_int2[i,ind2,ind1,kkj]/(E_0_deg[ind1]-E_0_deg[ind2])
                #         b += somma*fields[i,kkj]*self.muB
                #     a += b*fields[i,kki]*self.muB
                # if np.abs(a.imag)<1e-9:
                #     E2[ind1] += a.real
                # elif np.abs(a.imag)>1e-9:
                #     print('complex')
                # else:
                #     pass

            E_012 = np.sort(E_0_deg+E1+E2)
            E_pert += E_012
            print('\nEB\t\tE0\t\tE0+E1\t\tE0+E1+E2\n' if wordy else "", end = "")
            for ii in range(self.basis.shape[0]):
                print(f'{E_field[ii]:.5f}'+'\t'+f'{E_0_deg[ii]:.5f}'+'\t'+f'{E_01[ii]:.5f}'+'\t'+f'{E_012[ii]:.5f}'+'\n' if wordy else "", end = "")

            #calcolo chi approx
            #mu_int(campo x i x comp)
            #mu_int2(campo x i x j x comp)
            #chi_approx(field, 3, 3)
            for kki in range(3):
                for kkj in range(3):
                    num = 0
                    den = 0
                    for ind1 in range(self.basis.shape[0]):
                        somma = 0
                        for ind2 in range(self.basis.shape[0]):
                            if E_0_deg[ind1]-E_0_deg[ind2]!=0:
                                somma += (mu_int2[i,ind1,ind2,kki]*mu_int2[i,ind2,ind1,kkj] + mu_int2[i,ind1,ind2,kkj]*mu_int2[i,ind2,ind1,kki])/(E_0_deg[ind1]-E_0_deg[ind2])
                        num += (mu_int[i,ind1,kki]*mu_int[i,ind1,kkj]/kt - somma)*np.exp(-E_0_deg[ind1]/kt)
                        den += np.exp(-E_0_deg[ind1]/kt)
                    chi_approx[i,kki,kkj] = np.real(num/den) * self.mu0*self.muB**2*1.9865e-23

            #print(chi_approx[i])

            #calcolo chi not approx
            #chi_VV(field,3,3)
            E_2nd_order = E_0_deg+E1+E2 #non riordinato
            for kki in range(3):
                for kkj in range(3):
                    num = 0
                    den = 0
                    for ind1 in range(self.basis.shape[0]):
                        somma = 0
                        for ind2 in range(self.basis.shape[0]):
                            if E_0_deg[ind1]-E_0_deg[ind2]!=0:
                                somma += (mu_int2[i,ind1,ind2,kki]*mu_int2[i,ind2,ind1,kkj] + mu_int2[i,ind1,ind2,kkj]*mu_int2[i,ind2,ind1,kki])/(E_0_deg[ind1]-E_0_deg[ind2])
                        num += (- somma)*np.exp(-E_2nd_order[ind1]/kt)
                        den += np.exp(-E_2nd_order[ind1]/kt)
                    chi_VV[i,kki,kkj] = np.real(num/den) * self.mu0*self.muB**2*1.9865e-23

            #print(chi_VV[i])
            # E_2nd_order = E_0_deg+E1+E2 #non riordinato
            # for kki in range(3):
            #     for kkj in range(3):
            #         num = 0
            #         den = 0
            #         for ind1 in range(self.basis.shape[0]):
            #             num += (- somma)*np.exp(-E_2nd_order[ind1]/kt)
            #             den += np.exp(-E_2nd_order[ind1]/kt)
            #         chi_VV[i,kki,kkj] = np.real(num/den) * self.mu0*self.muB*1.9865e-23


        E_diag /= fields.shape[0]
        E_pert /= fields.shape[0]

        chi_approx_tensor = np.zeros((3,3))
        chi_approx_tensor = np.sum(chi_approx, axis=0)/fields.shape[0]
        chi_VV_tensor = np.zeros((3,3))
        chi_VV_tensor = np.sum(chi_VV, axis=0)/fields.shape[0]

        if save_energy[0]==True:
            with open('Energies'+str(int(save_energy[-1]))+'.txt', 'w') as f:
                f.write('EB\t\tE0\t\tE0+E1+E2\n')
                for ii in range(len(E_field)):
                    f.write(f'{E_diag[ii]:.5f}'+'\t'+f'{E_0_deg[ii]:.5f}'+'\t'+f'{E_pert[ii]:.5f}'+'\n')

        if ret_energy==True:
            return E_diag, E_pert, chi_approx_tensor, chi_VV_tensor
        else:
            return chi_approx_tensor, chi_VV_tensor

    ### NOT TESTED ###
    def susceptibility_Brill(self, fields, temp, delta=0.001, evaluation=True, wordy=False, **par):
        #parigi 2019 eq 54

        try:
            k=par['k']
        except:
            k=1

        delta1 = delta
        E = np.zeros((fields.shape[0], self.basis.shape[0], 4), dtype='float64')
        E_inc = np.zeros((fields.shape[0], 3, self.basis.shape[0], 4), dtype='float64')
        chi = np.zeros((fields.shape[0], 3, 3))  # (campo x tensore)
        kt = self.kB*temp/1.9865e-23  #in cm-1
        for i in range(fields.shape[0]):  #calcolo un tensore per campo
            print('FIELD: '+str(fields[i])+'\n' if wordy else "", end = "")
            par['field'] = fields[i]  #se ci metto abs viene come OctoYot
            result = self.calc.MatrixH(self.Hterms, self.basis, evaluation=evaluation, **par, wordy=wordy)  #mi calcolo energia e autovalori ad un certo campo
            E[i,:,0] = result[0,:].real#*1.9865e-23  # per convertire da cm-1 a J
            field_inc = np.zeros((3,3))
            for comp in range(3):    # incr x y z
                field_inc[comp] = np.copy(fields[i])
                field_inc[comp, comp] += delta1
                #print('field: ', field_inc[comp])
                par['field'] = field_inc[comp]
                result = self.calc.MatrixH(self.Hterms, self.basis, evaluation=evaluation, **par, wordy=wordy)
                E[i,:,1+comp] = result[0,:].real#*1.9865e-23    # per convertire da cm-1 a J
            M = np.zeros(3)
            for kk in range(3):
                somma_inc = 0
                somma = 0
                for j in range(self.basis.shape[0]):
                    somma_inc += np.exp(-(E[i,j,1+kk]-min(E[i,:,0]))/kt)
                    somma += np.exp(-(E[i,j,0]-min(E[i,:,0]))/kt)

                M[kk] = (np.log(somma_inc)-np.log(somma))/delta

            field_inc = np.zeros((3,3))
            M_inc = np.zeros((3,3))
            for comp in range(3):    # incr x y z
                field_inc[comp] = np.copy(fields[i])
                field_inc[comp, comp] += delta
                #print('FIELD: ', field_inc[comp])
                par['field'] = field_inc[comp]
                result = self.calc.MatrixH(self.Hterms, self.basis, evaluation=evaluation, **par, wordy=wordy)  #mi calcolo energia e autovalori ad un certo campo
                E_inc[i,comp,:,0] = result[0,:].real#*1.9865e-23  # per convertire da cm-1 a J
                field_inc_inc = np.zeros((3,3))
                for kk in range(3):    # incr x y z
                    field_inc_inc[kk] = np.copy(field_inc[comp])
                    field_inc_inc[kk, kk] += delta1
                    #print('field: ', field_inc_inc[kk])
                    par['field'] = field_inc_inc[kk]
                    result = self.calc.MatrixH(self.Hterms, self.basis, evaluation=evaluation, **par, wordy=wordy)
                    E_inc[i,comp,:,1+kk] = result[0,:].real#*1.9865e-23    # per convertire da cm-1 a J
                for kk in range(3):
                    somma_inc = 0
                    somma = 0
                    for j in range(self.basis.shape[0]):
                        somma_inc += np.exp(-(E_inc[i,comp,j,1+kk]-min(E[i,:,0]))/kt)
                        somma += np.exp(-(E_inc[i,comp,j,0]-min(E[i,:,0]))/kt)

                    M_inc[comp, kk] = (np.log(somma_inc)-np.log(somma))/delta
                #print('M_inc',M_inc[comp])

            for kki in range(3): #magnetizzazione
                for kkj in range(3):   #campo
                    chi[i,kki,kkj] = ((M_inc[kkj,kki]-M[kki])/(delta))*self.mu0*self.muB**2*1.9865e-23*kt  #chi_ab = dMa/dBb

            print('M (BM)\n'+str(M)+'\n' if wordy else "", end = "")
            print('M_inc (BM)\n'+str(M_inc)+'\n' if wordy else "", end = "")
            print('chi (m3)\n'+str(chi[i,:,:])+'\n' if wordy else "", end = "")

        chi_tensor = np.zeros((3,3))
        chi_tensor = np.sum(chi, axis=0)/fields.shape[0]

        return chi_tensor

#########PROJECTIONS###########

def projection(basis2, labels, basis1, energy, bin=1e-4, min_contr=10):
    #just implementation of projection operator

    matrix_coeff = np.zeros_like(basis1)
    matrix_coeff_re2 = np.zeros_like(basis1, dtype='float64')
    states = []
    states_red = {}

    for i in range(basis2.shape[0]):
        states.append([])
        states_red[i+1] = {}
        for j in range(basis1.shape[0]):
            matrix_coeff[j,i] = np.dot(basis1[:,j].T,basis2[:,i])
            matrix_coeff_re2[j,i] = np.abs(np.dot(np.conj(basis1[:,j]).T,basis2[:,i]))**2
            stato = [f'{round(energy[j],3)}', matrix_coeff_re2[j,i]]
            states[i].append(stato)
            key, value = stato[0], stato[1]
            if value>bin:
                if key in states_red[i+1].keys():
                    states_red[i+1][key] += value
                else:
                    states_red[i+1][key] = value
            else:
                pass
        # print(matrix_coeff[:,i])   #combinazione lineare per basis2[:,i]
        # print(matrix_coeff_re2[:,i])
        # exit()
        tot = sum(states_red[i+1].values())
        if round(tot,2) != 1:
            warnings.warn('The expantion coefficient do not sum to 1')
            print(tot)
        for key, value in states_red[i+1].items():
            states_red[i+1][key] = value*100/tot
        sortato = sorted(states_red[i+1].items(), key=lambda x:x[1], reverse=True)  #sort dict on values
        states_red[i+1] = dict(sortato)

    # print(states_red)

    states_red_2 = {}   #in questo ci vanno solo i livelli con percentuali superiori o uguali a min_contr
    for key1 in states_red.keys():
        states_red_2[key1] = {}
        for key2, value in states_red[key1].items():
            if value>=min_contr:
                states_red_2[key1][key2] = value

    # print(states_red_2)

    return states_red_2

def projection_basis(basis2, labels, bin=1e-5, J_label=False):
    #just implementation of projection operator

    basis1 = np.eye(basis2.shape[0])
    matrix_coeff = np.zeros_like(basis1, dtype='complex128')
    matrix_coeff_re2 = np.zeros_like(basis1, dtype='float64')
    states = []
    states_red = {}
    for i in range(basis2.shape[0]):
        states.append([])
        states_red[i+1] = {}
        for j in range(basis1.shape[0]):
            matrix_coeff[j,i] = np.dot(basis1[:,j].T,basis2[:,i])
            matrix_coeff_re2[j,i] = np.abs(np.dot(np.conj(basis1[:,j]).T,basis2[:,i]))**2
            stato = [labels[j], matrix_coeff_re2[j,i]]
            states[i].append(stato)
            if J_label==True:
                key, value = stato[0], stato[1]
            else:
                key, value = stato[0][:stato[0].index(' (')], stato[1]
            if value>bin:
                if key in states_red[i+1].keys():
                    states_red[i+1][key] += value
                else:
                    states_red[i+1][key] = value
            else:
                pass
        tot = sum(states_red[i+1].values())
        if round(tot,2) != 1:
            warnings.warn('The expantion coefficient do not sum to 1')
            print(tot)
        for key, value in states_red[i+1].items():
            states_red[i+1][key] = value*100/tot
        sortato = sorted(states_red[i+1].items(), key=lambda x:x[1], reverse=True)  #sort dict on values
        states_red[i+1] = dict(sortato)

    return states_red

def the_highest_MJ(proj, state_list):

    cont = np.zeros_like(state_list)
    for key,value in proj.items():
        for i in range(len(cont)):
            end = key.find(')')
            M_key = np.abs(eval(key[end+1:]))
            if state_list[i]==M_key:
                cont[i] += value

    M = state_list[cont.argmax()]
    perc = max(cont)

    return M, perc

def the_highest_L(proj, conf):
    
    state_list = terms_labels(conf)
    
    cont = np.zeros(len(state_list))
    for key,value in proj.items():
        
        for i in range(len(cont)):
            try:
                L_key = key[:(key.index('(')-1)]
            except:
                L_key = key
            if state_list[i]==L_key:
                cont[i] += value

    L = state_list[cont.argmax()]
    perc = max(cont)

    return L, perc


#######FIGURES###########

def tensor_rep(A, name=None, angles=(0,0)):
    #represents the rank2 tensors as linear combination of spherical harmonics

    import matplotlib.gridspec as gridspec
    from mpl_toolkits.mplot3d import Axes3D
    from scipy.special import sph_harm

    # Grids of polar and azimuthal angles
    theta = np.linspace(0, np.pi, 50)
    phi = np.linspace(0, 2*np.pi, 50)
    # Create a 2-D meshgrid of (theta, phi) angles.
    theta, phi = np.meshgrid(theta, phi)
    # Calculate the Cartesian coordinates of each point in the mesh.
    xyz = np.array([np.sin(theta) * np.sin(phi),
                    np.sin(theta) * np.cos(phi),
                    np.cos(theta)])

    def plot_Y(ax, el, m, k=1):
        """Plot the spherical harmonic of degree el and order m on Axes ax."""

        if len(el)>1:
            Yx, Yy, Yz = np.zeros_like(xyz)
            for i in range(len(el)):
                # NB In SciPy's sph_harm function the azimuthal coordinate, theta,
                # comes before the polar coordinate, phi.
                Y = k[i]*sph_harm(abs(m[i]), el[i], phi, theta)
                # Linear combination of Y_l,m and Y_l,-m to create the real form.
                if m[i] < 0:
                    Y = np.sqrt(2) * (-1)**m[i] * Y.imag
                elif m[i] > 0:
                    Y = np.sqrt(2) * (-1)**m[i] * Y.real
                yx, yy, yz = (np.abs(Y) * xyz)
                Yx += yx
                Yy += yy
                Yz += yz
        else:
            Y = sph_harm(abs(m), el, phi, theta)
            # Linear combination of Y_l,m and Y_l,-m to create the real form.
            if m < 0:
                Y = np.sqrt(2) * (-1)**m * Y.imag
            elif m > 0:
                Y = np.sqrt(2) * (-1)**m * Y.real
            Yx, Yy, Yz = k*(np.abs(Y) * xyz)

        # Colour the plotted surface according to the sign of Y.
        cmap = plt.cm.ScalarMappable(cmap=plt.get_cmap('RdBu_r'))  #rosso positivo, blu negativo
        cmap.set_clim(-0.5, 0.5)

        ax.plot_surface(Yx, Yy, Yz,
                        facecolors=cmap.to_rgba(Y.real),
                        rstride=2, cstride=2)

        # Draw a set of x, y, z axes for reference.
        delta=0.2
        ax_lim = [np.amax(np.abs(Yx)), np.amax(np.abs(Yy)), np.amax(np.abs(Yz))]
        ax.plot([-ax_lim[0], ax_lim[0]], [0,0], [0,0], c='0.5', lw=1, zorder=10)
        ax.plot([0,0], [-ax_lim[1], ax_lim[1]], [0,0], c='0.5', lw=1, zorder=10)
        ax.plot([0,0], [0,0], [-ax_lim[2], ax_lim[2]], c='0.5', lw=1, zorder=10)
        # Set the Axes limits and title, turn off the Axes frame.
        ax.set_title(r'$Y_{{{},{}}}$'.format(el, m))
        ax.set_xlim(-ax_lim[0]+delta, ax_lim[0]+delta)
        ax.set_ylim(-ax_lim[1]+delta, ax_lim[1]+delta)
        ax.set_zlim(-ax_lim[2]+delta, ax_lim[2]+delta)
        #ax.axis('off')

    S_A = np.zeros_like(A)
    N_A = np.zeros_like(A)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            S_A[i,j] = (1/2)*(A[i,j]+A[j,i])
            N_A[i,j] = S_A[i,j]-(1/3)*np.trace(A)

    el = [2,2,2,2,2]
    m = [0, 1, -1, 2, -2]
    k = np.array([np.sqrt(3/2)*N_A[2,2], (1/2)*(N_A[2,0]+1j*N_A[2,1]), (1/2)*(N_A[2,0]-1j*N_A[2,1]), (1/2)*(N_A[0,0]-N_A[1,1]+2*1j*N_A[0,1]), (1/2)*(N_A[0,0]-N_A[1,1]-2*1j*N_A[0,1])])
    k /= max(np.abs(np.real(k)))

    fig = plt.figure(figsize=plt.figaspect(1.))
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(angles[0], angles[1])  #0,90  0,180, -90,90
    plot_Y(ax, el, m, k)
    if name is not None:
        plt.savefig(name+'.png')
    else:
        plt.show()

    return 0

def calc_segm(E_val, x=1, spanx=0.5):
    """ Costruisce il segmento (x-spanx/x, E_val), (x+spanx/2, E_val) """

    x1 = x - spanx/2
    x2 = x + spanx/2

    segment = (x1, x2), (E_val, E_val) #tupla di tuple
    return segment

def plot_segm(ax, segment, lw=0.5, c='k', ls='-'):
    """ Plotta il segmentino, ritorna l'oggetto 'linea' """
    line, = ax.plot(segment[0], segment[1],
            lw=lw,      # linewidth
            c=c,        # color
            ls=ls,      # linestyle
            )
    return line

def text_on_segm(ax, segment, text, text_kwargs={}):
    """ Scrive text sul lato sinistro del segment che gli passi """
    x_t = segment[0][0]
    y_t = segment[1][0]
    text += ' '     # Per mettere text non appiccicato alla barra
    text_plot = ax.text(x_t+0.11, y_t+80, text,
            horizontalalignment='right',
            verticalalignment='center',
            **text_kwargs)
    return text_plot

def plot_lines(ax, S1, S2, e1, e2, l, c='k'):

    for i,e in enumerate(e2):
        for ii, item in enumerate(l[int(i+1)].items()):
            key, value = item
            if str(e1) == key:
                line, = ax.plot([S1[0][1], S2[ii][0][0]], [e1, e], lw=value/100, c=c)
            else:
                line=None
    return line

def level_fig_tot(E_matrix, theories, proj_LS_dict, proj_prev_dict):

    COLORS = COLORS_list()
    levels = [str(int(w)) for w in range(E_matrix.shape[1])]  #number of levels
    deg = {theory:np.ones(len(set(E_matrix[k,:])), dtype='int32') for k,theory in enumerate(theories)}
    segm = {}
    spanx = 0.5
    for k, theory in enumerate(theories):
        x = k + 1   # La scala delle x deve partire da 1 altrimenti fa schifo
        segm[theory] = [calc_segm(E, x, spanx) for E in np.sort(list(set(E_matrix[k,:])))]    # Costruisco i segmentini e li salvo nel dizionario di prima
                                                                                              # solo uno per valore di energia
        count = 0
        for i in range(len(deg[theory])):
            prec = E_matrix[k,count]
            for j in range(int(count),len(E_matrix[k,:])):
                #print(E_matrix[k,j],prec)
                if E_matrix[k,j]==prec:
                    deg[theory][i] += 1
                    #print(i,deg[theory][i])
                else:
                    break
            deg[theory][i] -= 1
            count += deg[theory][i]

    fig = plt.figure()
    fig.set_size_inches(11,8)  #larghezza altezza
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.10, top=0.95)
    ax = fig.add_subplot()

    for k, theory in enumerate(theories):
        [plot_segm(ax, S,
            lw=2.0,
            c=C,
            ls='-') for S, C in zip(segm[theory], COLORS[:len(segm[theory])])]

        if k==0:
            keys_ee = []
            for kk in range(len(deg[theory])):
                [keys_ee.append(key) for key in proj_LS_dict[theory][sum(deg[theory][:kk+1])].keys()]
            [text_on_segm(ax, S,
                keys_ee[kk]+' ({})'.format(deg[theory][kk]),
                text_kwargs={'fontsize':12, 'color':COLORS[kk]})
                for kk, S in enumerate(segm[theory])]

        else:
            [text_on_segm(ax, S,
                '({})'.format(deg[theory][kk]),
                text_kwargs={'fontsize':12, 'color':COLORS[kk]})
                for kk, S in enumerate(segm[theory]) if deg[theory][kk] > 1]

        if k>0:
            [plot_lines(ax, S1 = segm[theories[k-1]][kk], S2 = segm[theory],
                e1 = np.sort(list(set(E_matrix[k-1,:])))[kk], e2 = E_matrix[k,:],
                l = proj_prev_dict[theory], c = C)
                for kk,C in enumerate(COLORS[:len(segm[theories[k-1]])])]

    ax.tick_params(labelsize=12)
    ax.ticklabel_format(axis='y', style='scientific', useMathText=True)#, scilimits=(0,0))
    ax.set_xticks(np.arange(len(theories))+1)   # Scala finta coi numeri (che parte da 1)
    ax.set_xticklabels(theories, fontsize=12)                # Rimpiazzo i numeri con la lista delle teorie
    ax.set_ylabel('Energy (cm$^{-1})$', fontsize=12)

    plt.show()

def E_vs_field(field, Epert, Etrue, name=None):

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    fig.set_size_inches(12,8)
    plt.subplots_adjust(left=0.1, bottom=0.1, top=0.9, right=0.875)

    ax.set_xlim(1,11)
    ax.set_xticks(np.arange(int(min(field)),int(max(field))+2,2))
    ax.set_xlabel('B$_0$ (T)')
    ax.set_ylabel('Energy (cm$^{-1}$)')
    ax.plot(0,0,'^', label='2° order', c='grey', transform=fig.transFigure)
    ax.plot(0,0,'.', label='exact', c='grey', transform=fig.transFigure)

    for i in range(Etrue.shape[1]):
        dots, = ax.plot(field, Etrue[:,i], '.')
        ax.plot(field, Etrue[:,i], '-', c=dots.get_color(), label=str(i+1))
        ax.plot(field, Epert[:,i], '--', c=dots.get_color())
        ax.plot(field, Epert[:,i], '^', c=dots.get_color())

    ax.legend(loc='upper left', bbox_to_anchor=(0.89,0.9), bbox_transform=fig.transFigure)
    if name is None:
        plt.show()
    else:
        plt.savefig(name, dpi=300)

def plot_charge_density(A2, A4, A6):
    #Reproduce the plots in fig 8 of Jeffrey D. Rinehart and Jeffrey R. Long Chem.Sci., 2011, 2, 2078-2085

    theta = np.linspace(0, np.pi, 50)
    phi = np.linspace(0, 2*np.pi, 50)
    theta, phi = np.meshgrid(theta, phi)
    r = np.zeros_like(theta)
    for i in range(theta.shape[0]):
        for j in range(theta.shape[1]):
            r[i,j] = Freeion_charge_dist(theta[i,j], phi[i,j], A2, A4, A6)
    xyz = np.array([r*np.sin(theta) * np.sin(phi),
                    r*np.sin(theta) * np.cos(phi),
                    r*np.cos(theta)])
    X, Y, Z = xyz

    fig = plt.figure(figsize=plt.figaspect(1.))
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    asse = 1
    ax.set_xlim(-asse, asse)
    ax.set_ylim(-asse, asse)
    ax.set_zlim(-asse, asse)
    ax.plot_surface(X, Y, Z, edgecolor='royalblue', lw=0.5, rstride=2, cstride=2,
                alpha=0.3)
    plt.show()

def plot_charge_density_data(A2, A4, A6, data):
    #Reproduce the plots in fig 8 of Jeffrey D. Rinehart and Jeffrey R. Long Chem.Sci., 2011, 2, 2078-2085

    theta = np.linspace(0, np.pi, 50)
    phi = np.linspace(0, 2*np.pi, 50)
    theta, phi = np.meshgrid(theta, phi)
    r = np.zeros_like(theta)
    for i in range(theta.shape[0]):
        for j in range(theta.shape[1]):
            r[i,j] = Freeion_charge_dist(theta[i,j], phi[i,j], A2, A4, A6)
    xyz = np.array([r*np.sin(theta) * np.sin(phi),
                    r*np.sin(theta) * np.cos(phi),
                    r*np.cos(theta)])
    X, Y, Z = xyz

    fig = plt.figure(figsize=plt.figaspect(1.))
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    asse = 2
    ax.set_xlim(-asse, asse)
    ax.set_ylim(-asse, asse)
    ax.set_zlim(-asse, asse)
    ax.plot_surface(X, Y, Z, edgecolor='royalblue', lw=0.5, rstride=2, cstride=2,
                alpha=0.3)
    for i in range(data.shape[0]):
        vector = data[i,1:-1]
        ax.quiver(0.,0.,0.,vector[0],vector[1],vector[2],color='b')
        ax.text(vector[0],vector[1],vector[2],data[i,-1])
    plt.show()

#########CRYSTAL FIELD###############

def w_inp_dat(coord_tot, charges_tot, num=1, directory=None):
    """
    Creates a file named 'simpre.dat', which contains coordinates and charges, to be used as input for a SIMPRE calculation.
    As such, only the negative charges are considered.
    ----------
    Parameters:
    - coord_tot : 2darray
        cartesian coordinates of the ligands
    - charges_tot : sequence
        ligand charges (with their sign)
    - num : int
        SIMPRE index for a configuration
    """

    coord = []
    charges = []
    for i in range(coord_tot.shape[0]):
        if charges_tot[i]<=0:
            coord.append(coord_tot[i,:])
            charges.append(np.abs(charges_tot[i]))
    coord = np.array(coord)
    if coord.size==0:
        print('WARNING: Only positive charged ligands found. The "simpre.dat" file can not be generated.')
        return 0

    with open(directory+'simpre.dat', 'w') as f:
        f.write('1=Ce3+, 2=Pr3+, 3=Nd3+, 4=Pm3+, 5=Sm3+, 6=Tb3+, 7=Dy3+, 8=Ho3+, 9=Er3+, 10=Tm3+, 11=Yb3+, 12=user\n')
        f.write('\n')
        f.write(' '+f'{int(num):2.0f}'+'    !ion code (from 1 to 12, see above)\n')
        f.write(' '+f'{coord.shape[0]:2.0f}'+'    !number of effective charges\n')
        for i in range(len(coord[:,0])):
            f.write('{:-3}'.format(i+1))
            for j in range(3):
                f.write('{:13.7f}'.format(coord[i,j]))
            f.write('{:13.7f}'.format(charges[i])+'\n')
        f.write('\n')
        f.write('\n')
        f.close()

def read_data(filename, sph_flag = False):

    file = open(filename).readlines()
    coord_m = []
    charges = []
    labels = []
    rdisp = []
    for i,line in enumerate(file):
        splitline = line.split('\t')
        labels.append(splitline[0])
        charges.append(eval(splitline[1]))
        rdisp.append(float(splitline[2]))
        coord_m.append([float(splitline[j]) for j in range(3,len(splitline))])
    coord_m = np.array(coord_m)
    rdisp = np.array(rdisp)

    if not sph_flag:
        sph_coord = from_car_to_sph(coord_m)
    else:
        sph_coord = np.copy(coord_m)

    sph_coord[:,0] -= rdisp

    coord_m = from_sph_to_car(sph_coord)
    charges = np.array(charges)

    data = np.zeros((coord_m.shape[0],5), dtype='object')
    data[:,1:-1] = coord_m
    data[:,-1] = np.array(charges)
    data[:,0] = labels

    return data

def from_Vint_to_Bkq(dic_V, conf):
    #conversion V matrix element to the Ckq coefficents of the ligand field expanded as spherical harmonics
    #see Gerloch & McMeeking (1975) (Table 2)
    #or OctoYot f_e_LF.f90 subroutine AOMmatrixD()
    #prende in input V reale scritto nell'ordine di OctoYot: z2, yz, xz, xy, x2-y2
    #per q>0 coorisponde a BKQR, per q<0 corrisponde a BKQIM

    if conf[0]=='d':
        l = 2
        dic_ckq = {
                  '0':{'0':2./5.*np.sqrt(np.pi)*(dic_V['11']+dic_V['22']+dic_V['33']+dic_V['44']+dic_V['55'])},
                  '2':{'0':np.sqrt(np.pi/5.)*(2.*dic_V['11'] + dic_V['22'] + dic_V['33'] - 2.*dic_V['44'] - 2.*dic_V['55']),
                       '1':-np.sqrt(4.*np.pi/5.)*( np.sqrt(3)/np.sqrt(2)*(dic_V['42'] + dic_V['53']) + dic_V['31']/np.sqrt(2)),
                       '-1':np.sqrt(4.*np.pi/5.)*( -np.sqrt(3)/np.sqrt(2)*(dic_V['52'] - dic_V['43']) + dic_V['21']/np.sqrt(2)),
                       '2':-np.sqrt(4.*np.pi/5.)*(np.sqrt(2)*dic_V['51'] + np.sqrt(3)/(2.*np.sqrt(2))*(dic_V['22']-dic_V['33'])),
                       '-2':-np.sqrt(4.*np.pi/5.)*(-np.sqrt(2)*dic_V['41'] + np.sqrt(3)/np.sqrt(2)*dic_V['32'])},
                   '4':{'0':np.sqrt(np.pi)/5.*(6.*dic_V['11']-4.*dic_V['22']-4.*dic_V['33']+dic_V['44']+dic_V['55']),
                        '1':2*np.sqrt(2.*np.pi/5.)*(-np.sqrt(3)/np.sqrt(2)*dic_V['31']+1./(2.*np.sqrt(2))*(dic_V['42'] + dic_V['53'])),
                        '-1':2*np.sqrt(2.*np.pi/5.)*(np.sqrt(3)/np.sqrt(2)*dic_V['21']+1./(2.*np.sqrt(2))*(dic_V['52'] - dic_V['43'])),
                        '2':2.*np.sqrt(2.*np.pi/5.)*(np.sqrt(3)/2.*dic_V['51']+(dic_V['33']-dic_V['22'])/2.),
                        '-2':2.*np.sqrt(2.*np.pi/5.)*(-np.sqrt(3)/2.*dic_V['41']-dic_V['32']),
                        '3':np.sqrt(7*np.pi/5.)*(dic_V['42']-dic_V['53']),
                        '-3':np.sqrt(7*np.pi/5.)*(dic_V['43']+dic_V['52']),
                        '4':np.sqrt(7.*np.pi/10.)*(dic_V['55']-dic_V['44']),
                        '-4':-np.sqrt(7.*np.pi/5.)*np.sqrt(2)*dic_V['54']}
                    }
        
    #conversion V matrix element to the Ckq coefficents of the ligand field expanded as spherical harmonics
    #given by Urland, Chem.Phys.14, 393,(1976). Table 3
    #or OctoYot f_e_LF.f90 subroutine AOMmatrixF()
    #prende in input V reale scritto nell'ordine di OctoYot: |sigma>, |piS>, |piC>, |deltaS>, |deltaC>, |phiS>, |phiC>
    #                                             (in ORCA):     0      -1     1       -2        2        -3      3
    #(For a definition of these orbitals: see Harnung & Schaffer, Struct&Bond,12,201,(1972))
    #per q>0 coorisponde a BKQRE, per q<0 corrisponde a BKQIM

    elif conf[0]=='f':
        l = 3
        dic_ckq = {
                  '0':{'0':(2./7.)*np.sqrt(np.pi)*(dic_V['11']+dic_V['22']+dic_V['33']+dic_V['44']+dic_V['55']+dic_V['66']+dic_V['77'])},
                  '2':{'0':(2./7.)*np.sqrt(5*np.pi)*dic_V['11'] + (3./14.)*np.sqrt(5*np.pi)*(dic_V['22']+dic_V['33']) - (5./14.)*np.sqrt(5*np.pi)*(dic_V['66']+dic_V['77']),
                       '1': - (1./7.)*np.sqrt(5*np.pi)*dic_V['31'] + (5./14.)*np.sqrt(3*np.pi)*(-dic_V['42']-dic_V['53']) + (5./14.)*np.sqrt(5*np.pi)*(-dic_V['64']-dic_V['75']),
                       '-1': (1./7.)*np.sqrt(5*np.pi)*dic_V['21'] + (5./14.)*np.sqrt(3*np.pi)*(dic_V['43']-dic_V['52']) + (5./14.)*np.sqrt(5*np.pi)*(dic_V['65']-dic_V['74']),
                       '2': (1./7.)*np.sqrt(30*np.pi)*(-dic_V['22']/2 + dic_V['33']/2) + (5./7.)*np.sqrt(2*np.pi)*(-dic_V['51']) + (5./7.)*np.sqrt(np.pi/2)*(-dic_V['62']-dic_V['73']),
                       '-2': (1./7.)*np.sqrt(30*np.pi)*(-dic_V['32']) + (5./7.)*np.sqrt(2*np.pi)*(dic_V['41']) + (5./7.)*np.sqrt(np.pi/2)*(dic_V['63']-dic_V['72'])},
                   '4':{'0': -np.sqrt(np.pi)*(dic_V['44']+dic_V['55']) + (1./7.)*np.sqrt(np.pi)*(6*dic_V['11'] + dic_V['22'] + dic_V['33'] + 3*dic_V['66'] + 3*dic_V['77']),
                        '1': (1./7.)*np.sqrt(30*np.pi)*(-dic_V['31']) + (4./7.)*np.sqrt(2*np.pi)*(-dic_V['42']-dic_V['53']) + (1./7.)*np.sqrt(30*np.pi)*(dic_V['64']+dic_V['75']),
                        '-1': (1./7.)*np.sqrt(30*np.pi)*(dic_V['21']) + (4./7.)*np.sqrt(2*np.pi)*(dic_V['43']-dic_V['52']) + (1./7.)*np.sqrt(30*np.pi)*(-dic_V['65']+dic_V['74']),
                        '2': (2./7.)*np.sqrt(10*np.pi)*(-dic_V['22']/2 + dic_V['33']/2) + (1./7.)*np.sqrt(6*np.pi)*(-dic_V['51']) + (3./7.)*np.sqrt(6*np.pi)*(dic_V['62']+dic_V['73']),
                        '-2': (2./7.)*np.sqrt(10*np.pi)*(-dic_V['32']) + (1./7.)*np.sqrt(6*np.pi)*(dic_V['41'] + 3*(-dic_V['63']+dic_V['72'])),
                        '3': np.sqrt((2./7.)*np.pi)*(dic_V['42']-dic_V['53']) + 3*np.sqrt((2./7.)*np.pi)*dic_V['71'],
                        '-3': np.sqrt((2./7.)*np.pi)*(dic_V['43']+dic_V['52']) - 3*np.sqrt((2./7.)*np.pi)*dic_V['61'],
                        '4': np.sqrt((10./7.)*np.pi)*(-dic_V['44']/2 + dic_V['55']/2) + np.sqrt((6./7.)*np.pi)*(dic_V['62']-dic_V['73']),
                        '-4': np.sqrt((10./7.)*np.pi)*(-dic_V['54']) + np.sqrt((6./7.)*np.pi)*(dic_V['63']+dic_V['72'])},
                   '6':{'0':(1./7.)*np.sqrt(13*np.pi)*(2*dic_V['11'] - (3./2.)*(dic_V['22']+dic_V['33']) + (3./5.)*(dic_V['44']+dic_V['55']) - (1./10.)*(dic_V['66']+dic_V['77'])),
                        '1':np.sqrt((13./7.)*np.pi)*(-dic_V['31'] + (1./2.)*np.sqrt(3./5.)*(dic_V['42']+dic_V['53']) + (1./10.)*(-dic_V['64']-dic_V['75'])),
                        '-1':np.sqrt((13./7.)*np.pi)*(dic_V['21'] + (1./2.)*np.sqrt(3./5.)*(-dic_V['43']+dic_V['52']) + (1./10.)*(dic_V['65']-dic_V['74'])),
                        '2':np.sqrt((13./7.)*np.pi)*(-(1./2.)*np.sqrt(3./5.)*(dic_V['22']-dic_V['33']) + (4./5.)*dic_V['51'] - (1./5.)*(dic_V['62']+dic_V['73'])),
                        '-2':np.sqrt((13./7.)*np.pi)*(-np.sqrt(3./5.)*dic_V['32'] - (4./5.)*dic_V['41'] + (1./5.)*(dic_V['63']-dic_V['72'])),
                        '3': (3./5.)*np.sqrt((39./14.)*np.pi)*(dic_V['42']-dic_V['53']) + (1./5.)*(np.sqrt((78./7.)*np.pi))*(-dic_V['71']),
                        '-3': (3./5.)*np.sqrt((39./14.)*np.pi)*(dic_V['43']+dic_V['52']) + (1./5.)*(np.sqrt((78./7.)*np.pi))*(dic_V['61']),
                        '4': (3./5.)*np.sqrt((26./7.)*np.pi)*(-dic_V['44']/2 + dic_V['55']/2) + np.sqrt((39./70.)*np.pi)*(-dic_V['62']+dic_V['73']),
                        '-4': (3./5.)*np.sqrt((26./7.)*np.pi)*(-dic_V['54']) + np.sqrt((39./70.)*np.pi)*(-dic_V['63']-dic_V['72']),
                        '5': (1./5.)*np.sqrt((429./14.)*np.pi)*(dic_V['64']-dic_V['75']),
                        '-5': (1./5.)*np.sqrt((429./14.)*np.pi)*(dic_V['65']+dic_V['74']),
                        '6':(1./5.)*(np.sqrt((429./7.)*np.pi))*(-dic_V['66']/2 + dic_V['77']/2),
                        '-6':(1./5.)*(np.sqrt((429./7.)*np.pi))*(-dic_V['76'])}
                    }

    else:
        print('ERROR: in from_Vint_to_Bkq')
        exit()

    #Adjust phases so the Bkq and Bkq' correspond to those of Goeller-Walrand
    for k in range(0,2*l+1,2):
        for q in range(0,k+1):
            dic_ckq[str(k)][str(q)] *= np.sqrt((2*k+1)/(4*np.pi))*(-1)**q
            if q!=0:
                dic_ckq[str(k)][str(-q)] *= -np.sqrt((2*k+1)/(4*np.pi))*(-1)**q

    return dic_ckq

def from_AOM_to_Vint(dic_AOM, conf):
    #conversion from AOM dict [es, eps, epc, theta, phi, chi] to <|V|> = sum(lig) sum(modes) D*D*e

    if conf[0]=='d':
        l=2
    else:
        l=3

    ligand_label = [key for key in dic_AOM.keys()]
    matrix_elements = [value for value in dic_AOM.values()]
    matrix_elements = np.array(matrix_elements)
    ee = matrix_elements[:,:3]    # e_sigma, e_pi_s, e_pi_c
    theta = matrix_elements[:,3]
    phi = matrix_elements[:,4]
    if matrix_elements.shape[1]==5:
        chi = np.zeros(matrix_elements.shape[0])
    else:
        chi = matrix_elements[:,-1]

    D = np.zeros((2*l+1,3))
    V = np.zeros((2*l+1,2*l+1))
    for i in range(matrix_elements.shape[0]): #per ogni legante
        theta[i] *= np.pi/180
        phi[i] *= np.pi/180
        chi[i] *= np.pi/180

        a1 = np.cos(phi[i])*np.cos(theta[i])*np.cos(chi[i])-np.sin(phi[i])*np.sin(chi[i])
        a2 = np.sin(phi[i])*np.cos(theta[i])*np.cos(chi[i])+np.cos(phi[i])*np.sin(chi[i])
        a3 = -np.sin(theta[i])*np.cos(chi[i])
        b1 = -np.cos(phi[i])*np.cos(theta[i])*np.sin(chi[i])-np.sin(phi[i])*np.cos(chi[i])
        b2 = -np.sin(phi[i])*np.cos(theta[i])*np.sin(chi[i])+np.cos(phi[i])*np.cos(chi[i])
        b3 = np.sin(theta[i])*np.sin(chi[i])
        g1 = np.cos(phi[i])*np.sin(theta[i])
        g2 = np.sin(phi[i])*np.sin(theta[i])
        g3 = np.cos(theta[i])

        if l==3:
            #Tab 1 Urland 1976 Chem. Phys. 14, 393-401
            D[0,0] = 0.5*g3*(5.*g3**2-3)
            D[0,1] = np.sqrt(3./2.)*b3*0.5*(5*g3**2-1)
            D[0,2] = np.sqrt(3./2.)*a3*0.5*(5*g3**2-1)
            D[1,0] = (1./4.)*np.sqrt(6)*g2*(5*g3**2-1)
            D[1,1] = (1./4.)*b2*(5*g3**2-1)+(5./2.)*g2*g3*b3
            D[1,2] = (1./4.)*a2*(5*g3**2-1)+(5./2.)*g2*g3*a3
            D[2,0] = (1./4.)*np.sqrt(6)*g1*(5*g3**2-1)
            D[2,1] = (1./4.)*b1*(5*g3**2-1)+(5./2.)*g1*g3*b3
            D[2,2] = (1./4.)*a1*(5*g3**2-1)+(5./2.)*g1*g3*a3
            D[3,0] = np.sqrt(15)*g1*g2*g3
            D[3,1] = np.sqrt(5./2.)*(b1*g2*g3+b2*g1*g3+b3*g1*g2)
            D[3,2] = np.sqrt(5./2.)*(a1*g2*g3+a2*g1*g3+a3*g1*g2)
            D[4,0] = 0.5*np.sqrt(15)*g3*(g1**2-g2**2)
            D[4,1] = np.sqrt(5./8.)*(g1**2*b3-g2**2*b3+2*b1*g1*g3-2*b2*g2*g3)
            D[4,2] = np.sqrt(5./8.)*(g1**2*a3-g2**2*a3+2*a1*g1*g3-2*a2*g2*g3)
            D[5,0] = (1./4.)*np.sqrt(10)*g2*(3*g1**2-g2**2)
            D[5,1] = (1./4.)*np.sqrt(15)*b2*(g1**2-g2**2)+0.5*np.sqrt(15)*g1*g2*b1
            D[5,2] = (1./4.)*np.sqrt(15)*a2*(g1**2-g2**2)+0.5*np.sqrt(15)*g1*g2*a1
            D[6,0] = (1./4.)*np.sqrt(10)*g1*(g1**2-3*g2**2)
            D[6,1] = (1./4.)*np.sqrt(15)*b1*(g1**2-g2**2)-0.5*np.sqrt(15)*g1*g2*b2
            D[6,2] = (1./4.)*np.sqrt(15)*a1*(g1**2-g2**2)-0.5*np.sqrt(15)*g1*g2*a2
        elif l==2:
            #eq 9 Gerloch & McMeeking 1975
            D[0,0] = 0.5*(3*g3**2-1)
            D[0,1] = np.sqrt(3)*a3*g3
            D[0,2] = np.sqrt(3)*b3*g3
            D[1,0] = np.sqrt(3)*g1*g3
            D[1,1] = a1*g3+a3*g1
            D[1,2] = b1*g3+b3*g1
            D[2,0] = np.sqrt(3)*g2*g3
            D[2,1] = a2*g3+a3*g2
            D[2,2] = b2*g3+b3*g2
            D[3,0] = np.sqrt(3)*g1*g2
            D[3,1] = a1*g2+a2*g1
            D[3,2] = b1*g2+b2*g1
            D[4,0] = np.sqrt(3)*0.5*(g1**2-g2**2)
            D[4,1] = a1*g1-a2*g2
            D[4,2] = b1*g1-b2*g2
        else:
            print('ERROR: l != 2 and l != 3 in from_AOM_to_Vint')

        #M = A*e  (eq 10 Gerloch & McMeeking 1975)
        for ii in range(0,2*l+1):
            for jj in range(0,2*l+1):
                V[ii,jj] += ee[i,0]*D[ii,0]*D[jj,0] + ee[i,1]*D[ii,2]*D[jj,2] + ee[i,2]*D[ii,1]*D[jj,1]  #in input li fornisco come x-y ma D è costruita per y-x

    dic_V = {}
    for i in range(1,2*l+1+1):
        for j in range(1,i+1):
            dic_V['{}{}'.format(i,j)] = V[i-1,j-1]

    return dic_V

def from_Aqkrk_to_Bkq(Aqkrk, revers=False):  
    # conversion from Akq<rk> of stevens to Bkq of wybourne (o Lkq secondo: https://www2.cpfs.mpg.de/~rotter/homepage_mcphase/manual/node130.html)

    dic_bkq = {}
    for key1 in Aqkrk.keys():
        dic_bkq[key1] = {}
        for key2, value in Aqkrk[key1].items():
            if revers==False:
                dic_bkq[key1].update({key2: value*conv_Aqkrk_bkq(int(key1),np.abs(int(key2)))})
            else:
                dic_bkq[key1].update({key2: value/conv_Aqkrk_bkq(int(key1),np.abs(int(key2)))})

    return dic_bkq

def sph_harm(l,m,theta,phi):
    yL = scipy.special.lpmn(m, l, np.cos(theta))[0][-1][-1]
    y = np.sqrt(fact(l-m)/fact(l+m))*yL*np.exp(1j*m*phi)
    return y

def calc_Bqk(data, conf, sph_flag = False, sth_param = False, bin=1e-9):
    # calc Stevens coefficients, B^q_k, from data in the hard point charge model

    if conf[0]=='d':
        l=2
    else:
        l=3

    dic_Aqk_rk = calc_Aqkrk(data, conf, sph_flag, sth_param, bin)
    dic_Bqk = {}
    for k in range(2,2*l+1,2):
        dic_Bqk[f'{k}'] = {}
        for q in range(-k,k+1):
            dic_Bqk[f'{k}'].update({f'{q}': dic_Aqk_rk[f'{k}'][f'{q}']*Stev_coeff(str(k), conf)})

    return dic_Bqk
            

def calc_Aqkrk(data, conf, sph_flag = False, sth_param = False, bin=1e-9):
    #in data is (N. ligands x 5)
    #[labels, x, y, z, charge] if sph_falg==False
    #the calculation is performed in the hard point charge model
    #equation 9a, 9b, 9c from simpre2

    import scipy.special

    au_conv = [scipy.constants.physical_constants['hartree-inverse meter relationship'][0]*1e-2, 1.889725989] #convertion from atomic unit

    if conf[0]=='d':
        l=2
    else:
        l=3

    if sph_flag==False:
        coord_car = data[:,1:-1]
        coord_sph = from_car_to_sph(coord_car)
    else:
        coord_sph = data[:,1:-1]
        coord_sph[:,1:] *= np.pi/180

    Aqkrk = {}
    for k in range(2,2*l+1,2):
        Aqkrk[f'{k}'] = {}
        r_val = r_expect(str(k), conf)
        for q in range(-k,k+1):
            pref = plm(k,np.abs(q))*(4*np.pi/(2*k+1))**(0.5)
            somma = 0
            for i in range(data.shape[0]):
                r = coord_sph[i,0]*au_conv[1]
                sphharmp = sph_harm(k,np.abs(q),coord_sph[i,1], coord_sph[i,2])
                sphharmm = sph_harm(k,-np.abs(q),coord_sph[i,1], coord_sph[i,2])
                if q==0:
                    somma += pref*(data[i,-1]*au_conv[0]/r**(k+1))*sphharmp.real
                elif q>0:
                    somma += (-1)**q*pref*(data[i,-1]*au_conv[0]/r**(k+1))*(1/np.sqrt(2))*(sphharmm + (-1)**q*sphharmp).real
                elif q<0:
                    somma += -(-1)**q*pref*(data[i,-1]*au_conv[0]/r**(k+1))*(1/np.sqrt(2))*(sphharmm - (-1)**np.abs(q)*sphharmp).imag

            if sth_param == True:
                value = (1-sigma_k(str(k), conf))*somma*r_val
            else:
                value = somma*r_val
            if np.abs(value)<=bin:
                Aqkrk[f'{k}'].update({f'{q}': 0})
            else:
                Aqkrk[f'{k}'].update({f'{q}': value})

    return Aqkrk

def calc_Bkq(data, conf, sph_flag = False, sth_param = False, bin=1e-9):  
    #eq 11a-11b-11c-12 from Software package SIMPRE - Revisited (M. Karbowiak and C. Rudowicz)

    import scipy.special

    au_conv = [scipy.constants.physical_constants['hartree-inverse meter relationship'][0]*1e-2, 1.889725989] #convertion from atomic unit

    if conf[0]=='d':
        l=2
    else:
        l=3

    if sph_flag==False:
        coord_car = data[:,1:-1]
        coord_sph = from_car_to_sph(coord_car)
    else:
        coord_sph = data[:,1:-1]
        coord_sph[:,1:] *= np.pi/180

    Aqkrk = {}
    for k in range(2,2*l+1,2):
        Aqkrk[f'{k}'] = {}
        prefac = np.sqrt(4*np.pi/(2*k+1))
        r_val = r_expect(str(k), conf)
        for q in range(0,k+1):
            somma = 0
            for i in range(data.shape[0]):
                r = coord_sph[i,0]*au_conv[1]
                sphharm = scipy.special.sph_harm(q, k, coord_sph[i,2],coord_sph[i,1])
                if q==0:
                    somma += prefac*(data[i,-1]*au_conv[0]/r**(k+1))*sphharm.real
                else:
                    somma += prefac*(data[i,-1]*au_conv[0]/r**(k+1))*sphharm
            if sth_param == True:
                value = (1-sigma_k(str(k), conf))*somma*r_val
            else:
                value = somma*r_val
            if np.abs(value)<=bin:
                Aqkrk[f'{k}'].update({f'{q}': 0})
            else:
                if q!=0:
                    Aqkrk[f'{k}'].update({f'{q}': value.real})
                    Aqkrk[f'{k}'].update({f'{-q}': value.imag})
                else:
                    Aqkrk[f'{k}'].update({f'{q}': value})

    return Aqkrk


def from_Vreal_to_V(dic_V):
    #ordine -2, -1, 0, 1, 2
    #prende in input l'ordine di OctoYot: z2, yz, xz, xy, x2-y2

    dic_Vnew = {
                '11':0.5*(dic_V['55']+dic_V['44']),
                '21':0.5*(dic_V['53']-1j*dic_V['43']+1j*dic_V['52']+dic_V['42']), '22':0.5*(dic_V['33']+dic_V['22']),
                '31':(1/np.sqrt(2))*(dic_V['51']-1j*dic_V['41']), '32':(1/np.sqrt(2))*(dic_V['31']-1j*dic_V['21']), '33':dic_V['11'],
                '41':0.5*(-dic_V['53']+1j*dic_V['43']+1j*dic_V['52']+dic_V['42']), '42':0.5*(-dic_V['33']+2*1j*dic_V['32']+dic_V['22']), '43':(1/np.sqrt(2))*(-dic_V['31']+1j*dic_V['21']), '44':0.5*(dic_V['33']+dic_V['22']),
                '51':0.5*(dic_V['55']-2*1j*dic_V['54']-dic_V['44']), '52':0.5*(dic_V['53']-1j*dic_V['52']-1j*dic_V['43']-dic_V['42']), '53':(1/np.sqrt(2))*(dic_V['51']-1j*dic_V['41']), '54':0.5*(-dic_V['53']-1j*dic_V['52']+1j*dic_V['43']-dic_V['42']), '55':0.5*(dic_V['55']+dic_V['44'])
                }

    return dic_Vnew


def rota_LF(l, dic_Bkq, A=0, B=0, C=0):
    #qui sono implementate le small-d che servono per ruotare i Bkq
    #gli angoli vengono passati in radianti
    #convention of D-matrix: Z-Y-Z
    #per ogni k c'è una diversa matrice D

    dic_Bkq_new = {}

    for k in range(2,2*l+1,2):
        D = np.zeros((k*2+1,k*2+1), dtype='complex128')
        for ii,m1 in enumerate(range(-k,k+1)):
            for jj,m in enumerate(range(-k,k+1)):
                D[ii,jj] = Wigner_coeff.Wigner_Dmatrix(k, m1, m, A, B, C)

        dic_Bkq_new[str(k)] = {}

        Bkq_vec = []
        for q in range(-k,k+1):
            if q!=0:
                if q<0:
                    Bkq_vec.append(dic_Bkq[str(k)][str(np.abs(q))]+1j*dic_Bkq[str(k)][str(-np.abs(q))])
                else:
                    if q%2!=0:
                        Bkq_vec.append(-dic_Bkq[str(k)][str(np.abs(q))]+1j*dic_Bkq[str(k)][str(-np.abs(q))])
                    else:
                        Bkq_vec.append(dic_Bkq[str(k)][str(np.abs(q))]-1j*dic_Bkq[str(k)][str(-np.abs(q))])
            else:
                Bkq_vec.append(dic_Bkq[str(k)][str(np.abs(q))]+1j*0)


        Bkq_vec = np.array(Bkq_vec)
        Bkq_vec_new = D@Bkq_vec
        dic_Bkq_new[str(k)][str(0)] = Bkq_vec_new[k].real

        for i,q in enumerate(range(k,0,-1)):
            dic_Bkq_new[str(k)][str(q)] = Bkq_vec_new[i].real
            dic_Bkq_new[str(k)][str(-q)] = Bkq_vec_new[i].imag

    return dic_Bkq_new

#@cron
def rota_LF_quat(l, dic_Bkq, R, dict=None, coeff=None):
    #qui sono implementate le small-d che servono per ruotare i Bkq con quaternioni
    #per ogni k c'è una diversa matrice D

    dic_Bkq_new = {}

    for k in range(2,2*l+1,2):

        D = Wigner_coeff.Wigner_Dmatrix_quat_complete(k, R, dict = dict, coeff=coeff)

        dic_Bkq_new[str(k)] = {}

        Bkq_vec = []
        for q in range(-k,k+1):
            if q!=0:
                if q<0:
                    Bkq_vec.append(dic_Bkq[str(k)][str(np.abs(q))]+1j*dic_Bkq[str(k)][str(-np.abs(q))])
                else:
                    if q%2!=0:
                        Bkq_vec.append(-dic_Bkq[str(k)][str(np.abs(q))]+1j*dic_Bkq[str(k)][str(-np.abs(q))])
                    else:
                        Bkq_vec.append(dic_Bkq[str(k)][str(np.abs(q))]-1j*dic_Bkq[str(k)][str(-np.abs(q))])
            else:
                Bkq_vec.append(dic_Bkq[str(k)][str(np.abs(q))]+1j*0)


        Bkq_vec = np.array(Bkq_vec)
        Bkq_vec_new = D@Bkq_vec
        dic_Bkq_new[str(k)][str(0)] = Bkq_vec_new[k].real

        for i,q in enumerate(range(k,0,-1)):
            dic_Bkq_new[str(k)][str(q)] = Bkq_vec_new[i].real
            dic_Bkq_new[str(k)][str(-q)] = Bkq_vec_new[i].imag

    return dic_Bkq_new

def read_DWigner_quat():

    def coeff_op(k):
        matrix = np.ones((2*k+1,2*k+1))
        for i,ii in enumerate(range(k,-k-1,-1)):
            for j,jj in enumerate(range(k,-k-1,-1)):
                if j==0 and i==0:
                    pass
                elif j>0:
                    matrix[i,j] = matrix[i,j-1]*np.sqrt(k*(k+1)-(jj+1)*((jj+1)-1))
                    #print(j,jj,np.sqrt(k*(k+1)-(jj+1)*((jj+1)-1)))
            if i>0:
                matrix[i,:] = matrix[i-1,:]*np.sqrt(k*(k+1)-(ii+1)*((ii+1)-1))
        return matrix

    filename = ['tables/tab_wignerDquat.txt', 'tables/tab_wignerDquat_coeff_t.txt']
    list_dict = []
    for ii in range(len(filename)):
        file = open(filename[ii]).readlines()

        dict = {}
        last = 0
        for i,line in enumerate(file):
            if 'L=' in line:
                line = line.replace('\n','')
                splitline = line.split('=')
                dict[int(splitline[-1])]={}
                last = int(splitline[-1])
            elif 'L=' not in line and line!='\n':
                line = line.replace('\n','')
                splitline=line.split(' ')
                dict[last].update({splitline[0]+':'+splitline[1]:splitline[-1]})
        list_dict.append(dict)

    dic_matrix_coeff = {}
    for k in list_dict[0].keys():
        matrix_coeff = np.zeros((2*k+1,2*k+1), dtype='object')
        matrix_divide = coeff_op(k)
        for i,ii in enumerate(range(k,-1,-1)):
            #print(list(range(ii,-1,-1)))
            for j,jj in enumerate(range(ii,-1,-1)):
                j += i
                key = str(ii)+':'+str(jj)
                #print(key)
                #print(key, i,j, ii, jj, list_dict[1][k][key], matrix_divide[i,j])
                matrix_coeff[i,j] = np.abs(eval(list_dict[1][k][key]))/matrix_divide[i,j] #str(ii)+':'+str(jj)
                matrix_coeff[i,-j-1] = matrix_coeff[i,j]
                matrix_coeff[-i-1,j] = matrix_coeff[i,j]
                matrix_coeff[-i-1,-j-1] = matrix_coeff[i,j]
                matrix_coeff[j,i] = matrix_coeff[i,j]
                matrix_coeff[j,-i-1] = matrix_coeff[i,j]
                matrix_coeff[-j-1,i] = matrix_coeff[i,j]
                matrix_coeff[-j-1,-i-1] = matrix_coeff[i,j]
        #print(matrix_coeff)
        # # print(matrix_divide)
        #exit()
        dic_matrix_coeff[k]=matrix_coeff
    # dic_matrix_coeff = {}
    # for k in list_dict[0].keys():
    #     matrix_coeff = np.zeros((2*k+1,2*k+1))
    #     for key,value in list_dict[1][k].items():
    #         idx = [int(ii) for ii in key.split(':')]
    #         i = np.abs(idx[0]-k)
    #         j = np.abs(idx[1]-k)
    #         matrix_coeff[i,j] = np.abs(eval(value))
    #     #matrix_coeff /= coeff_op(k)
    #
    #     print(coeff_op(k))
    #     exit()
    #     dic_matrix_coeff[k]=matrix_coeff/coeff_op(k)
    # pprint(list_dict[0])
    # exit()

    return list_dict[0], dic_matrix_coeff


def R_zyz(alpha, beta, gamma):
    #matrice di rotazione in convenzione zyz (active rotation)

    A = alpha*np.pi/180
    B = beta*np.pi/180
    C = gamma*np.pi/180

    ca = np.cos(A)
    sa = np.sin(A)
    cb = np.cos(B)
    sb = np.sin(B)
    cg = np.cos(C)
    sg = np.sin(C)

    R = np.array([[ca*cb*cg-sa*sg,-cg*sa-ca*cb*sg,ca*sb],[ca*sg+cb*cg*sa,ca*cg-cb*sa*sg,sa*sb],[-cg*sb,sb*sg,cb]])

    return R

def from_matrix_to_result(matrix):
    w,v = diagonalisation(matrix)  #importante usare eigh
    result = np.vstack((w,v))
    result = np.copy(result[:, result[0,:].argsort()])
    return result


def order_two_level_dict(dict1):
    dict_or = {}
    key1 = np.sort(np.array([eval(i) for i in dict1.keys()]))
    key2 = []
    for key in key1:
        key2.append(np.sort(np.array([eval(i) for i in dict1[str(key)].keys()])))
    for i,keyi in enumerate(key1):
        dict_or[str(keyi)] = {}
        for j,keyj in enumerate(key2[i]):
            try:
                dict_or[str(keyi)].update({str(keyj):dict1[str(keyi)][str(keyj)]})
            except:
                dict_or[str(keyi)] = {}
                dict_or[str(keyi)].update({str(keyj):dict1[str(keyi)][str(keyj)]})
    return dict_or


def Freeion_charge_dist(theta, phi, A2, A4, A6, r=1, bin=1e-10):
    #Graphical representation and Discussion of the Charge Density
    #from Ch. III p 292 of Sievers "Asphericity of 4f-Shells in their Hund's rule ground states" (1981)
    #in scipy.special.sph_harm the angles are defined in the opposite way

    c2 = A2/np.sqrt(4*np.pi/(2*2+1))
    c4 = A4/np.sqrt(4*np.pi/(2*4+1))
    c6 = A6/np.sqrt(4*np.pi/(2*6+1))

    val = 3/(4*np.pi) + c2*scipy.special.sph_harm(0,2,phi,theta).real + c4*scipy.special.sph_harm(0,4,phi,theta).real + c6*scipy.special.sph_harm(0,6,phi,theta).real
    if np.abs(val)<bin:
        val=0
    return (val)**(1/3)

def coeff_multipole_moments(conf, J, M, L=0, S=0):
    #based on calculation of charge density based on the Wigner-Eckart Theorem
    #from Ch. II p 290 of Sievers "Asphericity of 4f-Shells in their Hund's rule ground states" (1981)
    #Ak = sqrt(4*pi/(2*k+1))ck

    coeff = []
    for k in range(2,6+1,2):
        if int(conf[1:])>7:
            pref = (-1)**(J-M)*7/np.sqrt(4*np.pi)*Wigner_coeff.threej_symbol([[J,k,J],[-M,0,M]])/Wigner_coeff.threej_symbol([[J,k,J],[-J,0,J]])
            pref2 = np.sqrt(2*k+1)*Wigner_coeff.threej_symbol([[k,3,3],[0,0,0]])
            if k==0:  #not active, just for completness
                delta = 1
            else:
                delta = 0
            somma = 0
            for i in range(1,int(conf[1:])-7+1):
                somma += (-1)**i*Wigner_coeff.threej_symbol([[k,3,3],[0,4-i,i-4]])
            value = pref*(somma*pref2+delta)*np.sqrt(4*np.pi/(2*k+1))
            coeff.append(value)
        else:
            pref = (-1)**(2*J-M+L+S)*7/np.sqrt(4*np.pi)*(2*J+1)*np.sqrt(2*k+1)*Wigner_coeff.threej_symbol([[J,k,J],[-M,0,M]])/Wigner_coeff.threej_symbol([[L,k,L],[-L,0,L]])
            pref2 = Wigner_coeff.sixj_symbol([[L,J,S],[J,L,k]])*Wigner_coeff.threej_symbol([[k,3,3],[0,0,0]])
            somma = 0
            for i in range(1,int(conf[1:])+1):
                somma += (-1)**i*Wigner_coeff.threej_symbol([[k,3,3],[0,4-i,i-4]])
            value = pref*pref2*somma*np.sqrt(4*np.pi/(2*k+1))
            coeff.append(value)

    return coeff


#######READ FROM FILE########

def cfp_from_file(conf):
    #i valori di cfp sono letti da Nielson e Koster e sono scritti con lo stesso formalismo.
    #la conf es per un d8 deve essere d3, per un d3 deve essere d3

    prime = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    file = open('cfp_d_conf.txt', 'r').readlines()
    check = False
    cfp = []
    for i,line in enumerate(file):
        if conf in line:
            check = True
        if check == True:
            if r'#' not in line:
                cfp.append(line)
            elif r'#' in line:
                break

    cfp_dic = {}
    for i,line in enumerate(cfp):
        if i>0:
            splitline = line.split('\t')
            factor = int(splitline[2])
            number = 1
            if len(splitline)>3:
                splitline[-1] = splitline[-1].split(' ')
                for j in range(len(splitline[-1])):
                    number *= prime[j]**int(splitline[-1][j])
            number = np.sqrt(number)
            number *= factor
            try:
                cfp_dic[splitline[0]].update({splitline[1]:number})
                #cfp_dic[splitline[0]] += [splitline[1],number]
            except:
                cfp_dic[splitline[0]] = {splitline[1]: number}

    return cfp_dic

def read_matrix_from_file(conf, closed_shell=False):
    #legge le matrici Uk e V11 dai file di OctoYot
    #le seniority sono come quelle di Nielson e Koster

    prime = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]

    file = open('tables/'+conf[0]+'_electron_mod.dat').readlines()
    check = False
    dizionario = {}
    for i,line in enumerate(file):
        if 'MATRIX FOR' in line and 'ELECTROSTATIC' not in line:
            line = line.replace('\n','')
            nomi = line.split(' ')
            if nomi[-1] in dizionario.keys():
                dizionario[nomi[-1]][nomi[0][1:]] = {}
            else:
                dizionario[nomi[-1]] = {}
                dizionario[nomi[-1]][nomi[0][1:]] = {}
            check = True
            continue
            #print('1', nomi)
        # print(dizionario)
        if check == True and 'ELECTROSTATIC' not in line and 'ZZ' not in line and '\n' != line:
            line = line.replace('\n','')
            splitline=line.split(' ')
            splitline = [elem for elem in splitline if elem != '']
            #print('2',splitline)
            numeri = []
            stringhe = []
            for j in range(len(splitline)):
                try:
                    numeri.append(int(splitline[j]))
                except:
                    stringhe.append(splitline[j])
            # print(numeri)
            valore = 1
            for j in range(1,len(numeri)):
                valore *= prime[j-1]**numeri[j]
            valore = np.sqrt(valore)
            try:
                valore *= numeri[0]
            except:
                print(splitline)
                exit()
            # print(valore)
            # exit()
            if len(stringhe)>1:
                # print('true')
                n1 = list(dizionario.keys())[-1]
                n2 = list(dizionario[n1].keys())[-1]
                n3 = stringhe[0]
                n4 = stringhe[1]
                dizionario[n1][n2][n3] = {}
            else:
                n4 = stringhe[0]

            dizionario[n1][n2][n3].update({n4:valore})

        if 'MATRIX FOR' in line or 'ELECTROSTATIC' in line or 'ZZ' in line or '\n' == line:
            #print('3',line)
            check=False

    if closed_shell==False:
        conf_str = conf
    else:
        conf_n = almost_closed_shells(conf)
        conf_str = conf[0]+str(conf_n)


    aggiunta = {}
    for key in dizionario[conf_str].keys():
        kk = int(key[-1])
        aggiunta[key] = {}
        for key1 in dizionario[conf_str][key].keys():
            L = state_legend(key1[1])
            S = int(key1[0])
            for key2 in dizionario[conf_str][key][key1].keys():
                L1 = state_legend(key2[1])
                S1 = int(key2[0])

                if key=='V11':
                    try:
                        dizionario[conf_str][key][key2][key1] = dizionario[conf_str][key][key1][key2]*(-1)**(L-L1-S/2+S1/2)
                    except:
                        try:
                            prova = aggiunta[key][key2]
                        except:
                            aggiunta[key][key2] = {}
                        aggiunta[key][key2].update({key1:dizionario[conf_str][key][key1][key2]*(-1)**(L-L1-S/2+S1/2)})
                else:
                    try:
                        dizionario[conf_str][key][key2][key1] = dizionario[conf_str][key][key1][key2]*(-1)**(L-L1)
                    except:
                        try:
                            prova = aggiunta[key][key2]
                        except:
                            aggiunta[key][key2] = {}
                        aggiunta[key][key2].update({key1:dizionario[conf_str][key][key1][key2]*(-1)**(L-L1)})


    for key in aggiunta.keys():
        dizionario[conf_str][key].update(aggiunta[key])

    # pprint(dizionario[conf_str]['U6'])
    # exit()

    return dizionario[conf_str]

def read_ee_int(conf, closed_shell):

    if closed_shell==False:
        conf_str = conf
    else:
        conf_n = almost_closed_shells(conf)
        conf_str = conf[0]+str(conf_n)

    file = open('tables/f_electron_mod.dat').readlines()
    check = False
    dizionario = {}

    for i,line in enumerate(file):
        if 'ELECTROSTATIC' in line:
            line = line.replace('\n','')
            nomi = line.split(' ')
            dizionario[nomi[-1]] = {}
            check = True
            continue
        elif check==True and 'ZZ' not in line and line != '\n':
            line = line.replace('\n','')
            splitline = line.split(' ')
            term_string = []
            for j in range(len(splitline)):
                if splitline[j]!='':
                    term_string.append(splitline[j])
            #print(term_string)
            try:
                if len(term_string)==6:
                    term1 = term_string[0]
                    term2 = term_string[1]
                else:
                    term2 = term_string[0]
                numero = [term_string[ii] for ii in range(len(term_string)) if term_string[ii]!=term1 and term_string[ii]!=term2]
            except:
                if len(term_string)==3:
                    term1 = term_string[0]
                    term2 = term_string[1]
                else:
                    term2 = term_string[0]
                numero = [0]
            try:
                dizionario[nomi[-1]][term1].update({term2:numero})
            except:
                dizionario[nomi[-1]][term1] = {}
                dizionario[nomi[-1]][term1].update({term2:numero})
        elif 'MATRIX FOR' in line or 'ZZ' in line:
            check=False

    return dizionario[conf_str]


#--------------------------------------------------#
#           TABLES, LEGENDS & CONVENTIONS          #
#--------------------------------------------------#

def points_on_sphere(num_pts=100, figure=False, angles=False):
    #golden spiral method (https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere)

    import mpl_toolkits.mplot3d

    indices = np.arange(0, num_pts, dtype='float64') + 0.5

    phi = np.arccos(1 - 2*indices/num_pts)
    theta = np.pi * (1 + 5**0.5) * indices

    x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)

    if figure==True:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z)
        plt.show()
    else:
        pass

    if angles == False:
        return x, y, z
    else:
        return phi, theta

def from_car_to_sph(coord):
    #for coord system centered in 0
    coord_conv = np.zeros_like(coord)
    for i in range(len(coord[:,0])):
        coord_conv[i,0] = np.linalg.norm(coord[i,:])
        coord_conv[i,1] = np.arccos(coord[i,2]/coord_conv[i,0])
        coord_conv[i,2] = np.arctan2(coord[i,1],coord[i,0])

    return coord_conv

def from_sph_to_car(coord_sph):
    #for coord system centered in 0
    #angles in radiants
    coord_car = np.zeros_like(coord_sph)
    for i in range(len(coord_car[:,0])):
        coord_car[i,0] = coord_sph[i,0]*np.cos(coord_sph[i,2])*np.sin(coord_sph[i,1])
        coord_car[i,1] = coord_sph[i,0]*np.sin(coord_sph[i,2])*np.sin(coord_sph[i,1])
        coord_car[i,2] = coord_sph[i,0]*np.cos(coord_sph[i,1])

    return coord_car

def terms_labels(conf):
    #questi dati sono presi da Boca, "theoretical fundations of molecular magnetism" (Ch 8, p 381, Tab 8.4)
    #oppure da OctoYot f_e_data.f90, TS_d_labels (seguono l'ordine di Nielson e Koster)

    legenda={'d1': ['2D'],
             'd2': ['3P','3F','1S','1D','1G'],
             'd3': ['4P','4F','2P','2D1','2D2','2F','2G','2H'],
             'd4': ['5D','3P1','3P2','3D','3F1','3F2','3G','3H','1S1','1S2','1D1','1D2','1F','1G1','1G2','1I'],
             'd5': ['6S','4P','4D','4F','4G','2S','2P','2D1','2D2','2D3','2F1','2F2','2G1','2G2','2H','2I'],
             'f1':['2F'],
             'f2':['3P','3F','3H','1S','1D','1G','1I'],
             'f3':['4S','4D','4F','4G','4I','2P','2D1','2D2','2F1','2F2','2G1','2G2','2H1','2H2','2I','2K','2L'],
             'f4':['5S','5D','5F','5G','5I','3P1','3P2','3P3','3D1','3D2','3F1','3F2','3F3','3F4','3G1','3G2','3G3','3H1','3H2','3H3','3H4','3I1','3I2','3K1','3K2','3L','3M','1S1','1S2','1D1','1D2','1D3','1D4','1F','1G1','1G2','1G3','1G4','1H1','1H2','1I1','1I2','1I3','1K','1L1','1L2','1N'],
             'f5':['6P','6F','6H','4S','4P1','4P2','4D1','4D2','4D3','4F1','4F2','4F3','4F4','4G1','4G2','4G3','4G4','4H1','4H2','4H3','4I1','4I2','4I3','4K1','4K2','4L','4M','2P1','2P2','2P3','2P4','2D1','2D2','2D3','2D4','2D5','2F1','2F2','2F3','2F4','2F5','2F6','2F7','2G1','2G2','2G3','2G4','2G5','2G6','2H1','2H2','2H3','2H4','2H5','2H6','2H7','2I1','2I2','2I3','2I4','2I5','2K1','2K2','2K3','2K4','2K5','2L1','2L2','2L3','2M1','2M2','2N','2O'],
             'f6':['7F','5S','5P','5D1','5D2','5D3','5F1','5F2','5G1','5G2','5G3','5H1','5H2','5I1','5I2','5K', '5L','3P1','3P2','3P3','3P4','3P5','3P6','3D1','3D2','3D3','3D4','3D5','3F1','3F2','3F3','3F4','3F5','3F6','3F7','3F8','3F9','3G1','3G2','3G3','3G4','3G5','3G6','3G7','3H1','3H2','3H3','3H4','3H5','3H6','3H7','3H8','3H9','3I1','3I2','3I3','3I4','3I5','3I6','3K1','3K2','3K3','3K4','3K5','3K6','3L1','3L2','3L3','3M1','3M2','3M3','3N','3O','1S1','1S2','1S3','1S4','1P','1D1','1D2','1D3','1D4','1D5','1D6','1F1','1F2','1F3','1F4','1G1','1G2','1G3','1G4','1G5','1G6','1G7','1G8','1H1','1H2','1H3','1H4','1I1','1I2','1I3','1I4','1I5','1I6','1I7','1K1','1K2','1K3','1L1','1L2','1L3','1L4','1M1','1M2','1N1','1N2','1Q'],
             'f7':['8S','6P','6D','6F','6G','6H','6I','4S1','4S2','4P1','4P2','4D1','4D2','4D3','4D4','4D5','4D6','4F1','4F2','4F3','4F4','4F5','4G1','4G2','4G3','4G4','4G5','4G6','4G7','4H1','4H2', '4H3','4H4','4H5','4I1','4I2','4I3','4I4','4I5','4K1','4K2','4K3','4L1','4L2','4L3','4M','4N','2S1','2S2','2P1','2P2','2P3','2P4','2P5','2D1','2D2','2D3','2D4','2D5','2D6','2D7','2F1','2F2','2F3','2F4','2F5','2F6','2F7','2F8','2F9','2F0','2G1','2G2','2G3','2G4','2G5','2G6','2G7','2G8','2G9','2G0','2H1','2H2','2H3','2H4','2H5','2H6','2H7','2H8','2H9','2I1','2I2','2I3','2I4','2I5','2I6','2I7','2I8','2I9','2K1','2K2','2K3','2K4','2K5','2K6','2K7','2L1','2L2','2L3','2L4','2L5','2M1','2M2','2M3','2M4','2N1','2N2','2O','2Q']}

    return legenda[conf]

def terms_basis(conf):
    #(2S, L, seniority) da OctoYot f_e_data.f90, TS_d_basis (seguono l'ordine di Nielson e Koster)
    if conf[0]=='d' and int(conf[1:])>5:
        conf = 'd'+str(almost_closed_shells(conf))
    elif conf[0]=='f' and int(conf[1:])>7:
        conf = 'f'+str(almost_closed_shells(conf))
    else:
        pass

    legenda={'d1': [[1, 2, 1]],
             'd2': [[2, 1, 2],  [2, 3, 2],  [0, 0, 2],  [0, 2, 2],  [0, 4, 2]],
             'd3': [[3, 1, 3],  [3, 3, 3],  [1, 1, 3],  [1, 2, 1],  [1, 2, 3],  [1, 3, 3],  [1, 4, 3],  [1, 5, 3]],
             'd4': [[4, 2, 4],  [2, 1, 2],  [2, 1, 4],  [2, 2, 4],  [2, 3, 2],  [2, 3, 4],  [2, 4, 4],  [2, 5, 4], [0, 0, 0],  [0, 0, 4],  [0, 2, 2],  [0, 2, 4],  [0, 3, 4],  [0, 4, 2],  [0, 4, 4],  [0, 6, 4]],
             'd5': [[5, 0, 5],  [3, 1, 3],  [3, 2, 5],  [3, 3, 3],  [3, 4, 5],  [1, 0, 5],  [1, 1, 3],  [1, 2, 1], [1, 2, 3],  [1, 2, 5],  [1, 3, 3],  [1, 3, 5],  [1, 4, 3],  [1, 4, 5],  [1, 5, 3],  [1, 6, 5]],
             'f1':[[1, 3, 1]],
             'f2':[[2, 1, 2],  [2, 3, 2],  [2, 5, 2],  [0, 0, 0],  [0, 2, 2],  [0, 4, 2],  [0, 6, 2]],
             'f3':[[3, 0, 3],  [3, 2, 3],  [3, 3, 3],  [3, 4, 3],  [3, 6, 3],  [1, 1, 3],  [1, 2, 3],  [1, 2, 3],  [1, 3, 1],  [1, 3, 3],  [1, 4, 3],  [1, 4, 3],  [1, 5, 3],  [1, 5, 3],  [1, 6, 3],  [1, 7, 3],  [1, 8, 3]],
             'f4':[[4, 0, 4],  [4, 2, 4],  [4, 3, 4],  [4, 4, 4],  [4, 6, 4],  [2, 1, 2],  [2, 1, 4],  [2, 1, 4],  [2, 2, 4],  [2, 2, 4],  [2, 3, 2],  [2, 3, 4],  [2, 3, 4],  [2, 3, 4],  [2, 4, 4],  [2, 4, 4],  [2, 4, 4],  [2, 5, 2],  [2, 5, 4],  [2, 5, 4],  [2, 5, 4],  [2, 6, 4],  [2, 6, 4],  [2, 7, 4],  [2, 7, 4],  [2, 8, 4],  [2, 9, 4],  [0, 0, 0],  [0, 0, 4],  [0, 2, 2],  [0, 2, 4],  [0, 2, 4],  [0, 2, 4],  [0, 3, 4],  [0, 4, 2],  [0, 4, 4],  [0, 4, 4],  [0, 4, 4],  [0, 5, 4],  [0, 5, 4],  [0, 6, 2],  [0, 6, 4],  [0, 6, 4],  [0, 7, 4],  [0, 8, 4],  [0, 8, 4],  [0, 10, 4]],
             'f5':[[5, 1, 5],  [5, 3, 5],  [5, 5, 5],  [3, 0, 3],  [3, 1, 5],  [3, 1, 5],  [3, 2, 3],  [3, 2, 5],  [3, 2, 5],  [3, 3, 3],  [3, 3, 5],  [3, 3, 5],  [3, 3, 5],  [3, 4, 3],  [3, 4, 5],  [3, 4, 5],  [3, 4, 5],  [3, 5, 5],  [3, 5, 5],  [3, 5, 5],  [3, 6, 3],  [3, 6, 5],  [3, 6, 5],  [3, 7, 5],  [3, 7, 5],  [3, 8, 5],  [3, 9, 5],  [1, 1, 3],  [1, 1, 5],  [1, 1, 5],  [1, 1, 5],  [1, 2, 3],  [1, 2, 3],  [1, 2, 5],  [1, 2, 5],  [1, 2, 5],  [1, 3, 1],  [1, 3, 3],  [1, 3, 5],  [1, 3, 5],  [1, 3, 5],  [1, 3, 5],  [1, 3, 5],  [1, 4, 3],  [1, 4, 3],  [1, 4, 5],  [1, 4, 5],  [1, 4, 5],  [1, 4, 5],  [1, 5, 3],  [1, 5, 3],  [1, 5, 5],  [1, 5, 5],  [1, 5, 5],  [1, 5, 5],  [1, 5, 5],  [1, 6, 3],  [1, 6, 5],  [1, 6, 5],  [1, 6, 5],  [1, 6, 5],  [1, 7, 3],  [1, 7, 5],  [1, 7, 5],  [1, 7, 5],  [1, 7, 5],  [1, 8, 3],  [1, 8, 5],  [1, 8, 5],  [1, 9, 5],  [1, 9, 5],  [1,10, 5],  [1,11, 5]],
             'f6':[[6, 3, 6],  [4, 0, 4],  [4, 1, 6],  [4, 2, 4],  [4, 2, 6],  [4, 2, 6],  [4, 3, 4],  [4, 3, 6],  [4, 4, 4],  [4, 4, 6],  [4, 4, 6],  [4, 5, 6],  [4, 5, 6],  [4, 6, 4],  [4, 6, 6],  [4, 7, 6],  [4, 8, 6],  [2, 1, 2],  [2, 1, 4],  [2, 1, 4],  [2, 1, 6],  [2, 1, 6],  [2, 1, 6],  [2, 2, 4],  [2, 2, 4],  [2, 2, 6],  [2, 2, 6],  [2, 2, 6],  [2, 3, 2],  [2, 3, 4],  [2, 3, 4],  [2, 3, 4],  [2, 3, 6],  [2, 3, 6],  [2, 3, 6],  [2, 3, 6],  [2, 3, 6],  [2, 4, 4],  [2, 4, 4],  [2, 4, 4],  [2, 4, 6],  [2, 4, 6],  [2, 4, 6],  [2, 4, 6],  [2, 5, 2],  [2, 5, 4],  [2, 5, 4],  [2, 5, 4],  [2, 5, 6],  [2, 5, 6],  [2, 5, 6],  [2, 5, 6],  [2, 5, 6],  [2, 6, 4],  [2, 6, 4],  [2, 6, 6],  [2, 6, 6],  [2, 6, 6],  [2, 6, 6],  [2, 7, 4],  [2, 7, 4],  [2, 7, 6],  [2, 7, 6],  [2, 7, 6],  [2, 7, 6],  [2, 8, 4],  [2, 8, 6],  [2, 8, 6],  [2, 9, 4],  [2, 9, 6],  [2, 9, 6],  [2,10, 6],  [2,11, 6],  [0, 0, 0],  [0, 0, 4],  [0, 0, 6],  [0, 0, 6],  [0, 1, 6],  [0, 2, 2],  [0, 2, 4],  [0, 2, 4],  [0, 2, 4],  [0, 2, 6],  [0, 2, 6],  [0, 3, 4],  [0, 3, 6],  [0, 3, 6],  [0, 3, 6],  [0, 4, 2],  [0, 4, 4],  [0, 4, 4],  [0, 4, 4],  [0, 4, 6],  [0, 4, 6],  [0, 4, 6],  [0, 4, 6],  [0, 5, 4],  [0, 5, 4],  [0, 5, 6],  [0, 5, 6],  [0, 6, 2],  [0, 6, 4],  [0, 6, 4],  [0, 6, 6],  [0, 6, 6],  [0, 6, 6],  [0, 6, 6],  [0, 7, 4],  [0, 7, 6],  [0, 7, 6],  [0, 8, 4],  [0, 8, 4],  [0, 8, 6],  [0, 8, 6],  [0, 9, 6],  [0, 9, 6],  [0,10, 4],  [0,10, 6],  [0,12, 6]],
             'f7':[[7, 0, 7],  [5, 1, 5],  [5, 2, 7],  [5, 3, 5],  [5, 4, 7],  [5, 5, 5],  [5, 6, 7],  [3, 0, 3],  [3, 0, 7],  [3, 1, 5],  [3, 1, 5],  [3, 2, 3],  [3, 2, 5],  [3, 2, 5],  [3, 2, 7],  [3, 2, 7],  [3, 2, 7],  [3, 3, 3],  [3, 3, 5],  [3, 3, 5],  [3, 3, 5],  [3, 3, 7],  [3, 4, 3],  [3, 4, 5],  [3, 4, 5],  [3, 4, 5],  [3, 4, 7],  [3, 4, 7],  [3, 4, 5],  [3, 5, 5],  [3, 5, 5],  [3, 5, 7],  [3, 5, 7],  [3, 5, 7],  [3, 6, 3],  [3, 6, 5],  [3, 6, 5],  [3, 6, 7],  [3, 6, 7],  [3, 7, 5],  [3, 7, 5],  [3, 7, 7],  [3, 8, 5],  [3, 8, 7],  [3, 8, 7],  [3, 9, 5],  [3,10, 7],  [1, 0, 7],  [1, 0, 7],  [1, 1, 3],  [1, 1, 5],  [1, 1, 5],  [1, 1, 5],  [1, 1, 7],  [1, 2, 3],  [1, 2, 3],  [1, 2, 5],  [1, 2, 5],  [1, 2, 5],  [1, 2, 7],  [1, 2, 7],  [1, 3, 1],  [1, 3, 3],  [1, 3, 5],  [1, 3, 5],  [1, 3, 5],  [1, 3, 5],  [1, 3, 5],  [1, 3, 7],  [1, 3, 7],  [1, 3, 7],  [1, 4, 3],  [1, 4, 3],  [1, 4, 5],  [1, 4, 5],  [1, 4, 5],  [1, 4, 5],  [1, 4, 7],  [1, 4, 7],  [1, 4, 7],  [1, 4, 7],  [1, 5, 3],  [1, 5, 3],  [1, 5, 5],  [1, 5, 5],  [1, 5, 5],  [1, 5, 5],  [1, 5, 5],  [1, 5, 7],  [1, 5, 7],  [1, 6, 3],  [1, 6, 5],  [1, 6, 5],  [1, 6, 5],  [1, 6, 5],  [1, 6, 7],  [1, 6, 7],  [1, 6, 7],  [1, 6, 7],  [1, 7, 3],  [1, 7, 5],  [1, 7, 5],  [1, 7, 5],  [1, 7, 5],  [1, 7, 7],  [1, 7, 7],  [1, 8, 3],  [1, 8, 5],  [1, 8, 5],  [1, 8, 7],  [1, 8, 7],  [1, 9, 5],  [1, 9, 5],  [1, 9, 7],  [1, 9, 7],  [1,10, 5],  [1,10, 7],  [1,11, 5],  [1,12, 7]]}

    return legenda[conf]

def conv_Aqkrk_bkq(l,m):
    #conversion from stevens coefficients to wybourne formalism

    if isinstance(l, int):
        l = str(l)
    if isinstance(m, int):
        m = str(m)

    legenda = {'2':{'0':2, '1':1/np.sqrt(6), '2':2/np.sqrt(6)},
               '4':{'0':8, '1':2/np.sqrt(5), '2':4/np.sqrt(10), '3':2/np.sqrt(35), '4':8/np.sqrt(70)},
               '6':{'0':16, '1':8/np.sqrt(42), '2':16/np.sqrt(105), '3':8/np.sqrt(105), '4':16/(3*np.sqrt(14)), '5':8/(3*np.sqrt(77)), '6':16/np.sqrt(231)}}
    return legenda[str(l)][str(m)]

def r_expect(k, conf):
    # <r^k> from S. Edvarsson, M. Klintenberg, J. Alloys Compd. 1998, 275–277, 230

    if isinstance(k, int):
        k = str(k)

    if k=='0':
        return 1
    else:
        legenda = {'2':{'f1':1.456, 'f2':1.327, 'f3':1.222, 'f4':1.135, 'f5':1.061, 'f6':0.997, 'f7':0.942, 'f8':0.893, 'f9':0.849, 'f10':0.810, 'f11':0.773, 'f12':0.740, 'f13':0.710},
                '4':{'f1':5.437, 'f2':4.537, 'f3':3.875, 'f4':3.366, 'f5':2.964, 'f6':2.638, 'f7':2.381, 'f8':2.163, 'f9':1.977, 'f10':1.816, 'f11':1.677, 'f12':1.555, 'f13':1.448},
                '6':{'f1':42.26, 'f2':32.65, 'f3':26.12, 'f4':21.46, 'f5':17.99, 'f6':15.34, 'f7':13.36, 'f8':11.75, 'f9':10.44, 'f10':9.345, 'f11':8.431, 'f12':7.659, 'f13':7.003}}
        return legenda[k][conf]

def sigma_k(k, conf):
    # Sternheimer shielding parameters from S. Edvardsson, M. Klinterberg, J. Alloys Compd. 1998, 275, 233.

    if isinstance(k, int):
        k = str(k)

    legenda = {'2':{'f1':0.510, 'f2':0.515, 'f3':0.518, 'f4':0.519, 'f5':0.519, 'f6':0.520, 'f7':0.521, 'f8':0.523, 'f9':0.527, 'f10':0.534, 'f11':0.544, 'f12':0.554, 'f13':0.571},
               '4':{'f1':0.0132, 'f2':0.0138, 'f3':0.0130, 'f4':0.0109, 'f5':0.0077, 'f6':0.0033, 'f7':-0.0031, 'f8':-0.0107, 'f9':-0.0199, 'f10':-0.0306, 'f11':-0.0427, 'f12':-0.0567, 'f13':-0.0725},
               '6':{'f1':-0.0294, 'f2':-0.0301, 'f3':-0.0310, 'f4':-0.0314, 'f5':-0.0317, 'f6':-0.0319, 'f7':-0.0318, 'f8':-0.0318, 'f9':-0.0316, 'f10':-0.0313, 'f11':-0.0310, 'f12':-0.0306, 'f13':-0.0300}}
    return legenda[k][conf]

def Stev_coeff(k, conf):
    # Stevens coefficients (alpha=2, beta=4, gamma=6) from K. W. H. Stevens, Proc. Phys. Soc. 1952, 65, 209.
    # or table 20 from A. Abragam and B. Bleaney, Electron Paramagnetic Resonance of Transition Ions, Dover, New York, 1986.
    
    legenda = {'2':{'f1':-2/35, 'f2':-52/(11*15**2), 'f3':-7/(33**2), 'f4':14/(11**2*15), 'f5':13/(7*45), 'f6':0, 'f7':0, 'f8':-1/99, 'f9':-2/(9*35), 'f10':-1/(30*15), 'f11':4/(45*35), 'f12':1/99, 'f13':2/63},
               '4':{'f1':2/(7*45), 'f2':-4/(55*33*3), 'f3':-8*17/(11**2*13*297), 'f4':952/(13*3**3*11**3*5), 'f5':26/(33*7*45), 'f6':0, 'f7':0, 'f8':2/(11*1485), 'f9':-8/(11*45*273), 'f10':-1/(11*2730), 'f11':2/(11*15*273), 'f12':8/(3*11*1485), 'f13':-2/(77*15)},
               '6':{'f1':0, 'f2':17*16/(7*11**2*13*5*3**4), 'f3':-17*19*5/(13**2*11**3*3**3*7), 'f4':2584/(11**2*13**2*3*63), 'f5':0, 'f6':0, 'f7':0, 'f8':-1/(13*33*2079), 'f9':4/(11**2*13**2*3**3*7), 'f10':-5/(13*33*9009), 'f11':8/(13**2*11**2*3**3*7), 'f12':-5/(13*33*2079), 'f13':4/(13*33*63)}}
    return legenda[k][conf]

def plm(l,m):
    #spherical harmonics prefactor

    legenda = {'2':{'0':(1/4)*np.sqrt(5/np.pi), '1':(1/2)*np.sqrt(15/np.pi), '2':(1/4)*np.sqrt(15/np.pi)},
               '4':{'0':(3/16)*np.sqrt(1/np.pi), '1':(3/4)*np.sqrt(5/(2*np.pi)), '2':(3/8)*np.sqrt(5/np.pi), '3':(3/8)*np.sqrt(70/np.pi), '4':(3/16)*np.sqrt(35/np.pi)},
               '6':{'0':(1/32)*np.sqrt(13/np.pi), '1':(1/8)*np.sqrt(273/(4*np.pi)), '2':(1/64)*np.sqrt(2730/np.pi), '3':(1/32)*np.sqrt(2730/np.pi), '4':(21/32)*np.sqrt(13/(np.pi*7)), '5':np.sqrt(9009/(512*np.pi)), '6':(231/64)*np.sqrt(26/(np.pi*231))}}
    return legenda[str(l)][str(m)]

def A_table(nel, MJ):  #NOT USED
    #multipole moments of trivalen rare earth ions for J=MJ
    #Table 1 p 292 of Sievers "Asphericity of 4f-Shells in their Hund's rule ground states" (1981)

    legend = {1: {'5/2':[-0.2857, 0.0476, 0.000]},
            2: {'4':[-0.2941, -0.0771, 0.0192]},
            3: {'9/2':[-0.1157, -0.0550, -0.0359]},
            4: {'4':[0.1080, 0.0428, 0.0191]},
            5: {'5/2':[0.2063, 0.0188, 0.0000]},
            7: {'7/2':[0.000, 0.000, 0.000]},
            8: {'6':[-0.3333, 0.0909, -0.0117]},
            9: {'15/2':[-0.3333, -0.1212, 0.0583]},
            10:{'8':[-0.1333, -0.0909, -0.1166]},
            11:{'15/2':[0.1333, 0.0909, 0.1166]},
            12:{'6':[0.3333,0.1212,-0.0583]},
            13:{'7/2':[0.3333,-0.0909,0.0117]}}

    return legend[nel][MJ]

def state_legend(L_str, inv=False):
    legenda = {'S':0,
               'P':1,
               'D':2,
               'F':3,
               'G':4,
               'H':5,
               'I':6,
               'K':7,
               'L':8,
               'M':9,
               'N':10,
               'O':11,
               'Q':12,
               'R':13,
               'T':14,
               'U':15,
               'V':16}
    if inv==False:
        return legenda[L_str]
    else:
        inv_map = {str(v): k for k, v in legenda.items()}
        return inv_map[L_str]

def almost_closed_shells(name):
    legenda = {'d6':4,
               'd7':3,
               'd8':2,
               'd9':1,
               'f8':6,
               'f9':5,
               'f10':4,
               'f11':3,
               'f12':2,
               'f13':1}
    return legenda[name]

def ground_term_legend(conf):

    legenda = {'d1':'2D',
               'd2':'3F',
               'd3':'4F',
               'd4':'5D',
               'd5':'6S',
               'f1':'2F (5/2)',
               'f2':'3H (4)',
               'f3':'4I (9/2)',
               'f4':'5I (4)',
               'f5':'6H (5/2)',
               'f6':'7F (0)',
               'f7':'8S (7/2)',
               'f8':'7F (6)',
               'f9':'6H (15/2)',
               'f10':'5I (8)',
               'f11':'4I (15/2)',
               'f12':'3H (6)',
               'f13':'2F (7/2)'}
    return legenda[conf]

def free_ion_param_f(conf):
    #Table 5 p 168 from C. Goerller-Walrand, K. Binnemans, Handbook of Physics & Chemistry of Rare Earths, Vol 23, Ch 155, (1996)
    dict = {'f2':{'F2': 68323, 'F4': 49979, 'F6': 32589, 'zeta': 747},
            'f3':{'F2': 72295, 'F4': 52281, 'F6': 35374, 'zeta': 879},
            'f4':{'F2': 75842, 'F4': 54319, 'F6': 38945, 'zeta': 1023},
            'f5':{'F2': 79012, 'F4': 56979, 'F6': 40078, 'zeta': 1170},
            'f6':{'F2': 82786, 'F4': 59401, 'F6': 42644, 'zeta': 1332},
            'f7':{'F2': 85300, 'F4': 60517, 'F6': 44731, 'zeta': 1504},
            'f8':{'F2': 89540, 'F4': 63485, 'F6': 44998, 'zeta': 1705},
            'f9':{'F2': 92373, 'F4': 65281, 'F6': 47642, 'zeta': 1915},
            'f10':{'F2': 95772, 'F4': 67512, 'F6': 48582, 'zeta': 2142},
            'f11':{'F2': 97909, 'F4': 70349, 'F6': 48861, 'zeta': 2358},
            'f12':{'F2': 101381, 'F4': 70230, 'F6': 51827, 'zeta': 2644}}
    return dict[conf]

def free_ion_param_f_HF(conf):
    # from Ma, C. G., Brik, M. G., Li, Q. X., & Tian, Y. (2014). Systematic analysis of spectroscopic characteristics of the lanthanide and actinide ions with the 4fN and 5fN (N= 1… 14) electronic configurations in a free state. Journal of alloys and compounds, 599, 93-101.
    dict = {'f2':{'F2': 96681, 'F4': 60533, 'F6': 43509, 'zeta': 808},
            'f3':{'F2': 100645, 'F4': 63030, 'F6': 45309, 'zeta': 937},
            'f4':{'F2': 104389, 'F4': 65383, 'F6': 47003, 'zeta': 1075},
            'f5':{'F2': 107971, 'F4': 67630, 'F6': 48619, 'zeta': 1225},
            'f6':{'F2': 111416, 'F4': 69786, 'F6': 50169, 'zeta': 1387},
            'f7':{'F2': 114742, 'F4': 71865, 'F6': 51662, 'zeta': 1561},
            'f8':{'F2': 117981, 'F4': 73886, 'F6': 53113, 'zeta': 1749},
            'f9':{'F2': 121132, 'F4': 75850, 'F6': 54523, 'zeta': 1950},
            'f10':{'F2': 124214, 'F4': 77768, 'F6': 55899, 'zeta': 2165},
            'f11':{'F2': 127240, 'F4': 79650, 'F6': 57248, 'zeta': 2396},
            'f12':{'F2': 130201, 'F4': 81489, 'F6': 58566, 'zeta': 2643},
            'f13':{'F2': 133119, 'F4': 83300, 'F6': 59864, 'zeta': 2906}}
    return dict[conf]

def COLORS_list():
    colors = [
    'tab:blue',
    'tab:red',
    'tab:green',
    'tab:orange',
    'tab:pink',
    'tab:purple',
    'tab:gray',
    'tab:cyan',
    'tab:brown',
    'tab:olive',
    'salmon',
    'indigo',
    'm',
    'c',
    'g',
    'r',
    'b',
    'k',
    ]

    for w in range(10):
        colors += colors
    COLORS = tuple(colors)
    return COLORS

def color_atoms():

    elem_cpk = {  # color codes for elements
            # Basics
            'H' : 'lightgray',
            'C' : 'k',
            'N' : 'b',
            'O' : 'r',
            # Halogens
            'F' : 'tab:green',
            'Cl': 'g',
            'Br': 'maroon',
            'I' : 'darkviolet',
            # Noble gases
            'He': 'c',
            'Ne': 'c',
            'Ar': 'c',
            'Kr': 'c',
            'Xe': 'c',
            # Common nonmetals
            'P' : 'orange',
            'S' : 'y',
            'B' : 'tan',
            # Metals
            #   Alkali
            'Li': 'violet',
            'Na': 'violet',
            'K' : 'violet',
            'Rb': 'violet',
            'Cs': 'violet',
            #   Alkali-earth
            'Be': 'darkgreen',
            'Mg': 'darkgreen',
            'Ca': 'darkgreen',
            'Sr': 'darkgreen',
            'Ba': 'darkgreen',
            #   Transition, I series
            'Sc': 'steelblue',
            'Ti': 'steelblue',
            'V' : 'steelblue',
            'Cr': 'steelblue',
            'Mn': 'steelblue',
            'Fe': 'steelblue',
            'Co': 'steelblue',
            'Ni': 'steelblue',
            'Cu': 'steelblue',
            'Zn': 'steelblue',
            #   Transition, II series
            'Y' : 'deepskyblue',
            'Zr': 'deepskyblue',
            'Nb': 'deepskyblue',
            'Mo': 'deepskyblue',
            'Tc': 'deepskyblue',
            'Ru': 'deepskyblue',
            'Rh': 'deepskyblue',
            'Pd': 'deepskyblue',
            'Ag': 'deepskyblue',
            'Cd': 'deepskyblue',
            #   Transition, III series
            'La': 'cadetblue',
            'Hf': 'cadetblue',
            'Ta': 'cadetblue',
            'W' : 'cadetblue',
            'Re': 'cadetblue',
            'Os': 'cadetblue',
            'Ir': 'cadetblue',
            'Pt': 'cadetblue',
            'Au': 'cadetblue',
            'Hg': 'cadetblue',
            #   Lanthanides
            'Ce': 'teal',
            'Pr': 'teal',
            'Nd': 'teal',
            'Pm': 'teal',
            'Sm': 'teal',
            'Eu': 'teal',
            'Gd': 'teal',
            'Tb': 'teal',
            'Dy': 'teal',
            'Ho': 'teal',
            'Er': 'teal',
            'Tm': 'teal',
            'Yb': 'teal',
            'Lu': 'teal',
            # Default color for all the others
            '_' : 'tab:pink',
            }
    
    return elem_cpk

#OLD FUNCTIONS

def read_structure(filexyz):
    #first 2 lines are not accounted for
    label = []
    coord = []
    file = open(filexyz, 'r').readlines()
    for i,line in enumerate(file):
        splitline = line.split('\t')
        if i>1:
            label.append(splitline[0])
            row = [float(splitline[j]) for j in range(1, len(splitline))]
            coord.append(row)
    label = np.array(label)
    coord = np.array(coord)
    return label, coord

def princ_comp(w, v=np.zeros((3,3))):

    from itertools import permutations

    permutazioni = list(permutations([0,1,2]))
    vst = np.zeros_like(v)
    for perm in permutazioni :
        sax = w[perm[0]]-(w[perm[1]]+w[perm[2]])/2.
        srh = w[perm[1]]-w[perm[2]]
        if np.abs(sax)>=np.abs(srh*3./2.) :
            zz = w[perm[0]]
            vst[:,2] = v[:,perm[0]]
            if np.sign(sax) == np.sign(srh) :
                xx = w[perm[2]]
                vst[:,0] = v[:,perm[2]]
                yy = w[perm[1]]
                vst[:,1] = v[:,perm[1]]
            else:
                xx = w[perm[1]]
                vst[:,0] = v[:,perm[1]]
                yy = w[perm[2]]
                vst[:,1] = v[:,perm[2]]
#        print('ax', zz-(xx+yy)/2.)
#        print('rh', xx-yy)
    wst = np.array([xx, yy, zz])

    return wst, vst

def princ_comp_sort(w, v=np.zeros((3,3))):

    indices = np.argsort(w)[::-1]
    wst = w[indices]
    vst = v[:,indices]
    return wst, vst

def def_fmtsusc(filesusc):
    fmtsusc = 'MOLCAS'
    for line in filesusc :

        if 'VAN VLECK SUSCEPTIBILITY' in line :
            fmtsusc = 'MOLCAS'
            break
        if 'TEMPERATURE DEPENDENT' in line :
            fmtsusc = 'ORCA'
            break
    return fmtsusc

def from_orca(filesusc, name):

    from itertools import islice

    if name=='D':
        count = 0
        for ii, line in enumerate(filesusc):
            #print(line)
            if 'Raw matrix (cm-1)' in line:
                count+=1
                if count%4==0:
                    Dmatstring=''.join(islice(filesusc,ii+1,ii+4,None))
                    evalstring=''.join(islice(filesusc,ii+6,ii+7,None))
                    evecstring=''.join(islice(filesusc,ii+12,ii+15,None))
                    break

        Dmatlist = [float(f) for f in Dmatstring.split()]
        evallist = [float(f) for f in evalstring.split()]
        eveclist = [float(f) for f in evecstring.split()]
        matrix = np.reshape(np.array(Dmatlist),(3,3))
        eigval = np.reshape(np.array(evallist),(3,))
        eigvec = np.reshape(np.array(eveclist),(3,3))

    elif name=='g':
        count = 0
        for ii, line in enumerate(filesusc):
            if 'ELECTRONIC G-MATRIX FROM EFFECTIVE HAMILTONIAN' in line:
                count+=1
                if count%2==0:
                    gmatstring=''.join(islice(filesusc,ii+5,ii+8,None))
                    evalstring=''.join(islice(filesusc,ii+10,ii+11,None))
                    evecstring=''.join(islice(filesusc,ii+15,ii+18,None))

        gmatlist = [float(f) for f in gmatstring.split()]
        evallist = []
        for i,f in enumerate(evalstring.split()):
            if i<3:
                try:
                    evallist.append(float(f))
                except:
                    pass
        eveclist = []
        for f in evecstring.split():
            try:
                eveclist.append(float(f))
            except:
                pass
        matrix = np.reshape(np.array(gmatlist),(3,3))
        eigval = np.reshape(np.array(evallist),(3,))
        eigvec = np.reshape(np.array(eveclist),(3,3))
    else:
        exit()

    return matrix, eigval, eigvec

def find_chi(fmtsusc, filesusc, temp):

    factor = np.pi*4/(1e6*scipy.constants.Avogadro*temp)
    cgs = True
    if fmtsusc == 'MOLCAS' :
        for line in filesusc :
            if 'cm3*K/mol' in line :
                factor = np.pi*4/(1e6*scipy.constants.Avogadro*temp)
                cgs = True
            try :
                if float(line.split()[0]) == temp :
                    chistring = line.split()
#                    print (chistring)
                    break
            except :
                pass
        if cgs :
            chicgs = np.array([[float(chiel) for chiel in chistring[1:4]],[float(chiel) for chiel in chistring[4:7]],[float(chiel) for chiel in chistring[7:10]]])
            chi = chicgs*factor

        else :
            chi = np.array([[float(chiel) for chiel in chistring[1:4]],[float(chiel) for chiel in chistring[4:7]],[float(chiel) for chiel in chistring[7:10]]])
    elif fmtsusc == 'ORCA' :
        factor = np.pi*4/(1e6*scipy.constants.Avogadro*temp)
        counter_ten = 0
        for idx,line in enumerate(filesusc):
            if 'TEMPERATURE/K:' in line:
                if float(line.split()[1])==int(temp):
                    counter_ten+=1
                    #if counter_ten%2==0:
                    chi = ''.join(islice(filesusc,idx+2,idx+5,None))
                    chixx = float(chi.split()[0])
                    chixy = float(chi.split()[1])
                    chixz = float(chi.split()[2])
                    chiyy = float(chi.split()[4])
                    chiyz = float(chi.split()[5])
                    chizz = float(chi.split()[8])
                    #break
        chicgs = np.array([[chixx,chixy,chixz],[chixy,chiyy,chiyz],[chixz,chiyz,chizz]])
#            print('chi_ORCA cgs', chicgs)
        chi = chicgs*factor

    return chi

def angle_between_vectors(v1, v2):

    v1 = np.array(v1)
    v2 = np.array(v2)

    dot_product = np.dot(v1, v2)

    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    cos_angle = dot_product / (norm_v1 * norm_v2)

    angle_rad = np.arccos(cos_angle)

    angle_deg = np.degrees(angle_rad)
    
    return angle_rad, angle_deg

#OTHER (not used)

def projection_LS(basis2, labels, basis1, bin=1e-4, J_label=False):   #!!!!!!!NOT USED (BUT WORKS)
    # calculates projection of basis2 on basis1 (the free ion case, with only e-e interaction)
    # expresses it in terms of SL basis or SLJ basis (if J_label = True)

    # print(labels)

    # LS_list = [labels[i][:2] for i in range(len(labels))]
    # LS = []
    # [LS.append(LS_list[i]) for i in range(len(LS_list)) if LS_list[i] not in LS]

    matrix_coeff = np.zeros_like(basis1)
    matrix_coeff_re2 = np.zeros_like(basis1, dtype='float64')
    states = []
    states_red = {}
    for i in range(basis2.shape[0]):
        states.append([])
        states_red[i+1] = {}
        for j in range(basis1.shape[0]):
            matrix_coeff[j,i] = np.dot(basis1[:,j].T,basis2[:,i])
            matrix_coeff_re2[j,i] = np.abs(np.dot(np.conj(basis1[:,j]).T,basis2[:,i]))**2
            for key in labels[j+1].keys():   #dovrebbe essere solo 1
                stato = [key, matrix_coeff_re2[j,i]]
            states[i].append(stato)
            if J_label==True:
                key, value = stato[0], stato[1]
            else:
                key, value = stato[0][:2], stato[1]
            if value>bin:
                if key in states_red[i+1].keys():
                    states_red[i+1][key] += value
                else:
                    states_red[i+1][key] = value
            else:
                pass
        tot = sum(states_red[i+1].values())
        if round(tot,2) != 1:
            warnings.warn('The expantion coefficient do not sum to 1')
            print(tot)
        for key, value in states_red[i+1].items():
            states_red[i+1][key] = value*100/tot
        sortato = sorted(states_red[i+1].items(), key=lambda x:x[1], reverse=True)  #sort dict on values
        states_red[i+1] = dict(sortato)
        # print(tot)
        # print(matrix_coeff[:,i])   #combinazione lineare per basis2[:,i]
        # print(matrix_coeff_re2[:,i])
        # print(states[i])
        # print(states_red[i+1])
    # print(states_red)
    # print(states_red)
    # exit()
    return states_red
