#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy
import scipy.constants
import scipy.spatial
import scipy.special
import matplotlib.pyplot as plt
from itertools import islice
from datetime import datetime
from pprint import pprint
import copy
import warnings
#import sympy   #activate this to use evaluation False in functions 
import crystdat

__version__ = "1.2.0"

def print_program_info():
    program_name = "NJA-CFS (Not Just Another - Crystal Field Software)"

    current_date = datetime.now().strftime("%B %d, %Y")

    print("\n")
    print(f"{'*' * 60}")
    print(f"{' ' * 3}{program_name}")
    print(f"{' ' * 3}Version: {__version__}")
    print(f"{' ' * 3}Date: {current_date}")
    print(f"{'*' * 60}")
    print("\n")

print_program_info()

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
    """
    This class is a wrapper around the functions to compute angular momentum coupling coefficients
    and Wigner D matrices for spherical harmonics rotations.

    References:
    1. Boca, R., "Theoretical Foundations of Molecular Magnetism" (1999)
    """

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
        # computes racah formula for 3j-symbols (p 52 Ch 1 from Boca)
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
        l (int): The one-electron orbital quantum number.
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
        l (int): The one-electron orbital quantum number.
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

        if np.abs(A)<bin:
            A = 0
        if np.abs(B)<bin:
            B = 0

        Z = R[0]**2 - R[1]**2 - R[2]**2 + R[3]**2
        Ac = np.conj(A)
        Bc = np.conj(B)
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
                D[i,j] = eval(value)
                D[i,j] *= coeff[l][i,j]

            D_rep = np.zeros_like(D, dtype='complex128')
            for i,ii in enumerate(range(l,-1,-1)):
                for j,jj in enumerate(range(l,-l-1,-1)):
                    D_rep[i,j] = D[i,j] 
                    D_rep[-i-1,-j-1] = (-1)**(np.abs(i-j))*np.conj(D[i,j]) 
            D=D_rep

        return D


class CFP():
    """
    Handles the coefficients of fractional parentage (cfp) for a given electron configuration.

    References:
    1. Nielson - Koster in "Spectroscopic Coefficients for the p^n, d^n and f^n Configurations" (1963)

    Attributes:
    dic_LS_inv_almost (dict): An inverse dictionary for the LS-coupling scheme for the almost-closed-shell configuration.
    l (int): The one-electron orbital quantum number.
    n (int): The number of electrons in the almost-closed-shell configuration.
    N (int): The number of electrons in the configuration.
    closed (bool): A flag indicating whether the configuration is almost-closed-shell or not.
    """

    def __init__(self, conf, dic_cfp, dic_LS_inv_almost=None):
        """
        Initializes the CFP object with the given configuration, CFP dictionary, and optional LS-coupling scheme dictionary.

        The configuration is represented by a string of the form 'l^x', where 'l' is orbital momentum quantum number 
        (represented as 'd' for l=2 and 'f' for l=3), and 'x' is the number of electrons in the configuration.

        Parameters:
        conf (str): The electron configuration.
        dic_cfp (dict): A dictionary containing the coefficients of fractional parentage (CFP) for the configuration.
        dic_LS_inv_almost (dict, optional): An inverse dictionary for the LS-coupling scheme for the almost-closed-shell configuration. Default is None.
        """

        if conf[0]=='d':
            self.l = 2
        else:
            self.l = 3

        self.n = int(conf[1:])  
        self.N = int(conf[1:])  

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
        #The equation for the almost-closed-shell parameters calculation is given in 
        # Nielson - Koster in "Spectroscopic Coefficients for the p^n, d^n and f^n Configurations" (1963)
        """
        Returns the cfps for a given state, identified by its L, S quantum number and the name of the state.

        Parameters:
        L (int): Orbital angular momentum quantum number
        S (float): Spin angular momentum quantum number
        name (str): The name of the state (from terms_labels()).

        Returns:
        cfp_list (numpy.ndarray): The list of CFPs for the given state.
        """

        dic = self.dic_cfp 

        if self.closed==True:  
            cfp_list = []
            for keys in dic.keys():
                values_list = [[key, float(val)] for key, val in dic[keys].items()]
                values = sum(values_list, [])
                for i in range(0,len(values),2):
                    zz=1
                    if self.N == 2*self.l:
                        zz = (-1)**(v-1/2)
                    if values[i]==name:
                        term = self.dic_LS_inv_almost[keys][0] 
                        Sk = term[0]/2.
                        Lk = term[1]
                        N = self.N-1
                        cfp_value = values[i+1]/(zz*(-1)**(Sk+S+Lk+L-self.l-0.5)*np.sqrt((N+1)*(2*S+1)*(2*L+1)/((4*self.l+2-N)*(2*Sk+1)*(2*Lk+1))))  
                        cfp_list.append([keys,cfp_value])
        else:
            cfp_list = [[key, float(val)] for key, val in dic[name].items()]

        cfp_list = np.array(cfp_list, dtype='object')

        return cfp_list


class RME():  #RME(CFP)
    #from E. Konig "Ligand Field Energy Diagrams" (eq 2.85,2.87)
    #or from Boca, "theoretical foundations of molecular magnetism" (Ch 8, p 516)
    """
    This class is used to perform a RME calculation for a given electron configuration. 
    The electron configuration is represented by a string of the form 'l^x', where 'l' is the azimuthal quantum number 
    (represented as 'd' for l=2 and 'f' for l=3), and 'x' is the number of electrons in the configuration.

    References:
    1. E. Konig "Ligand Field Energy Diagrams" (eq 2.85,2.87)
    2. R. Boca, "theoretical foundations of molecular magnetism" (Ch 8, p 516)

    Attributes:
    v, L, S, v1, L1, S1 (float): The quantum numbers for the state.
    label1, label2 (str): The labels for the states.
    dic_LS (dict): A dictionary for the LS-coupling scheme.
    dic_LS_inv_almost (dict): An inverse dictionary for the LS-coupling scheme for the almost-closed-shell configuration.
    s (float): The spin quantum number.
    l (int): The one-electron orbital quantum number.
    n (int): The number of electrons in the almost-closed-shell configuration.
    N (int): The number of electrons in the configuration.
    cfp (CFP): A CFP object for the configuration.
    closed (bool): A flag indicating whether the configuration is almost-closed-shell or not.
    """

    def __init__(self, state, conf, dic_cfp, labels, dic_LS, dic_LS_inv_almost):
        #state = [v, L, S, v1, L1, S1]
        #WARNING: the dic_LS is the one from the corresponding configuration in case of almost closed-shell + 1
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

        self.n = int(conf[1:])  #electrons closed conf
        self.N = int(conf[1:])  #n. of electrons true
        self.cfp = CFP(conf, dic_cfp, dic_LS_inv_almost) #cfp corrected for almost-closed-shell

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


    def Uk(self, k, n=None, v=None, L=None, S=None, v1=None, L1=None, S1=None, label1=None, label2=None):   
        """
        Calculate the Uk coefficient for a given set of quantum numbers and labels.

        This method calculates the Uk coefficient, which is used in the calculation of reduced matrix elements (RMEs) in atomic physics. 
        The quantum numbers and labels for the initial and final states are either provided as arguments or taken from the RME object itself. 
        The calculation involves summing over the coefficients of fractional parentage (CFP) for the initial and final states 
        and the 6-j symbol of the associated angular momenta.

        Parameters:
        k (int): The rank of the tensor operator.

        Returns:
        Uk (float): The calculated Uk coefficient.
        """

        if n is None:
            n = self.N
        if v is None or L is None or S is None or label1 is None:
            v, L, S, label1 = self.v, self.L, self.S, self.label1
        if v1 is None or L1 is None or S1 is None or label2 is None:
            v1, L1, S1, label2 = self.v1, self.L1, self.S1, self.label2

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

        if self.closed == True:
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

        cfp_list = self.cfp.cfp(self.v, self.L, self.S, self.label1)  #cfps are already corrected for almost_closed_shells
        cfp_list1 = self.cfp.cfp(self.v1, self.L1, self.S1, self.label2)

        somma = 0
        for i in range(len(cfp_list)):
            for j in range(len(cfp_list1)):
                if cfp_list[i,0] == cfp_list1[j,0]:
                    L_parent = state_legend(cfp_list[i,0][1])
                    S_parent = (int(cfp_list[i,0][0])-1)/2
                    matrix1 = [[self.S, self.S1, 1],[self.s, self.s, S_parent]]
                    matrix2 = [[self.L, self.L1, k],[self.l, self.l, L_parent]]
                    somma += cfp_list[i,-1]*cfp_list1[j,-1]*(-1)**(L_parent+S_parent+self.S+self.L+self.s+self.l+k+1)*Wigner_coeff.sixj_symbol(matrix1)*Wigner_coeff.sixj_symbol(matrix2)

        if self.closed == True:
            return (-1)**k*pref*somma
        else:
            return pref*somma


class Hamiltonian():
    """
    This class presents the routines for the computation of the Hamiltonian matrix element for the given integral. 

    References:
    1. R. Boca, "A handbook of magnetochemical formulae". Elsevier, 2012.
    2. R. Boca, "theoretical foundations of molecular magnetism" 1999. 
    3. E. Konig "Ligand Field Energy Diagrams" 2013.
    4. C. Goerller-Walrand, K. Binnemans, Handbook of Physics & Chemistry of Rare Earths, Vol 23, Ch 155, 1996.

    Attributes:
    v, L, S, v1, L1, S1 (float): The quantum numbers for the state.
    label1, label2 (str): The labels for the states.
    dic_LS (dict): A dictionary for the LS-coupling scheme.
    dic_LS_inv_almost (dict): An inverse dictionary for the LS-coupling scheme for the almost-closed-shell configuration.
    s (float): The spin quantum number.
    l (int): The one-electron orbital quantum number.
    n (int): The number of electrons in the almost-closed-shell configuration.
    N (int): The number of electrons in the configuration.
    cfp (CFP): A CFP object for the configuration.
    closed (bool): A flag indicating whether the configuration is almost-closed-shell or not.
    """

    def __init__(self, state, labels, conf, dic_cfp=None, tables=None, dic_LS=None, dic_LS_almost=None):
        #state = [v, L, S, v1, L1, S1, J, M, J1, M1]
        #labels = [label1, label2]
        """
        Initializes the Hamiltonian object with the given state, configuration, CFP dictionary, labels, and LS-coupling scheme dictionaries.

        The state is represented by a list of the form [v, L, S, v1, L1, S1, J, MJ, J1, MJ1], where 'v', 'L', 'S', 'J' and 'MJ' are the quantum numbers 
        for the initial state and 'v1', 'L1', 'S1', 'J1' and 'MJ1' are the quantum numbers for the final state. 
        The configuration is represented by a string of the form 'nl^x', where 'n' is the principal quantum number, 
        'l' is the azimuthal quantum number (represented as 'd' for l=2 and 'f' for l=3), 
        and 'x' is the number of electrons in the configuration.

        Parameters:
        state (list): The quantum numbers for the state.
        labels (list): The labels for the states.
        conf (str): The electron configuration.
        dic_cfp (dict): A dictionary containing the coefficients of fractional parentage (CFP) for the configuration.
        dic_LS (dict): A dictionary for the LS-coupling scheme.
        dic_LS_inv_almost (dict): An inverse dictionary for the LS-coupling scheme for the almost-closed-shell configuration.
        """

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
    
    def electrostatic_int(self, basis, F0=0, F2=0, F4=0, F6=0, evaluation=True, tab_ee=None):
        """
        Computes electron repulsion integrals <self.label1|Hee|self.label2> for the given basis set.
        Equations are taken from Boca 1999, "theoretical foundations of molecular magnetism" (Ch 8, p 518) (valid only for l2 conf)
        For the d^n configurations the equations are reported in Boca2012 (p 145 eq 4.66-4.69)

        Parameters:
        basis (numpy.ndarray): The basis set for the calculation.
        F0, F2, F4, F6 (float, optional): The F0, F2, F4, and F6 are the Slater-Condon parameters in cm^{-1}. F0 is irrelevant.
        evaluation (bool, optional): A flag indicating whether the parameters are symbolic or numerical. Default is True.
        tab_ee (dict, optional): A dictionary containing the electron-electron integrals. Default is None.

        Returns:
        integral (float): The calculated electron repulsion integral
        """

        def l_Ck_l1(l, k, l1):
            return (-1)**l*np.sqrt((2*l+1)*(2*l1+1))*Wigner_coeff.threej_symbol([[l, k, l1],[0, 0, 0]])

        def Vee(label1v, label1v1):
            #works only for d^n, f^1 and f^2

            idxv = term_label.index(label1v)
            Sv, Lv, vv = term_basis[idxv]
            Sv /= 2

            idxv1 = term_label.index(label1v1)
            Sv1, Lv1, vv1 = term_basis[idxv1]
            Sv1 /= 2

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
                                somma += self.rme.Uk(k, n=self.N, v=vv, L=Lv, S=Sv, v1=v1, L1=L1, S1=S1, label1=label1v, label2=label2)*self.rme.Uk(k, n=self.N, v=vv1, L=Lv1, S=Sv1, v1=v1, L1=L1, S1=S1, label1=label1v1, label2=label2)  
                        else:

                            label2 = term_label[ii]
                            if label2[0]==label1v[0]:
                                try:
                                    somma += self.rme['U'+str(k)][label1v][label2]*self.rme['U'+str(k)][label1v1][label2]
                                except:
                                    somma += 0
                    somma *= 1/(2*Lv+1)
                    if self.v==self.v1:
                        somma -= self.n/(2*self.l+1)
                    ck *= somma
                else:
                    if self.v==self.v1:
                        ck = self.n*(self.n-1)/2
                if self.closed==True:
                    if self.v==self.v1:
                        ck -= (2*self.l+1-self.n)/(2*self.l+1)*l_Ck_l1(self.l, k, self.l)**2
                else:
                    pass
                ck_list[i] = ck

            return ck_list
           
        #=======================================================================

        if evaluation==False:
            F0, F2, F4, F6 = sympy.symbols("F0, F2, F4, F6")
            coeff = [F0, F2, F4, F6]
        else:
            coeff = [F0, F2, F4, F6]

        if tab_ee is not None:
            coeff_ee = tab_ee.copy()

        term_basis = np.array(terms_basis(self.conf[0]+str(self.n)))
        term_label = terms_labels(self.conf[0]+str(self.n))
        ck_coeff = np.zeros(len(range(0,2*self.l+1,2)))
        for i, term in enumerate(term_basis):
            if tab_ee is not None:
                ck_coeff += coeff_ee[term_label[i]][term_label[i]]*(term[0]+1)*(2*term[1]+1)
            else:
                ck_coeff += np.array(Vee(term_label[i], term_label[i]), dtype='float64')*(term[0]+1)*(2*term[1]+1)

        ck_coeff /= basis.shape[0]

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
            if label1v==label1v1:
                integral += (ck[i] - ck_coeff[i])*coeff[i]
            else:
                integral += ck[i]*coeff[i]
        
        return integral

    def SO_coupling(self, zeta, k=1, evaluation=True):
        """
        Computes SOC integrals <self.label1|Hso|self.label2>.
        Equations are taken from Konig & Kremer (Ch 2, p 11, eq 2.82)

        Parameters:
        zeta (float): The SOC parameter in cm^{-1}.
        k (int, optional): The orbital reduction factor. Default is 1.
        evaluation (bool, optional): A flag indicating whether the parameters are symbolic or numerical. Default is True.

        Returns:
        integral (float): The calculated SOC integral.
        """

        if evaluation==False:
            zeta, k = sympy.symbols("zeta, k")
        else:
            pass

        pref = zeta*k*(-1)**(self.J+self.L+self.S1)*np.sqrt(self.l*(self.l+1)*(2*self.l+1))
        coeff = Wigner_coeff.sixj_symbol([[self.L, self.L1,1],[self.S1,self.S, self.J]])
        if self.TAB == False:
            rme_V1k = self.rme.V1k()
        else:
            try:
                rme_V1k = self.rme['V11'][self.label1][self.label2]
            except:
                rme_V1k = 0

        return pref*coeff*rme_V1k

    def LF_contribution(self, dic_kq, evaluation=True):
        """
        Computes the LF/CF contribution to the Hamiltonian matrix element <self.label1|HCF/LF|self.label2> using Bkq in Wybourne formalism.
        Equations are taken from C. Goerller-Walrand, K. Binnemans, Handbook of Physics & Chemistry of Rare Earths, Vol 23, Ch 155, (1996).

        Parameters:
        dic_kq (dict): A dictionary containing the Bkq coefficients in cm^{-1}, e.g. dic_bkq = {'2':{'-1':0.0}} where k=2 and q=-1. 
        The complete (complex) coefficient corresponds to Bkq + iBk-q. 
        evaluation (bool, optional): A flag indicating whether the parameters are symbolic or numerical. Default is True.

        Returns:
        result (complex): The calculated LF/CF contribution to the Hamiltonian matrix element.
        """

        if evaluation==False:
            for k in range(2,2*self.l+1,2):
                for q in range(-k,k+1,1):
                    dic_kq[f'{k}'][f'{q}'] = sympy.Symbol(f"B{k}{q}")

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
           
            coeff2 = Wigner_coeff.sixj_symbol([[self.J, self.J1, k],[self.L1, self.L, self.S]])
           
            integral = 0
            for q in range(0,k+1):
                if q==0:
                    coeff0 = Wigner_coeff.threej_symbol([[self.J, k, self.J1],[-self.M,0,self.M1]])
                    Yk0 = (-1)**(2*self.J + self.L1 + self.S - self.M + k)*np.sqrt((2*self.J+1)*(2*self.J1+1))*ckk*coeff0*coeff2*Uk
                    integral += dic_kq[f'{k}'][f'{0}']*Yk0
                else:
                    coeffp = Wigner_coeff.threej_symbol([[self.J, k, self.J1],[-self.M,q,self.M1]])
                    coeffm = Wigner_coeff.threej_symbol([[self.J, k, self.J1],[-self.M,-q,self.M1]])
                    Ckqp = (-1)**(2*self.J + self.L1 + self.S - self.M + k)*np.sqrt((2*self.J+1)*(2*self.J1+1))*ckk*coeffp*coeff2*Uk
                    Ckqm = (-1)**(2*self.J + self.L1 + self.S - self.M + k)*np.sqrt((2*self.J+1)*(2*self.J1+1))*ckk*coeffm*coeff2*Uk
                    if eval:
                        integral += dic_kq[f'{k}'][f'{q}']*(Ckqm+(-1)**q*Ckqp) + 1j*dic_kq[f'{k}'][f'{-q}']*(Ckqm - (-1)**q*Ckqp)
                    else:
                        integral += dic_kq[f'{k}'][f'{q}']*(Ckqm+(-1)**q*Ckqp) + sympy.I*dic_kq[f'{k}'][f'{-q}']*(Ckqm - (-1)**q*Ckqp)

            result += integral

        return result

    def Zeeman(self, field=np.array([0.,0.,0.]), k=1, evaluation=True, MM=False):
        """
        Computes the Zeeman contribution to the Hamiltonian matrix element <self.label1|HZeeman|self.label2> using the magnetic field.
        Equations are taken from Boca2012, "A handbook of magnetochemical formulae" (p 588).

        Parameters:
        field (numpy.ndarray): The magnetic field vector in T.
        k (int, optional): The rank of the tensor operator. Default is 1.
        evaluation (bool, optional): A flag indicating whether the parameters are symbolic or numerical. Default is True.
        MM (bool, optional): A flag indicating whether to return the L1 and S1 contributions. Default is False.

        Returns:
        int (complex): The calculated Zeeman contribution to the Hamiltonian matrix element.
        """

        if evaluation ==False:
            Bx, By, Bz = sympy.symbols("Bx, By, Bz")
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
            return L1, S1


def Full_basis(conf):
    """
    Produces the complete basis set for a given configuration, saved in different format:
    1. basis --> array: n. microstates x [2S, L, 2J, 2M, sen (,count)] 
    2. dic_LS --> dict: '2S:L:2J:2M:sen:count': label as N. and K.
    3. basis_l --> list: 2S+1 L (J)
    4. basis_l_JM --> list: 2S+1 L (J) MJ

    Parameters:
    conf (str): The electron configuration.

    Returns:
    basis (numpy.ndarray): The complete basis set for the configuration.
    dic_LS (dict): A dictionary for the LS-coupling scheme.
    basis_l (numpy.ndarray): The labels for the states.
    basis_l_JM (numpy.ndarray): The labels for the states (with MJ).
    """

    n_el = int(conf[1:])

    if conf[0]=='d' and n_el>5:
        n_el = 10-n_el
    elif conf[0]=='f' and n_el>7:
        n_el = 14-n_el
    else:
        pass

    if conf[0]=='d':
        n_freeion_SL = [1,5,8,16,16]
    else:
        n_freeion_SL = [1,7,17,47,73,119,119]

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
        label_l = terms_labels(conf)[st]

        if len(label_l)>2:
            sen_str = label_l[-1]
        else:
            sen_str = ''
        if sen_str=='10':
            sen_str = '0'

        J1 = np.abs(2*L-TwoS)
        J2 = 2*L + TwoS

        for TwoJ in range(J1,J2+2,2):
            if TwoJ%2==0:
                J_str = str(int(TwoJ/2))
            else:
                J_str = str(int(TwoJ))+'/2'
            for TwoMJ in range(-TwoJ,TwoJ+2,2):
                if conf[0]=='f':
                    if sen_str=='':
                        count = 0
                    else:
                        if sen_str=='0':
                            count = 10
                        else:
                            count = int(sen_str)
                    lista = [TwoS, L, TwoJ, TwoMJ, sen, count]
                else:
                    lista = [TwoS, L, TwoJ, TwoMJ, sen]
                basis.append(lista)
                dic_LS[':'.join(f'{n}' for n in lista)] = label_l
                if TwoMJ%2==0:
                    MJ_str = str(int(TwoMJ/2))
                else:
                    MJ_str = str(int(TwoMJ))+'/2'
                basis_l.append(str(TwoS+1)+state_legend(str(L), inv=True)+sen_str+' ('+J_str+')')  
                basis_l_JM.append(str(TwoS+1)+state_legend(str(L), inv=True)+sen_str+' ('+J_str+') '+MJ_str)

    basis = np.array(basis)
    basis_l = np.array(basis_l)
    basis_l_JM = np.array(basis_l_JM)

    return basis, dic_LS, basis_l, basis_l_JM

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

class calculation(): 
    """
    This is the main class of the code. Here the Hamiltonian matrix is computed and diagonalized.
    Additionally, in this class the basis set can be reduced and the wavefunctions optimized.

    Attributes:
    conf (str): The electron configuration.
    l (int): The one-electron orbital quantum number.
    n (int): The number of electrons in the almost-closed-shell configuration.
    N (int): The number of electrons in the configuration.
    closed (bool): A flag indicating whether the configuration is almost-closed-shell or not.
    basis (numpy.ndarray): The complete basis set for the configuration.
    dic_LS (dict): A dictionary for the LS-coupling scheme.
    basis_l (numpy.ndarray): The labels for the states.
    basis_l_JM (numpy.ndarray): The labels for the states (with MJ).   
    microst (int): The number of microstates.
    ground (bool): A flag indicating whether the basis set includes only the ground multiplet.
    dic_cfp (dict): A dictionary containing the coefficients of fractional parentage (cfp) for the configuration.
    tables (dict): A dictionary containing the calculated RMEs.
    dic_LS_almost (dict): An inverse dictionary for the LS-coupling scheme for the almost-closed-shell configuration.
    dic_ee (dict): A dictionary containing the electron-electron integrals. 
    """


    def __init__(self, conf, ground_only=False, TAB=False, wordy=True):
        """
        Initializes the calculation object through the definition of the type of configuration 
        (almost closed shell or not), the basis set and additional dictionaries.
        Additionally, it defines whether the RMEs are computed explicitly or read from tables 
        (the computation of the RMEs will result in an increase of the computational time).

        Parameters:
        conf (str): The electron configuration.
        ground_only (bool, optional): A flag indicating whether the basis set includes only the ground multiplet. Default is False.
        TAB (bool, optional): A flag indicating whether the RMEs are calculated or read from tables. Default is False.
        wordy (bool, optional): A flag indicating whether to print the output. Default is True.
        """

        conf_list_d = ['d1','d2','d3','d4','d5','d6','d7','d8','d9']
        conf_list_f = ['f1','f2','f3','f4','f5','f6','f7','f8','f9','f10','f11','f12','f13']
        if conf not in conf_list_d and conf not in conf_list_f:
            raise ValueError('Configuration not valid')

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
        #basis --> array: n. microstates x [2S, L, 2J, 2M, sen (,count)]
        #dic_LS --> dict: '2S:L:2J:2M:sen:count': label as N. and K.
        #basis_l --> list: 2S+1 L (J)
        #basis_l_JM --> list: 2S+1 L (J) MJ
        """
        Reduces the basis set to the ground multiplet.

        Parameters:
        ground (str, optional): The ground multiplet. Default is None.

        Returns:
        basis_red (numpy.ndarray): The reduced basis set for the ground multiplet.
        dic_LS_red (dict): A dictionary for the LS-coupling scheme for the ground multiplet.
        basis_l_red (numpy.ndarray): The labels for the states for the ground multiplet (hence, only one type).
        basis_l_JM_red (numpy.ndarray): The labels for the states (with MJ) for the ground multiplet.
        """
        if ground is None:
            ground = ground_term_legend(self.conf) #conf NOT almost closed
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
        """
        Reads the RMEs from tables or computes them explicitly.

        Parameters:
        TAB (bool): A flag indicating whether the RMEs are calculated or read from tables.
        stringa (str): The corresponding configuration.
        conf (str): The electron configuration.

        Returns:
        dic_cfp (dict): A dictionary containing the coefficients of fractional parentage (cfp) for the configuration.
        tables (dict): A dictionary containing the calculated RMEs.
        dic_LS_almost (dict): An inverse dictionary for the LS-coupling scheme for the almost-closed-shell configuration.
        dic_ee (dict): A dictionary containing the electron-electron integrals.
        """
        
        if conf=='d1' or conf=='f1':
            if conf=='d1':
                tables = {'U2': {'2D': {'2D': np.float64(1.0)}},
                        'U4': {'2D': {'2D': np.float64(1.0)}},
                        'V11': {'2D': {'2D': np.float64(1.224744871391589)}}}
                dic_ee = {'2D': {'2D': [0,0,0,0]}}
                dic_LS_almost = None
                dic_ee = None
                dic_cfp = None
            else:
                tables = {'U1': {'2F': {'2F': np.float64(1.0)}},
                    'U2': {'2F': {'2F': np.float64(1.0)}},
                    'U3': {'2F': {'2F': np.float64(1.0)}},
                    'U4': {'2F': {'2F': np.float64(1.0)}},
                    'U5': {'2F': {'2F': np.float64(1.0)}},
                    'U6': {'2F': {'2F': np.float64(1.0)}},
                    'V11': {'2F': {'2F': np.float64(1.224744871391589)}}}
                dic_ee = {'2F': {'2F': [0,0,0,0]}}
                dic_LS_almost = None
                dic_ee = None
                dic_cfp = None
        else:
            if TAB==False:
                dic_cfp = cfp_from_file(stringa)
                tables = None
                if self.closed==True:
                    dic_LS_almost = Full_basis(conf[0]+str(self.n+1))[1]   #, self.n+1)[1]  #questa è la dic_LS della configurazione corrispondente per i cfp
                else:
                    dic_LS_almost = None
            else:
                dic_cfp = None
                tables = read_matrix_from_file(self.conf, self.closed)
                dic_LS_almost = None
            dic_ee = None
            if self.l==3 and self.n>2:
                dic_ee = read_ee_int(self.conf, self.closed)

        return dic_cfp, tables, dic_LS_almost, dic_ee

    def reduce_basis(self, conf, roots, dic=None, contributes=None, wordy=False):
        # reduce the basis set according to spin multiplicity
        # for dN configurations the Hund's rules are applied
        # for fN configurations the weak field - High Spin situation is considered and 
        # default parameters are taken from Ma, C. G., Brik, M. G., Li, Q. X., & Tian, Y. (2014) Journal of alloys and compounds, 599, 93-101.

        # basis --> array: n. microstates x [2S, L, 2J, 2M, sen (,count)]
        # dic_LS --> dict: '2S:L:2J:2M:sen:count': label as N. and K.
        # basis_l --> list: 2S+1 L (J)
        # basis_l_JM --> list: 2S+1 L (J) MJ
        """
        Reduces the basis set according to the spin multiplicity.
        1. for dN configurations the Hund's rules are applied
        2. for fN configurations the weak field - High Spin situation is considered and 
        The default parameters are taken from Ma, C. G., Brik, M. G., Li, Q. X., & Tian, Y. (2014) Journal of alloys and compounds, 599, 93-101.

        Parameters:
        conf (str): The electron configuration.
        roots (list, optional): A list containing the selected multiplicities as list of (sum(2*L+1), 2*S+1) pairs.
        dic (dict, optional): A dictionary containing the parameters for the free ion. Default is None.
        contributes (list, optional): A list containing the contributions to the matrix. Default is None.
        """

        print('Performing basis reduction... \n' if wordy else "", end = "")

        def state_select(ground, basis, basis_l, basis_l_JM):

            term_num = [(int(ground[0])-1), state_legend(ground[1])]#, eval(ground[ground.index('(')+1:ground.index(')')])*2]
            basis_new = []
            basis_l_red = []
            basis_l_JM_red = []
            dic_LS_red = {}
            for ii in range(basis.shape[0]):
                if np.equal(basis[ii,:len(term_num)],term_num).all():
                    # print(basis[ii,:], term_num, basis_l[ii], basis_l_JM[ii])
                    if list(basis[ii]) not in basis_new:
                        basis_new.append(list(basis[ii]))
                        lista = list(basis[ii])
                        dic_LS_red[':'.join(f'{n}' for n in lista)]=self.dic_LS[':'.join(f'{n}' for n in lista)]
                        basis_l_red.append(basis_l[ii])
                        basis_l_JM_red.append(basis_l_JM[ii])
            basis_new = np.array(basis_new)
            
            basis_update = []
            basis_l_update = []
            basis_l_JM_update = []
            for ii in range(basis.shape[0]):
                if not any((basis[ii] == x).all() for x in basis_new):
                    basis_update.append(list(basis[ii]))
                    basis_l_update.append(basis_l[ii])
                    basis_l_JM_update.append(basis_l_JM[ii])
            basis_update = np.array(basis_update)

            return basis_update, basis_l_update, basis_l_JM_update, basis_new, dic_LS_red, basis_l_red, basis_l_JM_red

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

            if dic is None:
                dic = free_ion_param_f_HF(conf)
            if contributes is None:
                matrix = self.build_matrix(['Hee'], **dic)
            else:
                matrix = self.build_matrix(contributes, **dic)

            w,v = diagonalisation(matrix)
            v = np.round(v, 16)  #numbers are saved as complex128 data type
            result = np.vstack((w,v))
            result = np.copy(result[:, result[0,:].argsort()])

            projected = projection_basis(result[1:,:], self.basis_l)  #this gives just the order of spectroscopic terms of the free ion

            max_proj = []
            for i in projected.keys():
                s_proj = projected[i]
                dlabel = list(s_proj.keys())
                dvalue = np.array(list(s_proj.values()))
                max_proj.append(dlabel[dvalue.argmax()])

            unique_max_proj = []
            for item in max_proj:
                if item not in unique_max_proj:
                    unique_max_proj.append(item)
            max_proj = unique_max_proj

        basis_update = np.copy(self.basis)
        basis_l_update = np.copy(self.basis_l)
        basis_l_JM_update = np.copy(self.basis_l_JM)
        nroots = np.zeros(len(roots))
        for i in range(len(roots)):
            for j in range(len(max_proj)):
                if nroots[i] < roots[i][0]*roots[i][1] and int(max_proj[j][0])==roots[i][1]:
                    nroots[i] += eval(max_proj[j][0])*(2*state_legend(max_proj[j][1])+1) #(eval(max_proj[j][max_proj[j].index('(')+1:max_proj[j].index(')')]))*2 +1 #eval(max_proj[j][0])*(2*state_legend(max_proj[j][1])+1) #
                    basis_update, basis_l_update, basis_l_JM_update, basis_proj, dic_LS_proj, basis_l_proj, basis_l_JM_proj = state_select(max_proj[j], basis_update, basis_l_update, basis_l_JM_update)
                    
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

        self.basis, self.dic_LS, self.basis_l, self.basis_l_JM = basis_red, dic_LS_red, basis_l_red, basis_l_JM_red

        print('Calculation on reduced set\nBasis set reduced to: '+str(self.basis.shape[0])+'\n' if wordy else "", end = "")


    #@njit
    #@cron
    def MatrixH(self, elem, F0=0, F2=0, F4=0, F6=0, zeta=0, k=1, dic_V=None, dic_bkq = None,dic_AOM = None, PCM = None, field = [0.,0.,0.], cfp_angles = None,wordy=False,
                      Orth=False, Norm=False, eig_opt=False, ground_proj=False, return_proj=False, save_label=False, save_LF=False, save_matrix=False):
        """
        This is the core of the calculation. This functions converts the different CFPs formalism in the Wybourne one, it optimizes the wavefunction
        composition, calls the function that builts the Hamiltonian and computes the wavefunctions composition.

        Parameters:
        elem (list): The contributions to the Hamiltonian matrix among: 
                    'Hee'=interelectronic repulsion, 'Hso'=spin-orbit coupling, 
                    'Hcf'=crystal-field/ligand-field contribution, 'Hz'=Zeeman splitting.
        F0 (float, optional): The F0 Slater-Condon parameter in cm^{-1}. Default is 0.
        F2 (float, optional): The F2 Slater-Condon parameter in cm^{-1}. Default is 0.
        F4 (float, optional): The F4 Slater-Condon parameter in cm^{-1}. Default is 0.
        F6 (float, optional): The F6 Slater-Condon parameter in cm^{-1}. Default is 0.
        zeta (float, optional): The spin-orbit coupling parameter in cm^{-1}. Default is 0.
        k (float, optional): The orbital reduction factor. Default is 1.
        dic_V (dict, optional): A dictionary containing one-electron ligand field matrix elements. Default is None. 
                                dic_V = {'11':0.0, '12':0, ...}, with key='ij', with 1<=i<=2*l+1 and j<=i.
        dic_bkq (dict, optional): A dictionary containing the crystal field parameters in Wybourne formalism. Default is None.
                                dic_bkq = {'k':{'q':0.0, '-q':0.0}, ...}, with k=2,4,..2*l and q=-k,-k+1,...k-1,k.
        dic_AOM (dict, optional): A dictionary containing the crystal field parameters in AOM formalism. Default is None.
                                dic_AOM = {'<ligand name>:[e_{sigma}, e_{pic}, e_{pis}, theta, phi, chi]}, where the angles are in degrees.
        PCM (list, optional): A list for the definition of the PCM for crystal field calculation. Default is None.
                              The first element of the list is the array dtype=object where each row is: '<ligand name>', x, y, z, charge.
                              The charge is expressed as fraction of electronic charge. The coordinates are in Angstrom. 
                              The metal center is assumed in (0,0,0). 
                              The second element in the list is a boolean that defines if the coordinates are expressed as cartesian (False) or spherical (True).
                              The third element is again a boolean and defines if the Sternheimer shielding parameters are used (True) or not (False).
        field (list, optional): The magnetic field vector in Tesla. Default is [0.,0.,0.].  
        cfp_angles (list, optional): A list containing the Euler angles or quaternions for the rotation of the crystal field parameters. Default is None.
        wordy (bool, optional): A flag indicating whether to print the output. Default is False.
        Orth (bool, optional): A flag indicating whether to perform the orthogonality check. Default is False.
        Norm (bool, optional): A flag indicating whether to perform the normalization check. Default is False.
        eig_opt (bool, optional): A flag indicating whether to optimize the wavefunction. Default is False.
        ground_proj (bool, optional): A flag indicating whether to wavefunctions composition is computed or not. Default is False.
        return_proj (bool, optional): A flag indicating whether to return the wavefunctions composition. Default is False.
                                      If the calculation is perfomed in the complete basis set or on a reduced one (different from just considering the ground state only with ground_only=True)
                                      the projection is a list of two dictionaries. The first without, the second with, the MJ label.
        save_label (bool, optional): A flag indicating whether to save the labels of the states. Default is False.
        save_LF (bool, optional): A flag indicating whether to save the Hamiltonian matrix (without the Zeeman contribution). Default is False.
        save_matrix (bool, optional): A flag indicating whether to save the Hamiltonian matrix. Default is False.

        Returns:
        result (numpy.ndarray): The eigenvalues and eigenvectors of the Hamiltonian matrix. The first row of the matrix are the eigenvalues.
                                Each column of the matrix is an eigenvector.
        projected (dict): The wavefunctions composition. The states are labeled as progressive numbers.
        """

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
                dic_bkq = calc_Bkq(PCM[0], self.conf, PCM[1], PCM[2])
            else:
                print('ERROR: BKQ dict is missing in MatrixH')
                exit()

        if eig_opt:
            print('\nWavefunction optimization...')
            dic_bkq, quat = self.opt_eigenfunction_minimization(self, dic_bkq, self.basis.shape[0])
            print('...done')
            print('CFP rotation quaternion: ', quat)
        elif not eig_opt and cfp_angles is not None and len(cfp_angles)==3:
            dic_bkq = rota_LF(self.l, dic_bkq, *cfp_angles)
        elif not eig_opt and cfp_angles is not None and len(cfp_angles)>3:
            dict, coeff = read_DWigner_quat()
            dic_bkq = rota_LF_quat(self.l, dic_bkq, cfp_angles, dict, coeff)
        else:
            pass

        self.par_dict = {'F0':F0, 'F2':F2, 'F4':F4, 'F6':F6, 'zeta':zeta, 'k':k, 'dic_bkq':dic_bkq, 'field':field} 
        matrix = self.build_matrix(elem, F0, F2, F4, F6, zeta, k, dic_bkq, field, save_label, save_LF, save_matrix)

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
                    pprint(projected[1])  
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
    def build_matrix(self, elem, F0=0, F2=0, F4=0, F6=0, zeta=0, k=1, dic_bkq = None, field = [0.,0.,0.], save_label=False, save_LF=False, save_matrix=False):
        """
        Builds the Hamiltonian matrix.

        Parameters:
        elem (list): The contributions to the Hamiltonian matrix among: 
                    'Hee'=interelectronic repulsion, 'Hso'=spin-orbit coupling, 
                    'Hcf'=crystal-field/ligand-field contribution, 'Hz'=Zeeman splitting.
        F0 (float, optional): The F0 Slater-Condon parameter in cm^{-1}. Default is 0.
        F2 (float, optional): The F2 Slater-Condon parameter in cm^{-1}. Default is 0.
        F4 (float, optional): The F4 Slater-Condon parameter in cm^{-1}. Default is 0.
        F6 (float, optional): The F6 Slater-Condon parameter in cm^{-1}. Default is 0.
        zeta (float, optional): The spin-orbit coupling parameter in cm^{-1}. Default is 0.
        k (float, optional): The orbital reduction factor. Default is 1.
        dic_bkq (dict, optional): A dictionary containing the crystal field parameters in Wybourne formalism. Default is None.
                                  dic_bkq = {'k':{'q':0.0, '-q':0.0}, ...}, with k=2,4,..2*l and q=-k,-k+1,...k-1,k.
        field (list, optional): The magnetic field vector in Tesla. Default is [0.,0.,0.].
        save_label (bool, optional): A flag indicating whether to save the labels of the states. Default is False.
        save_LF (bool, optional): A flag indicating whether to save the Hamiltonian matrix (without the Zeeman contribution). Default is False.
        save_matrix (bool, optional): A flag indicating whether to save the Hamiltonian matrix. Default is False.

        Returns:
        matrix (numpy.ndarray): The Hamiltonian matrix (complex).
        """

        F = F0, F2, F4, F6

        basis = self.basis
        dic_LS = self.dic_LS

        if self.closed==True:
            fac = -1.
        else:
            fac = 1.

        matrix = np.zeros((basis.shape[0],basis.shape[0]),dtype='complex128')
        if save_label:
            label_matrix = []
        if save_LF:
            LF_matrixs = np.zeros_like(matrix)

        for i in range(basis.shape[0]):
            statei = basis[i]
            Si = statei[0]/2.
            Li = statei[1]
            Ji = statei[2]/2.
            MJi = statei[3]/2.
            seni = statei[4]
            labeli = dic_LS[':'.join([f'{qq}' for qq in statei])]

            if save_label:
                label_matrix.append([Si*2,Li,Ji*2,MJi*2,seni])

            for j in range(0,i+1):
                statej = basis[j]
                Sj = statej[0]/2.
                Lj = statej[1]
                Jj = statej[2]/2.
                MJj = statej[3]/2.
                senj = statej[4]
                labelj = dic_LS[':'.join([f'{qq}' for qq in statej])]
                
                H = Hamiltonian([seni,Li,Si,senj,Lj,Sj,Ji,MJi,Jj,MJj], [labeli,labelj], self.conf, self.dic_cfp, self.tables, dic_LS, self.dic_LS_almost)  

                if 'Hee' in elem:
                    if Ji==Jj and MJi==MJj:
                        if Li == Lj and Si == Sj:
                            if self.l==3:
                                Hee = H.electrostatic_int(basis, *F, tab_ee = self.dic_ee)
                            else:
                                Hee = H.electrostatic_int(basis, *F)
                            matrix[i,j] += Hee
                            if save_LF:
                                LF_matrixs[i,j] += Hee
                                if i!=j:
                                    LF_matrixs[j,i] += np.conj(Hee)
                            if i != j:
                                matrix[j,i] += Hee

                if 'Hso' in elem:
                    if Ji==Jj and MJi==MJj:
                        Hso = fac*H.SO_coupling(zeta, k)
                        matrix[i,j] += Hso
                        if save_LF:
                            LF_matrixs[i,j] += Hso
                            if i!=j:
                                LF_matrixs[j,i] += np.conj(Hso)
                        if i != j:
                            matrix[j,i] += Hso

                if 'Hcf' in elem:
                    if Si==Sj:
                        Hcf = fac*H.LF_contribution(dic_bkq)
                        matrix[i,j] += Hcf
                        if save_LF:
                            LF_matrixs[i,j] += Hcf
                            if i!=j:
                                LF_matrixs[j,i] += np.conj(Hcf)
                        if i != j:
                            matrix[j,i] += np.conj(Hcf)

                if 'Hz' in elem:
                    if Li==Lj and Si==Sj and seni==senj:
                        Hz = H.Zeeman(field, k)
                        matrix[i,j] += Hz
                        if i != j:
                            matrix[j,i] += np.conj(Hz)

                else:
                    pass

        if save_label:
            np.savetxt('matrix_label.txt', np.array(label_matrix))
        if save_LF:
            np.save('matrix_LF', LF_matrixs, allow_pickle=True, fix_imports=False)
        if save_matrix:
            np.save('matrix', matrix, allow_pickle=True, fix_imports=False)

        return matrix

    @staticmethod
    def opt_eigenfunction_minimization(calc, dic_Bkq, shape):
        """
        Wavefunction optimization algorithm. It's based on the maximization of a single component of the ground state wavefunction
        using the dual anneling minimization algorithm implemented in scipy.

        Parameters:
        calc (object): The instance of the class.
        dic_Bkq (dict): A dictionary containing the crystal field parameters in Wybourne Default is None.
                        dic_bkq = {'k':{'q':0.0, '-q':0.0}, ...}, with k=2,4,..2*l and q=-k,-k+1,...k-1,k.
        shape (int): The shape of the ground state wavefunction.

        Returns:
        dic_rot (dict): A dictionary containing the rotated crystal field parameters.
        quat (list): A list containing the quaternion for the rotation of the crystal field parameters.
        """

        from scipy.optimize import dual_annealing
        from scipy.optimize import least_squares
        import time

        def use_nja_(calc, dic, wordy=False):
            result = calc.MatrixH(['Hcf'], **dic, eig_opt=False, wordy=wordy)
            ground_state = np.abs(result[1:,0])**2
            or_ground_state = np.sort(ground_state)
            return or_ground_state

        cycle = 0
        def target(quat, x):
            nonlocal cycle
            cycle += 1
            dic_rot = rota_LF_quat(3, dic_Bkq, quat, dict, coeff)
            dic = {'dic_bkq': dic_rot}
            ground_comp = use_nja_(calc, dic, wordy=False)*100
            print(f'Opt cycle: {cycle}, target: {np.sum((ground_comp-x)**2):.4e}, highest comp: {ground_comp[-1]:.2f}', end='\r')
            return np.sum((ground_comp-x)**2) 
        
        dict, coeff = read_DWigner_quat()

        x = np.zeros(shape)
        x[-1] = 100

        start_time = time.time()
        bounds = [(-1, 1), (-1, 1), (-1, 1), (-1, 1)]
        bounds_ls = ([-1, -1, -1, -1], [1, 1, 1, 1])
        result_global = dual_annealing(target, bounds, args=(x,), accept=-5.0, maxiter=500, maxfun=1000, seed=666, no_local_search=True)
        result = least_squares(target, result_global.x, bounds=bounds_ls, args=(x, ))
        end_time = time.time()
        elapsed_time = end_time - start_time
        print('\n'+f"Opt time: {elapsed_time:.2f} seconds"+'\n')

        dic_rot = rota_LF_quat(calc.l, dic_Bkq, result.x, dict, coeff)

        R = result.x
        quat = [R[-1], R[0], R[1], R[2]]

        return dic_rot, quat

    @staticmethod
    def opt_eigenfunction_grid(calc, dic_Bkq, shape):
        """
        Wavefunction optimization algorithm. It's based on a grid search, using REPULSION angles (evenly sampled on a sphere), 
        read from crystdat.py.

        Parameters:
        calc (object): The instance of the class.
        dic_Bkq (dict): A dictionary containing the crystal field parameters in Wybourne Default is None.
                        dic_bkq = {'k':{'q':0.0, '-q':0.0}, ...}, with k=2,4,..2*l and q=-k,-k+1,...k-1,k.
        shape (int): The shape of the ground state wavefunction.

        Returns:
        dic_rot (dict): A dictionary containing the rotated crystal field parameters.
        quat (list): A list containing the quaternion for the rotation of the crystal field parameters.
        """

        import time

        def use_nja_(calc, dic, wordy=False):
            result = calc.MatrixH(['Hcf'], **dic, eig_opt=False, wordy=wordy)
            ground_state = np.abs(result[1:,0])**2
            or_ground_state = np.sort(ground_state)
            return or_ground_state
        
        dict, coeff = read_DWigner_quat()

        x = np.zeros(shape)
        x[-1] = 1

        rep_cryst = np.array(crystdat.rep678_cryst)  #change here the name of the grid to select smaller or bigger grids from cristdat.py

        target_list = []

        start_time = time.time()

        for i in range(rep_cryst.shape[0]):

            angles = rep_cryst[i,:]
            a = angles[0]
            b = angles[1]

            r = scipy.spatial.transform.Rotation.from_euler('ZYZ', [0,b,a], degrees=True)
            R = r.as_quat()
            quat = [R[-1], R[0], R[1], R[2]]

            dic_rot = rota_LF_quat(calc.l, dic_Bkq, quat, dict, coeff)
            dic = {'dic_bkq': dic_rot}
            
            ground_comp = use_nja_(calc, dic, wordy=False)

            target = np.sum((np.abs(ground_comp-x))**2)
            target_list.append(target)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Opt time: {elapsed_time:.2f} seconds"+'\n')

        target_list = np.array(target_list)

        min_index = np.argmin(target_list)
        angles = rep_cryst[min_index,:]
        a = angles[0]
        b = angles[1]
        r = scipy.spatial.transform.Rotation.from_euler('ZYZ', [0,b,a], degrees=True)
        R = r.as_quat()
        quat = [R[-1], R[0], R[1], R[2]]
        dic_rot = rota_LF_quat(calc.l, dic_Bkq, quat, dict, coeff)

        return dic_rot, quat

#======================= DETACHED FUNCTIONS for MAGNETIC PROPERTIES ==============================

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
        seni = statei[4]

        for j in range(0,i+1):
            statej = basis[j]
            Sj = statej[0]/2.
            Lj = statej[1]
            Jj = statej[2]/2.
            Mj = statej[3]/2.
            senj = statej[4]

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
    result = None

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
                result = a[j, i,...]
        if norm(a[i, i,...] - a[i - 1, i - 1,...]) >= SAFE * err:
            # print('safe exit', a[i, i], a[i - 1, i - 1])
            return result, err

    return result, err

@jit(complex128[:,:](float64[:],float64[:,:],complex128[:,:]))
def add_Zeeman(field_vec, basis, LF_matrix):

    matrix = np.zeros((basis.shape[0],basis.shape[0]),dtype=complex128)
    for i in range(basis.shape[0]):
        statei = basis[i]
        Si = statei[0]/2.
        Li = statei[1]
        Ji = statei[2]/2.
        Mi = statei[3]/2.
        seni = statei[4]

        for j in range(0,i+1):
            statej = basis[j]
            Sj = statej[0]/2.
            Lj = statej[1]
            Jj = statej[2]/2.
            Mj = statej[3]/2.
            senj = statej[4]

            matrix[i,j] += LF_matrix[i,j]
            if i!=j:
                matrix[j,i] += LF_matrix[j,i]

            if Li==Lj and Si==Sj and seni==senj:
                integral, kL, gS = Zeeman(Li,Si,Ji,Mi,Lj,Sj,Jj,Mj,field_vec)
                matrix[i,j] += integral[0]
                if i!=j:
                    matrix[j,i] += np.conj(integral[0])

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

        den = 0
        num = 0
        for ii in range(len(E)):

            mu_single = np.dot(np.conj(np.ascontiguousarray(result[1:, ii]).T), np.dot(mu_matrix[kk, ...], np.ascontiguousarray(result[1:, ii])))

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

def effGval(levels, mu_matrix, v_matrix): 
    #v_matrix = result[1:,:]

    levels = np.array(levels)  
                                                            
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
                    gk[kk,ii,jj] = np.dot(np.conj(v_matrix[:,ii]).T, np.dot(mu_matrix[kk,...],v_matrix[:,jj]))

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

def calc_torque(B0, T, LF_matrix, basis, plane='xz', step=3.1415/40, figure=False, show_fig=False):

    NA = 6.022e23
    muB = 0.4668517532494337
    Jconv = 1.9865e-23

    def missing_axis(plane_label):

        axes = ['x', 'y', 'z']
        
        missing_axis_label = next(axis for axis in axes if axis not in plane_label)
        missing_axis_index = axes.index(missing_axis_label)
        B_direction = axes.index(plane_label[0])
        
        return missing_axis_label, missing_axis_index, B_direction

    angles = np.arange(0, 2 * np.pi + np.pi/180, step)
    tau_vecs = []
    mu_matrix = mag_moment(basis)

    str_ax, idx_ax, idx_B = missing_axis(plane)
    vec_field = np.array([0.0, 0.0, 0.0])

    # Calculate tau_vec for each B0
    for idx in range(len(B0)):
        vec_field[idx_B] = B0[idx]
        tau_vec = np.zeros((len(angles), 3))
        for i in range(len(angles)):
            rot_vec = rotate_vectors(vec_field, angles[i], str_ax)
            M = M_vector(rot_vec, mu_matrix, LF_matrix, basis, T)
            tau_vec[i, :] = np.cross(M, rot_vec)*NA*muB*Jconv
        tau_vecs.append(tau_vec[:, idx_ax])
    
    if figure:
        fig, ax1 = plt.subplots()
        fig.set_size_inches(7, 4)
        
        line1, = ax1.plot(angles * 180 / np.pi, tau_vecs[0], 'b-', label=f'{B0[0]} T')
        ax1.set_xlabel('Angle / degrees')
        ax1.set_ylabel(r'$\tau$ / Nm/mol', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        
        axes = [ax1]
        lines = [line1]
        colors = ['r', 'g', 'c', 'm', 'y', 'k']
        for idx in range(1, len(B0)):
            ax = ax1.twinx()
            ax.spines['right'].set_position(('outward', 60 * (idx - 1)))  # Adjust the position
            line, = ax.plot(angles * 180 / np.pi, tau_vecs[idx], color=colors[idx % len(colors)], label=f'{B0[idx]} T')
            ax.set_ylabel(r'$\tau$ / Nm/mol', color=colors[idx % len(colors)])
            ax.tick_params(axis='y', labelcolor=colors[idx % len(colors)])
            axes.append(ax)
            lines.append(line)
        
        labs = [l.get_label() for l in lines]
        ax1.legend(lines, labs, loc='upper right')
        
        ax1.axhline(0, color='black', lw=0.5)
        ax1.set_xlim(0, 360)
        ax1.set_xticks(np.arange(0, 361, 45))
        for i in np.arange(0, 361, 45):
            ax1.axvline(i, color='gray', lw=0.5, alpha=0.5)
        
        ax1.set_title('B direction: ' + plane[0] + ', torque component: ' + str_ax)
        plt.tight_layout()
        if isinstance(figure, str):
            plt.savefig(figure, dpi=600)
        if show_fig:
            plt.show()

    return angles, np.array(tau_vecs)

def susceptibility_field(fields, temp, basis, LF_matrix, delta=0.001):

    kB = 1.380649e-23
    mu0 = 1.25663706212e-06
    muB = 0.4668517532494337

    M_list = np.zeros(fields.shape[0])
    susc_list = np.zeros(fields.shape[0])
    mu = np.zeros((fields.shape[0], basis.shape[0], 3), dtype='complex128')

    mu_matrix = mag_moment(basis)

    for i in range(fields.shape[0]): 

        matrix = add_Zeeman(fields[i], basis, LF_matrix)
        result = from_matrix_to_result_copy(matrix)
        E = (result[0,:].real-min(result[0,:].real)) #* 1.9865e-23   

        den = 0
        num = 0
        for ii in range(len(E)):
            for kk in range(3):
                mu_single = np.dot(np.conj(result[1:,ii]).T, np.dot(mu_matrix[kk,...],result[1:,ii]))
                mu[i,ii,0] += np.copy(mu_single.real)

            num += np.real(mu[i,ii,0]*np.exp(-E[ii]/(kB/1.9865e-23*temp)))
            den += np.real(np.exp(-E[ii]/(kB/1.9865e-23*temp)))
        E_av = num/den

        B_inc = fields[i]/np.linalg.norm(fields[i])*delta
        matrix = add_Zeeman(fields[i]+B_inc, basis, LF_matrix)
        result = from_matrix_to_result_copy(matrix)
        E = (result[0,:].real-min(result[0,:].real)) 
        den = 0
        num = 0
        for ii in range(len(E)):
            for kk in range(3):
                mu_single = np.dot(np.conj(result[1:,ii]).T, np.dot(mu_matrix[kk,...],result[1:,ii]))
                mu[i,ii,1] += np.copy(mu_single.real)

            num += np.real(mu[i,ii,1]*np.exp(-E[ii]/(kB/1.9865e-23*temp)))
            den += np.real(np.exp(-E[ii]/(kB/1.9865e-23*temp)))
        E_av_inc = num/den

        B_inc = fields[i]/np.linalg.norm(fields[i])*delta
        matrix = add_Zeeman(fields[i]-B_inc, basis, LF_matrix)
        result = from_matrix_to_result_copy(matrix)
        result = from_matrix_to_result(matrix)
        E = (result[0,:].real-min(result[0,:].real)) 
        den = 0
        num = 0
        for ii in range(len(E)):
            for kk in range(3):
                mu_single = np.dot(np.conj(result[1:,ii]).T, np.dot(mu_matrix[kk,...],result[1:,ii]))
                mu[i,ii,2] += np.copy(mu_single.real)

            num += np.real(mu[i,ii,2]*np.exp(-E[ii]/(kB/1.9865e-23*temp)))
            den += np.real(np.exp(-E[ii]/(kB/1.9865e-23*temp)))
        E_av_incm = num/den

        M_list[i] = E_av*muB*1.9865e-23
        susc_list[i] = (E_av_inc - E_av_incm)/(2*delta)*mu0*muB*1.9865e-23

    return M_list, susc_list

#======================= MAGNETIC PROPERTIES ==============================

class Magnetics():
    """
    Class for magnetic properties calculations.

    References:
    1. R. Boca, "theoretical foundations of molecular magnetism" 1999.

    Attributes:
    result (np.array): eigenvalue and eigenvectors of the Hamiltonian matrix.
    par_dict (dict): dictionary of parameters used in the calculation.
    calc (Calculation): instance of the Calculation class.
    basis (np.array): basis of the Hamiltonian matrix.
    Hterms (list): The contributions to the Hamiltonian matrix among: 
                    'Hee'=interelectronic repulsion, 'Hso'=spin-orbit coupling, 
                    'Hcf'=crystal-field/ligand-field contribution, 'Hz'=Zeeman splitting.
    kB (float): Boltzmann constant.
    mu0 (float): vacuum permeability.
    muB (float): Bohr magneton.
    """

    def __init__(self, calc, contributes, par):
        """
        Initializes the Magnetics class, computing the 0 field results.

        Parameters:
        calc (Calculation): instance of the Calculation class.
        contributes (list): The contributions to the Hamiltonian matrix among: 
                            'Hee'=interelectronic repulsion, 'Hso'=spin-orbit coupling,
                            'Hcf'=crystal-field/ligand-field contribution, 'Hz'=Zeeman splitting.
        par (dict): dictionary of parameters used in the calculation.
        """

        self.result = calc.MatrixH(contributes, **par, wordy=True)
        self.par_dict = calc.par_dict

        self.calc = calc
        self.basis = self.calc.basis
        self.Hterms = contributes

        self.kB = scipy.constants.k
        self.mu0 = scipy.constants.mu_0
        self.muB = scipy.constants.physical_constants['Bohr magneton'][0]/1.9865e-23

    def mag_moment(self, k=1, evaluation=True):
        #costruction of magnetic moment matrix as kL+geS
        #y component is divided by i (imaginary unit)
        """
        Computes the magnetic moment matrix.

        Parameters:
        k (int, optional): orbital reduction factor.
        evaluation (bool, optional): A flag indicating whether the parameters are symbolic or numerical. Default is True.
        
        Returns:
        matrix (numpy.ndarray): the magnetic moment matrix (complex).
        """

        if not evaluation:
            matrix = np.zeros((3, self.basis.shape[0],self.basis.shape[0]),dtype=object)
        else:
            matrix = np.zeros((3, self.basis.shape[0],self.basis.shape[0]),dtype='complex128')
        for i in range(self.basis.shape[0]):
            statei = self.basis[i]
            Si = statei[0]/2.
            Li = statei[1]
            Ji = statei[2]/2.
            MJi = statei[3]/2.
            seni = statei[4]
            labeli = self.calc.dic_LS[':'.join([f'{qq}' for qq in statei])]
            for j in range(0,i+1):
                statej = self.basis[j]
                Sj = statej[0]/2.
                Lj = statej[1]
                Jj = statej[2]/2.
                MJj = statej[3]/2.
                senj = statej[4]
                labelj = self.calc.dic_LS[':'.join([f'{qq}' for qq in statej])]
                H = Hamiltonian([seni,Li,Si,senj,Lj,Sj,Ji,MJi,Jj,MJj], [labeli,labelj], self.calc.conf, self.calc.dic_cfp, self.calc.tables, self.calc.dic_LS, self.calc.dic_LS_almost)  #self.conf è quella del main
                if Li==Lj and Si==Sj and seni==senj:

                    kL, gS = H.Zeeman(k=k, evaluation=evaluation, MM=True)

                    # x,y,z  -->  -1,0,+1
                    matrix[0,i,j] += (kL[0]+gS[0] - (kL[2]+gS[2]))*1/(np.sqrt(2))
                    matrix[1,i,j] += (kL[0]+gS[0] + kL[2]+gS[2])*1j/(np.sqrt(2))
                    matrix[2,i,j] += kL[1]+gS[1]

                    if i!=j:
                        for kk in range(3):
                            matrix[kk,j,i] += np.conj(matrix[kk,i,j])

        matrix = - matrix #the minus sign is because mu = - kL - 2S

        if evaluation:
            matrix = np.round(matrix, 16)

        return matrix

    #@staticmethod
    def effGval(self, levels, v_matrix=None):
        """
        Computes the effective g-matrix for the Kramers doublets specified.

        Parameters:
        levels (list): list of the Kramers doublet levels, e.g. [(1,2), (3,4)].
        v_matrix (numpy.ndarray, optional): eigenvectors of the Hamiltonian matrix. Default is None.

        Returns:
        np.sqrt(2*w) (numpy.ndarray): eigenvalues of the g-matrix.
        v (numpy.ndarray): eigenvectors of the g-matrix.
        """ 

        if v_matrix is None:
            par = self.par_dict
            matrix = self.calc.build_matrix(self.Hterms, **par) 
            result = from_matrix_to_result(matrix)
            v_matrix = np.copy(result[1:,:])
        else:
            pass

        levels = np.array(levels)

        mu_matrix = self.mag_moment(1)  
                                                             
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
                        gk[kk,ii,jj] = np.dot(np.conj(v_matrix[:,ii]).T, np.dot(mu_matrix[kk,...],v_matrix[:,jj]))

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


    def susceptibility_field(self, fields, temp, delta=0.001, wordy=False):
        """
        Computes the scalar magnetization and magnetic susceptibility fields, for a set of field vectors (evenly sampled on a sphere) at a certain temperature.

        Parameters:
        fields (numpy.ndarray): field vectors (shape = N x 3, in T).
        temp (float): temperature in K.
        delta (float, optional): differentiation step in T. Default is 0.001.
        wordy (bool, optional): A flag indicating whether to print the results. Default is False.

        Returns:
        M_list (numpy.ndarray): scalar magnetization field (SI unit).
        susc_list (numpy.ndarray): scalar magnetic susceptibility field (SI unit).
        """

        par = self.par_dict

        M_list = np.zeros(fields.shape[0])
        susc_list = np.zeros(fields.shape[0])
        mu = np.zeros((fields.shape[0], self.basis.shape[0], 3), dtype='complex128')

        mu_matrix = self.mag_moment(1)

        for i in range(fields.shape[0]): 
            print(str(i)+' FIELD: '+str(fields[i])+'\n' if wordy else "", end = "")

            par['field'] = fields[i] 
            weights = fields[i]/np.linalg.norm(fields[i])
            matrix = self.calc.build_matrix(self.Hterms, **par) 
            result = from_matrix_to_result(matrix)
            E = (result[0,:].real-min(result[0,:].real)) 

            den = 0
            num = 0
            for ii in range(len(E)):
                for kk in range(3):
                    mu_single = np.dot(np.conj(result[1:,ii]).T, np.dot(mu_matrix[kk,...],result[1:,ii]))
                    if np.abs(np.copy(mu_single).imag)<1e-9:
                        mu[i,ii,0] += np.copy(mu_single.real)*weights[kk]
                    else:
                        print('complex')

                num += np.real(mu[i,ii,0]*np.exp(-E[ii]/(self.kB/1.9865e-23*temp)))
                den += np.real(np.exp(-E[ii]/(self.kB/1.9865e-23*temp)))
            E_av = num/den

            B_inc = fields[i]/np.linalg.norm(fields[i])*delta
            par['field'] = fields[i]+B_inc
            matrix = self.calc.build_matrix(self.Hterms, **par)  
            result = from_matrix_to_result(matrix)
            E = (result[0,:].real-min(result[0,:].real)) 
            den = 0
            num = 0
            for ii in range(len(E)):
                for kk in range(3):
                    mu_single = np.dot(np.conj(result[1:,ii]).T, np.dot(mu_matrix[kk,...],result[1:,ii]))
                    if np.abs(np.copy(mu_single).imag)<1e-9:
                        mu[i,ii,1] += np.copy(mu_single.real)*weights[kk]
                    else:
                        print('complex')

                num += np.real(mu[i,ii,1]*np.exp(-E[ii]/(self.kB/1.9865e-23*temp)))
                den += np.real(np.exp(-E[ii]/(self.kB/1.9865e-23*temp)))
            E_av_inc = num/den

            B_inc = fields[i]/np.linalg.norm(fields[i])*delta
            par['field'] = fields[i]-B_inc
            matrix = self.calc.build_matrix(self.Hterms, **par)  
            result = from_matrix_to_result(matrix)
            E = (result[0,:].real-min(result[0,:].real)) 
            den = 0
            num = 0
            for ii in range(len(E)):
                for kk in range(3):
                    mu_single = np.dot(np.conj(result[1:,ii]).T, np.dot(mu_matrix[kk,...],result[1:,ii]))
                    if np.abs(np.copy(mu_single).imag)<1e-9:
                        mu[i,ii,2] += np.copy(mu_single.real)*weights[kk]
                    else:
                        print('complex')

                num += np.real(mu[i,ii,2]*np.exp(-E[ii]/(self.kB/1.9865e-23*temp)))
                den += np.real(np.exp(-E[ii]/(self.kB/1.9865e-23*temp)))
            E_av_incm = num/den

            M_list[i] = E_av*self.muB*1.9865e-23
            susc_list[i] = (E_av_inc - E_av_incm)/(2*delta)*self.mu0*self.muB*1.9865e-23

            print('M '+str(M_list[i])+'\n' if wordy else "", end = "")
            print('chi '+str(susc_list[i])+'\n' if wordy else "", end = "")

        return M_list, susc_list
    

    def susceptibility_B_copy(self, fields, temp, delta=0.001, wordy=False):
        """
        Computes the magnetic susceptibility tensor as the derivative of the magnetization vector in x,y,z,
        for a set of field vectors (evenly sampled on a sphere). The values are then averaged among the 
        different directions of the magnetic field vector, hence it is generally enough to pass a single field vector 
        as e.g. np.array([[0,0,1]]) (fields needs to be bidimensional).

        Parameters:
        fields (numpy.ndarray): field vectors (shape = N x 3, in T). 
        temp (float): temperature in K.
        delta (float, optional): differentiation step in T. Default is 0.001.
        wordy (bool, optional): A flag indicating whether to print the results. Default is False.

        Returns:
        chi_tensor (numpy.ndarray): magnetic susceptibility tensor (SI unit).
        Mav (numpy.ndarray): magnetization vector (SI unit).
        """

        def M_vector_in(field_vec):
            mu = np.zeros((self.basis.shape[0], 3), dtype='complex128')
            par['field'] = field_vec
            matrix = self.calc.build_matrix(self.Hterms, **par)  #hamiltonian matrix for a field vector B
            result = from_matrix_to_result(matrix)
            E = (result[0,:].real-min(result[0,:].real)) #* 1.9865e-23
            E -= min(E)

            M = np.zeros(3)

            for kk in range(3):
                den = 0
                num = 0
                for ij in range(len(E)):
                    mu_single = np.dot(np.conj(result[1:,ij]).T, np.dot(mu_matrix[kk,...],result[1:,ij]))   # <i|mu_kk|i> for i in range N and for kk=x,y,z
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
        chi = np.zeros((fields.shape[0], 3, 3))
        mu_matrix = self.mag_moment(k)  #computes the magnetic moments matrix <i|kL + geS|j>

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

            for kki in range(3):
                for kkj in range(3):
                    chi[i,kki,kkj] = ((M_inc[kkj,kki]-M_dec[kkj,kki])/(2*delta))*self.mu0*self.muB*1.9865e-23

            print('M (BM)\n'+str(M)+'\n' if wordy else "", end = "")
            print('M_inc (BM)\n'+str(M_inc)+'\n' if wordy else "", end = "")
            print('chi (m3)\n'+str(chi[i,:,:])+'\n' if wordy else "", end = "")

        chi_tensor = np.zeros((3,3))
        chi_tensor = np.sum(chi, axis=0)/fields.shape[0]   #since I've calculated the tensor for different directions of the magnetic field vector, here there is the average calculation
        Mav = np.sum(Mag_vector, axis=0)/fields.shape[0]    #same for magnetization

        return chi_tensor, Mav*self.muB*1.9865e-23

    # @staticmethod
    # @cron
    ### COPY OF THE NUMBA FUNCTIONS ###
    def dfridr(self, func, x, h, idxi, shape, fargs):
        """
        returns the derivative of the function at a point x by Ridders' method of polynomial extrapolation. 
        The value h is input as an estimated initial stepsize. It has to be bigger than the one you would pass to a standard numerical differentiation method. 
        The stepsize is decreased by CON at each iteeration. Max size of tableau is set by NTAB.
        See NJA-CFS documentation for references.

        Parameters:
        func (function): function to differentiate.
        x (numpy.ndarray): differentiation variable.
        h (float): step size.
        idxi (int): index of the differentiation variable.
        shape (tuple): shape of the output.
        fargs (tuple): additional arguments for the function.

        Returns:
        result (numpy.ndarray): result of the differentiation.
        err (float): error estimation of the differentiation.
        """

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
        result = None

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
                    result = a[j, i,...]
            if norm(a[i, i,...] - a[i - 1, i - 1,...]) >= SAFE * err:
                return result, err

        return result, err

    #@cron
    ### COPY OF THE NUMBA FUNCTIONS ###
    def susceptibility_B_ord1(self, fields, temp, basis, LF_matrix, delta=1.):
        """
        Computes the magnetic susceptibility tensor as the derivative of the magnetization vector in x,y,z,
        for a set of field vectors (evenly sampled on a sphere). The values are then averaged among the 
        different directions of the magnetic field vector, hence it is generally enough to pass a single field vector
        as e.g. np.array([[0,0,1]]) (fields needs to be bidimensional).
        The differentiation method is based on Ridders' method of polynomial extrapolation. 
        The value h is input as an estimated initial stepsize (it has to be larger that the one you would pass to a standard differentiation procedure).
        An estimate of the error is also computed.

        Parameters:
        fields (numpy.ndarray): field vectors (shape = N x 3, in T).
        temp (float): temperature in K.
        basis (numpy.ndarray): basis of the Hamiltonian matrix.
        LF_matrix (numpy.ndarray): 0 field Hamiltonian matrix.
        delta (float, optional): differentiation step in T. Default is 1.

        Returns:
        chi_tensor (numpy.ndarray): magnetic susceptibility tensor (SI unit).
        err_tensor (numpy.ndarray): error estimation for the computation of the magnetic susceptibility tensor (SI unit).
        """

        mu0 = 1.25663706212e-06
        muB = 0.4668517532494337
        Jconv = 1.9865e-23

        mu_matrix = mag_moment(basis)  #complex128[:,:,:]
        chi = np.zeros((fields.shape[0], 3, 3), dtype='float64')
        err = np.zeros_like(chi)
        if len(fields.shape)<2:
            fields = np.array([fields])
        for i in range(fields.shape[0]):
            for idx in range(3):
                chi_comp, err_comp = self.dfridr(M_vector, fields[i], delta, idx, chi.shape[1:], fargs=(mu_matrix, LF_matrix, basis, temp))
                chi[i,idx] = chi_comp * mu0*muB*Jconv
                err[i,idx] = np.ones(chi_comp.shape)*err_comp * mu0*muB*Jconv

        chi_tensor = np.zeros((3,3))
        chi_tensor = np.sum(chi, axis=0)/fields.shape[0]
        err_tensor = np.sum(err, axis=0)/fields.shape[0]

        return chi_tensor, err_tensor


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

def plot_energy_levels(eigenvalues, ax=None, color='b', label=None, tolerance=0.05, offset=0, delta=0):
    """
    Plot crystal field energy levels from a splitting matrix with a horizontal offset.
    
    Parameters:
    - splitting_matrix: 2D numpy array, the crystal field splitting matrix to analyze.
    - ax: matplotlib axis, optional. If provided, plots on this axis.
    - color: str, color of the levels (e.g., 'b' for blue).
    - label: str, optional label to identify different crystal fields.
    - tolerance: float, maximum difference to group levels as degenerate.
    - offset: float, horizontal offset to place the energy levels of this crystal field.
    """
    
    # Sort and group nearly degenerate energy levels
    unique_levels = []
    grouped_levels = []

    for ev in sorted(eigenvalues):
        if not unique_levels or abs(ev - unique_levels[-1]) > tolerance:
            unique_levels.append(ev)
            grouped_levels.append([ev])
        else:
            grouped_levels[-1].append(ev)
    
    # Create the plot if no axis was provided
    if ax is None:
        fig, ax = plt.subplots()
    
    # Set up offsets for degenerate states
    x_offset = 0.15  # Small offset within a group of degenerate levels

    # Plot each level with offsets for degenerate states, shifted by the main offset
    for level_group in grouped_levels:
        energy = level_group[0]  # Common energy level for the degenerate group
        n_deg = len(level_group)  # Number of degenerate states
        x_positions = np.linspace(-x_offset * (n_deg - 1) / 2, x_offset * (n_deg - 1) / 2, n_deg) + offset

        # Plot each degenerate level with the specified color
        for x in x_positions:
            ax.hlines(y=energy, xmin=x - 0.05 +delta, xmax=x + 0.05+delta, color=color, linewidth=2)

    # Add label if provided
    if label:
        ax.text(offset + 0.22+delta, max(unique_levels) + 0.2, label, ha='center', color=color)

    # Update plot appearance
    ax.set_ylabel("Energy Levels")
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    
    ax.get_xaxis().set_visible(False)

    return ax  # Return axis to allow further modifications

def fig_tensor_rep_1(tensor, n_points=40):

    u = np.linspace(0, 2 * np.pi, n_points)
    v = np.linspace(0, np.pi, n_points)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    points = np.stack((x.ravel(), y.ravel(), z.ravel()), axis=1)  # Shape: (N, 3)
    points = points / np.linalg.norm(points, axis=1)[:, np.newaxis]

    N = points.shape[0]
    magnitudes = np.zeros(N)

    # Compute M(n) for each vector n
    for idx in range(N):
        n = points[idx]  # Extract the vector
        magnitudes[idx] = n.T @ tensor @ n  # Matrix-vector operations

    x_scaled = (x.ravel() * magnitudes).reshape(x.shape)
    y_scaled = (y.ravel() * magnitudes).reshape(y.shape)
    z_scaled = (z.ravel() * magnitudes).reshape(z.shape)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    norm = plt.Normalize(magnitudes.min(), magnitudes.max())
    colors = plt.cm.turbo(norm(magnitudes).reshape(x.shape))

    ax.plot_surface(x_scaled, y_scaled, z_scaled, facecolors=colors, edgecolor='k', alpha=1, linewidth=0.5)

    # Set transparent background
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    # Hide the axes and labels
    # ax.set_axis_off()

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    x_limits = [x_scaled.min(), x_scaled.max()]
    y_limits = [y_scaled.min(), y_scaled.max()]
    z_limits = [z_scaled.min(), z_scaled.max()]

    all_limits = np.array([x_limits, y_limits, z_limits])
    widest_range = all_limits.max() - all_limits.min()

    ax.set_xlim(all_limits.min(), all_limits.min() + widest_range)
    ax.set_ylim(all_limits.min(), all_limits.min() + widest_range)
    ax.set_zlim(all_limits.min(), all_limits.min() + widest_range)

    mappable = plt.cm.ScalarMappable(cmap='turbo', norm=norm)
    mappable.set_array(magnitudes)
    plt.colorbar(mappable, ax=ax, shrink=0.5, aspect=8)

    plt.show()

def fig_tensor_rep_2(tensor, n_points=100):

    from mpl_toolkits.mplot3d import Axes3D

    eigenvalues, eigenvectors = np.linalg.eigh(tensor)

    u = np.linspace(0, 2 * np.pi, n_points)
    v = np.linspace(0, np.pi, n_points)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    sphere = np.stack((x.ravel(), y.ravel(), z.ravel()))

    ellipsoid = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ sphere
    x_ellipsoid, y_ellipsoid, z_ellipsoid = ellipsoid.reshape(3, *x.shape)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(x_ellipsoid, y_ellipsoid, z_ellipsoid, color='b', alpha=0.6, edgecolor='k')

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    x_limits = [x_ellipsoid.min(), x_ellipsoid.max()]
    y_limits = [y_ellipsoid.min(), y_ellipsoid.max()]
    z_limits = [z_ellipsoid.min(), z_ellipsoid.max()]

    all_limits = np.array([x_limits, y_limits, z_limits])
    widest_range = all_limits.max() - all_limits.min()

    ax.set_xlim(all_limits.min(), all_limits.min() + widest_range)
    ax.set_ylim(all_limits.min(), all_limits.min() + widest_range)
    ax.set_zlim(all_limits.min(), all_limits.min() + widest_range)

    plt.show()

def fig_susc_field(conf, dic_Bkq, temp=2., n_points=20, delta=0.01):

    from mpl_toolkits.mplot3d import Axes3D

    def use_nja_(conf, dic_Bkq, field_vecs, wordy=False):

        calc = calculation(conf, ground_only=True, TAB=True, wordy=wordy)
        
        dic = {}
        dic['dic_bkq'] = dic_Bkq

        Magn = Magnetics(calc, ['Hcf','Hz'], dic)
        _, susc_field = Magn.susceptibility_field(fields=field_vecs, temp=temp, delta = delta, wordy=wordy)

        return susc_field

    u = np.linspace(0, 2 * np.pi, n_points)
    v = np.linspace(0, np.pi, n_points)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    points = np.stack((x.ravel(), y.ravel(), z.ravel()), axis=1)  # Shape: (N, 3)

    points = points / np.linalg.norm(points, axis=1)[:, np.newaxis]

    magnitudes = use_nja_(conf, dic_Bkq, points)

    #subtract the average
    #magnitudes -= np.average(magnitudes)

    x_scaled = (x.ravel() * magnitudes).reshape(x.shape)
    y_scaled = (y.ravel() * magnitudes).reshape(y.shape)
    z_scaled = (z.ravel() * magnitudes).reshape(z.shape)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x_scaled, y_scaled, z_scaled, cmap='viridis', edgecolor='k', alpha=0.8)

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    x_limits = [x_scaled.min(), x_scaled.max()]
    y_limits = [y_scaled.min(), y_scaled.max()]
    z_limits = [z_scaled.min(), z_scaled.max()]

    all_limits = np.array([x_limits, y_limits, z_limits])
    widest_range = all_limits.max() - all_limits.min()

    ax.set_xlim(all_limits.min(), all_limits.min() + widest_range)
    ax.set_ylim(all_limits.min(), all_limits.min() + widest_range)
    ax.set_zlim(all_limits.min(), all_limits.min() + widest_range)

    plt.show()

    return magnitudes, points

def fig_rep_magnfield(Mvec, xyz, data=None):

    import matplotlib
    import matplotlib.cm as cm

    def cmap2list(cmap, N=10, start=0, end=1):
        x = np.linspace(start, end, N)
        colors = cmap(x)
        return colors

    ### plot magnetization surface
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d', facecolor='white')
    Mvec = np.reshape(Mvec,(len(Mvec),1))
    data_p = np.hstack((Mvec,xyz))
    data_p = data_p[data_p[:, 0].argsort()]
    vectors = data_p[:,1:]
    norm_or = data_p[:,0]
    
    colorlist = cmap2list(cm.coolwarm, N=vectors.shape[0])

    ax.scatter(vectors[:,0],vectors[:,1],vectors[:,2], color=colorlist)

    box = plt.axes([0.75, 0.3, 0.02, 0.45])
    norm = matplotlib.colors.Normalize(norm_or[0],norm_or[-1])
    plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.coolwarm), cax=box)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    if data is not None:
        for i in range(data.shape[0]):
            vector = data[i,1:-1]
            ax.plot([0.,vector[0]],[0.,vector[1]],[0.,vector[2]],'--',lw=0.2,c='k')
            if data[i,0] in color_atoms().keys():
                ax.scatter(vector[0],vector[1],vector[2],'o',c = color_atoms()[data[i,0]],lw=3)
            else:
                ax.scatter(vector[0],vector[1],vector[2],'o',c = color_atoms()['_'],lw=3)
            ax.text(vector[0]+0.4*np.sign(vector[0]),vector[1]+0.4*np.sign(vector[1]),vector[2]+0.4*np.sign(vector[2]),data[i,-1], size=8)

    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_zlim(-4, 4)

    # Remove the grid
    ax.grid(False)

    plt.show()

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
    #the angles must be in radiants

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
    # conversion from Akq<rk> of stevens to Bkq of wybourne (o Lkq according to: https://www2.cpfs.mpg.de/~rotter/homepage_mcphase/manual/node130.html)

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
    # calc Stevens coefficients, B^q_k, from data in the point charge model

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
    #equation 9a, 9b, 9c from Software package SIMPRE - Revisited (M. Karbowiak and C. Rudowicz)

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
            pref = plm(k,np.abs(q))*(4*np.pi/(2*k+1))
            somma = 0
            for i in range(data.shape[0]):
                sphharmp = scipy.special.sph_harm(np.abs(q), k, coord_sph[i,2],coord_sph[i,1])  #sph_harm(k,q,coord_sph[i,1], coord_sph[i,2])/np.sqrt(4*np.pi/(2*k+1))
                sphharmm = scipy.special.sph_harm(-np.abs(q), k, coord_sph[i,2],coord_sph[i,1])  #sph_harm(k,-q,coord_sph[i,1], coord_sph[i,2])/np.sqrt(4*np.pi/(2*k+1))
                r = coord_sph[i,0]*au_conv[1]
                if q==0:
                    somma += pref*(data[i,-1]*au_conv[0]/r**(k+1))*scipy.special.sph_harm(0, k, coord_sph[i,2],coord_sph[i,1]).real
                elif q>0:
                    somma += pref*(data[i,-1]*au_conv[0]/r**(k+1))*(1/np.sqrt(2))*(sphharmm + (-1)**q*sphharmp).real
                elif q<0:
                    somma += (1j*pref*(data[i,-1]*au_conv[0]/r**(k+1))*(1/np.sqrt(2))*(sphharmm - (-1)**q*sphharmp)).real

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

    Bkq = {}
    for k in range(2,2*l+1,2):
        Bkq[f'{k}'] = {}
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
                    somma += (-1)**q*prefac*(data[i,-1]*au_conv[0]/r**(k+1))*sphharm
            if sth_param == True:
                value = (1-sigma_k(str(k), conf))*somma*r_val
            else:
                value = somma*r_val
            if np.abs(value)<=bin:
                Bkq[f'{k}'].update({f'{q}': 0})
            else:
                if q!=0:
                    Bkq[f'{k}'].update({f'{q}': value.real})
                    Bkq[f'{k}'].update({f'{-q}': value.imag})
                else:
                    Bkq[f'{k}'].update({f'{q}': value})

    return Bkq


def rota_LF(l, dic_Bkq, A=0, B=0, C=0):
    #rotation of Bkq with Euler angles
    #the angles must be in radiants
    #convention of D-matrix: Z-Y-Z

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
    #rotation of Bkq with quaternions

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
    #rotation matrix zyz (active rotation)

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
    w,v = diagonalisation(matrix)  
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

    prime = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    if conf[0]=='d':
        file = open('tables/cfp_d_conf.txt', 'r').readlines()
    elif conf[0]=='f':
        file = open('tables/cfp_f_conf.txt', 'r').readlines()[1:] #skip the first line
    else:
        raise ValueError('conf must be dn or fn')
    
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
                splitline[-1] = splitline[-1].split()
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

def read_matrix_from_file(conf_print, closed_shell=False):

    file = open('tables/tables_'+conf_print[0]+'conf.txt').readlines()
    
    dizionario = {}
    conf = None
    for i,line in enumerate(file):
        if 'CONFIGURATION' in line:
            line = line.strip()
            conf = line.split()[-1]
            dizionario[conf] = {}
        elif 'MATRIX' in line:
            line = line.strip()
            key1 = line.split()[-1]
            dizionario[conf][key1] = {}
        elif '#' in line:
            pass
        else:
            splitline = line.split()
            if splitline[0] not in dizionario[conf][key1].keys():
                dizionario[conf][key1][splitline[0]] = {}
            dizionario[conf][key1][splitline[0]][splitline[1]] = float(splitline[-1])

            L = state_legend(splitline[0][1])
            S = int(splitline[0][0])
            L1 = state_legend(splitline[1][1])
            S1 = int(splitline[1][0])

            if key1=='V11':
                if splitline[1] not in dizionario[conf][key1].keys():
                    dizionario[conf][key1][splitline[1]] = {}
                dizionario[conf][key1][splitline[1]][splitline[0]] = float(splitline[-1])*(-1)**(L-L1-S/2+S1/2)
            else:
                if splitline[1] not in dizionario[conf][key1].keys():
                    dizionario[conf][key1][splitline[1]] = {}
                dizionario[conf][key1][splitline[1]][splitline[0]] = float(splitline[-1])*(-1)**(L-L1)

    if closed_shell==False:
        conf_str = conf_print
    else:
        conf_n = almost_closed_shells(conf_print)
        conf_str = conf[0]+str(conf_n)

    return dizionario[conf_str]

def read_ee_int(conf, closed_shell):

    if closed_shell==False:
        conf_str = conf
    else:
        conf_n = almost_closed_shells(conf)
        conf_str = conf[0]+str(conf_n)

    file = open('tables/dic_ee_values.txt', 'r').readlines()
    dic_ee_loaded = {}
    conf = None
    for line in file:
        if line=='#':
            conf = None
        else:
            splitline = line.split()
            if len(splitline)==1:
                conf = splitline[0].strip()
                dic_ee_loaded[conf] = {}
            else:
                if splitline[0] not in dic_ee_loaded[conf].keys():
                    dic_ee_loaded[conf][splitline[0]] = {}
               
                if splitline[0]!=splitline[1]:
                    if splitline[1] not in dic_ee_loaded[conf].keys():
                        dic_ee_loaded[conf][splitline[1]] = {}
                    dic_ee_loaded[conf][splitline[1]].update({splitline[0]:np.array(splitline[2:], dtype=float)})
                dic_ee_loaded[conf][splitline[0]].update({splitline[1]:np.array(splitline[2:], dtype=float)})

   
    return dic_ee_loaded[conf_str]

def calc_ee_int(conf, closed_shell=False):

    if closed_shell==False:
        conf_str = conf
    else:
        conf_n = almost_closed_shells(conf)
        conf_str = conf[0]+str(conf_n)

    import sympy

    def calc_dic_ek(conf, dic_ek_out):

        def omegaUL(U,L):
            gU = {99:0,
                10:6,
                11:12,
                20:14,
                21:21,
                30:24,
                22:30,
                31:32,
                40:36}
            return 1/2*state_legend(L)*(state_legend(L)+1) - gU[U]

        S_list = [7/2, 3, 5/2, 5/2, 2, 2, 3/2, 3/2, 1, 1/2]
        v_list = [7, 6, 5, 7, 4, 6, 5, 7, 6, 7]
        Sv_list = [(S_list[i],v_list[i]) for i in range(len(S_list))]

        AB_label = {'f5': {'2F':[6,7], '2H':[6,7], '2I':[4,5], '2K':[4,5]},
                    'f6': {'3F':[8,9], '3H':[8,9], '3I':[5,6], '3K':[5,6], '1G':[7,8], '1I':[6,7], '1L':[3,4]},
                    'f7': {'2F':[6,7], '2G':[9,0], '2H':[6,7], '2I':[4,5,8,9], '2K':[4,5], '2L':[4,5]}}

        #W, U, U1
        x_g = {200:
                {20:{20:2}},  #questo è back calcolato da 1D1:1D1 sulla base delle tabelle di ryley
               210:
                {11:
                {21:12*np.sqrt(455)},
                20:
                {20:-6/7, 21:6*np.sqrt(66)/7},
                21:
                {11:12*np.sqrt(455), 20:6*np.sqrt(66)/7, 21:[3/7, 0]}},
               211:
                {10:
                {30:-20*np.sqrt(143)},
                11:
                {21:10*np.sqrt(182), 30:10},
                20:
                {20:-8/7, 21:4*np.sqrt(33)/7, 30:4*np.sqrt(3)},
                21:
                {11:10*np.sqrt(182), 20:4*np.sqrt(33)/7, 21:[4/7, 3], 30:2},
                30:
                {10:-20*np.sqrt(143), 11:10, 20:4*np.sqrt(3), 21:2, 30:2}},
               220:
                {20:
                {20:3/14, 21:3*np.sqrt(55)/7, 22:-3*np.sqrt(5/28)},
                21:
                {20:3*np.sqrt(55)/7, 21:[-6/7,-3], 22:3/np.sqrt(7)},
                22:
                {20:-3*np.sqrt(5/28), 21:3/np.sqrt(7), 22:3/2}},
               221:
                {10:
                {30:5*np.sqrt(143), 31:-15*np.sqrt(429)},
                11:
                {21:14*np.sqrt(910/11), 30:2*np.sqrt(10), 31:2*np.sqrt(39)/11},
                20:
                {20:2/7, 21:-10*np.sqrt(6)/7, 30:np.sqrt(3), 31:9*np.sqrt(3/7)},
                21:
                {11:14*np.sqrt(910/11), 20:-10*np.sqrt(6)/7, 21:[-1/7,12/11], 30: 5*np.sqrt(2/11), 31:3*np.sqrt(2)/11},
                30:
                {10: 5*np.sqrt(143), 11:2*np.sqrt(10), 20: np.sqrt(3), 21:5*np.sqrt(2/11), 30:-1/2, 31:3/(2*np.sqrt(11))},
                31:
                {10:-15*np.sqrt(429), 11:2*np.sqrt(39)/11, 20:9*np.sqrt(3/7), 21:3*np.sqrt(2)/11, 30:3/(2*np.sqrt(11)), 31:1/22}},
               222:
                {99:
                {40:-30*np.sqrt(143)},
                10:
                {30:-3*np.sqrt(1430), 40:9*np.sqrt(1430)},
                20:
                {20:6/11, 30:-3*np.sqrt(42/11), 40:9*np.sqrt(2)/11},
                30:
                {10:-3*np.sqrt(1430), 20:-3*np.sqrt(42/11), 30:-3, 40:1/np.sqrt(11)},
                40:
                {99:-30*np.sqrt(143), 10:9*np.sqrt(1430), 20:9*np.sqrt(2)/11, 30:1/np.sqrt(11), 40:3/11}}}
        #U1 L U
        chi_L = {20:
                {"D":{20:143},
                "G":{20:-130},
                "I":{20:35}},
                21:
                {"H":{11:1, 20:0, 21:[49,-75]},
                "D":{20:-39*np.sqrt(2), 21:[377, 13]},
                "F":{21:[455, -65]},
                "G":{20:4*np.sqrt(65), 21:[-561, 55]},
                "K":{21:[-315, 133]},
                "L":{21:[245, -75]}},
                30:
                {"P":{11:-13*np.sqrt(11), 30:-52},
                "F":{10:1, 21:12*np.sqrt(195), 30:38},
                "G":{20:-13*np.sqrt(5), 21:8*np.sqrt(143), 30:-52},
                "H":{11:np.sqrt(39), 21:11*np.sqrt(42), 30:88},
                "I":{20:30, 30:25},
                "K":{21:-4*np.sqrt(17), 30:-94},
                "M":{30:25}},
                22:
                {"S":{22:260},
                "D":{20:3*np.sqrt(429), 21:45*np.sqrt(78), 22:-25},
                "G":{20:-38*np.sqrt(65), 21:12*np.sqrt(11), 22:94},
                "H":{21:-12*np.sqrt(546), 22:104},
                "I":{20:21*np.sqrt(85), 22:-181},
                "L":{21:-8*np.sqrt(665), 22:-36},
                "N":{22:40}},
                31:
                {"P":{11:11*np.sqrt(330), 30:76*np.sqrt(143), 31:-6644},
                "D":{20:-8*np.sqrt(78), 21:-60*np.sqrt(39/7), 31:4792},
                "F":{10:[0,1], 21:[-312*np.sqrt(5), 12*np.sqrt(715)], 30:[-48*np.sqrt(39), -98*np.sqrt(33)], 31:[4420, -902, 336*np.sqrt(143)]},
                "G":{20:5*np.sqrt(65), 21:2024/np.sqrt(7), 30:20*np.sqrt(1001), 31:-2684},
                "H":{11:[11*np.sqrt(85), -25*np.sqrt(77)], 21:[31*np.sqrt(1309/3), 103*np.sqrt(5/3)], 30:[-20*np.sqrt(374), -44*np.sqrt(70)], 31:[-2024,2680,-48*np.sqrt(6545)]},
                "I":{20:[10*np.sqrt(21),0], 30:[-57*np.sqrt(33), 18*np.sqrt(1122)], 31:[-12661/5,17336/5,-3366*np.sqrt(34)/5]},
                "K":{21:[-52*np.sqrt(323/23), -336*np.sqrt(66/23)], 30:[-494*np.sqrt(19/23), 73*np.sqrt(1122/23)], 31:[123506/23, -85096/23, 144*np.sqrt(21318)/23]},
                "L":{21:-24*np.sqrt(190), 31:-4712},
                "M":{30:-21*np.sqrt(385), 31:-473},
                "N":{31:1672},
                "O":{31:220}},
                40:
                {"S":{99:1, 40:-1408},
                "D":{20:-88*np.sqrt(13), 40:-44},
                "F":{10:1, 30:90*np.sqrt(11), 40:1078},
                "G":{20:[53*np.sqrt(715/27), 7*np.sqrt(15470/27)], 30:[-16*np.sqrt(1001), 64*np.sqrt(442)], 40:[-16720/9, 10942/9, -34*np.sqrt(2618)/9]},
                "H":{30:-72*np.sqrt(462), 40:-704},
                "I":{20:[34*np.sqrt(1045/31), -12*np.sqrt(1785/31)], 30:[-9*np.sqrt(21945/31), 756*np.sqrt(85/31)], 40:[-2453/31, 36088/31, 60*np.sqrt(74613)/31]},
                "K":{30:-84*np.sqrt(33), 40:-132},
                "L":{40:[-4268/31, 11770/31, 924*np.sqrt(1995)/31]},
                "M":{30:-99*np.sqrt(15), 40:-1067},
                "N":{40:528},
                "Q":{40:22}}}
        #U1 L U
        phi_L = {11:
                {"P":{11:-11},
                "H":{11:3}},
                20:
                {"D":{20:-11},
                "G":{20:-4},
                "I":{20:7}},
                21:
                {"D":{20:6*np.sqrt(2), 21:-57},
                "F":{10:1, 21:63},
                "G":{20:np.sqrt(65), 21:55},
                "H":{21:-105},
                "K":{21:-14},
                "L":{21:42}},
                30:
                {"P":{11:np.sqrt(11), 30:83},
                "F":{21:np.sqrt(195), 30:-72},
                "G":{20:2*np.sqrt(5), 21:-np.sqrt(143), 30:20},
                "H":{11:np.sqrt(39), 21:-2*np.sqrt(42), 30:-15},
                "I":{20:3, 30:42},
                "K":{21:-4*np.sqrt(17), 30:-28},
                "M":{30:6}},
                22:
                {"S":{99:1, 22:144},
                "D":{20:3*np.sqrt(429), 22:69},
                "G":{20:4*np.sqrt(65), 22:-148},
                "H":{22:72},
                "I":{20:3*np.sqrt(85), 22:39},
                "L":{22:-96},
                "N":{22:56}},
                31:
                {"P":{11:np.sqrt(330), 30:17*np.sqrt(143), 31:209},
                "D":{21:12*np.sqrt(273), 31:-200},
                "F":{10:[1,0], 21:[-36*np.sqrt(5), -3*np.sqrt(715)], 30:[-16*np.sqrt(39), 24*np.sqrt(33)], 31:[624, -616, -80*np.sqrt(143)]},
                "G":{21:11*np.sqrt(7), 30:4*np.sqrt(1001), 31:836},
                "H":{11:[np.sqrt(85), np.sqrt(77)], 21:[-2*np.sqrt(1309/3), -74*np.sqrt(5/3)], 30:[np.sqrt(187/2), 31*np.sqrt(35/2)], 31:[-1353/2, 703/2, -5*np.sqrt(6545)/2]},
                "I":{30:[30*np.sqrt(33), 0], 31:[-2662/5, -88/5, 528*np.sqrt(34)/5]},
                "K":{21:[-28*np.sqrt(323/23), 42*np.sqrt(66/23)], 30:[4*np.sqrt(437),0], 31:[6652/23, -5456/23, 96*np.sqrt(21318)/23]},
                "L":{21:-6*np.sqrt(190), 31:-464},
                "M":{30:-6*np.sqrt(385), 31:814},
                "N":{31:-616},
                "O":{31:352}},
                40:
                {"S":{22:2*np.sqrt(2145)},
                "D":{20:11*np.sqrt(13), 21:-6*np.sqrt(26), 22:9*np.sqrt(33)},
                "F":{21:3*np.sqrt(455)},
                "G":{20:[-4*np.sqrt(715/27),np.sqrt(15470/27)], 21:[-131*np.sqrt(11/27), 17*np.sqrt(238/27)], 22:[-4*np.sqrt(11/27), -17*np.sqrt(238/27)]},
                "H":{21:-12*np.sqrt(21), 22:3*np.sqrt(286)},
                "I":{20:[7*np.sqrt(1045/31),3*np.sqrt(1785/31)], 22:[3*np.sqrt(3553/31),75*np.sqrt(21/31)]},
                "K":{21:-2*np.sqrt(119)},
                "L":{21:[22*np.sqrt(105/31), -84*np.sqrt(19/31)], 22:[4*np.sqrt(627/31), 12*np.sqrt(385/31)]},
                "N":{22:-np.sqrt(2530)}}}
        #conf v:(2*S+1):U v1:(2*S1+1):U1
        y_g = {"f3":
                {"1:2:10":{"3:2:21":-6*np.sqrt(22)},
                "3:2:11":{"3:2:11":2},
                "3:2:20":{"3:2:20":10/7, "3:2:21":2*np.sqrt(66)/7},
                "3:2:21":{"3:2:20":2*np.sqrt(66)/7, "3:2:21":2/7}},
                "f4":
                {"2:3:10":{"4:3:21":-12*np.sqrt(33/5)},
                "2:3:11":{"4:3:11":6/5, "4:3:30":6},
                "4:3:10":{"4:3:21":8*np.sqrt(11/15)},
                "4:3:11":{"4:3:11":29/15,"4:3:30":-1/3},
                "4:3:20":{"4:3:20":6/7, "4:3:21":-8*np.sqrt(11/147), "4:3:30":4/np.sqrt(3)},
                "4:3:21":{"4:3:10":8*np.sqrt(11/15), "4:3:20":-8*np.sqrt(11/147), "4:3:21":-2/21, "4:3:30":-4/3},
                "4:3:30":{"4:3:11":-1/3, "4:3:20":4/np.sqrt(3), "4:3:21":-4/3, "4:3:30":1/3},
                "0:1:99":{"4:1:22":-12*np.sqrt(22)},
                "2:1:20":{"4:1:20":3*np.sqrt(3/175), "4:1:21":-4*np.sqrt(33/35), "4:1:22":-np.sqrt(3/5)},
                "4:1:20":{"4:1:20":221/140, "4:1:21":8*np.sqrt(11/245), "4:1:22":-np.sqrt(7/80)},
                "4:1:21":{"4:1:20":8*np.sqrt(11/245), "4:1:21":2/7},
                "4:1:22":{"4:1:20":-np.sqrt(7/80), "4:1:22":1/4}},
                "f5":
                {"3:4:10":{"5:4:21":9*np.sqrt(11)},
                "3:4:20":{"5:4:20":3/np.sqrt(7), "5:4:21":np.sqrt(33/7), "5:4:30":-2*np.sqrt(21)},
                "5:4:10":{"5:4:21":-np.sqrt(55/3)},
                "5:4:11":{"5:4:11":-1/3, "5:4:30":-5/3},
                "5:4:20":{"5:4:20":5/7, "5:4:21":5*np.sqrt(11/147), "5:4:30":2/np.sqrt(3)},
                "5:4:21":{"5:4:10":-np.sqrt(55/3), "5:4:20":5*np.sqrt(11/147), "5:4:21":-4/21, "5:4:30":-2/3},
                "5:4:30":{"5:4:11":-5/3, "5:4:20":2/np.sqrt(3), "5:4:21":-2/3, "5:4:30":-1/3},
                "1:2:10":{"5:2:21":36/np.sqrt(5), "5:2:31":-36*np.sqrt(2)},
                "3:2:11":{"5:2:11":3/np.sqrt(2), "5:2:30":3*np.sqrt(5)/2, "5:2:31":-np.sqrt(39/8)},
                "3:2:20":{"5:2:20":3/7, "5:2:21":-11*np.sqrt(6)/7, "5:2:30":-4*np.sqrt(3)},
                "3:2:21":{"5:2:10":3*np.sqrt(33/10), "5:2:20":-3*np.sqrt(33/98), "5:2:21":3/(7*np.sqrt(11)), "5:2:30":-3/(2*np.sqrt(2)), "5:2:31":3/(2*np.sqrt(22))},
                "5:2:10":{"5:2:21":43/np.sqrt(30), "5:2:31":4*np.sqrt(3)},
                "5:2:11":{"5:2:11":-5/6, "5:2:30":-5*np.sqrt(5/72), "5:2:31":-np.sqrt(13/48)},
                "5:2:20":{"5:2:20":11/7, "5:2:21":-11/(7*np.sqrt(6)), "5:2:30":4/np.sqrt(3)},
                "5:2:21":{"5:2:10":43/np.sqrt(30), "5:2:20":-11/(7*np.sqrt(6)), "5:2:21":25/231, "5:2:30":29/(6*np.sqrt(22)), "5:2:31":1/(22*np.sqrt(2))},
                "5:2:30":{"5:2:11":-5*np.sqrt(5/72), "5:2:20":4/np.sqrt(3), "5:2:21":29/(6*np.sqrt(22)), "5:2:30":-1/12, "5:2:31":1/(4*np.sqrt(11))},
                "5:2:31":{"5:2:10":4*np.sqrt(3), "5:2:11":-np.sqrt(13/48), "5:2:21":1/(22*np.sqrt(2)), "5:2:30":1/(4*np.sqrt(11)), "5:2:31":1/44}},
                "f6":
                {"4:5:10":{"6:5:21":-6*np.sqrt(11)},
                "4:5:20":{"6:5:20":-2*np.sqrt(2/7), "6:5:21":2*np.sqrt(33/7)},
                "2:3:10":{"6:3:21":-48*np.sqrt(2/5), "6:3:31":-36},
                "2:3:11":{"6:3:11":np.sqrt(6/5), "6:3:30":np.sqrt(3), "6:3:31":3*np.sqrt(13/10)},
                "4:3:10":{"6:3:21":46/np.sqrt(15), "6:3:31":-8*np.sqrt(6)},
                "4:3:11":{"6:3:11":11/(3*np.sqrt(5)), "6:3:30":-19/(3*np.sqrt(2)), "6:3:31":np.sqrt(13/60)},
                "4:3:20":{"6:3:20":-6*np.sqrt(2)/7, "6:3:21":-22/(7*np.sqrt(3)), "6:3:30":8*np.sqrt(2/3)},
                "4:3:21":{"6:3:10":-np.sqrt(110/3), "6:3:20":np.sqrt(22/147), "6:3:21":-16/(21*np.sqrt(11)), "6:3:30":5/(3*np.sqrt(2)), "6:3:31":1/np.sqrt(22)},
                "4:3:30":{"6:3:11":-np.sqrt(5)/3, "6:3:20":4*np.sqrt(2/3), "6:3:21":4/(3*np.sqrt(11)), "6:3:30":1/(3*np.sqrt(2)), "6:3:31":-1/np.sqrt(22)},
                "2:1:20":{"6:1:20":6/np.sqrt(55), "6:1:30":2*np.sqrt(42/5), "6:1:40":6*np.sqrt(2/55)},
                "4:1:20":{"6:1:20":-61/np.sqrt(770), "6:1:30":8*np.sqrt(3/5), "6:1:40":-6/np.sqrt(385)},
                "4:1:21":{"6:1:10":3*np.sqrt(22), "6:1:20":np.sqrt(2/7), "6:1:30":-np.sqrt(3), "6:1:40":1/np.sqrt(7)},
                "4:1:22":{"6:1:99":-4*np.sqrt(33/5), "6:1:20":-1/np.sqrt(22), "6:1:40":2/np.sqrt(11)}},
                "f7":
                {"3:4:99":{"7:4:22":-12*np.sqrt(11)},
                "3:4:10":{"7:4:21":6*np.sqrt(33)},
                "3:4:20":{"7:4:20":-np.sqrt(5/7), "7:4:21":2*np.sqrt(11/7), "7:4:22":-1},
                "3:2:11":{"7:2:30":2*np.sqrt(10)},
                "3:2:20":{"7:2:20":-16/np.sqrt(77), "7:2:30":-2*np.sqrt(6), "7:2:40":6*np.sqrt(2/77)},
                "3:2:21":{"7:2:10":-np.sqrt(66), "7:2:20":np.sqrt(6/7), "7:2:30":1, "7:2:40":np.sqrt(3/7)}}}

        def calc_ek(conf, label1, label2, S, L, dic_ek): 

            #dic_ek [label1:n:v:U:2S+1:L][label2:n:v1:U1:2S+1:L] 
            v,W,U = terms_labels_symm(conf)[label1]
            v1,W1,U1 = terms_labels_symm(conf)[label2]

            ek_coeff = np.zeros(4)
            n = int(conf[1:]) 
            if label1==label2:
                ek_coeff[0] = n*(n-1)/2 
                ek_coeff[1] = 9*(n - v)/2 + 1/4*v*(v+2) - S*(S+1)

            if v==v1:

                if v!=2*S and W==W1 and int(str(W)[0])==2:
                    factor1 = x_g.get(W, {}).get(U, {}).get(U1)
                    if factor1 is None: 
                        factor1 = x_g.get(W, {}).get(U1, {}).get(U)
                    factor2 = chi_L.get(U1, {}).get(L, {}).get(U)
                    if factor2 is None:  
                        factor2 = chi_L.get(U, {}).get(L, {}).get(U1)
                    if factor1 is not None and factor2 is not None:
                        if (isinstance(factor1, float) or isinstance(factor1, int)) and (isinstance(factor2, float) or isinstance(factor2, int)):
                            ek_coeff[2] = factor1*factor2
                        elif (isinstance(factor1, float) or isinstance(factor1, int)) and isinstance(factor2, list):
                            idx = 1
                            try:       
                                if AB_label[conf][label1[:2]].index(int(label1[-1]))%2==0:
                                    idx = 0
                            except KeyError:
                                if AB_label[conf][label2[:2]].index(int(label2[-1]))%2==0:
                                    idx = 0
                            if label1!=label2 and v==v1 and U==U1 and W==W1:
                                idx = -1
                            ek_coeff[2] = factor2[idx]*factor1
                        else:
                            ek_coeff[2] = np.sum(np.array(factor1)*np.array(factor2))
                        
                    if (S,v) in Sv_list:
                        ek_coeff[2] *= -1
                    
                if n==2*S and U1==U:
                    ek_coeff[3] = -3*omegaUL(U,L)+omegaUL(U,L)  #I'll remove it later
                if v==n and (v==6 or v==7):
                    ek_coeff[3] = 0
                key1 = ':'.join([label1,str(v),str(v),str(U),str(int(2*S+1)),L])
                key2 = ':'.join([label2,str(v),str(v),str(U1),str(int(2*S+1)),L])
                key1_cut = ':'.join([label1[:2],str(v),str(v),str(U),str(int(2*S+1)),L])
                key2_cut = ':'.join([label2[:2],str(v),str(v),str(U1),str(int(2*S+1)),L])

                if dic_ek['f'+str(v)].get(key1, {}).get(key2) is not None:
                    prev_int = dic_ek['f'+str(v)][key1][key2][3].copy()
                    if U1==U and label1==label2:
                        prev_int += omegaUL(U,L)
                    if n==v+2:
                        ek_coeff[3] = prev_int*(1-v)/(7-v)
                    elif n==v+4:
                        ek_coeff[3] = prev_int*(-4)/(7-v)
                elif dic_ek['f'+str(v)].get(key1_cut, {}).get(key2_cut) is not None:
                    prev_int = dic_ek['f'+str(v)][key1_cut][key2_cut][3].copy()
                    if U1==U and label1==label2:
                        prev_int += omegaUL(U,L)
                    if n==v+2:
                        ek_coeff[3] = prev_int*(1-v)/(7-v)
                    elif n==v+4:
                        ek_coeff[3] = prev_int*(-4)/(7-v)
                        
            else:

                if n==5 and ((v,v1)==(1,3) or (v1,v)==(1,3)) and (2*S+1)==2:
                    key1 = ':'.join([label1,'3',str(v),str(U),str(int(2*S+1)),L])
                    key2 = ':'.join([label2,'3',str(v1),str(U1),str(int(2*S+1)),L])
                    if dic_ek['f3'].get(key1, {}).get(key2) is not None:
                        ek_coeff[3] = dic_ek['f3'][key1][key2][3]*np.sqrt(2/5)
                    elif dic_ek['f3'].get(key2, {}).get(key1) is not None:
                        ek_coeff[3] = dic_ek['f3'][key2][key1][3]*np.sqrt(2/5)
                if n==6 and ((v,v1)==(0,4) or (v,v1)==(4,0)) and (2*S+1)==1:
                    key1 = ':'.join([label1,'4',str(v),str(U),str(int(2*S+1)),L])
                    key2 = ':'.join([label2,'4',str(v1),str(U1),str(int(2*S+1)),L])
                    if dic_ek['f4'].get(key1, {}).get(key2) is not None:
                        ek_coeff[3] = dic_ek['f4'][key1][key2][3]*np.sqrt(9/5)
                    elif dic_ek['f4'].get(key2, {}).get(key1) is not None:
                        ek_coeff[3] = dic_ek['f4'][key2][key1][3]*np.sqrt(9/5)
                if n==6 and ((v,v1)==(2,4) or (v,v1)==(4,2)):
                    key1 = ':'.join([label1,'4',str(v),str(U),str(int(2*S+1)),L])
                    key2 = ':'.join([label2,'4',str(v1),str(U1),str(int(2*S+1)),L])
                    if dic_ek['f4'].get(key1, {}).get(key2) is not None:
                        ek_coeff[3] = dic_ek['f4'][key1][key2][3]*np.sqrt(1/6)
                    elif dic_ek['f4'].get(key2, {}).get(key1) is not None:
                        ek_coeff[3] = dic_ek['f4'][key2][key1][3]*np.sqrt(1/6)
                if n==7 and ((v,v1)==(1,5) or (v,v1)==(5,1)) and (2*S+1)==2:
                    key1 = ':'.join([label1,'5',str(v),str(U),str(int(2*S+1)),L])
                    key2 = ':'.join([label2,'5',str(v1),str(U1),str(int(2*S+1)),L])
                    if dic_ek['f5'].get(key1, {}).get(key2) is not None:
                        ek_coeff[3] = dic_ek['f5'][key1][key2][3]*np.sqrt(3/2)
                    elif dic_ek['f5'].get(key2, {}).get(key1) is not None:
                        ek_coeff[3] = dic_ek['f5'][key2][key1][3]*np.sqrt(3/2)

            key1 = ':'.join([str(v),str(int(2*S+1)),str(U)])
            key2 = ':'.join([str(v1),str(int(2*S+1)),str(U1)])
            factor1 = y_g.get(conf, {}).get(key1, {}).get(key2)
            if factor1 is None: 
                factor1 = y_g.get(conf, {}).get(key2, {}).get(key1)
            factor2 = phi_L.get(U1, {}).get(L, {}).get(U)
            if factor2 is None:  
                factor2 = phi_L.get(U, {}).get(L, {}).get(U1)
            if factor1 is not None and factor2 is not None:
                if (isinstance(factor1, float) or isinstance(factor1, int)) and (isinstance(factor2, float) or isinstance(factor2, int)):
                    ek_coeff[3] = factor1*factor2
                else:
                    idx = 1
                    try:       
                        if AB_label[conf][label1[:2]].index(int(label1[-1]))%2==0:
                            idx = 0
                    except KeyError:
                        if AB_label[conf][label2[:2]].index(int(label2[-1]))%2==0:
                            idx = 0
                    if label1!=label2 and v==v1 and U==U1 and W==W1:
                        idx = -1

                    ek_coeff[3] = factor2[idx]*factor1

            if U1==U and v1==v and label1==label2:
                ek_coeff[3] -= omegaUL(U,L)

            key1 = ':'.join([label1,str(n),str(v),str(U),str(int(2*S+1)),L])
            key2 = ':'.join([label2,str(n),str(v1),str(U1),str(int(2*S+1)),L])
            if key1 not in dic_ek[conf].keys():
                dic_ek[conf][key1] = {}
            if key2 not in dic_ek[conf].keys():
                dic_ek[conf][key2] = {}
            else:
                if key2 not in dic_ek[conf][key1].keys():
                    dic_ek[conf][key1][key2] = ek_coeff
                    dic_ek[conf][key2][key1] = ek_coeff

            return dic_ek 

        calc = calculation(conf, TAB=True, wordy=False)
        basis = calc.basis
        dic_ek_out[conf] = {}
        labels_list = []
        for i in range(basis.shape[0]):
            statei = basis[i]
            Si = statei[0]/2.
            Li = statei[1]
            Ji = statei[2]/2.
            MJi = statei[3]/2.
            labeli = calc.dic_LS[':'.join([f'{qq}' for qq in statei])]

            for j in range(0,i+1):
                statej = basis[j]
                Sj = statej[0]/2.
                Lj = statej[1]
                Jj = statej[2]/2.
                MJj = statej[3]/2.
                labelj = calc.dic_LS[':'.join([f'{qq}' for qq in statej])]

                if Ji==Jj and MJi==MJj and Li == Lj and Si == Sj:
                    if labeli+':'+labelj not in labels_list and labelj+':'+labeli not in labels_list:
                        labels_list.append(labeli+':'+labelj)
                        dic_ek_out = calc_ek(conf, labeli, labelj, Si, state_legend(str(Li), inv=True), dic_ek_out)
                

        return dic_ek_out

    def calc_ee_int(conf, label, label1, S, L, dic_ek):

        n = int(conf[1:])
        v,W,U = terms_labels_symm(conf)[label]
        v1,W1,U1 = terms_labels_symm(conf)[label1]
        key1 = ':'.join([label,str(n),str(v),str(U),str(int(2*S+1)),L])
        key2 = ':'.join([label1,str(n),str(v1),str(U1),str(int(2*S+1)),L])
        ek_coeff = dic_ek[key1][key2]

        F0, F2, F4, F6 = sympy.Symbol("F0, F2, F4, F6")
        coeff = [F0, F2, F4, F6]  

        Ek_coeff = []
        Ek_coeff.append(coeff[0] - 10/225*coeff[1] - 33/1089*coeff[2] - 286*25/184041*coeff[3])
        Ek_coeff.append(1/9*(70/225*coeff[1] + 231/1089*coeff[2] + 2002*25/184041*coeff[3]))
        Ek_coeff.append(1/9*(1/225*coeff[1] - 3/1089*coeff[2] + 7*25/184041*coeff[3]))
        Ek_coeff.append(1/3*(5/225*coeff[1] + 6/1089*coeff[2] - 91*25/184041*coeff[3]))

        ee_int = sympy.simplify(Ek_coeff[0]*ek_coeff[0] + Ek_coeff[1]*ek_coeff[1] + Ek_coeff[2]*ek_coeff[2] + Ek_coeff[3]*ek_coeff[3])

        return ee_int, ek_coeff

    #create the dictionry with e-e interaction
    conf_list = ['f3', 'f4', 'f5', 'f6', 'f7']
    dic_ek_conf = {'f0':{},
                   'f1':{},
                   'f2':
                   {'3P:2:2:11:3:P':{'3P:2:2:11:3:P':np.array([0,0,0,22+11])},
                   '3F:2:2:10:3:F':{'3F:2:2:10:3:F':np.array([0,0,0,0])},
                   '3H:2:2:11:3:H':{'3H:2:2:11:3:H':np.array([0,0,0,-6-3])},
                   '1S:2:0:99:1:S':{'1S:2:0:99:1:S':np.array([0,0,0,0])},
                   '1D:2:2:20:1:D':{'1D:2:2:20:1:D':np.array([0,0,0,-22+11])},
                   '1G:2:2:20:1:G':{'1G:2:2:20:1:G':np.array([0,0,0,-8+4])},
                   '1I:2:2:20:1:I':{'1I:2:2:20:1:I':np.array([0,0,0,14-7])}}}
    
    dic_ee_expr = {}
    dic_ee_values = {}

    for conf in conf_list:

        dic_ek_conf = calc_dic_ek(conf, dic_ek_conf)
        dic_ee_expr[conf] = {}
        dic_ee_values[conf] = {}

        calc = calculation(conf, TAB=True, wordy=False)
        basis = calc.basis
        for i in range(basis.shape[0]):
            statei = basis[i]
            Si = statei[0]/2.
            Li = statei[1]
            Ji = statei[2]/2.
            MJi = statei[3]/2.
            labeli = calc.dic_LS[':'.join([f'{qq}' for qq in statei])]

            for j in range(0,i+1):
                statej = basis[j]
                Sj = statej[0]/2.
                Lj = statej[1]
                Jj = statej[2]/2.
                MJj = statej[3]/2.
                labelj = calc.dic_LS[':'.join([f'{qq}' for qq in statej])]

                if Ji==Jj and MJi==MJj and Li == Lj and Si == Sj:
                    ee_value, ee_c = calc_ee_int(conf, labeli, labelj, Si, state_legend(str(Li), inv=True), dic_ek_conf[conf])
                    dic_ee_expr[conf][labeli+':'+labelj] = ee_value
                    dic_ee_values[conf][labeli+':'+labelj] = ee_c
                    if labeli!=labelj:
                        dic_ee_expr[conf][labelj+':'+labeli] = ee_value
                        dic_ee_values[conf][labelj+':'+labeli] = ee_c

    return dic_ee_expr[conf_str], dic_ee_values[conf_str]

#--------------------------------------------------#
#           TABLES, LEGENDS & CONVENTIONS          #
#--------------------------------------------------#

def rotate_vectors(vectors, angle, axis):
    """
    Rotate Cartesian vectors with respect to one of the x, y, or z axes.

    Parameters:
    vectors (np.ndarray): Array of shape (N, 3) representing N Cartesian vectors.
    angle (float): Rotation angle in radians.
    axis (str): Axis of rotation ('x', 'y', or 'z').

    Returns:
    np.ndarray: Array of rotated vectors of shape (N, 3).
    """
    if axis == 'x':
        r = scipy.spatial.transform.Rotation.from_euler('x', angle, degrees=False)
    elif axis == 'y':
        r = scipy.spatial.transform.Rotation.from_euler('y', angle, degrees=False)
    elif axis == 'z':
        r = scipy.spatial.transform.Rotation.from_euler('z', angle, degrees=False)
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'")

    rotated_vectors = r.apply(vectors)
    return rotated_vectors

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
    #from Boca, "theoretical foundations of molecular magnetism" (Ch 8, p 381, Tab 8.4)
    #or OctoYot f_e_data.f90, TS_d_labels (following the order of Nielson e Koster)

    if conf[0]=='d' and int(conf[1:])>5:
        conf = 'd'+str(almost_closed_shells(conf))
    elif conf[0]=='f' and int(conf[1:])>7:
        conf = 'f'+str(almost_closed_shells(conf))
    else:
        pass

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
    #(2S, L, seniority) da OctoYot f_e_data.f90, TS_d_basis (order from Nielson e Koster)
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

def terms_labels_symm(conf):
    #In this classification of states 99 and 999 substitute 00 and 000 respectively

    if conf[0]=='d' and int(conf[1:])>5:
        conf = 'd'+str(almost_closed_shells(conf))
    elif conf[0]=='f' and int(conf[1:])>7:
        conf = 'f'+str(almost_closed_shells(conf))
    else:
        pass

    legenda = {'f1':{'2F':[1, 100, 10]},
        'f2':{'3P':[2, 110, 11],
            '3F':[2, 110, 10],
            '3H':[2, 110, 11],
            '1S':[0, 999, 99],
            '1D':[2, 200, 20],
            '1G':[2, 200, 20],
            '1I':[2, 200, 20]},
        'f3':{'4S':[3, 111, 99],
            '4D':[3, 111, 20],
            '4F':[3, 111, 10],
            '4G':[3, 111, 20],
            '4I':[3, 111, 20],
            '2P':[3, 210, 11],
            '2D1':[3, 210, 20],
            '2D2':[3, 210, 21],
            '2F1':[1, 100, 10],
            '2F2':[3, 210, 21],
            '2G1':[3, 210, 20],
            '2G2':[3, 210, 21],
            '2H1':[3, 210, 11],
            '2H2':[3, 210, 21],
            '2I':[3, 210, 20],
            '2K':[3, 210, 21],
            '2L':[3, 210, 21]},
        'f4':{'5S':[4, 111,99],
            '5D':[4, 111, 20],
            '5F':[4, 111, 10],
            '5G':[4, 111, 20],
            '5I':[4, 111, 20],
            '3P1':[2, 110, 11],
            '3P2':[4, 211, 11],
            '3P3':[4, 211, 30],
            '3D1':[4, 211, 20],
            '3D2':[4, 211, 21],
            '3F1':[2, 110, 10],
            '3F2':[4, 211, 10],
            '3F3':[4, 211, 21],
            '3F4':[4, 211, 30],
            '3G1':[4, 211, 20],
            '3G2':[4, 211, 21],
            '3G3':[4, 211, 30],
            '3H1':[2, 110, 11],
            '3H2':[4, 211, 11],
            '3H3':[4, 211, 21],
            '3H4':[4, 211, 30],
            '3I1':[4, 211, 20],
            '3I2':[4, 211, 30],
            '3K1':[4, 211, 21],
            '3K2':[4, 211, 30],
            '3L':[4, 211, 21],
            '3M':[4, 211, 30],
            '1S1':[0,999,99],
            '1S2':[4, 220, 22],
            '1D1':[2, 200, 20],
            '1D2':[4, 220, 20],
            '1D3':[4, 220, 21],
            '1D4':[4, 220, 22],
            '1F':[4, 220, 21],
            '1G1':[2, 200, 20],
            '1G2':[4, 220, 20],
            '1G3':[4, 220, 21],
            '1G4':[4, 220, 22],
            '1H1':[4, 220, 21],
            '1H2':[4, 220, 22],
            '1I1':[2, 200, 20],
            '1I2':[4, 220, 20],
            '1I3':[4, 220, 22],
            '1K':[4, 220, 21],
            '1L1':[4, 220, 21],
            '1L2':[4, 220, 22],
            '1N':[4, 220, 22]},
        'f5':{'6P':[5, 110, 11],
            '6F':[5, 110, 10],
            '6H':[5, 110, 11],
            '4S':[3, 111, 99],
            '4P1':[5, 211, 11],
            '4P2':[5, 211, 30],
            '4D1':[3, 111, 20],
            '4D2':[5, 211, 20],
            '4D3':[5, 211, 21],
            '4F1':[3, 111, 10],
            '4F2':[5, 211, 10],
            '4F3':[5, 211, 21],
            '4F4':[5, 211, 30],
            '4G1':[3, 111, 20],
            '4G2':[5, 211, 20],
            '4G3':[5, 211, 21],
            '4G4':[5, 211, 30],
            '4H1':[5, 211, 11],
            '4H2':[5, 211, 21],
            '4H3':[5, 211, 30],
            '4I1':[3, 111, 20],
            '4I2':[5, 211, 20],
            '4I3':[5, 211, 30],
            '4K1':[5, 211, 21],
            '4K2':[5, 211, 30],
            '4L':[5, 211, 21],
            '4M':[5, 211, 30],
            '2P1':[3, 210, 11],
            '2P2':[5, 221, 11],
            '2P3':[5, 221, 30],
            '2P4':[5, 221, 31],
            '2D1':[3, 210, 20],
            '2D2':[3, 210, 21],
            '2D3':[5, 221, 20],
            '2D4':[5, 221, 21],
            '2D5':[5, 221, 31],
            '2F1':[1, 100, 10],
            '2F2':[3, 210, 21],
            '2F3':[5, 221, 10],
            '2F4':[5, 221, 21],
            '2F5':[5, 221, 30],
            '2F6':[5, 221, 31],
            '2F7':[5, 221, 31],
            '2G1':[3, 210, 20],
            '2G2':[3, 210, 21],
            '2G3':[5, 221, 20],
            '2G4':[5, 221, 21],
            '2G5':[5, 221, 30],
            '2G6':[5, 221, 31],
            '2H1':[3, 210, 11],
            '2H2':[3, 210, 21],
            '2H3':[5, 221, 11],
            '2H4':[5, 221, 21],
            '2H5':[5, 221, 30],
            '2H6':[5, 221, 31],
            '2H7':[5, 221, 31],
            '2I1':[3, 210, 20],
            '2I2':[5, 221, 20],
            '2I3':[5, 221, 30],
            '2I4':[5, 221, 31],
            '2I5':[5, 221, 31],
            '2K1':[3, 210, 21],
            '2K2':[5, 221, 21],
            '2K3':[5, 221, 30],
            '2K4':[5, 221, 31],
            '2K5':[5, 221, 31],
            '2L1':[3, 210, 21],
            '2L2':[5, 221, 21],
            '2L3':[5, 221, 31],
            '2M1':[5, 221, 30],
            '2M2':[5, 221, 31],
            '2N':[5, 221, 31],
            '2O':[5, 221, 31]},
        'f6':{'7F':[6, 100, 10],
            '5S':[4, 111, 99],
            '5P':[6, 210, 11],
            '5D1':[4, 111, 20],
            '5D2':[6, 210, 20],
            '5D3':[6, 210, 21],
            '5F1':[4, 111, 10],
            '5F2':[6, 210, 21],
            '5G1':[4, 111, 20],
            '5G2':[6, 210, 20],
            '5G3':[6, 210, 21],
            '5H1':[6, 210, 11],
            '5H2':[6, 210, 21],
            '5I1':[4, 111, 20],
            '5I2':[6, 210, 20],
            '5K':[6, 210, 21],
            '5L':[6, 210, 21],
            '3P1':[2, 110, 11],
            '3P2':[4, 211, 11],
            '3P3':[4, 211, 30],
            '3P4':[6, 221, 11],
            '3P5':[6, 221, 30],
            '3P6':[6, 221, 31],
            '3D1':[4, 211, 20],
            '3D2':[4, 211, 21],
            '3D3':[6, 221, 20],
            '3D4':[6, 221, 21],
            '3D5':[6, 221, 31],
            '3F1':[2, 110, 10],
            '3F2':[4, 211, 10],
            '3F3':[4, 211, 21],
            '3F4':[4, 211, 30],
            '3F5':[6, 221, 10],
            '3F6':[6, 221, 21],
            '3F7':[6, 221, 30],
            '3F8':[6, 221, 31],
            '3F9':[6, 221, 31],
            '3G1':[4, 211, 20],
            '3G2':[4, 211, 21],
            '3G3':[4, 211, 30],
            '3G4':[6, 221, 20],
            '3G5':[6, 221, 21],
            '3G6':[6, 221, 30],
            '3G7':[6, 221, 31],
            '3H1':[2, 110, 11],
            '3H2':[4, 211, 11],
            '3H3':[4, 211, 21],
            '3H4':[4, 211, 30],
            '3H5':[6, 221, 11],
            '3H6':[6, 221, 21],
            '3H7':[6, 221, 30],
            '3H8':[6, 221, 31],
            '3H9':[6, 221, 31],
            '3I1':[4, 211, 20],
            '3I2':[4, 211, 30],
            '3I3':[6, 221, 20],
            '3I4':[6, 221, 30],
            '3I5':[6, 221, 31],
            '3I6':[6, 221, 31],
            '3K1':[4, 211, 21],
            '3K2':[4, 211, 30],
            '3K3':[6, 221, 21],
            '3K4':[6, 221, 30],
            '3K5':[6, 221, 31],
            '3K6':[6, 221, 31],
            '3L1':[4, 211, 21],
            '3L2':[6, 221, 21],
            '3L3':[6, 221, 31],
            '3M1':[4, 211, 30],
            '3M2':[6, 221, 30],
            '3M3':[6, 221, 31],
            '3N':[6, 221, 31],
            '3O':[6, 221, 31],
            '1S1':[0, 999, 99],
            '1S2':[4, 220, 22],
            '1S3':[6, 222, 99],
            '1S4':[6, 222, 40],
            '1P':[6, 222, 30],
            '1D1':[2, 200, 20],
            '1D2':[4, 220, 20],
            '1D3':[4, 220, 21],
            '1D4':[4, 220, 22],
            '1D5':[6, 222, 20],
            '1D6':[6, 222, 40],
            '1F1':[4, 220, 21],
            '1F2':[6, 222, 10],
            '1F3':[6, 222, 30],
            '1F4':[6, 222, 40],
            '1G1':[2, 200, 20],
            '1G2':[4, 220, 20],
            '1G3':[4, 220, 21],
            '1G4':[4, 220, 22],
            '1G5':[6, 222, 20],
            '1G6':[6, 222, 30],
            '1G7':[6, 222, 40],
            '1G8':[6, 222, 40],
            '1H1':[4, 220, 21],
            '1H2':[4, 220, 22],
            '1H3':[6, 222, 30],
            '1H4':[6, 222, 40],
            '1I1':[2, 200, 20],
            '1I2':[4, 220, 20],
            '1I3':[4, 220, 22],
            '1I4':[6, 222, 20],
            '1I5':[6, 222, 30],
            '1I6':[6, 222, 40],
            '1I7':[6, 222, 40],
            '1K1':[4, 220, 21],
            '1K2':[6, 222, 30],
            '1K3':[6, 222, 40],
            '1L1':[4, 220, 21],
            '1L2':[4, 220, 22],
            '1L3':[6, 222, 40],
            '1L4':[6, 222, 40],
            '1M1':[6, 222, 30],
            '1M2':[6, 222, 40],
            '1N1':[4, 220, 22],
            '1N2':[6, 222, 40],
            '1Q':[6, 222, 40]},
        'f7':{'8S':[7, 999, 99],
            '6P':[5, 110, 11],
            '6D':[7, 200, 20],
            '6F':[5, 110, 10],
            '6G':[7, 200, 20],
            '6H':[5, 110, 11],
            '6I':[7, 200, 20],
            '4S1':[3, 111, 99],
            '4S2':[7, 220, 22],
            '4P1':[5, 211, 11],
            '4P2':[5, 211, 30],
            '4D1':[3, 111, 20],
            '4D2':[5, 211, 20],
            '4D3':[5, 211, 21],
            '4D4':[7, 220, 20],
            '4D5':[7, 220, 21],
            '4D6':[7, 220, 22],
            '4F1':[3, 111, 10],
            '4F2':[5, 211, 10],
            '4F3':[5, 211, 21],
            '4F4':[5, 211, 30],
            '4F5':[7, 220, 21],
            '4G1':[3, 111, 20],
            '4G2':[5, 211, 20],
            '4G3':[5, 211, 21],
            '4G4':[5, 211, 30],
            '4G5':[7, 220, 20],
            '4G6':[7, 220, 21],
            '4G7':[7, 220, 22],
            '4H1':[5, 211, 11],
            '4H2':[5, 211, 21],
            '4H3':[5, 211, 30],
            '4H4':[7, 220, 21],
            '4H5':[7, 220, 22],
            '4I1':[3, 111, 20],
            '4I2':[5, 211, 20],
            '4I3':[5, 211, 30],
            '4I4':[7, 220, 20],
            '4I5':[7, 220, 22],
            '4K1':[5, 211, 21],
            '4K2':[5, 211, 30],
            '4K3':[7, 220, 21],
            '4L1':[5, 211, 21],
            '4L2':[7, 220, 21],
            '4L3':[7, 220, 22],
            '4M':[5, 211, 30],
            '4N':[7, 220, 22],
            '2S1':[7, 222, 99],
            '2S2':[7, 222, 40],
            '2P1':[3, 210, 11],
            '2P2':[5, 221, 11],
            '2P3':[5, 221, 30],
            '2P4':[5, 221, 31],
            '2P5':[7, 222, 30],
            '2D1':[3, 210, 20],
            '2D2':[3, 210, 21],
            '2D3':[5, 221, 20],
            '2D4':[5, 221, 21],
            '2D5':[5, 221, 31],
            '2D6':[7, 222, 20],
            '2D7':[7, 222, 40],
            '2F1':[1, 100, 10],
            '2F2':[3, 210, 21],
            '2F3':[5, 221, 10],
            '2F4':[5, 221, 21],
            '2F5':[5, 221, 30],
            '2F6':[5, 221, 31],
            '2F7':[5, 221, 31],
            '2F8':[7, 222, 10],
            '2F9':[7, 222, 30],
            '2F0':[7, 222, 40],
            '2G1':[3, 210, 20],
            '2G2':[3, 210, 21],
            '2G3':[5, 221, 20],
            '2G4':[5, 221, 21],
            '2G5':[5, 221, 30],
            '2G6':[5, 221, 31],
            '2G7':[7, 222, 20],
            '2G8':[7, 222, 30],
            '2G9':[7, 222, 40],
            '2G0':[7, 222, 40],
            '2H1':[3, 210, 11],
            '2H2':[3, 210, 21],
            '2H3':[5, 221, 11],
            '2H4':[5, 221, 21],
            '2H5':[5, 221, 30],
            '2H6':[5, 221, 31],
            '2H7':[5, 221, 31],
            '2H8':[7, 222, 30],
            '2H9':[7, 222, 40],
            '2I1':[3, 210, 20],
            '2I2':[5, 221, 20],
            '2I3':[5, 221, 30],
            '2I4':[5, 221, 31],
            '2I5':[5, 221, 31],
            '2I6':[7, 222, 20],
            '2I7':[7, 222, 30],
            '2I8':[7, 222, 40],
            '2I9':[7, 222, 40],
            '2K1':[3, 210, 21],
            '2K2':[5, 221, 21],
            '2K3':[5, 221, 30],
            '2K4':[5, 221, 31],
            '2K5':[5, 221, 31],
            '2K6':[7, 222, 30],
            '2K7':[7, 222, 40],
            '2L1':[3, 210, 21],
            '2L2':[5, 221, 21],
            '2L3':[5, 221, 31],
            '2L4':[7, 222, 40],
            '2L5':[7, 222, 40],
            '2M1':[5, 221, 30],
            '2M2':[5, 221, 31],
            '2M3':[7, 222, 30],
            '2M4':[7, 222, 40],
            '2N1':[5, 221, 31],
            '2N2':[7, 222, 40],
            '2O':[5, 221, 31],
            '2Q':[7, 222, 40]}}

    return legenda[conf]

def conv_Aqkrk_bkq(l,m):
    #conversion factor Stevens coefficients to Wybourne formalism

    if isinstance(l, int):
        l = str(l)
    if isinstance(m, int):
        m = str(m)

    legenda = {'2':{'0':2, '1':1/np.sqrt(6), '2':2/np.sqrt(6)},
               '4':{'0':8, '1':2/np.sqrt(5), '2':4/np.sqrt(10), '3':2/np.sqrt(35), '4':8/np.sqrt(70)},
               '6':{'0':16, '1':8/np.sqrt(42), '2':16/np.sqrt(105), '3':8/np.sqrt(105), '4':16/(3*np.sqrt(14)), '5':8/(3*np.sqrt(77)), '6':16/np.sqrt(231)}}
    return legenda[str(l)][str(m)]

def r_expect(k, conf):
    # in atomic units

    if isinstance(k, int):
        k = str(k)

    if k=='0':
        return 1

    if conf[0]=='d':
        # <r^k> from table 7.6 in A. Abragam and B. Bleaney, Electron Paramagnetic Resonance of Transition Ions, Dover, New York, 1986.
        legenda = {'2':{'d2': 2.447,'d3': 2.070,'d4': 1.781,'d5': 1.548,'d6': 1.393,'d7': 1.251,'d8': 1.130,'d9': 1.028},
                '4':{'d2': 13.17,'d3': 9.605,'d4': 7.211,'d5': 5.513,'d6': 4.496,'d7': 3.655,'d8': 3.003,'d9': 2.498}}
    else:
        # <r^k> from S. Edvarsson, M. Klintenberg, J. Alloys Compd. 1998, 275–277, 230
        legenda = {'2':{'f1':1.456, 'f2':1.327, 'f3':1.222, 'f4':1.135, 'f5':1.061, 'f6':0.997, 'f7':0.942, 'f8':0.893, 'f9':0.849, 'f10':0.810, 'f11':0.773, 'f12':0.740, 'f13':0.710},
                '4':{'f1':5.437, 'f2':4.537, 'f3':3.875, 'f4':3.366, 'f5':2.964, 'f6':2.638, 'f7':2.381, 'f8':2.163, 'f9':1.977, 'f10':1.816, 'f11':1.677, 'f12':1.555, 'f13':1.448},
                '6':{'f1':42.26, 'f2':32.65, 'f3':26.12, 'f4':21.46, 'f5':17.99, 'f6':15.34, 'f7':13.36, 'f8':11.75, 'f9':10.44, 'f10':9.345, 'f11':8.431, 'f12':7.659, 'f13':7.003}}
    
    return legenda[k][conf]

def sigma_k(k, conf):
    # Sternheimer shielding parameters from S. Edvardsson, M. Klinterberg, J. Alloys Compd. 1998, 275, 233.

    if isinstance(k, int):
        k = str(k)

    if conf[0] == 'd':
        raise NotImplementedError("Sternheimer shielding parameters for d^n configurations are not implemented.")

    legenda = {'2':{'f1':0.510, 'f2':0.515, 'f3':0.518, 'f4':0.519, 'f5':0.519, 'f6':0.520, 'f7':0.521, 'f8':0.523, 'f9':0.527, 'f10':0.534, 'f11':0.544, 'f12':0.554, 'f13':0.571},
               '4':{'f1':0.0132, 'f2':0.0138, 'f3':0.0130, 'f4':0.0109, 'f5':0.0077, 'f6':0.0033, 'f7':-0.0031, 'f8':-0.0107, 'f9':-0.0199, 'f10':-0.0306, 'f11':-0.0427, 'f12':-0.0567, 'f13':-0.0725},
               '6':{'f1':-0.0294, 'f2':-0.0301, 'f3':-0.0310, 'f4':-0.0314, 'f5':-0.0317, 'f6':-0.0319, 'f7':-0.0318, 'f8':-0.0318, 'f9':-0.0316, 'f10':-0.0313, 'f11':-0.0310, 'f12':-0.0306, 'f13':-0.0300}}
    return legenda[k][conf]

def Stev_coeff(k, conf):
    # Stevens coefficients (alpha=2, beta=4, gamma=6) from K. W. H. Stevens, Proc. Phys. Soc. 1952, 65, 209.
    # or table 20 from A. Abragam and B. Bleaney, Electron Paramagnetic Resonance of Transition Ions, Dover, New York, 1986.
    
    if conf[0] == 'd':
        raise NotImplementedError("Stevens coefficients for d^n configurations are not implemented.")

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

    if conf=='f13' or conf=='f1':
        print('Warning: '+conf+' is not included in free_ion_param_f(), redirected toward free_ion_param_f_HF()')
        return free_ion_param_f_HF(conf)

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
    dict = {'f1':{'F2': 0, 'F4': 0, 'F6': 0, 'zeta': 689},
            'f2':{'F2': 96681, 'F4': 60533, 'F6': 43509, 'zeta': 808},
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

def free_ion_param_AB(conf):
    # from Electron paramagnetic resonance of transition metal ions from Abragam e Bleany 

    def from_BC_to_Fk(B,C,A=0):
        ABC = np.array([A,B,C])
        conv = np.array([[1,0,7/5],[0,49,49/7],[0,0,441/35]])
        return np.dot(conv,ABC)

    if conf[0]=='d':
        # table 7.3
        dict = {'d1':{'F2': from_BC_to_Fk(0,0)[1], 'F4': from_BC_to_Fk(0,0)[2], 'zeta':79.0},
                'd2':{'F2': from_BC_to_Fk(694,2910)[1], 'F4': from_BC_to_Fk(694,2910)[2], 'zeta':120.0},
                'd3':{'F2': from_BC_to_Fk(755,3257)[1], 'F4': from_BC_to_Fk(755,3257)[2], 'zeta':168.0},
                'd4':{'F2': from_BC_to_Fk(810,3565)[1], 'F4': from_BC_to_Fk(810,3565)[2], 'zeta':236.0},
                'd5':{'F2': from_BC_to_Fk(860,3850)[1], 'F4': from_BC_to_Fk(860,3850)[2], 'zeta':335.0},
                'd6':{'F2': from_BC_to_Fk(917,4040)[1], 'F4': from_BC_to_Fk(917,4040)[2], 'zeta':404.0},
                'd7':{'F2': from_BC_to_Fk(971,4497)[1], 'F4': from_BC_to_Fk(971,4497)[2], 'zeta':528.0},
                'd8':{'F2': from_BC_to_Fk(1030,4850)[1], 'F4': from_BC_to_Fk(1030,4850)[2], 'zeta':644.0},
                'd9':{'F2': from_BC_to_Fk(0,0)[1], 'F4': from_BC_to_Fk(0,0)[2], 'zeta':829.0}}

    elif conf[0]=='f':
        warnings.warn("Slater-Condon F^k from Abragam e Bleany are not available for f^n configurations.\nPlease use free_ion_param_f_HF() or free_ion_param_f() if .", UserWarning)
        # table 5.3
        dict = {'f1':{'zeta': 740},
            'f2':{'zeta': 878},
            'f3':{'zeta': 1024},
            'f5':{'zeta': 1342},
            'f7':{'zeta': 1717},
            'f8':{'zeta': 1915},
            'f9':{'zeta': 2182},
            'f10':{'zeta': 2360},
            'f11':{'zeta': 2610},
            'f12':{'zeta': 2866},
            'f13':{'zeta': 3161}}

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

#=========================================================================================
#                                    FROM ON_DEV                                         #
#=========================================================================================

def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
    
def read_AILFT_orca6(filename, conf, method='CASSCF', return_V=False, rotangle_V=False, return_orcamatrix=False):
    # rotate_V is a list of angles to rotate the V matrix
    # rotangle_V = [alpha, beta, gamma] or [q0, q1, q2, q3]
    # euler angles are in scipy convention 'ZYZ'

    from_au = 27.2113834*8065.54477

    file = open(filename).readlines()

    if conf[0] == 'f':
        collected = []
        check_name = []
        passed = False
        F2 = 0
        F4 = 0
        F6 = 0
        zeta = 0
        for i, line in enumerate(file):
            if "AILFT MATRIX ELEMENTS ("+method+")" in line:
                for j in range(4, 11):
                    collected.append(file[i+j][10:])
                    check_name.append(file[i+j][:10])
                    passed = True
            if passed and not F2 and 'F2' in line:
                splitline = line.split(' ')
                numbers = [float(num) for num in splitline if is_float(num)]
                F2 = numbers[-1]
            if passed and not F4 and 'F4' in line:
                splitline = line.split(' ')
                numbers = [float(num) for num in splitline if is_float(num)]
                F4 = numbers[-1]
            if passed and not F6 and 'F6' in line:
                splitline = line.split(' ')
                numbers = [float(num) for num in splitline if is_float(num)]
                F6 = numbers[-1]
            if passed and not zeta and 'SOC constant zeta' in line:
                splitline = line.split(' ')
                numbers = [float(num) for num in splitline if is_float(num)]
                zeta = numbers[-1]
                passed = False  #since in orca output zeta is usually after F2

        names = ['f0', 'f+1', 'f-1', 'f+2', 'f-2', 'f+3', 'f-3']
        for i,item in enumerate(check_name):
            if names[i] not in item:
                raise ValueError(f"Error in reading the file: {names[i]} not found in {item} while reading ORCA file {filename}")
        matrix = []
        for i in range(len(collected)):
            splitline = collected[i].split(' ')
            row = []
            for j in range(len(splitline)):
                if splitline[j] != '' and is_float(splitline[j]):
                    row.append(float(splitline[j]))
            matrix.append(row)
        matrix = np.array(matrix) #order 0 1 -1 2 -2 3 -3 

        if rotangle_V:
            if len(rotangle_V) > 3:
                matrix = rotate_V_quat(matrix, 3, rotangle_V)
            else:
                matrix = rotate_V(matrix, 3, *rotangle_V)

        matrix = adjust_phase_matrix(matrix)

        if return_orcamatrix:
            return matrix

        ### change to order 0 -1 1 -2 2 -3 3 
        mask = [[(0,0), (2,0), (1,0), (4,0), (3,0), (6,0), (5,0)],
                [(2,0), (2,2), (1,2), (4,2), (3,2), (6,2), (5,2)],
                [(1,0), (1,2), (1,1), (4,1), (3,1), (6,1), (5,1)],
                [(4,0), (4,2), (4,1), (4,4), (3,4), (6,4), (5,4)],
                [(3,0), (3,2), (3,1), (3,4), (3,3), (6,3), (5,3)],
                [(6,0), (6,2), (6,1), (6,4), (6,3), (6,6), (5,6)],
                [(5,0), (5,2), (5,1), (5,4), (5,3), (5,6), (5,5)]]
        mask = np.array(mask)
        row_indices = mask[:,:,0]
        col_indices = mask[:,:,1]
        matrix = matrix[row_indices, col_indices]

        dic = {}

        dic_V = {
            '11':matrix[0,0]*from_au,
            '21':matrix[1,0]*from_au, '22':matrix[1,1]*from_au,
            '31':matrix[2,0]*from_au,'32':matrix[2,1]*from_au,'33':matrix[2,2]*from_au,
            '41':matrix[3,0]*from_au,'42':matrix[3,1]*from_au,'43':matrix[3,2]*from_au,'44':matrix[3,3]*from_au,
            '51':matrix[4,0]*from_au,'52':matrix[4,1]*from_au,'53':matrix[4,2]*from_au,'54':matrix[4,3]*from_au,'55':matrix[4,4]*from_au,
            '61':matrix[5,0]*from_au,'62':matrix[5,1]*from_au,'63':matrix[5,2]*from_au,'64':matrix[5,3]*from_au,'65':matrix[5,4]*from_au,'66':matrix[5,5]*from_au,
            '71':matrix[6,0]*from_au,'72':matrix[6,1]*from_au,'73':matrix[6,2]*from_au,'74':matrix[6,3]*from_au,'75':matrix[6,4]*from_au,'76':matrix[6,5]*from_au,'77':matrix[6,6]*from_au
        }
        
        dic_Bkq = from_Vint_to_Bkq(dic_V, conf)
        dic['dic_bkq'] = dic_Bkq
        dic['F2'] = F2
        dic['F4'] = F4
        dic['F6'] = F6
        dic['zeta'] = zeta


    else:

        collected = []
        check_name = []
        passed = False
        F2 = 0
        F4 = 0
        zeta = 0
        for i, line in enumerate(file):
            if "AILFT MATRIX ELEMENTS ("+method+")" in line:
                for j in range(5, 10):
                    collected.append(file[i+j][10:])
                    check_name.append(file[i+j][:10])
                    passed = True
            if passed and not F2 and 'F2' in line:
                splitline = line.split(' ')
                numbers = [float(num) for num in splitline if is_float(num)]
                F2 = numbers[-1]
            if passed and not F4 and 'F4' in line:
                splitline = line.split(' ')
                numbers = [float(num) for num in splitline if is_float(num)]
                F4 = numbers[-1]
            if passed and not zeta and 'SOC constant zeta' in line:
                splitline = line.split(' ')
                numbers = [float(num) for num in splitline if is_float(num)]
                zeta = numbers[-1]
                passed = False  #since in orca output zeta is usually after F2

        names = ['dz2', 'dxz', 'dyz', 'dx2-y2', 'dxy']
                  
        for i,item in enumerate(check_name):
            if names[i] not in item:
                raise ValueError(f"Error in reading the file: {names[i]} not found in {item} while reading ORCA file {filename}")
        matrix = []
        for i in range(len(collected)):
            splitline = collected[i].split(' ')
            row = []
            for j in range(len(splitline)):
                if splitline[j] != '' and is_float(splitline[j]):
                    row.append(float(splitline[j]))
            matrix.append(row)
        matrix = np.array(matrix)

        if rotangle_V:
            if len(rotangle_V) > 3:
                matrix = rotate_V_quat(matrix, 2, rotangle_V)
            else:
                matrix = rotate_V(matrix, 2, *rotangle_V)

        if return_orcamatrix:
            return matrix

        ### change to order  z2, yz, xz, xy, x2-y2 
        mask = [[(0,0), (2,0), (1,0), (4,0), (3,0)],
                [(2,0), (2,2), (1,2), (4,2), (3,2)],
                [(1,0), (1,2), (1,1), (4,1), (3,1)],
                [(4,0), (4,2), (4,1), (4,4), (3,4)],
                [(3,0), (3,2), (3,1), (3,4), (3,3)]]
        mask = np.array(mask)
        row_indices = mask[:,:,0]
        col_indices = mask[:,:,1]
        matrix = matrix[row_indices, col_indices]

        dic = {}

        dic_V = {
            '11':matrix[0,0]*from_au,
            '21':matrix[1,0]*from_au, '22':matrix[1,1]*from_au,
            '31':matrix[2,0]*from_au,'32':matrix[2,1]*from_au,'33':matrix[2,2]*from_au,
            '41':matrix[3,0]*from_au,'42':matrix[3,1]*from_au,'43':matrix[3,2]*from_au,'44':matrix[3,3]*from_au,
            '51':matrix[4,0]*from_au,'52':matrix[4,1]*from_au,'53':matrix[4,2]*from_au,'54':matrix[4,3]*from_au,'55':matrix[4,4]*from_au,
        }
        
        dic_Bkq = from_Vint_to_Bkq(dic_V, conf)
        dic['dic_bkq'] = dic_Bkq
        dic['F2'] = F2
        dic['F4'] = F4
        dic['zeta'] = zeta

    if return_V:
        return dic_V
    else:
        return dic

def adjust_phase_matrix(matrix):
    #adjust the phase of the LF matrix from ORCA
    for i in range(7):  # Python uses 0-based indexing
        phasei = 1.0
        if i == 5 or i == 6:  # Adjusted for 0-based indexing
            phasei = -1.0
        for j in range(7):
            phasej = 1.0
            if j == 5 or j == 6:  # Adjusted for 0-based indexing
                phasej = -1.0
            matrix[i, j] *= phasei * phasej
    return matrix

def rotate_V(V, l, alpha=0.0, beta=0.0, gamma=0.0, real=True):
    #in rad
    #V has to be in the order 0, 1, -1, 2, -2, 3, -3 (like directly from orca)
    V_rot = np.zeros_like(V, dtype='complex128')
    M_num = [0, 1, -1, 2, -2, 3, -3]
    M1_num = [0, 1, -1, 2, -2, 3, -3]
    U = U_complex2real(l)
    D = np.zeros_like(V, dtype='complex128')
    for im1, m1 in enumerate(np.arange(-l, l+1, 1)):
        for jm, m in enumerate(np.arange(-l, l+1, 1)):
            D[im1, jm] = Wigner_coeff.Wigner_Dmatrix(l, m1, m, alpha, beta, gamma)
    if real:
        R = np.conj(U).T@D@U   
    else:
        R = D.copy()
    mask = []
    perm = [3,4,2,5,1,6,0]
    for i,ii in enumerate(perm):
        row = []
        for j,jj in enumerate(perm):
            row.append((ii,jj))
        mask.append(row)
    mask = np.array(mask)
    row_indices = mask[:,:,0]
    col_indices = mask[:,:,1]
    R = R[row_indices, col_indices]
    for iM,M in enumerate(M_num):
        for jM,M1 in enumerate(M1_num):
            for im1,m1 in enumerate(np.arange(-l, l+1, 1)):
                for jm2,m2 in enumerate(np.arange(-l, l+1, 1)):
                    V_rot[iM,jM] += R[iM,im1]*V[im1,jm2]*R[jM,jm2]
    V_rot = V_rot.real
    return V_rot

def rotate_V_quat(V, l, q=[1.0,0.0,0.0,0.0], real=True):
    #V has to be in the order 0, 1, -1, 2, -2, 3, -3 (like directly from orca)
    V_rot = np.zeros_like(V, dtype='complex128')
    M_num = [0, 1, -1, 2, -2, 3, -3]
    M1_num = [0, 1, -1, 2, -2, 3, -3]
    dict, coeff = read_DWigner_quat()
    U = U_complex2real(l)
    D = Wigner_coeff.Wigner_Dmatrix_quat_complete(l, q, dict = dict, coeff = coeff)   #order: -3,-2,-1,0,1,2,3
    if real:
        R = np.conj(U).T@D@U
    else:
        R = D.copy()
    mask = []
    perm = [3,4,2,5,1,6,0]
    for i,ii in enumerate(perm):
        row = []
        for j,jj in enumerate(perm):
            row.append((ii,jj))
        mask.append(row)
    mask = np.array(mask)
    row_indices = mask[:,:,0]
    col_indices = mask[:,:,1]
    R = R[row_indices, col_indices]
    for iM,M in enumerate(M_num):
        for jM,M1 in enumerate(M1_num):
            for im1,m1 in enumerate(np.arange(-l, l+1, 1)):
                for jm2,m2 in enumerate(np.arange(-l, l+1, 1)):
                    V_rot[iM,jM] += R[iM,im1]*V[im1,jm2]*np.conj(R[jM,jm2])
    V_rot = V_rot.real
    return V_rot

def rotate_dicV(dic, l, rotangle_V=False, real=True, return_orcaV = False):
    #ruota V a partire dal dizionario nell'ordine 0, -1, 1, -2, 2, -3, 3
    #quindi prima lo converte in matrice 0, 1, -1, 2, -2, 3, -3
    #poi lo ruota
    #poi lo converte in dizionario 0, -1, 1, -2, 2, -3, 3

    def permutation(V):
        perm = [0,2,1,4,3,6,5]
        mask = []
        for i,ii in enumerate(perm):
            row = []
            for j,jj in enumerate(perm):
                row.append((ii,jj))
            mask.append(row)
        mask = np.array(mask)
        row_indices = mask[:,:,0]
        col_indices = mask[:,:,1]
        V = V[row_indices, col_indices]
        return V
    
    #-----------------------------------------------------------------

    V = np.zeros((2*l+1, 2*l+1))
    for i in range(2*l+1):
        for j in range(0,i+1):
            V[i,j] = dic[str(i+1)+str(j+1)]
            V[j,i] = dic[str(i+1)+str(j+1)]
    
    #da [0, -1, 1, -2, 2, -3, 3] a [0, 1, -1, 2, -2, 3, -3]
    V = permutation(V)
    if return_orcaV:
        return V/27.2113834/8065.54477

    #ruoto
    if rotangle_V is not False:
        if len(rotangle_V) > 3:
            V_rot = rotate_V_quat(V, l, rotangle_V, real)
        else:
            V_rot = rotate_V(V, l, *rotangle_V, real)

    #da [0, 1, -1, 2, -2, 3, -3] a [0, -1, 1, -2, 2, -3, 3]
    V_rot = permutation(V_rot)

    dic_rot = {}
    for i in range(2*l+1):
        for j in range(0,i+1):
            dic_rot[str(i+1)+str(j+1)] = V_rot[i,j]

    return dic_rot
    
def U_complex2real(l):
    # order: ... -3, -2, -1, 0, 1, 2, 3 ...
    dim = 2*l + 1
    U = 1j * np.zeros((dim, dim))
    mvalues = range(l, -l - 1, -1)
    for i, m in enumerate(mvalues):
        if m < 0:
            U[i, i] = (-1)**m / np.sqrt(2)
            U[dim - i - 1, i] = 1 / np.sqrt(2)
        elif m == 0:
            U[i, i] = 1.0
        else:
            U[i, i] = 1j / np.sqrt(2)
            U[dim - i - 1, i] = -(-1)**m * 1j / np.sqrt(2)
    return U

def read_xyz(filename):
    #the first two lines must be the number of atoms and a comment line
    #in the subsequent lines the first column must be the atom name

    filecoord = []
    for i,line in enumerate(open(filename).readlines()):
        if i>1:
            splitline = line.split()
            num = [float(num) for num in splitline if is_float(num)]
            filecoord.append([splitline[0]]+num)

    filecoord = np.array(filecoord)
    return filecoord

def from_Vint_to_Bkq_2(l, dic_, reverse=False):
    #equivalent to from_Vint_to_Bkq but with the possibility of doing the reverse operation

    dic_conv = copy.deepcopy(dic_)

    if reverse:
        #fill the dictionary with zeros
        for k in range(0,2*l+1,2):
            if str(k) not in dic_conv.keys():
                dic_conv[str(k)] = {}
            for q in range(k,-k-1,-1):
                if str(q) not in dic_conv[str(k)].keys():
                    dic_conv[str(k)][str(q)] = 0.0
        V = []
        for k in range(0,2*l+1,2):
            for q in range(0,k+1):
                dic_conv[str(k)][str(q)] /= np.sqrt((2*k+1)/(4*np.pi))*(-1)**q
                V.append(dic_conv[str(k)][str(q)])
                if q!=0:
                    dic_conv[str(k)][str(-q)] /= -np.sqrt((2*k+1)/(4*np.pi))*(-1)**q
                    V.append(dic_conv[str(k)][str(-q)])
    else:
        V = [dic_conv[key] for key in dic_conv.keys()]

    if l==2:

        B0_0 = [2./5.*np.sqrt(np.pi), 0, 2./5.*np.sqrt(np.pi), 0, 0, 2./5.*np.sqrt(np.pi), 0, 0, 0, 2./5.*np.sqrt(np.pi), 0, 0, 0, 0, 2./5.*np.sqrt(np.pi)]
        B2_0 = [np.sqrt(np.pi/5.)*(2.),0, np.sqrt(np.pi/5.),0, 0, np.sqrt(np.pi/5.), 0, 0, 0, np.sqrt(np.pi/5.)*(- 2.), 0, 0, 0, 0, np.sqrt(np.pi/5.)*(- 2.)]
        B2_1 = [0, 0, 0, -np.sqrt(4.*np.pi/5.)*1/np.sqrt(2), 0, 0, 0, -np.sqrt(4.*np.pi/5.)*( np.sqrt(3)/np.sqrt(2)), 0, 0, 0, 0, -np.sqrt(4.*np.pi/5.)*( np.sqrt(3)/np.sqrt(2)), 0, 0]
        B2_m1 = [0, np.sqrt(4.*np.pi/5.)*1/np.sqrt(2), 0, 0, 0, 0, 0, 0,  np.sqrt(4.*np.pi/5.)*(-np.sqrt(3)/np.sqrt(2)*(-1)), 0, 0, np.sqrt(4.*np.pi/5.)*(-np.sqrt(3)/np.sqrt(2)), 0, 0, 0]
        B2_2 = [0, 0, -np.sqrt(4.*np.pi/5.)*np.sqrt(3)/(2.*np.sqrt(2)), 0, 0, -np.sqrt(4.*np.pi/5.)*(np.sqrt(3)/(2.*np.sqrt(2))*(-1)), 0, 0, 0, 0, -np.sqrt(4.*np.pi/5.)*np.sqrt(2), 0, 0, 0, 0]
        B2_m2 = [0, 0, 0, 0, -np.sqrt(4.*np.pi/5.)* np.sqrt(3)/np.sqrt(2), 0, -np.sqrt(4.*np.pi/5.)*(-np.sqrt(2)), 0, 0, 0, 0, 0, 0, 0, 0]
        B4_0 = [np.sqrt(np.pi)/5.*(6.),0, np.sqrt(np.pi)/5.*(-4.),0, 0, np.sqrt(np.pi)/5.*(-4.), 0, 0, 0, np.sqrt(np.pi)/5., 0, 0, 0, 0, np.sqrt(np.pi)/5.]
        B4_1 = [0, 0, 0, 2*np.sqrt(2.*np.pi/5.)*(-np.sqrt(3)/np.sqrt(2)), 0, 0, 0, 2*np.sqrt(2.*np.pi/5.)*1./(2.*np.sqrt(2)), 0, 0, 0, 0, 2*np.sqrt(2.*np.pi/5.)*(1./(2.*np.sqrt(2))), 0, 0]
        B4_m1 = [0, 2*np.sqrt(2.*np.pi/5.)*(np.sqrt(3)/np.sqrt(2)), 0, 0, 0, 0, 0, 0, 2*np.sqrt(2.*np.pi/5.)*(1./(2.*np.sqrt(2))*(-1)), 0, 0, 2*np.sqrt(2.*np.pi/5.)*1./(2.*np.sqrt(2)), 0, 0, 0]
        B4_2 = [0, 0, 2.*np.sqrt(2.*np.pi/5.)*(-1)/2., 0, 0, 2.*np.sqrt(2.*np.pi/5.)*1/2, 0, 0, 0, 0, 2.*np.sqrt(2.*np.pi/5.)*(np.sqrt(3)/2.), 0, 0, 0, 0]
        B4_m2 = [0, 0, 0, 0, 2.*np.sqrt(2.*np.pi/5.)*(-1), 0, 2.*np.sqrt(2.*np.pi/5.)*(-np.sqrt(3)/2.), 0, 0, 0, 0, 0, 0, 0, 0]
        B4_3 = [0, 0, 0, 0, 0, 0, 0, np.sqrt(7*np.pi/5.), 0, 0, 0, 0, np.sqrt(7*np.pi/5.)*(-1), 0, 0]
        B4_m3 = [0, 0, 0, 0, 0, 0, 0, 0, np.sqrt(7*np.pi/5.), 0, 0, np.sqrt(7*np.pi/5.), 0, 0, 0]
        B4_4 = [0, 0, 0, 0, 0, 0, 0, 0, 0, np.sqrt(7.*np.pi/10.)*(-1), 0, 0, 0, 0, np.sqrt(7.*np.pi/10.)]
        B4_m4 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -np.sqrt(7.*np.pi/5.)*np.sqrt(2), 0]

        conv_matrix = np.vstack((B0_0, B2_0, B2_1, B2_m1, B2_2, B2_m2, B4_0, B4_1, B4_m1, B4_2, B4_m2, B4_3, B4_m3, B4_4, B4_m4))


    elif l==3:

        B0_0 = [(2./7.)*np.sqrt(np.pi), 0, (2./7.)*np.sqrt(np.pi), 0, 0, (2./7.)*np.sqrt(np.pi), 0, 0, 0, (2./7.)*np.sqrt(np.pi), 0, 0, 0, 0, (2./7.)*np.sqrt(np.pi), 0, 0, 0, 0, 0, (2./7.)*np.sqrt(np.pi), 0, 0, 0, 0, 0, 0, (2./7.)*np.sqrt(np.pi)]
        B2_0 = [(2./7.)*np.sqrt(5*np.pi), 0, (3./14.)*np.sqrt(5*np.pi), 0, 0, (3./14.)*np.sqrt(5*np.pi), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, - (5./14.)*np.sqrt(5*np.pi), 0, 0, 0, 0, 0, 0, - (5./14.)*np.sqrt(5*np.pi)]
        B2_1 = [0, 0, 0, - (1./7.)*np.sqrt(5*np.pi), 0, 0, 0,  (5./14.)*np.sqrt(3*np.pi)*(-1), 0, 0, 0, 0, (5./14.)*np.sqrt(3*np.pi)*(-1), 0, 0, 0, 0, 0, (5./14.)*np.sqrt(5*np.pi)*(-1), 0, 0, 0, 0, 0, 0, (5./14.)*np.sqrt(5*np.pi)*(-1), 0, 0]
        B2_m1 = [0, (1./7.)*np.sqrt(5*np.pi), 0, 0, 0, 0, 0, 0, (5./14.)*np.sqrt(3*np.pi), 0, 0, (5./14.)*np.sqrt(3*np.pi)*(-1), 0, 0, 0, 0, 0, 0, 0, (5./14.)*np.sqrt(5*np.pi), 0, 0, 0, 0, (5./14.)*np.sqrt(5*np.pi)*(-1), 0, 0, 0]
        B2_2 = [0, 0, (1./7.)*np.sqrt(30*np.pi)*(-1/2), 0, 0, (1./7.)*np.sqrt(30*np.pi)*(1/2), 0, 0, 0, 0, (5./7.)*np.sqrt(2*np.pi)*(-1), 0, 0, 0, 0, 0, (5./7.)*np.sqrt(np.pi/2)*(-1), 0, 0, 0, 0, 0, 0, (5./7.)*np.sqrt(np.pi/2)*(-1), 0, 0, 0, 0]
        B2_m2 = [0, 0, 0, 0, (1./7.)*np.sqrt(30*np.pi)*(-1), 0, (5./7.)*np.sqrt(2*np.pi), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (5./7.)*np.sqrt(np.pi/2), 0, 0, 0, 0, (5./7.)*np.sqrt(np.pi/2)*(-1), 0, 0, 0, 0, 0]
        B4_0 = [(1./7.)*np.sqrt(np.pi)*(6), 0, (1./7.)*np.sqrt(np.pi), 0, 0, (1./7.)*np.sqrt(np.pi), 0, 0, 0, -np.sqrt(np.pi), 0, 0, 0, 0,  -np.sqrt(np.pi), 0, 0, 0, 0, 0, (1./7.)*np.sqrt(np.pi)*(3), 0, 0, 0, 0, 0, 0, (1./7.)*np.sqrt(np.pi)*(3)]
        B4_1 = [0, 0, 0, (1./7.)*np.sqrt(30*np.pi)*(-1), 0, 0, 0, (4./7.)*np.sqrt(2*np.pi)*(-1), 0, 0, 0, 0, (4./7.)*np.sqrt(2*np.pi)*(-1), 0, 0, 0, 0, 0, (1./7.)*np.sqrt(30*np.pi), 0, 0, 0, 0, 0, 0, (1./7.)*np.sqrt(30*np.pi), 0, 0]
        B4_m1 = [0, (1./7.)*np.sqrt(30*np.pi), 0, 0, 0, 0, 0, 0, (4./7.)*np.sqrt(2*np.pi), 0, 0, (4./7.)*np.sqrt(2*np.pi)*(-1), 0, 0, 0, 0, 0, 0, 0, (1./7.)*np.sqrt(30*np.pi)*(-1), 0, 0, 0, 0, (1./7.)*np.sqrt(30*np.pi), 0, 0, 0]
        B4_2 = [0, 0, (2./7.)*np.sqrt(10*np.pi)*(-1/2), 0, 0, (2./7.)*np.sqrt(10*np.pi)*(1/2), 0, 0, 0, 0, (1./7.)*np.sqrt(6*np.pi)*(-1), 0, 0, 0, 0, 0, (3./7.)*np.sqrt(6*np.pi), 0, 0, 0, 0, 0, 0, (3./7.)*np.sqrt(6*np.pi), 0, 0, 0, 0]
        B4_m2 = [0, 0, 0, 0, (2./7.)*np.sqrt(10*np.pi)*(-1), 0, (1./7.)*np.sqrt(6*np.pi), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (3./7.)*np.sqrt(6*np.pi)*(-1), 0, 0, 0, 0, (3./7.)*np.sqrt(6*np.pi), 0, 0, 0, 0, 0]
        B4_3 = [0, 0, 0, 0, 0, 0, 0, np.sqrt((2./7.)*np.pi), 0, 0, 0, 0, np.sqrt((2./7.)*np.pi)*(-1), 0, 0, 0, 0, 0, 0, 0, 0, 3*np.sqrt((2./7.)*np.pi), 0, 0, 0, 0, 0, 0]
        B4_m3 = [0, 0, 0, 0, 0, 0, 0, 0, np.sqrt((2./7.)*np.pi), 0, 0, np.sqrt((2./7.)*np.pi), 0, 0, 0, - 3*np.sqrt((2./7.)*np.pi), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        B4_4 = [0, 0, 0, 0, 0, 0, 0, 0, 0, np.sqrt((10./7.)*np.pi)*(-1/2), 0, 0, 0, 0, np.sqrt((10./7.)*np.pi)*(1/2), 0, np.sqrt((6./7.)*np.pi), 0, 0, 0, 0, 0, 0, np.sqrt((6./7.)*np.pi)*(-1), 0, 0, 0, 0] 
        B4_m4 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, np.sqrt((10./7.)*np.pi)*(-1), 0, 0, 0, np.sqrt((6./7.)*np.pi), 0, 0, 0, 0, np.sqrt((6./7.)*np.pi), 0, 0, 0, 0, 0]
        B6_0 = [(1./7.)*np.sqrt(13*np.pi)*(2), 0, (1./7.)*np.sqrt(13*np.pi)*(-(3./2.)), 0, 0, (1./7.)*np.sqrt(13*np.pi)*(-(3./2.)), 0, 0, 0, (1./7.)*np.sqrt(13*np.pi)*(3./5.), 0, 0, 0, 0, (1./7.)*np.sqrt(13*np.pi)*(3./5.), 0, 0, 0, 0, 0, -(1./7.)*np.sqrt(13*np.pi)*(1./10.), 0, 0, 0, 0, 0, 0, -(1./7.)*np.sqrt(13*np.pi)*(1./10.)]
        B6_1 = [0, 0, 0, np.sqrt((13./7.)*np.pi)*(-1), 0, 0, 0, np.sqrt((13./7.)*np.pi)*(1./2.)*np.sqrt(3./5.), 0, 0, 0, 0, np.sqrt((13./7.)*np.pi)*(1./2.)*np.sqrt(3./5.), 0, 0, 0, 0, 0, np.sqrt((13./7.)*np.pi)*(1./10.)*(-1), 0, 0, 0, 0, 0, 0, np.sqrt((13./7.)*np.pi)*(1./10.)*(-1), 0, 0]
        B6_m1 = [0, np.sqrt((13./7.)*np.pi), 0, 0, 0, 0, 0, 0, np.sqrt((13./7.)*np.pi)*(1./2.)*np.sqrt(3./5.)*(-1), 0, 0, np.sqrt((13./7.)*np.pi)*(1./2.)*np.sqrt(3./5.), 0, 0, 0, 0, 0, 0, 0, np.sqrt((13./7.)*np.pi)*(1./10.), 0, 0, 0, 0, np.sqrt((13./7.)*np.pi)*(1./10.)*(-1), 0, 0, 0]
        B6_2 = [0, 0, np.sqrt((13./7.)*np.pi)*(-1./2.)*np.sqrt(3./5.), 0, 0, np.sqrt((13./7.)*np.pi)*(1./2.)*np.sqrt(3./5.), 0, 0, 0, 0, np.sqrt((13./7.)*np.pi)*4/5, 0, 0, 0, 0, 0, np.sqrt((13./7.)*np.pi)*(1./5.)*(-1), 0, 0, 0, 0, 0, 0, np.sqrt((13./7.)*np.pi)*(1./5.)*(-1), 0, 0, 0, 0]
        B6_m2 = [0, 0, 0, 0, np.sqrt((13./7.)*np.pi)*(-np.sqrt(3./5.)), 0, np.sqrt((13./7.)*np.pi)*(- 4/5), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, np.sqrt((13./7.)*np.pi)*(1./5.), 0, 0, 0, 0, np.sqrt((13./7.)*np.pi)*(1./5.)*(-1), 0, 0, 0, 0, 0]
        B6_3 = [0, 0, 0, 0, 0, 0, 0, (3./5.)*np.sqrt((39./14.)*np.pi), 0, 0, 0, 0, (3./5.)*np.sqrt((39./14.)*np.pi)*(-1), 0, 0, 0, 0, 0, 0, 0, 0, (1./5.)*(np.sqrt((78./7.)*np.pi))*(-1), 0, 0, 0, 0, 0, 0]
        B6_m3 = [0, 0, 0, 0, 0, 0, 0, 0, (3./5.)*np.sqrt((39./14.)*np.pi), 0, 0, (3./5.)*np.sqrt((39./14.)*np.pi), 0, 0, 0, (1./5.)*(np.sqrt((78./7.)*np.pi)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        B6_4 = [0, 0, 0, 0, 0, 0, 0, 0, 0, (3./5.)*np.sqrt((26./7.)*np.pi)*(-1/2), 0, 0, 0, 0, (3./5.)*np.sqrt((26./7.)*np.pi)*(1/2), 0, np.sqrt((39./70.)*np.pi)*(-1), 0, 0, 0, 0, 0, 0, np.sqrt((39./70.)*np.pi), 0, 0, 0, 0]
        B6_m4 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (3./5.)*np.sqrt((26./7.)*np.pi)*(-1), 0, 0, 0, np.sqrt((39./70.)*np.pi)*(-1), 0, 0, 0, 0, np.sqrt((39./70.)*np.pi)*(-1), 0, 0, 0, 0, 0]
        B6_5 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (1./5.)*np.sqrt((429./14.)*np.pi), 0, 0, 0, 0, 0, 0, (1./5.)*np.sqrt((429./14.)*np.pi)*(-1), 0, 0]
        B6_m5 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (1./5.)*np.sqrt((429./14.)*np.pi), 0, 0, 0, 0, (1./5.)*np.sqrt((429./14.)*np.pi), 0, 0, 0]
        B6_6 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (1./5.)*(np.sqrt((429./7.)*np.pi))*(-1/2), 0, 0, 0, 0, 0, 0, (1./5.)*(np.sqrt((429./7.)*np.pi))*(1/2)]  
        B6_m6 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (1./5.)*(np.sqrt((429./7.)*np.pi))*(-1), 0] 

        conv_matrix = np.vstack((B0_0, B2_0, B2_1, B2_m1, B2_2, B2_m2, B4_0, B4_1, B4_m1, B4_2, B4_m2, B4_3, B4_m3, B4_4, B4_m4, B6_0, B6_1, B6_m1, B6_2, B6_m2, B6_3, B6_m3, B6_4, B6_m4, B6_5, B6_m5, B6_6, B6_m6))

    
    if not reverse:
        B = np.dot(conv_matrix, V)
        dic_Bkq = {}
        count = 0
        for k in range(0,2*l+1,2):
            dic_Bkq[str(k)] = {}
            for q in range(0,k+1):
                dic_Bkq[str(k)][str(q)] = np.sqrt((2*k+1)/(4*np.pi))*(-1)**q*B[count]
                count += 1
                if q!=0:
                    dic_Bkq[str(k)][str(-q)] = -np.sqrt((2*k+1)/(4*np.pi))*(-1)**q*B[count]
                    count += 1 

        return dic_Bkq
    
    else:
        B = np.dot(np.linalg.inv(conv_matrix), V)
        dic_V = {}
        count = 0
        for k in range(0,2*l+1):
            for q in range(0,k+1):
                dic_V[str(k+1)+str(q+1)] = B[count]
                count += 1

        return dic_V

def Rzyz_active(alpha=0,beta=0,gamma=0):
    #proper-euler angles
    #active rotation

    ca = np.cos(alpha)
    cb = np.cos(beta)
    cc = np.cos(gamma)
    sa = np.sin(alpha)
    sb = np.sin(beta)
    sc = np.sin(gamma)
    
    R1 = np.array([ca*cb*cc-(sa*sc), -ca*cb*sc-(sa*cc), ca*sb])
    R2 = np.array([ca*sc+(sa*cb*cc), -sa*cb*sc+(ca*cc), sa*sb])
    R3 = np.array([-sb*cc, sb*sc, cb])
    R = np.stack((R1, R2, R3), axis=0)
    
    return R 

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
