#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: letizia  (8/11/2023)
"""

import nja_cfs_v0 as nja
import functools
from datetime import datetime
import numpy as np
import scipy
from pprint import pprint
import matplotlib.pyplot as plt
import sympy
import copy
import crystdat

##############
class color_term:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
##############

#if the second number of the version is even, numba is used
numba_flag = False
if eval(nja.__version__.split('.')[1])%2==0:
    numba_flag = True

def test(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"\nRunning test: "+color_term.BOLD+func.__name__+color_term.END)
        start_time = datetime.now()
        try:
            func(*args, **kwargs)
            print(f"Test "+color_term.GREEN+"PASSED"+color_term.END)
        except AssertionError:
            print(f"Test "+color_term.RED+"NOT PASSED"+color_term.END)
        finally:
            end_time = datetime.now()
            print('Execution time: {}'.format(end_time - start_time))
    return wrapper

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

@test
def test_CF_splitting():

    # Example usage to add multiple crystal field splittings
    fig, ax = plt.subplots()

    conf = 'd3'
    contributes = ['Hee', 'Hcf', 'Hso']

    data = nja.read_data('test/Td_cube.inp', sph_flag = False)
    data[:,-1] *= -1
    data[:,1:-1] *= 2/(2*np.sqrt(3))

    dic_Bkq = nja.calc_Bkq(data, conf, False, False)
    dic_V = nja.from_Vint_to_Bkq_2(2, dic_Bkq, reverse=True)
    matrix = np.zeros((5,5))
    for i in range(5):
        for j in range(5):
            if i>=j:
                matrix[i,j] = dic_V[str(i+1)+str(j+1)]
                matrix[j,i] = dic_V[str(i+1)+str(j+1)]

    w,v = np.linalg.eigh(matrix)

    nja.plot_energy_levels(w-np.min(w), ax=ax, color='magenta', label="Td", delta=0)  

    data = nja.read_data('test/Oh_cube.inp', sph_flag = False)
    data[:,-1] *= -1
    dic_Bkq = nja.calc_Bkq(data, conf, False, False)
    dic_V = nja.from_Vint_to_Bkq_2(2, dic_Bkq, reverse=True)
    matrix = np.zeros((5,5))
    for i in range(5):
        for j in range(5):
            if i>=j:
                matrix[i,j] = dic_V[str(i+1)+str(j+1)]
                matrix[j,i] = dic_V[str(i+1)+str(j+1)]

    w,v = np.linalg.eigh(matrix)

    nja.plot_energy_levels(w-np.min(w), ax=ax, color='green', label="Oh", delta=0.5)

    plt.show()

@test
def test_plot_Ediagram():

    conf = 'd8'
    calc = nja.calculation(conf, TAB=False, wordy=False)
    basis, dic_LS, basis_l, basis_l_JM = nja.Full_basis(conf)

    dic_orca = nja.read_AILFT_orca6('test/calcsuscenisalfix.out', conf, method='CASSCF', return_V=False, rotangle_V=False, return_orcamatrix=False)

    contributes = ['Hee', 'Hcf', 'Hso']
    theories = ['Hee', 'Hee + Hcf', 'Hee + Hcf + Hso']
    list_contr = []
    E_matrix = []
    proj_LS_dict = {}
    proj_prev_dict = {}
    prev = np.zeros((basis.shape[0]+1,basis.shape[0]), dtype='complex128')
    for i in range(len(contributes)):
        list_contr.append(contributes[i])
        result = calc.MatrixH(list_contr, **dic_orca, field=[0.0,0.0,28.0], wordy=False, ground_proj=False, return_proj=False)
        if i==0:
            E0 = np.min(result[0,:].real)
        result[0,:] = result[0,:].real-E0

        proj_LS = nja.projection_basis(result[1:,:], basis_l, J_label=False)

        proj_LS_dict[theories[i]] = proj_LS
        if i==0:
            pass
        else:
            proj_prev = nja.projection(result[1:,:], basis_l, prev[1:,:], prev[0,:].real)
            proj_prev_dict[theories[i]] = proj_prev

        E_matrix.append([round(result[0,ii].real,3) for ii in range(result.shape[-1])])  

        prev = result.copy()

    E_matrix = np.array(E_matrix)  # dim = (n. contributi x n. livelli)

    #plot energy levels
    nja.level_fig_tot(E_matrix, theories, proj_LS_dict, proj_prev_dict)

@test
def test_plot_Ediagram_PCM():

    conf = 'f12'
    calc = nja.calculation(conf, TAB=False, wordy=False)
    basis, _, basis_l, _ = nja.Full_basis(conf)

    data = nja.read_data('test/beta.inp', sph_flag = False)
    data[:,-1] *= -1
    dic_Bkq = nja.calc_Bkq(data, conf, False, True)
    dic_PCM = nja.free_ion_param_f_HF(conf)
    dic_PCM['dic_bkq'] = dic_Bkq
    
    contributes = ['Hee', 'Hso', 'Hcf']
    theories = ['Hee', 'Hee + Hso', 'Hee + Hso + Hcf']
    list_contr = []
    E_matrix = []
    proj_LS_dict = {}
    proj_prev_dict = {}
    prev = np.zeros((basis.shape[0]+1,basis.shape[0]), dtype='complex128')
    for i in range(len(contributes)):
        list_contr.append(contributes[i])
        result = calc.MatrixH(list_contr, **dic_PCM, field=[0.0,0.0,28.0], wordy=False, ground_proj=False, return_proj=False)
        if i==0:
            E0 = np.min(result[0,:].real)
        result[0,:] = result[0,:].real-E0

        proj_LS = nja.projection_basis(result[1:,:], basis_l, J_label=False)

        proj_LS_dict[theories[i]] = proj_LS
        if i==0:
            pass
        else:
            proj_prev = nja.projection(result[1:,:], basis_l, prev[1:,:], prev[0,:].real)
            proj_prev_dict[theories[i]] = proj_prev

        E_matrix.append([round(result[0,ii].real,3) for ii in range(result.shape[-1])])  

        prev = result.copy()

    E_matrix = np.array(E_matrix)  # dim = (n. contributi x n. livelli)

    #plot energy levels
    nja.level_fig_tot(E_matrix, theories, proj_LS_dict, proj_prev_dict)

@test
def test_TanabeSugano():

    def red_proj_LS(proj_LS):
        state_order = []
        for num in proj_LS.keys():
            values = list(proj_LS[num].values())
            state_n = np.argmax(values)
            state_order.append(list(proj_LS[num].keys())[state_n])
        return state_order

    conf = 'd8'
    B = 1030

    data = nja.read_data('test/D4h.inp', sph_flag = False)
    data[:,-1] *= -1*0.06

    calc = nja.calculation(conf, ground_only=False, TAB=True, wordy=True)

    #first point at 0 CF
    dic = nja.free_ion_param_AB(conf)
    result = calc.MatrixH(['Hee', 'Hso'], **dic, eig_opt=False, wordy=False)
    proj_LS = nja.projection_basis(result[1:,:], calc.basis_l)

    diagram = [(result[0,:]-np.min(result[0,:]))/B]
    x_axis = [0.0]
    spacing = np.arange(0.01,12,0.2)
    print(len(spacing))
    for i in range(len(spacing)):
        data_mult = copy.deepcopy(data)
        data_mult[:,-1] *= spacing[i]
        dic = nja.free_ion_param_AB(conf)
        dic_Bkq = nja.calc_Bkq(data_mult, conf, False, False)
        dic_V = nja.from_Vint_to_Bkq_2(2, dic_Bkq, reverse=True)
        matrix = np.zeros((5,5))
        for ii in range(5):
            for j in range(5):
                if ii>=j:
                    matrix[ii,j] = dic_V[str(ii+1)+str(j+1)]
                    matrix[j,ii] = dic_V[str(ii+1)+str(j+1)]
        w,v = np.linalg.eigh(matrix)
        # proj_LS = nja.projection_basis(result[1:,:], calc.basis_l)
        # names = red_proj_LS(proj_LS)
        # names_list.append(names)
        x_axis.append(-(w[0]-w[-1])/B)
        print(f'{i}    {x_axis[-1]/10}       ',end='\r')
        dic['dic_bkq'] = dic_Bkq
        result = calc.MatrixH(['Hee','Hcf', 'Hso'], **dic, eig_opt=False, wordy=False)
        diagram.append((result[0,:]-np.min(result[0,:]))/B)

    diagram = np.array(diagram)
    # names = np.array(names_list)
    # terms = nja.terms_labels(conf)
    # colormap = plt.cm.get_cmap('tab10', 10)  # You can choose any colormap
    # colors = [colormap(i) for i in range(10)]
    # fig, ax = plt.subplots()
    # for i in range(diagram.shape[1]):
    #     for j in range(diagram.shape[0]):
    #         ax.scatter(np.array(x_axis[j])/10, diagram[j,i], color=colors[terms.index(names[j,i])], lw=0.5)
    # plt.show()

    fig, ax = plt.subplots()
    for i in range(diagram.shape[1]):
        ax.plot(np.array(x_axis)/10, diagram[:,i].real, 'k', lw=0.5)
    plt.ylabel('E/B')
    plt.xlabel('Dq/B')
    plt.show()

@test
def test_plot_magnetization_field():

    def use_nja_(conf, data, field_vecs, wordy=False):

        calc = nja.calculation(conf, ground_only=True, TAB=True, wordy=wordy)
        data_nja = np.copy(data)
        data_nja[:,-1] *= (-1)
        
        dic = nja.free_ion_param_f(conf)
        dic_Bkq = nja.calc_Bkq(data_nja, conf, False, True)
        dic['dic_bkq'] = dic_Bkq

        Magn = nja.Magnetics(calc, ['Hcf','Hz'], dic)
        Magnv, *_ = Magn.susceptibility_field(fields=field_vecs, temp=298., delta = 0.01, wordy=wordy)

        chi_B = Magn.susceptibility_B_copy(np.array([[0.0,0.0,1.0]]), 298., delta = 0.1)[0]

        w,v = np.linalg.eigh(chi_B)
        w,v = nja.princ_comp(w,v)
        plot_chi(v, w, data)

        return Magnv
    
    def plot_chi(v, w, data):   
        fig = plt.figure(figsize=(4,4))
        ax = fig.add_subplot(111, projection='3d')

        for i in range(data.shape[0]):
            vector = data[i,1:-1]
            ax.plot([0.,vector[0]],[0.,vector[1]],[0.,vector[2]],'--',lw=1,c='k')
            if data[i,0] in nja.color_atoms().keys():
                ax.scatter(vector[0],vector[1],vector[2],'o',c = nja.color_atoms()[data[i,0]],lw=8)
            else:
                ax.scatter(vector[0],vector[1],vector[2],'o',c = nja.color_atoms()['_'],lw=8)

        vectors = v.T
        labels = ['x','y','z']
        for i in range(v.shape[0]):
            xline, yline, zline = [0.,vectors[i,0]], [0.,vectors[i,1]], [0.,vectors[i,2]]
            ax.quiver(0.,0.,0.,vectors[i,0],vectors[i,1],vectors[i,2],color='r')
            ax.text(vectors[i,0],vectors[i,1],vectors[i,2], labels[i])

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.set_zlim(-6, 6)
        plt.show()
        #exit()

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
                if data[i,0] in nja.color_atoms().keys():
                    ax.scatter(vector[0],vector[1],vector[2],'o',c = nja.color_atoms()[data[i,0]],lw=3)
                else:
                    ax.scatter(vector[0],vector[1],vector[2],'o',c = nja.color_atoms()['_'],lw=3)
                ax.text(vector[0]+0.4*np.sign(vector[0]),vector[1]+0.4*np.sign(vector[1]),vector[2]+0.4*np.sign(vector[2]),data[i,-1], size=8)

        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_zlim(-4, 4)

        # Remove the grid
        ax.grid(False)

        plt.show()

    conf = 'f9'

    vec_field = np.array([0,0,1])

    data = nja.read_data('test/bbpn.inp', sph_flag = False)
    # coord = data[:,1:-1]

    rep2000_cryst = np.array(crystdat.rep168_cryst)

    xyz = np.zeros((rep2000_cryst.shape[0],3))

    for i in range(rep2000_cryst.shape[0]):

        angles = rep2000_cryst[i,:]
        a = angles[0]
        b = angles[1]

        r = scipy.spatial.transform.Rotation.from_euler('ZYZ', [0,b,a], degrees=True)
        R = r.as_matrix()

        xyz[i,:] = R.T@vec_field    

    Mvec = use_nja_(conf, data, xyz, wordy=False)

    fig_rep_magnfield(Mvec, xyz, data)

@test
def test_torque():
    # Aqkrk from https://pubs.acs.org/doi/10.1021/jp0209244 table 3
    # equation for torque computation from https://www.sciencedirect.com/science/article/pii/S0010854517302515?via%3Dihub 

    conf = 'f8'
    contributes = ['Hcf']
    
    dic_Aqkrk = {'2':{'0':293.0},
               '4':{'0':-197.0, '4':863.0},
               '6':{'0':15.1, '4':357.0}}

    dic_Bkq = nja.from_Aqkrk_to_Bkq(dic_Aqkrk)
    dic = {}
    dic['dic_bkq'] = dic_Bkq

    calc = nja.calculation(conf, ground_only=True, TAB=True, wordy=False)
    _, _ = calc.MatrixH(contributes, **dic, eig_opt=False, wordy=False, ground_proj=True, return_proj=True, save_label=True, save_LF=True)
    basis = np.loadtxt('matrix_label.txt')
    LF_matrix = np.load('matrix_LF.npy', allow_pickle=True, fix_imports=False)

    B0 = [0.1, 2, 5] #T
    T = 2.0 #K
    #returns angles and torque values
    #if plane is 'zx' then the returned tau is along the y axis and the field is placed on z (i.e. the first axis in 'zx' label)
    x,y = nja.calc_torque(B0, T, LF_matrix, basis, plane='zx', figure='test/tau_vecs.png', show_fig=True)
       
@test
def test_PCM():  #with SIMPRE

    def read_out(J, directory=None):

        if directory is None:
            directory = 'test/'
        out = open(directory+'simpre.out').readlines()

        eigen_num = None
        cfp_num = None
        dim = int(2*J+1)
        matrix = np.zeros((dim,dim+1), dtype=float)
        cfp_matrix = np.zeros((27,4), dtype=float)
        count = 0
        count2 = 0
        for i,line in enumerate(out):

            if 'Eigenvalues (cm-1) and eigenvectors (modulus square)' in line:
                eigen_num = i+4

            if eigen_num is not None:
                if eigen_num%2 != 0:
                    splitline = out[eigen_num].split()
                    matrix[count,:] = [float(splitline[ii]) for ii in range(0,dim+1)]
                    count += 1
                eigen_num += 1
                if count == dim:
                    break

            if '   k   q         Akq <rk>             Bkq' in line:
                cfp_num = i+3

            if cfp_num is not None:
                splitline = out[cfp_num].split()
                cfp_matrix[count2,:] = [float(num) for num in splitline]
                count2 += 1
                cfp_num += 1
                if count2 == 27:
                    cfp_num = None

        return matrix, cfp_matrix

    J = 15/2
    data = nja.read_data('test/beta.inp', sph_flag = False)
    data[:,-1] *= -1

    matrix, cfp_matrix = read_out(J, directory='test/')
    MJ_list = np.arange(-J,J+1,1)
    dic_proj = {}
    for j in range(matrix.shape[0]):
        dic_proj[str(j+1)] = {str(MJ_list[i-1]): matrix[j,i] for i in range(1,matrix.shape[1])}

    conf = 'f9'
    contributes = ['Hcf']

    dic_Aqkrk = nja.calc_Aqkrk(data, conf, False, True) 
    dic_Bqk = nja.calc_Bqk(data, conf, False, True)

    ### There's a sign mismatch between Bkq computed with SIMPRE and those from NJA
    ### using the Bkq from SIMPRE the susceptibility tensor orientation in Dybbpn complex is incorrect
    
    # Aqkrk_simpre = cfp_matrix[:,2]
    # Bqk_simpre = cfp_matrix[:,3]
    # Aqkrk_nja = np.array([dic_Aqkrk[str(int(k))][str(int(q))] for k,q in cfp_matrix[:,0:2]])
    # Bqk_nja = np.array([dic_Bqk[str(int(k))][str(int(q))] for k,q in cfp_matrix[:,0:2]])
    # pprint(Aqkrk_simpre)
    # pprint(Aqkrk_nja)
    # assert np.allclose(Aqkrk_simpre, Aqkrk_nja, rtol=1e-5, atol=1e-5)
    # assert np.allclose(Bqk_simpre, Bqk_nja, rtol=1e-5, atol=1e-5)

    dic_Bkq2 = nja.from_Aqkrk_to_Bkq(dic_Aqkrk, revers=False)
    dic_Bkq = nja.calc_Bkq(data, conf, False, True)
    Bkq_recalc = np.array([dic_Bkq2[str(int(k))][str(int(q))] for k,q in cfp_matrix[:,0:2]])
    Bkq_calc = np.array([dic_Bkq[str(int(k))][str(int(q))] for k,q in cfp_matrix[:,0:2]])

    assert np.allclose(Bkq_recalc, Bkq_calc, rtol=1e-5, atol=1e-5)

    dic = {'dic_bkq': dic_Bkq2}
    calc = nja.calculation(conf, ground_only=True, TAB=True, wordy=False)
    result, proj_nja = calc.MatrixH(contributes, **dic, eig_opt=False, return_proj=True, ground_proj=True)

    E_nja = result[0,:].real-min(result[0,:].real)
    E_simpre = matrix[:,0]

    assert np.allclose(E_nja, E_simpre, rtol=1e-5, atol=1e-5)

    j=0
    keys_list = list(proj_nja[j+1].keys())
    sliced_keys = [eval(key[key.index(') ')+1:]) for key in keys_list]
    for i in range(1,matrix.shape[1]):
        if MJ_list[i-1] in sliced_keys:
            for ii, key in enumerate(sliced_keys): #in the composition MJ and -MJ are swapped due to the sign mismatch
                if MJ_list[i-1] == key:
                    ind = ii
            #assert np.allclose(proj_nja[j+1][keys_list[ind]]/100, matrix[j,i], rtol=1e-3, atol=1e-3)

@test
def test_PCM_2():  
    #from simpre2, table 2

    data = nja.read_data('test/example_simpre.inp', sph_flag = True)
    data[:,-1] *= -1
    
    conf = 'f10'

    dic_Aqkrk = nja.calc_Aqkrk(data, conf, False, True) 

    assert np.allclose(dic_Aqkrk['2']['0'],238.2, rtol=1e-3, atol=1.0)
    assert np.allclose(dic_Aqkrk['4']['0'],-83.4, rtol=1e-3, atol=1.0)
    assert np.allclose(dic_Aqkrk['4']['4'],872.8, rtol=1e-3, atol=1.0)
    assert np.allclose(dic_Aqkrk['4']['-4'],39.9, rtol=1e-3, atol=1.0)
    assert np.allclose(dic_Aqkrk['6']['0'],-7.0, rtol=1e-3, atol=1.0)
    assert np.allclose(dic_Aqkrk['6']['4'],382.5, rtol=1e-3, atol=1.0)
    assert np.allclose(dic_Aqkrk['6']['-4'],68.2, rtol=1e-3, atol=1.0)

    dic_Bkq2 = nja.from_Aqkrk_to_Bkq(dic_Aqkrk, revers=False)
    dic_Bkq = nja.calc_Bkq(data, conf, False, True)

    for k in range(2,7,2):
        for q in range(-k, k+1, 1):
            if k in dic_Bkq.keys() and q in dic_Bkq[str(int(k))].keys():
                assert dic_Bkq2[str(int(k))][str(int(q))]==dic_Bkq[str(int(k))][str(int(q))]
            
@test
def test_conv_AqkrkBkq():
    #conversion test from Bkq(Wyb) in PCM and Aqkrk(Stev) in PCM

    conf = 'f11'
    data = nja.read_data('test/beta.inp', sph_flag = False)
    data[:,-1] *= -1
    dicAqkrk = nja.calc_Aqkrk(data, conf)
    dicBkq = nja.calc_Bkq(data, conf)
    dicBkq_conv = nja.from_Aqkrk_to_Bkq(dicAqkrk, revers=False)

    for k in range(2,7,2):
        for q in range(-k, k+1, 1):
            assert round(dicBkq_conv[f'{k}'][f'{q}'], 9) == round(dicBkq[f'{k}'][f'{q}'], 9)

@test
def test_StevensfromMOLCAS():

    conf = 'f9'
    contributes = ['Hcf']
    
    #comparison with the results from: J. Am. Chem. Soc. 2021, 143, 8108−8115
    cfp_list = np.loadtxt('test/CFP_DyDOTA.txt')
    dic_Aqkrk = {}
    count = 0
    for k in range(2,7,2):
        dic_Aqkrk[f'{k}'] = {}
        for q in range(k,-k-1,-1):
            dic_Aqkrk[f'{k}'][f'{q}'] = cfp_list[count]/nja.Stev_coeff(str(k), conf)
            count += 1

    dic_Bkq = nja.from_Aqkrk_to_Bkq(dic_Aqkrk)
    dic = {}
    dic['dic_bkq'] = dic_Bkq
    dic_Bkq['0'] = {}
    dic_Bkq['0']['0'] = 0

    calc = nja.calculation(conf, ground_only=True, TAB=True)
    _, projected = calc.MatrixH(contributes, **dic, ground_proj=True, return_proj=True)
    Mground, perc = nja.the_highest_MJ(projected[1], np.arange(15/2,-0.5,-1)) 
    
    assert Mground == 15/2 and perc > 94.0

@test
def test_conv_Vint_Bkq_d():

    conf = 'd8'
    contributes = ['Hee', 'Hcf', 'Hso']

    #from NiSAL-HDPT calcsuscenisal_10.out NEVPT2
    #(xy yz z2 xz x2-y2)
    dic_V1_orca = {'11':-1537343.193973,
        '21':-197.966117, '22':-1536481.975521,
        '31':2138.341330, '32':2620.966044, '33':-1534906.147670,
        '41':2944.032701, '42':1955.080014, '43':3930.351693, '44':-1531161.910464,
        '51':-599.165743, '52':1115.150600, '53':102.275178, '54':-2462.285886, '55':-1535571.155802}
    
    #(z2 yz xz xy x2-y2)
    dic_V = {'11':dic_V1_orca['33'],
        '21':dic_V1_orca['32'], '22':dic_V1_orca['22'],
        '31':dic_V1_orca['43'], '32':dic_V1_orca['42'], '33':dic_V1_orca['44'],
        '41':dic_V1_orca['31'], '42':dic_V1_orca['21'], '43':dic_V1_orca['41'], '44':dic_V1_orca['11'],
        '51':dic_V1_orca['53'], '52':dic_V1_orca['52'], '53':dic_V1_orca['54'], '54':dic_V1_orca['51'], '55':dic_V1_orca['55']}

    dic_Bkq = nja.from_Vint_to_Bkq(dic_V, conf)
    dic_Bkq['0']['0'] = 0
    dic = {'F2': 85687.2, 'F4': 48274.5, 'zeta': 643.4, 'dic_bkq': dic_Bkq}

    calc = nja.calculation(conf, ground_only=False, TAB=True, wordy=False)
    result1, _ = calc.MatrixH(contributes, **dic, eig_opt=False, wordy=False, ground_proj=True, return_proj=True)
    
    dic_V1 = nja.from_Vint_to_Bkq_2(2, dic_Bkq, reverse=True)
    dic_Bkq = nja.from_Vint_to_Bkq(dic_V1, conf)
    dic_Bkq['0']['0'] = 0
    dic = {'F2': 85687.2, 'F4': 48274.5, 'zeta': 643.4, 'dic_bkq': dic_Bkq}
    
    calc = nja.calculation(conf, ground_only=False, TAB=True, wordy=False)
    result2, _ = calc.MatrixH(contributes, **dic, eig_opt=False, wordy=False, ground_proj=True, return_proj=True)
    
    assert np.allclose(result1.real, result2.real)
    assert np.allclose(result1.imag, result2.imag)

    dic = {'F2': 85687.2, 'F4': 48274.5, 'zeta': 643.4, 'dic_V': dic_V1}

    calc = nja.calculation(conf, ground_only=False, TAB=True, wordy=False)
    result3, _ = calc.MatrixH(contributes, **dic, eig_opt=False, wordy=False, ground_proj=True, return_proj=True)

    assert np.allclose(result1.real, result3.real)
    assert np.allclose(result1.imag, result3.imag)
    
@test
def test_conv_Vint_Bkq_f():

    conf = 'f13'
    contributes = ['Hee', 'Hcf', 'Hso']

    dic = nja.read_AILFT_orca6('test/run_YbDOTA.out', conf)
    # pprint(dic['dic_bkq'])
    # dic2 = dic.copy()

    calc = nja.calculation(conf, ground_only=False, TAB=False, wordy=False)
    result1, _ = calc.MatrixH(contributes, **dic, eig_opt=False, wordy=False, ground_proj=True, return_proj=True)
    
    dic_V1 = nja.from_Vint_to_Bkq_2(3, dic['dic_bkq'], reverse=True)
    # pprint(dic_V1)
    dic_Bkq = nja.from_Vint_to_Bkq_2(3, nja.read_AILFT_orca6('test/run_YbDOTA.out', conf, return_V=True))
    # pprint(dic_Bkq)
    # exit()
    #dic_Bkq['0']['0'] = 0
    dic['dic_bkq'] = dic_Bkq

    # for k in range(0,7,2):
    #     for q in range(-k, k+1, 1):
    #         if k in dic2.keys() and q in dic2[str(int(k))].keys():
    #             assert dic_Bkq[str(int(k))][str(int(q))]==dic2[str(int(k))][str(int(q))]
    # assert dic['zeta']==dic2['zeta']
    # assert dic['F2']==dic2['F2']
    # assert dic['F4']==dic2['F4']
    # assert dic['F6']==dic2['F6']
    #pprint(dic)
    
    calc = nja.calculation(conf, ground_only=False, TAB=False, wordy=False)
    result2, _ = calc.MatrixH(contributes, **dic, eig_opt=False, wordy=False, ground_proj=True, return_proj=True)
    
    assert np.allclose(result1.real, result2.real, rtol=2, atol=1e-3)
    assert np.allclose(result1.imag, result2.imag, rtol=2, atol=1e-3)

    # remove dic_bkq from dic
    del dic['dic_bkq']
    dic['dic_V'] = dic_V1

    calc = nja.calculation(conf, ground_only=False, TAB=False, wordy=False)
    result3, _ = calc.MatrixH(contributes, **dic, eig_opt=False, wordy=False, ground_proj=True, return_proj=True)

    assert np.allclose(result1.real, result3.real, rtol=2, atol=1e-3)
    assert np.allclose(result1.imag, result3.imag, rtol=2, atol=1e-3)

@test
def test_Wigner_Euler_quat():

    A = 5.23375087
    B = 0.68208491
    C = 5.56775468
    r = scipy.spatial.transform.Rotation.from_euler('ZYZ',[A,B,C])  
    R = r.as_quat()
    quat = [R[-1], R[0], R[1], R[2]]
    dict, coeff = nja.read_DWigner_quat()
    for k in range(2,7,2):
        D1 = np.zeros((2*k+1,2*k+1), dtype='complex128')
        for ii,m1 in enumerate(range(-k,k+1)):
            for jj,m in enumerate(range(-k,k+1)):
                D1[ii,jj] = nja.Wigner_coeff.Wigner_Dmatrix(k, m1, m, A, B, C)  

        D = nja.Wigner_coeff.Wigner_Dmatrix_quat_complete(k, quat, dict = dict, coeff=coeff)

        Dre = np.round(D.real,8)
        D1re = np.round(D1.real,8)
        Dim = np.round(D.imag,8)
        D1im = np.round(D1.imag,8)

        assert np.array_equal(D1re, Dre)
        assert np.array_equal(D1im, Dim)

@test
def test_Wigner_Euler_quat2():

    A = 5.23375087
    B = 0.68208491
    C = 5.56775468
    r = scipy.spatial.transform.Rotation.from_euler('ZYZ',[A,B,C])  
    R = r.as_quat()
    quat = [R[-1], R[0], R[1], R[2]]
    dict, coeff = nja.read_DWigner_quat()
    k=3
    D1 = np.zeros((2*k+1,2*k+1), dtype='complex128')
    for ii,m1 in enumerate(range(-k,k+1)):
        for jj,m in enumerate(range(-k,k+1)):
            D1[ii,jj] = nja.Wigner_coeff.Wigner_Dmatrix(k, m1, m, A, B, C)  

    D = nja.Wigner_coeff.Wigner_Dmatrix_quat_complete(k, quat, dict = dict, coeff=coeff)

    Dre = np.round(D.real,8)
    D1re = np.round(D1.real,8)
    Dim = np.round(D.imag,8)
    D1im = np.round(D1.imag,8)

    assert np.array_equal(D1re, Dre)
    assert np.array_equal(D1im, Dim)

@test
def test_LF_rotation_euler():

    conf = 'f9'
    contributes = ['Hee', 'Hcf', 'Hso']
    ground = nja.ground_term_legend(conf)
    splitg = ground.split('(')
    J = eval(splitg[-1][:-1])
    
    #comparison with the results from: J. Am. Chem. Soc. 2021, 143, 8108−8115 (by MOLCAS)
    cfp_list = np.loadtxt('test/CFP_DyDOTA.txt')
    dic_Aqkrk = {}
    count = 0
    for k in range(2,7,2):
        dic_Aqkrk[f'{k}'] = {}
        for q in range(k,-k-1,-1):
            dic_Aqkrk[f'{k}'][f'{q}'] = cfp_list[count]/nja.Stev_coeff(str(k), conf)
            count += 1

    dic_Bkq = nja.from_Aqkrk_to_Bkq(dic_Aqkrk)
    dic_Bkq['0'] = {}
    dic_Bkq['0']['0'] = 0

    #g ref syst for Dy from J. Am. Chem. Soc. 2021, 143, 8108−8115
    Rot_mat = np.array([[0.696343, 0.027550, -0.717180],[0.216884, 0.944468, 0.246864],[0.684155, -0.327447, 0.651698]])  
    # #chi ref syst for Dy from J. Am. Chem. Soc. 2021, 143, 8108−8115
    # Rot_mat = np.array([[ 0.72415202, -0.05048323, -0.68779015],[-0.14026268,  0.96569099, -0.21855747],[ 0.67522606,  0.25473979,  0.69222636]])

    R = scipy.spatial.transform.Rotation.from_matrix(Rot_mat).as_euler('ZYZ')

    #ruoto i bkq
    dic_Bkq_rot1 = nja.rota_LF(3, dic_Bkq, *R)
    dic_Bkq_rot1['0'] = {}
    dic_Bkq_rot1['0']['0'] = 0

    #converto i Bkq in V, ruoto V e poi ricalcolo i Bkq
    dic_V = nja.from_Vint_to_Bkq_2(3, dic_Bkq, reverse=True)
    dic_V = nja.rotate_dicV(dic_V, 3, rotangle_V=R, real=True)
    dic_Bkq_rot2 = nja.from_Vint_to_Bkq_2(3, dic_V, reverse=False)

    for k in dic_Bkq_rot1.keys():
        for q in dic_Bkq_rot1[k]:
            assert np.round(dic_Bkq_rot1[k][q],10)==np.round(dic_Bkq_rot2[k][q],10)

@test
def test_LF_rotation_quat():

    conf = 'f9'
    
    #comparison with the results from: J. Am. Chem. Soc. 2021, 143, 8108−8115 (by MOLCAS)
    cfp_list = np.loadtxt('test/CFP_DyDOTA.txt')
    dic_Aqkrk = {}
    count = 0
    for k in range(2,7,2):
        dic_Aqkrk[f'{k}'] = {}
        for q in range(k,-k-1,-1):
            dic_Aqkrk[f'{k}'][f'{q}'] = cfp_list[count]/nja.Stev_coeff(str(k), conf)
            count += 1

    dic_Bkq = nja.from_Aqkrk_to_Bkq(dic_Aqkrk)
    dic_Bkq['0'] = {}
    dic_Bkq['0']['0'] = 0

    #g ref syst for Dy from J. Am. Chem. Soc. 2021, 143, 8108−8115
    Rot_mat = np.array([[0.696343, 0.027550, -0.717180],[0.216884, 0.944468, 0.246864],[0.684155, -0.327447, 0.651698]])  #non serve il trasposto perché altrimenti a 5K viene l'asse z storto
    # #chi ref syst for Dy from J. Am. Chem. Soc. 2021, 143, 8108−8115
    # Rot_mat = np.array([[ 0.72415202, -0.05048323, -0.68779015],[-0.14026268,  0.96569099, -0.21855747],[ 0.67522606,  0.25473979,  0.69222636]])

    R = scipy.spatial.transform.Rotation.from_matrix(Rot_mat).as_quat()
    quat = [R[-1], R[0], R[1], R[2]]

    #ruoto i bkq
    dict, coeff = nja.read_DWigner_quat()
    dic_Bkq_rot1 = nja.rota_LF_quat(3, dic_Bkq, quat, dict=dict, coeff=coeff)
    dic_Bkq_rot1['0'] = {}
    dic_Bkq_rot1['0']['0'] = 0

    #converto i Bkq in V, ruoto V e poi ricalcolo i Bkq
    dic_V = nja.from_Vint_to_Bkq_2(3, dic_Bkq, reverse=True)
    dic_V = nja.rotate_dicV(dic_V, 3, rotangle_V=quat, real=True)
    dic_Bkq_rot2 = nja.from_Vint_to_Bkq_2(3, dic_V, reverse=False)

    for k in dic_Bkq_rot1.keys():
        for q in dic_Bkq_rot1[k]:
            assert np.round(dic_Bkq_rot1[k][q],10)==np.round(dic_Bkq_rot2[k][q],10)

@test
def test_mag_moment():

    conf = 'd1'
    dic = {}
    calc = nja.calculation(conf, ground_only=False, TAB=True, wordy=False)
    _ = calc.MatrixH([], **dic, save_label=True)
    basis = np.loadtxt('matrix_label.txt')
    mu_matrix = nja.mag_moment(basis)

    assert np.round(mu_matrix[0,0,1],4) == -0.6924
    assert np.round(mu_matrix[1,0,1],4) == -0.6924*1j
    assert np.round(mu_matrix[2,0,0],4) == 1.1993

@test
def test_mag_moment2():

    basis = np.loadtxt('test/matrix_label_f9complete.txt')
    mu_matrix = nja.mag_moment(basis)
    
    assert np.round(mu_matrix[0,0,1],4) == -2.0812
    assert np.round(mu_matrix[1,0,1],4) == -2.0812*1j
    assert np.round(mu_matrix[2,0,0],4) == 3.6048

@test
def test_M_vector():

    conf = 'd3'
    contributes = ['Hee', 'Hcf', 'Hso']

    #conv for orca: 27.2113834*8065.54477 from a.u. to cm-1
    from_au = 27.2113834*8065.54477

    dic = nja.read_AILFT_orca6('test/CrF63-.out', conf)

    calc = nja.calculation(conf, ground_only=False, TAB=True, wordy=False)
    result, projected = calc.MatrixH(contributes, **dic, eig_opt=False, wordy=False, ground_proj=True, return_proj=True, save_label=True, save_LF=True)
    basis = np.loadtxt('matrix_label.txt')
    LF_matrix = np.load('matrix_LF.npy', allow_pickle=True, fix_imports=False)

    mu_matrix = nja.mag_moment(basis)

    #print('dal test di M',mu_matrix)
    #conversion = a.u. * 2.35051756758e5 T
    M = nja.M_vector(np.array([0.0,0.0,23.5051756758]), mu_matrix, LF_matrix, basis, temp=1.0)  #in Bohr magnetons

    #from CrF63-.out CASSCF
    M_AILFT = np.array([-0.0003645214756898332, -1.2322563787476262e-13, 1.4631881898029349])  #in atomic units

    ratio = M/M_AILFT/2

    for i in range(len(ratio)):
        if np.abs(np.round(ratio[i],16)) > 0:
            assert np.round(ratio[i], 2) == 1.0

# At low temperature, magnetization should be that of the ground state (approximately MS=-1/2,
# having magnetization of <-1/2| Mz | -1/2> = - <-1/2|Sz|-1/2> = +1/2)
@test
def test_M_vector2():

    conf = 'd9'
    contributes = ['Hee', 'Hso', 'Hcf']

    #conv for orca: 27.2113834*8065.54477 from a.u. to cm-1
    from_au = 27.2113834*8065.54477

    #(z2 yz xz xy x2-y2)
    dic_V = {'11':0.000000,
        '21':0.000000, '22':0.05*from_au,
        '31':0.000000, '32':0.000000, '33':0.05*from_au,
        '41':0.000000, '42':0.000000, '43':0.000000, '44':0.1*from_au,
        '51':0.000000, '52':0.000000, '53':0.000000, '54':0.000000, '55':0.3*from_au}
    
    dic_Bkq = nja.from_Vint_to_Bkq(dic_V, conf)
    dic = {'F2':0, 'F4':0, 'zeta':0, 'dic_bkq': dic_Bkq}
    calc = nja.calculation(conf, ground_only=False, TAB=True, wordy=False)
    result, projected = calc.MatrixH(contributes, **dic, eig_opt=False, wordy=False, ground_proj=True, return_proj=True, save_label=True, save_LF=True)
    basis = np.loadtxt('matrix_label.txt')
    LF_matrix = np.load('matrix_LF.npy', allow_pickle=True, fix_imports=False)

    mu_matrix = nja.mag_moment(basis)

    M = nja.M_vector(np.array([0.0,0.0,1e-7*2.35051756758e5]), mu_matrix, LF_matrix, basis, temp=0.0001)

    assert np.array_equal(np.round(M, 2)/2, np.array([0.0, 0.0, 1/2]))  

@test
def test_gtensor():

    conf = 'f13'
    contributes = ['Hcf']
    
    #comparison with the results from: J. Am. Chem. Soc. 2021, 143, 8108−8115
    cfp_list = np.loadtxt('test/CFP_YbDOTA.txt')
    dic_Aqkrk = {}
    count = 0
    for k in range(2,7,2):
        dic_Aqkrk[f'{k}'] = {}
        for q in range(k,-k-1,-1):
            dic_Aqkrk[f'{k}'][f'{q}'] = cfp_list[count]/nja.Stev_coeff(str(k), conf)
            count += 1

    dic_Bkq = nja.from_Aqkrk_to_Bkq(dic_Aqkrk)
    dic = {}
    dic['dic_bkq'] = dic_Bkq
    dic_Bkq['0'] = {}
    dic_Bkq['0']['0'] = 0

    #g ref for Yb from J. Am. Chem. Soc. 2021, 143, 8108−8115
    Rot_mat = np.array([[0.513134, -0.634873, 0.577606],[0.437125, 0.772449, 0.460700],[-0.738658, 0.016085, 0.673889]])

    R = scipy.spatial.transform.Rotation.from_matrix(Rot_mat.T).as_quat()
    quat = [R[-1], R[0], R[1], R[2]]
    dict, coeff = nja.read_DWigner_quat()
    dic_Bkq = nja.rota_LF_quat(3, dic_Bkq, quat, dict=dict, coeff=coeff)
    dic['dic_bkq'] = dic_Bkq
    dic_Bkq['0'] = {}
    dic_Bkq['0']['0'] = 0

    calc = nja.calculation(conf, ground_only=True, TAB=True, wordy=False)
    result = calc.MatrixH(contributes, **dic, eig_opt=False)
    E = np.copy(result[0,:]).real
    energy_print = np.around(E-min(E),8)
    energy_list, energy_count = np.unique(energy_print, return_counts=True)

    dic['field'] = np.array([0.0,0.0,1e-7*2.35051756758e5])
    Magn = nja.Magnetics(calc, ['Hcf','Hz'], dic)
    gw, gv = Magn.effGval((1,2))
    gw, gv = nja.princ_comp(gw, gv)

    _, angle1 = nja.angle_between_vectors(np.real(gv)[:,0], Rot_mat[0,:])
    _, angle2 = nja.angle_between_vectors(np.real(gv)[:,1], Rot_mat[1,:])
    _, angle3 = nja.angle_between_vectors(np.real(gv)[:,2], Rot_mat[2,:])

    assert round(gw[0],1)==np.round(0.094,1) and round(gw[1],1)==np.round(0.358,1) and round(gw[2],1)==np.round(7.688,1)
    assert int(energy_list[1]) == 257
    assert (angle1 < 1 or (angle1>179 and angle1<180)) and (angle2 < 1 or (angle2>179 and angle2<180)) and (angle3 < 1 or (angle3>179 and angle3<180))

@test
def test_susceptibility_B_ord1():

    conf = 'd8'
    contributes = ['Hee', 'Hcf', 'Hso']

    #from NiSAL-HDPT calcsuscenisal_10.out NEVPT2
    #(xy yz z2 xz x2-y2)
    dic_V1_orca = {'11':-1537343.193973,
        '21':-197.966117, '22':-1536481.975521,
        '31':2138.341330, '32':2620.966044, '33':-1534906.147670,
        '41':2944.032701, '42':1955.080014, '43':3930.351693, '44':-1531161.910464,
        '51':-599.165743, '52':1115.150600, '53':102.275178, '54':-2462.285886, '55':-1535571.155802}
    
    #(z2 yz xz xy x2-y2)
    dic_V = {'11':dic_V1_orca['33'],
        '21':dic_V1_orca['32'], '22':dic_V1_orca['22'],
        '31':dic_V1_orca['43'], '32':dic_V1_orca['42'], '33':dic_V1_orca['44'],
        '41':dic_V1_orca['31'], '42':dic_V1_orca['21'], '43':dic_V1_orca['41'], '44':dic_V1_orca['11'],
        '51':dic_V1_orca['53'], '52':dic_V1_orca['52'], '53':dic_V1_orca['54'], '54':dic_V1_orca['51'], '55':dic_V1_orca['55']}
    
    dic_Bkq = nja.from_Vint_to_Bkq(dic_V, conf)
    dic = {'F2': 85687.2, 'F4': 48274.5, 'zeta': 643.4, 'dic_bkq': dic_Bkq}

    calc = nja.calculation(conf, ground_only=False, TAB=True, wordy=False)
    result, projected = calc.MatrixH(contributes, **dic, eig_opt=False, wordy=False, ground_proj=True, return_proj=True, save_label=True, save_LF=True)
    basis = np.loadtxt('matrix_label.txt')
    LF_matrix = np.load('matrix_LF.npy', allow_pickle=True, fix_imports=False)

    #from NiSAL-HDPT calcsuscenisal_10.out NEVPT2
    chi_AILFT = np.array([[1.01328435e-31, 4.82628730e-33, 1.44940570e-32],[4.82628730e-33, 8.36113452e-32, 6.77342576e-33],[1.44940570e-32, 6.77342576e-33, 1.04818118e-31]])
    w,_ = np.linalg.eigh(chi_AILFT)
    w_av = np.sum(w)/3
    
    chi_B_diff, err_B = nja.susceptibility_B_ord1(np.array([[0.0,0.0,0.0]]), 298., basis, LF_matrix, delta=1)
    
    Magn = nja.Magnetics(calc, ['Hee', 'Hcf', 'Hso','Hz'], dic)

    chi_B = Magn.susceptibility_B_copy(np.array([[0.0,0.0,0.0]]), 298., delta = 0.1)   #this also works

    diff_AILFT = np.average(np.abs(chi_AILFT-chi_B[0]))

    if not np.array_equal(np.round(chi_B_diff, 37),np.round(chi_B[0], 37)) or diff_AILFT*100/w_av > 5:
        print('chi_B_diff',chi_B_diff)
        print('chi_B',chi_B[0])
        print('diff_AILFT',diff_AILFT)
        print('w_av',w_av)

    assert np.array_equal(np.round(chi_B_diff, 37),np.round(chi_B[0], 37))
    assert diff_AILFT*100/w_av <= 5

@test
def test_susceptibility_B_ord1_2():

    conf = 'd8'
    contributes = ['Hee', 'Hcf', 'Hso']

    from_au = 27.2113834*8065.54477

    #from NiSAL-HDPT NiSAL_HDPT.out CASSCF
    #(z2 xz yz x2-y2 xy)
    dic_V1_orca = {'11':-6.985156,
        '21':0.015107, '22':-6.970417,
        '31':0.010379, '32':0.007287, '33':-6.991948,
        '41':-0.000188, '42':-0.009316, '43':0.004099, '44':-6.988073,
        '51':0.008072, '52':0.011502, '53':-0.000819, '54':-0.002250, '55':-6.995262}
    
    #(z2 yz xz xy x2-y2)
    dic_V = {'11':dic_V1_orca['11']*from_au,
        '21':dic_V1_orca['31']*from_au, '22':dic_V1_orca['33']*from_au,
        '31':dic_V1_orca['21']*from_au, '32':dic_V1_orca['32']*from_au, '33':dic_V1_orca['22']*from_au,
        '41':dic_V1_orca['51']*from_au, '42':dic_V1_orca['53']*from_au, '43':dic_V1_orca['52']*from_au, '44':dic_V1_orca['55']*from_au,
        '51':dic_V1_orca['41']*from_au, '52':dic_V1_orca['43']*from_au, '53':dic_V1_orca['42']*from_au, '54':dic_V1_orca['54']*from_au, '55':dic_V1_orca['44']*from_au}
 
    dic_Bkq = nja.from_Vint_to_Bkq(dic_V, conf)
    dic = {'F2': 93649.1, 'F4': 58398.0, 'zeta': 648.1, 'dic_bkq': dic_Bkq}

    calc = nja.calculation(conf, ground_only=False, TAB=True, wordy=False)
    result, projected = calc.MatrixH(contributes, **dic, eig_opt=False, wordy=False, ground_proj=True, return_proj=True, save_label=True, save_LF=True)
    basis = np.loadtxt('matrix_label.txt')
    LF_matrix = np.load('matrix_LF.npy', allow_pickle=True, fix_imports=False)
    
    chi_B_diff, err_B = nja.susceptibility_B_ord1(np.array([[0.0,0.0,0.0]]), 298., basis, LF_matrix, delta=1)
    
    Magn = nja.Magnetics(calc, ['Hee', 'Hcf', 'Hso','Hz'], dic)

    chi_B = Magn.susceptibility_B_copy(fields=np.array([[0.0,0.0,0.0]]), temp=298., delta = 0.01)   #provare a mettere + e - differenza invece che + e 0

    if not np.array_equal(np.round(chi_B_diff, 37),np.round(chi_B[0], 37)):
        print('chi_B_diff',chi_B_diff)
        print('chi_B',chi_B[0])

    assert np.array_equal(np.round(chi_B_diff, 37),np.round(chi_B[0], 37))

@test
def test_susceptibility_B_ord1_3():

    conf = 'f9'
    contributes = ['Hcf']
    
    #comparison with the results from: J. Am. Chem. Soc. 2021, 143, 8108−8115
    cfp_list = np.loadtxt('test/CFP_DyDOTA.txt')
    dic_Aqkrk = {}
    count = 0
    for k in range(2,7,2):
        dic_Aqkrk[f'{k}'] = {}
        for q in range(k,-k-1,-1):
            dic_Aqkrk[f'{k}'][f'{q}'] = cfp_list[count]/nja.Stev_coeff(str(k), conf)
            count += 1

    dic_Bkq = nja.from_Aqkrk_to_Bkq(dic_Aqkrk)
    dic = {}
    dic['dic_bkq'] = dic_Bkq
    dic_Bkq['0'] = {}
    dic_Bkq['0']['0'] = 0

    # #g ref syst for Dy from J. Am. Chem. Soc. 2021, 143, 8108−8115
    Rot_mat = np.array([[0.696343, 0.027550, -0.717180],[0.216884, 0.944468, 0.246864],[0.684155, -0.327447, 0.651698]])  
 
    R = scipy.spatial.transform.Rotation.from_matrix(Rot_mat.T).as_quat()
    quat = [R[-1], R[0], R[1], R[2]]
    dict, coeff = nja.read_DWigner_quat()
    dic_Bkq = nja.rota_LF_quat(3, dic_Bkq, quat, dict=dict, coeff=coeff)
    dic['dic_bkq'] = dic_Bkq
    dic_Bkq['0'] = {}
    dic_Bkq['0']['0'] = 0

    calc = nja.calculation(conf, ground_only=True, TAB=True, wordy=False)
    _, _ = calc.MatrixH(contributes, **dic, eig_opt=False, wordy=False, ground_proj=True, return_proj=True, save_label=True, save_LF=True)
    basis = np.loadtxt('matrix_label.txt')
    LF_matrix = np.load('matrix_LF.npy', allow_pickle=True, fix_imports=False)
    
    chi_B_diff, err_B = nja.susceptibility_B_ord1(np.array([[0.0,0.0,0.0]]), 2., basis, LF_matrix, delta=1)

    w,v = np.linalg.eig(chi_B_diff)

    nja.fig_tensor_rep_1(chi_B_diff)
    #nja.fig_susc_field(conf, dic_Bkq)

    w,v = nja.princ_comp(w,v)

    dic['field'] = np.array([0.0,0.0,1e-7*2.35051756758e5])
    Magn = nja.Magnetics(calc, ['Hcf','Hz'], dic)
    gw, gv = Magn.effGval((1,2))
    gw, gv = nja.princ_comp(gw, gv)

    gw_paper = np.array([0.255, 0.668, 19.238])

    for i in range(3):
        assert gw[i]/gw_paper[i] > 0.95 and gw[i]/gw_paper[i] < 1.05

    assert chi_B_diff[0,0] > 1e-33
    assert err_B[0,0] < chi_B_diff[0,0]*1e-3

@test
def test_susceptibility_B_ord1_4():

    conf = 'f2'
    contributes = ['Hcf']
    
    #comparison with the results from: J. Am. Chem. Soc. 2021, 143, 8108−8115
    cfp_list = np.loadtxt('test/CFP_PrDOTA.txt')
    dic_Aqkrk = {}
    count = 0
    for k in range(2,7,2):
        dic_Aqkrk[f'{k}'] = {}
        for q in range(k,-k-1,-1):
            dic_Aqkrk[f'{k}'][f'{q}'] = cfp_list[count]/nja.Stev_coeff(str(k), conf)
            count += 1

    dic_Bkq = nja.from_Aqkrk_to_Bkq(dic_Aqkrk)
    dic = {}
    dic['dic_bkq'] = dic_Bkq
    dic_Bkq['0'] = {}
    dic_Bkq['0']['0'] = 0

    # chi ref syst for Pr from J. Am. Chem. Soc. 2021, 143, 8108−8115
    Rot_mat = np.array([[-0.671039,-0.017267,-0.741221],[-0.027448,-0.998465,0.048109],[-0.740914,0.052628,0.669535]])  
 
    R = scipy.spatial.transform.Rotation.from_matrix(Rot_mat.T).as_quat()
    quat = [R[-1], R[0], R[1], R[2]]
    dict, coeff = nja.read_DWigner_quat()
    dic_Bkq = nja.rota_LF_quat(3, dic_Bkq, quat, dict=dict, coeff=coeff)
    dic['dic_bkq'] = dic_Bkq
    dic_Bkq['0'] = {}
    dic_Bkq['0']['0'] = 0

    calc = nja.calculation(conf, ground_only=True, TAB=False, wordy=False)
    _, _ = calc.MatrixH(contributes, **dic, eig_opt=False, wordy=False, ground_proj=True, return_proj=True, save_label=True, save_LF=True)
    basis = np.loadtxt('matrix_label.txt')
    LF_matrix = np.load('matrix_LF.npy', allow_pickle=True, fix_imports=False)
    
    chi_B_diff, err_B = nja.susceptibility_B_ord1(np.array([[0.0,0.0,0.0]]), 2., basis, LF_matrix, delta=1)

    w,v = np.linalg.eig(chi_B_diff)

    nja.fig_tensor_rep_1(chi_B_diff)
    #nja.fig_susc_field(conf, dic_Bkq)

    w,v = nja.princ_comp(w,v)

    print(v)

    print(w/(np.pi*4/(1e6*scipy.constants.Avogadro)))

# in weak-field / high-temperature limit, finite-field magnetization should be linear in external field
@test
def test_calc_susceptibility_zerofield():

    conf = 'd8'
    contributes = ['Hee', 'Hcf', 'Hso']

    #conv for orca: 27.2113834*8065.54477 from a.u. to cm-1
    from_au = 27.2113834*8065.54477
    mu0 = 1.25663706212e-06
    muB = 0.4668517532494337

    #from NiSAL_HDPT.out CASSCF
    #(z2 xz yz x2-y2 xy)
    dic_V1_orca = {'11':-6.985156,
        '21':0.015107, '22':-6.970417,
        '31':0.010379, '32':0.007287, '33':-6.991948,
        '41':-0.000188, '42':-0.009316, '43':0.004099, '44':-6.988073,
        '51':0.008072, '52':0.011502, '53':-0.000819, '54':-0.002250, '55':-6.995262}
    
    #(z2 yz xz xy x2-y2)
    dic_V = {'11':dic_V1_orca['11']*from_au,
        '21':dic_V1_orca['31']*from_au, '22':dic_V1_orca['33']*from_au,
        '31':dic_V1_orca['21']*from_au, '32':dic_V1_orca['32']*from_au, '33':dic_V1_orca['22']*from_au,
        '41':dic_V1_orca['51']*from_au, '42':dic_V1_orca['53']*from_au, '43':dic_V1_orca['52']*from_au, '44':dic_V1_orca['55']*from_au,
        '51':dic_V1_orca['41']*from_au, '52':dic_V1_orca['43']*from_au, '53':dic_V1_orca['42']*from_au, '54':dic_V1_orca['54']*from_au, '55':dic_V1_orca['44']*from_au}
    
    dic_Bkq = nja.from_Vint_to_Bkq(dic_V, conf)
    dic = {'F2':93649.1, 'F4':58398.0, 'zeta':648.1, 'dic_bkq': dic_Bkq}

    calc = nja.calculation(conf, ground_only=False, TAB=True, wordy=False)
    result, projected = calc.MatrixH(contributes, **dic, eig_opt=False, wordy=False, ground_proj=True, return_proj=True, save_label=True, save_LF=True)
    basis = np.loadtxt('matrix_label.txt')
    LF_matrix = np.load('matrix_LF.npy', allow_pickle=True, fix_imports=False)

    mu_matrix = nja.mag_moment(basis)

    field = np.array([0.0,0.0,1e-7*2.35051756758e5])
    M = nja.M_vector(field, mu_matrix, LF_matrix, basis, temp=298.0)  #in Bohr magnetons
    
    #from NiSAL-HDPT.out CASSCF
    M_AILFT = np.array([2.103121567385288e-5, 9.908963872165094e-6, 0.0001212081853327746])  #in atomic units

    ratio = M/M_AILFT/2

    for i in range(len(ratio)):
        if np.abs(np.round(ratio[i],16)) > 0:
            assert np.round(ratio[i], 2) == 1.0

    chi, err_B = nja.susceptibility_B_ord1(np.array([field]), 298., basis, LF_matrix, delta=1.0)
    M_av_linear = np.dot(chi,field)/(mu0*muB*1.9865e-23)

    ratio = M/M_av_linear

    for i in range(len(ratio)):
        if np.abs(np.round(ratio[i],16)) > 0:
            assert np.round(ratio[i], 2) == 1.0

@test
def test_reduction():

    conf = 'f9'
    contributes = ['Hee','Hso','Hcf']
    dic = nja.read_AILFT_orca6('test/run_DOTA1_21sextets.out', conf)

    calc = nja.calculation(conf, ground_only=False, TAB=True)
    calc.reduce_basis(conf, roots = [(21,6)], wordy=True)  
    result, _ = calc.MatrixH(contributes, **dic, ground_proj=True, return_proj=True)

    assert result.shape[1]==126

def eigenfunction_optimization_opt():

    conf = 'f9'

    data = nja.read_data('test/bbpn.inp', sph_flag = False)
    data[:,-1] *= -1
    dic_Bkq = nja.calc_Bkq(data, conf)
    calc = nja.calculation(conf, ground_only=True, TAB=True, wordy=False)

    # dic_Bkq_rot, _ = nja.calculation.opt_eigenfunction_minimization(calc, dic_Bkq, 16)
    # dic_rot = {'dic_bkq': dic_Bkq_rot}

    dic = {'dic_bkq': dic_Bkq}
    result, _ = calc.MatrixH(['Hcf'], **dic, eig_opt=True, wordy=True, ground_proj=True, return_proj=True)
    # Magn = nja.Magnetics(calc, ['Hcf','Hz'], dic_rot)
    # chi_B_rot, _ = Magn.susceptibility_B_copy(fields=np.array([[0.0,0.0,0.0]]), temp=298., delta = 0.01)   

    plt.figure(figsize=(5,5))
    plt.imshow(np.abs(result[1:,:]), cmap='viridis', interpolation='none')
    plt.colorbar(label='Value')
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
    plt.xticks(ticks=np.arange(result.shape[1]), labels=np.arange(result.shape[1]))
    plt.yticks(ticks=np.arange(result.shape[1]), labels=np.arange(result.shape[1]))
    plt.savefig('f9_eigenfunction_opt_true.png', dpi=600)
    plt.close()

    dic = {'dic_bkq': dic_Bkq}
    result, _ = calc.MatrixH(['Hcf'], **dic, eig_opt=False, wordy=True, ground_proj=True, return_proj=True)
    # Magn = nja.Magnetics(calc, ['Hcf','Hz'], dic)
    # chi_B, _ = Magn.susceptibility_B_copy(fields=np.array([[0.0,0.0,0.0]]), temp=298., delta = 0.01)

    plt.figure(figsize=(5,5))
    plt.imshow(np.abs(result[1:,:]), cmap='viridis', interpolation='none')
    plt.colorbar(label='Value')
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
    plt.xticks(ticks=np.arange(result.shape[1]), labels=np.arange(result.shape[1]))
    plt.yticks(ticks=np.arange(result.shape[1]), labels=np.arange(result.shape[1]))
    plt.savefig('f9_eigenfunction_opt_false.png', dpi=600)
    plt.close()

    #check with susceptibility and eigenfunction composition

def comparison_exp_chi():
    #this compares the performances of different CFPs sets in reproducing the chi temp. dependence for the elpasolite Cs2NaDyCl6 complex

    ## experimental data
    x = np.array([6.229508196721311, 5.027322404371585, 4.918032786885246, 7.3224043715847, 8.633879781420765, 9.94535519125683, 11.256830601092895, 12.6775956284153, 16.174863387978142, 14.863387978142075, 19.12568306010929, 17.595628415300546, 20.87431693989071, 23.934426229508198, 27.10382513661202, 28.524590163934427, 30.163934426229506, 31.584699453551913, 33.333333333333336, 36.502732240437155, 39.34426229508197, 41.4207650273224, 44.37158469945355, 47.431693989071036, 50.60109289617486, 53.66120218579235, 56.612021857923494, 59.78142076502732, 62.51366120218579, 65.68306010928961, 68.85245901639344, 71.91256830601093, 75.08196721311475, 78.25136612021858, 81.31147540983606, 84.48087431693989, 86.775956284153, 2.841530054644809])
    y = np.array([7.863436123348018, 7.995594713656388, 7.995594713656388, 8.524229074889869, 9.383259911894273, 9.581497797356828, 9.97797356828194, 10.440528634361234, 11.674008810572689, 11.806167400881058, 11.872246696035242, 12.114537444933921, 12.533039647577093, 12.753303964757709, 12.907488986784141, 13.01762114537445, 13.105726872246697, 13.171806167400883, 13.237885462555067, 13.392070484581499, 13.325991189427313, 13.502202643171806, 13.41409691629956, 13.392070484581499, 13.546255506607931, 13.502202643171806, 13.524229074889869, 13.612334801762115, 13.590308370044054, 13.656387665198238, 13.546255506607931, 13.546255506607931, 13.546255506607931, 13.502202643171806, 13.590308370044054, 13.612334801762115, 13.436123348017622, 6.299559471365639])    
    
    conf = 'f9'

    calc = nja.calculation(conf, ground_only=True, TAB=True, wordy=False)

    #calc = nja.calculation(conf, ground_only=False, TAB=True, wordy=False)
    #calc.reduce_basis(conf, roots = [(21,6)], wordy=True)

    ## AILFT
    dic = {}
    dic_V = {
        '11':12140.0,
        '21':0, '22':11971.4,
        '31':0,'32':0,'33':11971.1,
        '41':0,'42':0,'43':0,'44':11735.3,
        '51':0,'52':0,'53':0,'54':0,'55':11870.1,
        '61':0,'62':130.7,'63':0,'64':0,'65':0,'66':12038.3,
        '71':0,'72':0,'73':-130.6,'74':0,'75':0,'76':0,'77':12038.9
    }
    dic_Bkq = nja.from_Vint_to_Bkq(dic_V, conf)
    dic['dic_bkq'] = dic_Bkq
    dic['F2'] = 109890.1
    dic['zeta'] = 1737.8
    result, projected = calc.MatrixH(['Hee','Hso','Hcf'], **dic, eig_opt=False, wordy=True, ground_proj=True, return_proj=True)
    Magn = nja.Magnetics(calc, ['Hee','Hso','Hcf','Hz'], dic)
    chi_nja = []
    x1 = np.arange(1,90,4)
    for T in x1:
        print(str(T), end='\r')
        chi= Magn.susceptibility_field(np.array([[0.0,0.0,1e-7]]), T, delta=0.01)[1]
        chi_nja.append(chi[0])
    chi_nja = np.array(chi_nja)

    ## PCM dycl6^-3
    data = nja.read_data('test/dycl63-.inp', sph_flag = False)
    data[:,-1] *= -1
    dic_Bkq = nja.calc_Bkq(data, conf)
    dic['dic_bkq'] = dic_Bkq
    result, projected = calc.MatrixH(['Hee','Hso','Hcf'], **dic, eig_opt=False, wordy=True, ground_proj=True, return_proj=True)
    Magn = nja.Magnetics(calc, ['Hee','Hso','Hcf','Hz'], dic)
    chi_nja2 = []
    x1 = np.arange(1,90,4)
    for T in x1:
        print(str(T), end='\r')
        chi= Magn.susceptibility_field(np.array([[0.0,0.0,1e-7]]), T, delta=0.01)[1]
        chi_nja2.append(chi[0])
    chi_nja2 = np.array(chi_nja2)

    ## PCM cs2nadycl6
    data = nja.read_data('test/cs2nadycl6.inp', sph_flag = False)
    data[:,-1] *= -1
    dic_Bkq = nja.calc_Bkq(data, conf)
    dic['dic_bkq'] = dic_Bkq
    result, projected = calc.MatrixH(['Hee','Hso','Hcf'], **dic, eig_opt=False, wordy=True, ground_proj=True, return_proj=True)
    Magn = nja.Magnetics(calc, ['Hee','Hso','Hcf','Hz'], dic)
    chi_nja4 = []
    x1 = np.arange(1,90,4)
    for T in x1:
        print(str(T), end='\r')
        chi= Magn.susceptibility_field(np.array([[0.0,0.0,1e-7]]), T, delta=0.01)[1]
        chi_nja4.append(chi[0])
    chi_nja4 = np.array(chi_nja4)

    ## AILFT corrected for PCM cs2nadycl6 
    data = nja.read_data('test/cs2nadycl6_empty.inp', sph_flag = False)
    data[:,-1] *= -1
    dic_Bkq = nja.calc_Bkq(data, conf)
    dic_V2 = nja.from_Vint_to_Bkq_2(3, dic_Bkq, reverse=True)
    dic_V3 = {key:dic_V2[key]+dic_V[key] for key in dic_V2.keys()}
    dic_Bkq = nja.from_Vint_to_Bkq_2(3, dic_V3)
    dic['dic_bkq'] = dic_Bkq
    result, projected = calc.MatrixH(['Hee','Hso','Hcf'], **dic, eig_opt=False, wordy=True, ground_proj=True, return_proj=True)
    Magn = nja.Magnetics(calc, ['Hee','Hso','Hcf','Hz'], dic)
    chi_nja5 = []
    x1 = np.arange(1,90,4)
    for T in x1:
        print(str(T), end='\r')
        chi= Magn.susceptibility_field(np.array([[0.0,0.0,1e-7]]), T, delta=0.01)[1]
        chi_nja5.append(chi[0])
    chi_nja5 = np.array(chi_nja5)

    ## AOM dycl6^-3
    data = nja.read_data('test/dycl63-.inp', sph_flag = False)
    data[:,-1] *= -1
    dic_Bkq = nja.calc_Bkq(data, conf)
    del dic['dic_bkq']
    AOM = np.array([[300.5,138,138,0.0,0.0,0.0],
                            [300.5,138,138,0.0,0.0,0.0],
                            [300.5,138,138,0.0,0.0,0.0],
                            [300.5,138,138,0.0,0.0,0.0],
                            [300.5,138,138,0.0,0.0,0.0],
                            [300.5,138,138,0.0,0.0,0.0]])
    sph_coord = nja.from_car_to_sph(data[:,1:-1])
    AOM[:,3:-1] = sph_coord[:,1:]*180/np.pi
    dic_AOM = {}
    for i in range(6):
        dic_AOM['Cl'+str(i+1)] = AOM[i,:]
    dic['dic_AOM'] = dic_AOM
    dic['F2'] = 412.1*225
    dic['F4'] = 60.9*1089
    dic['F6'] = 6.3*184041/25
    dic['zeta'] = 1920
    pprint(dic)
    calc = nja.calculation(conf, ground_only=False, TAB=True, wordy=False)
    calc.reduce_basis(conf, roots = [(21,6),(13,4)], wordy=True)
    result, projected = calc.MatrixH(['Hee','Hso','Hcf'], **dic, eig_opt=False, wordy=True, ground_proj=True, return_proj=True)
    Magn = nja.Magnetics(calc, ['Hee','Hso','Hcf','Hz'], dic)
    chi_nja3 = []
    x1 = np.arange(1,90,4)
    for T in x1:
        print(str(T), end='\r')
        chi= Magn.susceptibility_field(np.array([[0.0,0.0,1e-7]]), T, delta=0.01)[1]
        chi_nja3.append(chi[0])
    chi_nja3 = np.array(chi_nja3)

    ## plot of the results
    conv = 1/(np.pi*4/(1e6*scipy.constants.Avogadro*x1)) #cm^3 K/mol
    plt.figure(figsize=(5, 4.5))
    plt.plot(x,y,'o',c='b',label='Experiment')
    plt.ylim(0,15)
    plt.plot(x1,chi_nja*conv,'-.',c='r',label='NJA-CFS (AILFT)')
    plt.plot(x1,chi_nja2*conv,'--',c='r',label=r'NJA-CFS (PCM [DyCl$_6$]$^{-3}$)')
    plt.plot(x1,chi_nja4*conv,'--',c='g',label=r'NJA-CFS (PCM Cs$_2$NaDyCl$_6$)')
    plt.plot(x1,chi_nja5*conv,'-.',c='g',label='NJA-CFS (AILFT corrected)')
    plt.plot(x1,chi_nja3*conv,':',c='magenta',label='NJA-CFS (AOM)')
    plt.xlabel('Temperature (K)')
    plt.ylabel(r'$\chi$T (cm$^3$ K/mol)')
    plt.legend()
    #plt.show()
    plt.savefig('dycl6_tempdep_PCM4_wAOM.png', dpi=600)
    exit()

def Susceptibility_Tdep():

    tmax = 301

    conf = 'f9'
    contributes = ['Hcf']
    calc = nja.calculation(conf, ground_only=True, TAB=True, wordy=False)
    dic = nja.read_AILFT_orca6('test/run_DOTA1_21sextets.out', conf)
    _,_ = calc.MatrixH(contributes, **dic, eig_opt=False, wordy=False, ground_proj=True, return_proj=True, save_label=True, save_LF=True)
    basis = np.loadtxt('matrix_label.txt')
    LF_matrix = np.load('matrix_LF.npy', allow_pickle=True, fix_imports=False)
    
    filename = open('test/run_DOTA1_21sextets.out').readlines()
    fmtsusc = nja.def_fmtsusc(filename)
    chi_orca = []
    for T in range(1,tmax,2):
        print(str(T), end='\r')
        chi_AILFT = nja.find_chi(fmtsusc, filename, T)
        w,v = np.linalg.eig(chi_AILFT)
        w,v = nja.princ_comp(w,v)
        chi_orca.append(w)
    chi_orca = np.array(chi_orca)
    print(' ')
    chi_nja = []
    for T in range(1,tmax,2):
        print(str(T), end='\r')
        chi_B_diff, _ = nja.susceptibility_B_ord1(np.array([[0.0,0.0,0.0]]), T, basis, LF_matrix, delta=1)
        w,v = np.linalg.eig(chi_B_diff)
        w,v = nja.princ_comp(w,v)
        chi_nja.append(w)
    chi_nja = np.array(chi_nja)

    Temp = np.arange(1,tmax,2)*1/(np.pi*4/(1e6*scipy.constants.Avogadro))

    plt.figure(figsize=(8, 4.5))
    plt.plot(np.arange(1,tmax,2), chi_orca[:,0]*Temp,'--',c='r', label=r'CASSCF, $\chi_x$')
    plt.plot(np.arange(1,tmax,2), chi_nja[:,0]*Temp,c='r', label=r'NJA-CFS, $\chi_x$')
    plt.plot(np.arange(1,tmax,2), chi_orca[:,1]*Temp,'--',c='g', label=r'CASSCF, $\chi_y$')
    plt.plot(np.arange(1,tmax,2), chi_nja[:,1]*Temp,c='g', label=r'NJA-CFS, $\chi_y$')
    plt.plot(np.arange(1,tmax,2), chi_orca[:,2]*Temp,'--',c='b', label=r'CASSCF, $\chi_z$')
    plt.plot(np.arange(1,tmax,2), chi_nja[:,2]*Temp,c='b', label=r'NJA-CFS, $\chi_z$')
    plt.xlabel('Temperature (K)')
    plt.ylabel(r'$\chi$T (cm$^3$ K/mol)')
    plt.legend()
    plt.savefig('dydota_tempdep.png', dpi=600)

if __name__ == '__main__':

    #### test graphical representations
    # test_plot_Ediagram()
    # test_plot_Ediagram_PCM()
    # test_CF_splitting()
    # test_TanabeSugano()
    # test_plot_magnetization_field()

    #### actual tests
    test_conv_AqkrkBkq() #f11
    test_conv_Vint_Bkq_d() #d8
    test_PCM() #f9
    test_PCM_2() #f10
    test_StevensfromMOLCAS() #f9
    test_Wigner_Euler_quat()
    test_Wigner_Euler_quat2()
    test_LF_rotation_euler()
    test_LF_rotation_quat()
    test_mag_moment()  #d1
    test_mag_moment2()  #f9
    test_M_vector()  #d3
    test_M_vector2()  #d9
    test_gtensor()  #f13
    test_susceptibility_B_ord1()  #d8
    test_susceptibility_B_ord1_2()  #d8
    test_susceptibility_B_ord1_3()  #f9
    test_susceptibility_B_ord1_4()  #f2
    test_calc_susceptibility_zerofield()  #d8
    test_torque()
    test_reduction()

    #### other examples
    # eigenfunction_optimization_opt()
    # comparison_exp_chi()
    # Susceptibility_Tdep()

    
    

    

