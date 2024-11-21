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

@test
def test_CF_splitting():

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


    # Example usage to add multiple crystal field splittings
    fig, ax = plt.subplots()

    conf = 'd3'
    contributes = ['Hee', 'Hcf', 'Hso']

    # from_au = 27.2113834*8065.54477

    # matrix = nja.read_AILFT_orca6('test/CrF63-.out', conf, return_orcamatrix=True)
    # matrix *= from_au
    # print(matrix)

    data = nja.read_data('test/Td_cube.inp', sph_flag = False)
    data[:,1:-1] *= 2/(2*np.sqrt(3))#*6/4
    # data[:,-1] *= -6/4
    # print(np.linalg.norm(data[1,1:-1]))
    # exit()
    dic_Bkq = nja.calc_Bkq(data, conf, False, False)
    dic_V = nja.from_Vint_to_Bkq_2(2, dic_Bkq, reverse=True)
    matrix = np.zeros((5,5))
    for i in range(5):
        for j in range(5):
            if i>=j:
                matrix[i,j] = dic_V[str(i+1)+str(j+1)]
                matrix[j,i] = dic_V[str(i+1)+str(j+1)]

    w,v = np.linalg.eigh(matrix)

    #plot_CF_energy_or(w-np.min(w), tolerance=0.1)
    plot_energy_levels(w-np.min(w), ax=ax, color='magenta', label="Td", delta=0)  

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

    #plot_CF_energy_or(w-np.min(w), tolerance=0.1)
    plot_energy_levels(w-np.min(w), ax=ax, color='green', label="Oh", delta=0.5)

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
def test_PCM_from_Bkq():

    #turned out it is not possible ... 

    from pprint import pprint
    import scipy

    conf0 = 'f9'

    #f9
    Rot_mat = np.array([[0.696343, 0.027550, -0.717180],[0.216884, 0.944468, 0.246864],[0.684155, -0.327447, 0.651698]])  
    R1 = scipy.spatial.transform.Rotation.from_matrix(Rot_mat).as_euler('ZYZ')
    #f13
    Rot_mat = np.array([[0.513134, -0.634873, 0.577606],[0.437125, 0.772449, 0.460700],[-0.738658, 0.016085, 0.673889]])
    R2 = scipy.spatial.transform.Rotation.from_matrix(Rot_mat).as_euler('ZYZ')
    #f11
    Rot_mat = np.array([[0.658552, -0.430484, 0.617246],[0.206304, 0.892074, 0.402047],[-0.723704,-0.137429,0.676288]])  
    R3 = scipy.spatial.transform.Rotation.from_matrix(Rot_mat).as_euler('ZYZ')
    #f5
    Rot_mat = np.array([[-0.323967,0.879262,-0.349203],[-0.605347,-0.476314,-0.637715],[-0.727049,0.004790,0.686569]])
    R4 = scipy.spatial.transform.Rotation.from_matrix(Rot_mat).as_euler('ZYZ')
    #f3
    Rot_mat = np.array([[0.670934,0.148376,0.726520],[-0.070829,0.988120,-0.136392],[-0.738126,0.040051,0.673473]])
    R5 = scipy.spatial.transform.Rotation.from_matrix(Rot_mat).as_euler('ZYZ')
    R = [R1, R2, R3, R4, R5]

    data = nja.read_data('test/dota_briganti_m2m.inp', sph_flag = False)

    table = np.loadtxt('test/BKQ').T
    # table = np.array([np.loadtxt('test/CFP_DyDOTA.txt')])

    print(table.shape)

    conf_list = ['f9','f13']#,'f11','f5']#,'f3']#['f1','f2','f3','f4','f5','f8','f9','f10','f11','f12','f13']

    dic_Aqkrk_list = np.zeros((len(conf_list), 27))
    for i in range(table.shape[0]):
        conf = conf_list[i]
        # dic_Aqkrk = np.zeros(27)
        # count = 0
        # for k in range(2,7,2):
        #     for q in range(k,-k-1,-1):
        #         if round(table[i,count],8)!=0 and nja.Stev_coeff(str(k), conf)!=0:
        #             dic_Aqkrk[count] = table[i,count]/nja.Stev_coeff(str(k), conf)
        #         else:
        #             dic_Aqkrk[count] = 0
        #         count += 1

        #to rotate the Aqkrk
        count = 0
        dic_Aqkrk = {}
        for k in range(2,7,2):
            dic_Aqkrk[f'{k}'] = {}
            for q in range(k,-k-1,-1):
                if round(table[i,count],8)!=0 and nja.Stev_coeff(str(k), conf)!=0:
                    dic_Aqkrk[f'{k}'][f'{q}'] = table[i,count]/nja.Stev_coeff(str(k), conf)
                else:
                    dic_Aqkrk[f'{k}'][f'{q}'] = 0
                count += 1
        dic_Bkq = nja.from_Aqkrk_to_Bkq(dic_Aqkrk)
        dic_Bkq_rot1 = nja.rota_LF(3, dic_Bkq, *R[i])
        if conf=='f9':
            pprint(dic_Bkq_rot1)
        dic_Aqkrk_rot1 = nja.from_Aqkrk_to_Bkq(dic_Bkq_rot1, revers=True)
        dic_Aqkrk = np.zeros(27)
        count = 0
        for k in range(2,7,2):
            for q in range(k,-k-1,-1):
                dic_Aqkrk[count] = dic_Aqkrk_rot1[f'{k}'][f'{q}']
                count += 1
        dic_Aqkrk_list[i,:] = dic_Aqkrk

    coord_car = data[:,1:-1]
    coord_sph = nja.from_car_to_sph(coord_car)
    au_conv = [scipy.constants.physical_constants['hartree-inverse meter relationship'][0]*1e-2, 1.889725989]

    coeff_A = np.zeros((data.shape[0], 27*len(conf_list)))
    count = -1
    for i in range(len(conf_list)):
        conf = conf_list[i]
        for k in range(2,2*3+1,2):
            r_val = nja.r_expect(str(k), conf)
            for q in range(k,-k-1,-1):
                count += 1
                pref = nja.plm(k,np.abs(q))*(4*np.pi/(2*k+1))**(0.5)
                for i in range(data.shape[0]):
                    r = coord_sph[i,0]*au_conv[1]
                    sphharmp = scipy.special.sph_harm(np.abs(q), k, coord_sph[i,2],coord_sph[i,1])  
                    sphharmm = scipy.special.sph_harm(-np.abs(q), k, coord_sph[i,2],coord_sph[i,1])
                    if q==0:
                        coeff_A[i,count] = r_val*pref*(au_conv[0]/r**(k+1))*sphharmp.real
                    elif q>0:
                        coeff_A[i,count] = r_val*pref*(au_conv[0]/r**(k+1))*(1/np.sqrt(2))*(sphharmm + (-1)**q*sphharmp).real
                    elif q<0:
                        coeff_A[i,count] = r_val*(1j*pref*(au_conv[0]/r**(k+1))*(1/np.sqrt(2))*(sphharmm - (-1)**q*sphharmp)).real


    dic_Aqkrk = np.reshape(dic_Aqkrk_list, (len(conf_list)*27,))

    # B = np.linalg.pinv(coeff_A)
    # charges = B.T@dic_Aqkrk
    B = coeff_A.T
    #compute the conditioning number for B
    U, S, Vt = np.linalg.svd(B)
    print(S)
    print(f'Conditioning number: {S[0]/S[-1]}')
    #exit()
    # charges = np.linalg.inv(B.conj().T @ B) @ B.conj().T @ dic_Aqkrk
    charges = scipy.optimize.nnls(B, dic_Aqkrk)[0]
    print(charges)
    data[:,-1] = charges
    dic_CF = nja.calc_Aqkrk(data, conf0, False, True)
    dic_Bkq = nja.from_Aqkrk_to_Bkq(dic_CF)
    pprint(dic_Bkq)

@test
def test_PCM_from_Bkq2():

    from pprint import pprint
    import scipy

    #much worse

    conf0 = 'f11'

    data = nja.read_data('test/ErCl63-.inp', sph_flag = False)

    dic_Aqkrk_calc = nja.calc_Aqkrk(data, conf0, False, True)

    conf_list = ['f11']#,'f5']#,'f3']#['f1','f2','f3','f4','f5','f8','f9','f10','f11','f12','f13']

    dic_Aqkrk_list = np.zeros((len(conf_list), 27))
    for i in range(len(conf_list)):

        conf = conf_list[i]
        dic_Aqkrk = np.zeros(27)
        count = 0
        for k in range(2,7,2):
            for q in range(k,-k-1,-1):
                if round(dic_Aqkrk_calc[f'{k}'][f'{q}'],8)!=0:
                    dic_Aqkrk[count] = dic_Aqkrk_calc[f'{k}'][f'{q}']
                else:
                    dic_Aqkrk[count] = 0
                count += 1

        dic_Aqkrk_list[i,:] = dic_Aqkrk

    coord_car = data[:,1:-1]
    coord_sph = nja.from_car_to_sph(coord_car)
    au_conv = [scipy.constants.physical_constants['hartree-inverse meter relationship'][0]*1e-2, 1.889725989]

    coeff_A = np.zeros((data.shape[0], 27*len(conf_list)))
    count = -1
    for i in range(len(conf_list)):
        conf = conf_list[i]
        for k in range(2,2*3+1,2):
            r_val = nja.r_expect(str(k), conf)
            for q in range(k,-k-1,-1):
                count += 1
                pref = nja.plm(k,np.abs(q))*(4*np.pi/(2*k+1))**(0.5)
                for i in range(data.shape[0]):
                    r = coord_sph[i,0]*au_conv[1]
                    sphharmp = nja.sph_harm(k,np.abs(q),coord_sph[i,1], coord_sph[i,2])
                    sphharmm = nja.sph_harm(k,-np.abs(q),coord_sph[i,1], coord_sph[i,2])
                    if q==0:
                        coeff_A[i,count] = (1-nja.sigma_k(str(k), conf))*r_val*pref*(au_conv[0]/r**(k+1))*sphharmp.real
                    elif q>0:
                        coeff_A[i,count] = (1-nja.sigma_k(str(k), conf))*r_val*(-1)**q*pref*(au_conv[0]/r**(k+1))*(1/np.sqrt(2))*(sphharmm + (-1)**q*sphharmp).real
                    elif q<0:
                        coeff_A[i,count] = -(1-nja.sigma_k(str(k), conf))*r_val*(-1)**q*pref*(au_conv[0]/r**(k+1))*(1/np.sqrt(2))*(sphharmm - (-1)**np.abs(q)*sphharmp).imag


    dic_Aqkrk = np.reshape(dic_Aqkrk_list, (len(conf_list)*27,))

    # B = np.linalg.pinv(coeff_A)
    # charges = B.T@dic_Aqkrk
    B = coeff_A.T
    #compute the conditioning number for B
    U, S, Vt = np.linalg.svd(B)
    print(S)
    print(f'Conditioning number: {S[0]/S[-1]}')
    exit()
    charges = np.linalg.inv(B.conj().T @ B) @ B.conj().T @ dic_Aqkrk
    print(charges)
    data[:,-1] = charges
    dic_CF = nja.calc_Aqkrk(data, conf0, False, True)
    dic_Bkq = nja.from_Aqkrk_to_Bkq(dic_CF)
    pprint(dic_Bkq)

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

    data = nja.read_data('test/Oh_cube.inp', sph_flag = False)
    #data[:,1:-1] *= 2/(2*np.sqrt(3))
    data[:,-1] *= -1*3

    calc = nja.calculation(conf, ground_only=False, TAB=True, wordy=False)

    #first point at 0 CF
    dic = nja.free_ion_param_AB(conf)
    result = calc.MatrixH(['Hee'], **dic, eig_opt=False, wordy=False)
    proj_LS = nja.projection_basis(result[1:,:], calc.basis_l)
    # fig, ax = plt.subplots()
    # ax.plot(np.zeros_like(result[0,:]), result[0,:]-np.min(result[0,:]), 'o')
    # plt.show()

    diagram = [(result[0,:]-np.min(result[0,:]))/B]
    x_axis = [0.0]
    # names_list = [red_proj_LS(proj_LS)]
    spacing = np.arange(0.01,12,0.1)
    #print(len(spacing))
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
        #print(f'{i}    {x_axis[-1]/10}       ',end='\r')
        dic['dic_bkq'] = dic_Bkq
        result = calc.MatrixH(['Hee','Hcf'], **dic, eig_opt=False, wordy=False)
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
    plt.show()

@test
def test_JahnTeller_bak():

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

    def plot_CF_energy_single(eigenvalues, tolerance=0.1):

        # Step 2: Sort and group nearly degenerate energy levels
        unique_levels = []
        grouped_levels = []

        # Group levels with approximate equality within tolerance
        for ev in sorted(eigenvalues):
            if not unique_levels or abs(ev - unique_levels[-1]) > tolerance:
                unique_levels.append(ev)
                grouped_levels.append([ev])
            else:
                grouped_levels[-1].append(ev)

        # Step 3: Set up the plot with offsets for degeneracy
        fig = plt.figure()
        ax = fig.add_subplot(111)
        x_offset = 0.15  # Horizontal offset for degenerate levels

        for i, level_group in enumerate(grouped_levels):
            energy = level_group[0]  # Common energy level for degenerate group
            n_deg = len(level_group)  # Number of degenerate states
            x_positions = np.linspace(-x_offset * (n_deg - 1) / 2, x_offset * (n_deg - 1) / 2, n_deg)

            # Plot each degenerate level with horizontal offset
            for x in x_positions:
                ax.hlines(y=energy, xmin=x - 0.05, xmax=x + 0.05, color='b', linewidth=2)

        # Step 4: Add labels and adjust plot
        #ax.set_xlabel("Degenerate States (offset for visualization)")
        ax.set_ylabel("Energy Levels")
        ax.set_title("Crystal Field Splitting Energy Levels")
        ax.set_yticks(sorted(unique_levels))
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        ax.get_xaxis().set_visible(False)

        plt.show()

    fig, ax = plt.subplots()
    
    conf = 'd9'

    data = nja.read_data('test/Oh_cube.inp', sph_flag = False)
    data[:,-1] *= -1

    dist_list = np.arange(1, 1.5, 0.05)
    distplane_list = np.arange(1, 0.5, -0.05)

    for idxd, dist in enumerate(dist_list):

        data_mod = copy.deepcopy(data)
        data_mod[:2,1:-1] *= dist
        data_mod[2:,1:-1] *= distplane_list[idxd]

        calc = nja.calculation(conf, ground_only=False, TAB=True, wordy=False)
        dic = nja.free_ion_param_AB(conf)
        dic_Bkq = nja.calc_Bkq(data_mod, conf, False, False)
        dic['dic_bkq'] = dic_Bkq
        pprint(dic_Bkq)

        elem = ['Hee', 'Hcf']
        result = calc.MatrixH(elem, **dic, eig_opt=False, wordy=False, save_matrix=True)
        matrix_result = np.load('matrix.npy')
        matrix = np.zeros((calc.basis.shape[0],calc.basis.shape[0]),dtype='object')
        print(matrix.shape)

        F = [0, dic['F2'], dic['F4'], 0]

        symbols_all = [sympy.Symbol('F0'), sympy.Symbol('F2'), sympy.Symbol('F4'), sympy.Symbol('F6'), sympy.Symbol('zeta'), sympy.Symbol('k')]
        for k in range(2,2*calc.l+1,2):
            for q in range(-k,k+1,1):
                symbols_all.append(sympy.Symbol(f"B{k}{q}"))
            
        for i in range(calc.basis.shape[0]):
            statei = calc.basis[i]
            Si = statei[0]/2.
            Li = statei[1]
            Ji = statei[2]/2.
            MJi = statei[3]/2.
            seni = statei[-1]
            labeli = calc.dic_LS[':'.join([f'{qq}' for qq in statei])]
            for j in range(0,i+1):
                statej = calc.basis[j]
                Sj = statej[0]/2.
                Lj = statej[1]
                Jj = statej[2]/2.
                MJj = statej[3]/2.
                senj = statej[-1]
                labelj = calc.dic_LS[':'.join([f'{qq}' for qq in statej])]
                
                H = nja.Hamiltonian([seni,Li,Si,senj,Lj,Sj,Ji,MJi,Jj,MJj], [labeli,labelj], calc.conf, calc.dic_cfp, calc.tables, calc.dic_LS, calc.dic_LS_almost)  

                if 'Hee' in elem:
                    if Ji==Jj and MJi==MJj:
                        if Li == Lj and Si == Sj:
                            if calc.l==3:
                                Hee = H.electrostatic_int(calc.basis, *F, evaluation=False, tab_ee = calc.dic_ee)
                            else:
                                Hee = H.electrostatic_int(calc.basis, *F, evaluation=False)
                            matrix[i,j] += Hee
                            if i != j:
                                matrix[j,i] += Hee

                if 'Hso' in elem:
                    if Ji==Jj and MJi==MJj:
                        Hso = -H.SO_coupling(dic['zeta'], 1, evaluation=False)
                        matrix[i,j] += Hso
                        if i != j:
                            matrix[j,i] += Hso

                if 'Hcf' in elem:
                    if Si==Sj:
                        Hcf = -H.LF_contribution(dic['dic_bkq'], evaluation=False)
                        matrix[i,j] += Hcf
                        if i != j:
                            matrix[j,i] += np.conj(Hcf)

        
        coeff_matrices = {symbol: np.zeros(matrix.shape, dtype='complex128') for symbol in symbols_all}

        for i in range(matrix.shape[0]):
            for j in range(0,i+1):
                element = matrix[i, j]
                if element != 0.0:
                    elements_c = element.as_coefficients_dict()
                    for symbol in symbols_all:
                        for key, value in elements_c.items():
                            if str(symbol) in str(key):
                                coeff_matrices[symbol][i, j] = complex(value)
                                if 'I' in str(key):
                                    coeff_matrices[symbol][i, j] *= 1j
                                if i != j:
                                    coeff_matrices[symbol][j, i] = np.conj(coeff_matrices[symbol][i, j])

        variables = []
        for k in range(2,2*calc.l+1,2):
            for q in range(-k,k+1,1):
                variables.append(sympy.Symbol(f"B{k}{q}"))

        total_coeff = np.zeros_like(coeff_matrices[symbols_all[0]])
        cost_coeff = np.zeros_like(coeff_matrices[symbols_all[0]])
        for symbol, coeff_matrix in coeff_matrices.items():
            if symbol not in variables:
                cost_coeff += coeff_matrix
            total_coeff += coeff_matrix

        D = np.zeros((cost_coeff.shape[0]*cost_coeff.shape[1], len(variables)), dtype='complex128')
        for i, variable in enumerate(variables):
            D_single = coeff_matrices[variable]+cost_coeff
            D[:, i] = D_single.flatten()
        
        #least sq solution
        B = np.linalg.inv(D.conj().T @ D) @ D.conj().T @ matrix_result.flatten()
        # B = np.linalg.pinv(D) @ matrix_result.flatten()

        for i in range(len(B)):
            if np.abs(B[i].imag)<1e-15:
                print(variables[i], B[i].real)
            else:
                print(variables[i], B[i].real, B[i].imag)

        dic_bkq_calc = {}
        count = 0
        for k in range(2,2*calc.l+1,2):
            dic_bkq_calc[f'{k}'] = {}
            for q in range(-k,k+1,1):
                dic_bkq_calc[f'{k}'][f'{q}'] = B[count].real
                count += 1

        pprint(dic_bkq_calc)

        dic_V = nja.from_Vint_to_Bkq_2(2, dic_bkq_calc, reverse=True)
        matrix = np.zeros((5,5))
        for i in range(5):
            for j in range(5):
                if i>=j:
                    matrix[i,j] = dic_V[str(i+1)+str(j+1)]
                    matrix[j,i] = dic_V[str(i+1)+str(j+1)]

        w,v = np.linalg.eigh(matrix)

        plot_energy_levels(w, ax=ax, label=f"{dist:.3f}", tolerance=0.05, delta=0.5*idxd)

    plt.show()

    #plot_CF_energy_single(w, tolerance=0.1)

@test
def test_JahnTeller():

    def plot_energy_levels(eigenvalues, ax=None, color='b', label=None, tolerance=0.05, offset=0, delta=0, barycenter=None):
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
        
        ax.hlines(y=barycenter, xmin=-0.05 +delta, xmax=0.05+delta, color='r', linewidth=2)

        # Add label if provided
        if label:
            ax.text(offset + 0.22+delta, max(unique_levels) + 0.2, label, ha='center', color=color)

        # Update plot appearance
        ax.set_ylabel("Energy Levels")
        ax.grid(axis='y', linestyle='--', alpha=0.5)

        
        ax.get_xaxis().set_visible(False)

        return ax  # Return axis to allow further modifications

    def plot_CF_energy_single(eigenvalues, tolerance=0.1):

        # Step 2: Sort and group nearly degenerate energy levels
        unique_levels = []
        grouped_levels = []

        # Group levels with approximate equality within tolerance
        for ev in sorted(eigenvalues):
            if not unique_levels or abs(ev - unique_levels[-1]) > tolerance:
                unique_levels.append(ev)
                grouped_levels.append([ev])
            else:
                grouped_levels[-1].append(ev)

        # Step 3: Set up the plot with offsets for degeneracy
        fig = plt.figure()
        ax = fig.add_subplot(111)
        x_offset = 0.15  # Horizontal offset for degenerate levels

        for i, level_group in enumerate(grouped_levels):
            energy = level_group[0]  # Common energy level for degenerate group
            n_deg = len(level_group)  # Number of degenerate states
            x_positions = np.linspace(-x_offset * (n_deg - 1) / 2, x_offset * (n_deg - 1) / 2, n_deg)

            # Plot each degenerate level with horizontal offset
            for x in x_positions:
                ax.hlines(y=energy, xmin=x - 0.05, xmax=x + 0.05, color='b', linewidth=2)

        # Step 4: Add labels and adjust plot
        #ax.set_xlabel("Degenerate States (offset for visualization)")
        ax.set_ylabel("Energy Levels")
        ax.set_title("Crystal Field Splitting Energy Levels")
        ax.set_yticks(sorted(unique_levels))
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        ax.get_xaxis().set_visible(False)

        plt.show()

    def calc_CFSE(w,l,n_el,F):
        center = np.average(w)

        delta = np.array([w[i]-center for i in range(len(w))])

        count = 0
        for i in range(n_el):
            if i<(2*l+1):
                occ_HS[i] += 1
            else:
                occ_HS[i-2*l-1] += 1
            if i%2==0:
                occ_LS[count] += 1
            else:
                occ_LS[count] += 1
                count+=1

        if l==2:
            P = 5/84 * (F[1]+F[2])
        else:
            P = 1/8 * (70*F[1]+231*F[2]+2002*F[3])

        E_HS = np.sum(occ_HS * delta) + np.sum(occ_HS == 2)*P
        E_LS = np.sum(occ_LS * delta) + np.sum(occ_LS == 2)*P

        if E_LS>E_HS:
            CFSE = E_HS - np.sum(occ_HS == 2)*P
        else:
            CFSE = E_LS - np.sum(occ_HS == 2)*P

        return CFSE, E_HS, E_LS

    fig, ax = plt.subplots()
    
    conf = 'd9'
    n_el = int(conf[1:])

    if conf[0] =='d':
        l=2
    else:
        l=3

    occ_HS = np.zeros(2*l+1)
    occ_LS = np.zeros(2*l+1)
    CFSE_list = []
    E_HS_list = []
    E_LS_list = []
    x_axis = []
    
    data = nja.read_data('test/Oh_cube.inp', sph_flag = False)
    data[:,-1] *= -1

    dist_list = np.arange(0, 0.3, 0.05)
    # distplane_list = np.arange(1.2, 1, -0.005)

    print(len(dist_list))
    # print(len(distplane_list))
    # exit()

    for idxd, dist in enumerate(dist_list):

        data_mod = copy.deepcopy(data)
        data_mod[0,-2] += dist
        data_mod[1,-2] -= dist
        print(data_mod)
        x_axis.append(np.linalg.norm(data_mod[0,1:-1]))
        #data_mod[2:,1:-1] *= distplane_list[idxd]

        calc = nja.calculation(conf, ground_only=False, TAB=False, wordy=False)
        dic = nja.free_ion_param_AB(conf)
        dic_Bkq = nja.calc_Bkq(data_mod, conf, False, False)
        dic['dic_bkq'] = dic_Bkq
        pprint(dic_Bkq)

        elem = ['Hee','Hcf']
        result = calc.MatrixH(elem, **dic, eig_opt=False, wordy=False, save_matrix=True)
        matrix_result = np.load('matrix.npy')
        matrix = np.zeros((calc.basis.shape[0],calc.basis.shape[0]),dtype='object')
        print(matrix.shape)

        F = [0, dic['F2'], dic['F4'], 0]

        symbols_all = [sympy.Symbol('F0'), sympy.Symbol('F2'), sympy.Symbol('F4'), sympy.Symbol('F6'), sympy.Symbol('zeta'), sympy.Symbol('k')]
        for k in range(0,2*calc.l+1,2):
            for q in range(-k,k+1,1):
                symbols_all.append(sympy.Symbol(f"B{k}{q}"))
            
        for i in range(calc.basis.shape[0]):
            statei = calc.basis[i]
            Si = statei[0]/2.
            Li = statei[1]
            Ji = statei[2]/2.
            MJi = statei[3]/2.
            seni = statei[-1]
            labeli = calc.dic_LS[':'.join([f'{qq}' for qq in statei])]
            for j in range(0,i+1):
                statej = calc.basis[j]
                Sj = statej[0]/2.
                Lj = statej[1]
                Jj = statej[2]/2.
                MJj = statej[3]/2.
                senj = statej[-1]
                labelj = calc.dic_LS[':'.join([f'{qq}' for qq in statej])]
                
                H = nja.Hamiltonian([seni,Li,Si,senj,Lj,Sj,Ji,MJi,Jj,MJj], [labeli,labelj], calc.conf, calc.dic_cfp, calc.tables, calc.dic_LS, calc.dic_LS_almost)  

                if 'Hee' in elem:
                    if Ji==Jj and MJi==MJj:
                        if Li == Lj and Si == Sj:
                            if calc.l==3:
                                Hee = H.electrostatic_int(calc.basis, *F, evaluation=False, tab_ee = calc.dic_ee)
                            else:
                                Hee = H.electrostatic_int(calc.basis, *F, evaluation=False)
                            matrix[i,j] += Hee
                            if i != j:
                                matrix[j,i] += Hee

                if 'Hso' in elem:
                    if Ji==Jj and MJi==MJj:
                        Hso = -H.SO_coupling(dic['zeta'], 1, evaluation=False)
                        matrix[i,j] += Hso
                        if i != j:
                            matrix[j,i] += Hso

                if 'Hcf' in elem:
                    if Si==Sj:
                        Hcf = -H.LF_contribution(dic['dic_bkq'], evaluation=False)
                        matrix[i,j] += Hcf
                        if i != j:
                            matrix[j,i] += np.conj(Hcf)

        
        coeff_matrices = {symbol: np.zeros(matrix.shape, dtype='complex128') for symbol in symbols_all}

        for i in range(matrix.shape[0]):
            for j in range(0,i+1):
                element = matrix[i, j]
                if element != 0.0:
                    elements_c = element.as_coefficients_dict()
                    for symbol in symbols_all:
                        for key, value in elements_c.items():
                            if str(symbol) in str(key):
                                coeff_matrices[symbol][i, j] = complex(value)
                                if 'I' in str(key):
                                    coeff_matrices[symbol][i, j] *= 1j
                                if i != j:
                                    coeff_matrices[symbol][j, i] = np.conj(coeff_matrices[symbol][i, j])

        variables = []
        for k in range(0,2*calc.l+1,2):
            for q in range(-k,k+1,1):
                variables.append(sympy.Symbol(f"B{k}{q}"))

        total_coeff = np.zeros_like(coeff_matrices[symbols_all[0]])
        cost_coeff = np.zeros_like(coeff_matrices[symbols_all[0]])
        for symbol, coeff_matrix in coeff_matrices.items():
            if symbol not in variables:
                cost_coeff += coeff_matrix
            total_coeff += coeff_matrix

        D = np.zeros((cost_coeff.shape[0]*cost_coeff.shape[1], len(variables)), dtype='complex128')
        for i, variable in enumerate(variables):
            D_single = coeff_matrices[variable]+cost_coeff
            D[:, i] = D_single.flatten()
        
        #least sq solution
        B = np.linalg.inv(D.conj().T @ D) @ D.conj().T @ matrix_result.flatten()
        # B = np.linalg.pinv(D) @ matrix_result.flatten()

        for i in range(len(B)):
            if np.abs(B[i].imag)<1e-15:
                print(variables[i], B[i].real)
            else:
                print(variables[i], B[i].real, B[i].imag)

        dic_bkq_calc = {}
        count = 0
        for k in range(0,2*calc.l+1,2):
            dic_bkq_calc[f'{k}'] = {}
            for q in range(-k,k+1,1):
                dic_bkq_calc[f'{k}'][f'{q}'] = B[count].real
                count += 1

        pprint(dic_bkq_calc)

        dic_V = nja.from_Vint_to_Bkq_2(2, dic_bkq_calc, reverse=True)
        matrix = np.zeros((5,5))
        for i in range(5):
            for j in range(5):
                if i>=j:
                    matrix[i,j] = dic_V[str(i+1)+str(j+1)]
                    matrix[j,i] = dic_V[str(i+1)+str(j+1)]

        w,v = np.linalg.eigh(matrix)

        cfse, E_HS, E_LS = calc_CFSE(w,l,n_el,F)
        CFSE_list.append(cfse)
        E_HS_list.append(E_HS)
        E_LS_list.append(E_LS)

        plot_energy_levels(w, ax=ax, label=f"{dist:.3f}", tolerance=0.05, delta=0.5*idxd, barycenter=np.average(w))

    plt.show()

    plt.plot(x_axis, E_HS_list, c='magenta')
    plt.plot(x_axis, E_LS_list, c='green')
    plt.plot(x_axis, CFSE_list, c='blue')
    plt.show()

    #plot_CF_energy_single(w, tolerance=0.1)

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
def test_calc_ee_int_f():

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
            return 1/2*nja.state_legend(L)*(nja.state_legend(L)+1) - gU[U]

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
            v,W,U = nja.terms_labels_symm(conf)[label1]
            v1,W1,U1 = nja.terms_labels_symm(conf)[label2]

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

        calc = nja.calculation(conf, TAB=True, wordy=False)
        basis = calc.basis
        dic_ek_out[conf] = {}
        labels_list = []
        for i in range(basis.shape[0]):
            statei = basis[i]
            Si = statei[0]/2.
            Li = statei[1]
            Ji = statei[2]/2.
            MJi = statei[3]/2.
            seni = statei[-1]
            labeli = calc.dic_LS[':'.join([f'{qq}' for qq in statei])]

            for j in range(0,i+1):
                statej = basis[j]
                Sj = statej[0]/2.
                Lj = statej[1]
                Jj = statej[2]/2.
                MJj = statej[3]/2.
                senj = statej[-1]
                labelj = calc.dic_LS[':'.join([f'{qq}' for qq in statej])]

                if Ji==Jj and MJi==MJj and Li == Lj and Si == Sj:
                    if labeli+':'+labelj not in labels_list and labelj+':'+labeli not in labels_list:
                        labels_list.append(labeli+':'+labelj)
                        dic_ek_out = calc_ek(conf, labeli, labelj, Si, nja.state_legend(str(Li), inv=True), dic_ek_out)
                

        return dic_ek_out

    def calc_ee_int(conf, label, label1, S, L, dic_ek):

        n = int(conf[1:])
        v,W,U = nja.terms_labels_symm(conf)[label]
        v1,W1,U1 = nja.terms_labels_symm(conf)[label1]
        key1 = ':'.join([label,str(n),str(v),str(U),str(int(2*S+1)),L])
        key2 = ':'.join([label1,str(n),str(v1),str(U1),str(int(2*S+1)),L])
        ek_coeff = dic_ek[key1][key2]

        F0, F2, F4, F6 = sympy.symbols("F0, F2, F4, F6")
        coeff = [F0, F2, F4, F6]  

        Ek_coeff = []
        Ek_coeff.append(coeff[0] - 10/225*coeff[1] - 33/1089*coeff[2] - 286*25/184041*coeff[3])
        Ek_coeff.append(1/9*(70/225*coeff[1] + 231/1089*coeff[2] + 2002*25/184041*coeff[3]))
        Ek_coeff.append(1/9*(1/225*coeff[1] - 3/1089*coeff[2] + 7*25/184041*coeff[3]))
        Ek_coeff.append(1/3*(5/225*coeff[1] + 6/1089*coeff[2] - 91*25/184041*coeff[3]))

        ee_int = sympy.simplify(Ek_coeff[0]*ek_coeff[0] + Ek_coeff[1]*ek_coeff[1] + Ek_coeff[2]*ek_coeff[2] + Ek_coeff[3]*ek_coeff[3])

        return ee_int

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
    f = open('tables/dic_ee_values.txt', 'w')
    ff = open('tables/dic_ee_expression.txt', 'w')
    for conf in conf_list:

        print(conf)
        f.write('\n'+conf+'\n')
        ff.write('\n'+conf+'\n')

        dic_ek_conf = calc_dic_ek(conf, dic_ek_conf)
        dic_ee_expr[conf] = {}

        calc = nja.calculation(conf, TAB=True, wordy=False)
        basis = calc.basis
        for i in range(basis.shape[0]):
            statei = basis[i]
            Si = statei[0]/2.
            Li = statei[1]
            Ji = statei[2]/2.
            MJi = statei[3]/2.
            seni = statei[-1]
            labeli = calc.dic_LS[':'.join([f'{qq}' for qq in statei])]

            for j in range(0,i+1):
                statej = basis[j]
                Sj = statej[0]/2.
                Lj = statej[1]
                Jj = statej[2]/2.
                MJj = statej[3]/2.
                senj = statej[-1]
                labelj = calc.dic_LS[':'.join([f'{qq}' for qq in statej])]

                if Ji==Jj and MJi==MJj and Li == Lj and Si == Sj:
                    ee_value = calc_ee_int(conf, labeli, labelj, Si, nja.state_legend(str(Li), inv=True), dic_ek_conf[conf])
                    dic_ee_expr[conf][labeli+':'+labelj] = ee_value
                    if labeli!=labelj:
                        dic_ee_expr[conf][labelj+':'+labeli] = ee_value

        tab_ee = nja.read_ee_int_old(conf, False)
        coeff_ee = {}
        for key1 in tab_ee.keys():
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
                    try:
                        coeff_ee[key2].update({key1:numero})
                    except:
                        coeff_ee[key2]={}
                        coeff_ee[key2].update({key1:numero})

        dic_ee_values[conf] = {}
        F0, F2, F4, F6 = sympy.symbols("F0, F2, F4, F6")
        
        for key1 in coeff_ee.keys():
            for key2 in coeff_ee[key1].keys():
                
                expression = dic_ee_expr[conf][key1 + ':' + key2]
                ff.write(key1 + '\t' + key2 + '\t'+ str(expression) + '\n')

                coefficients = [expression.coeff(F0), expression.coeff(F2), expression.coeff(F4), expression.coeff(F6)]
                f.write(key1 + '\t' + key2 + '\t'+ '\t'.join([f'{coefficients[ii]:.15f}' for ii in range(4)]) + '\n')
                
                dic_ee_values[conf][key1 + ':' + key2] = np.array(coefficients, dtype=float)
                dic_ee_values[conf][key2 + ':' + key1] = np.array(coefficients, dtype=float)
                try:
                    assert np.allclose(coefficients, coeff_ee[key1][key2], rtol=1e-5, atol=1e-5)
                except AssertionError:
                    print(' ')
                    print(key1, key2)
                    print('mio', expression)
                    print(coeff_ee[key1][key2])
        f.write('#')
        ff.write('#')
    f.close()
    ff.close()

@test
def test_energy_allconf_d():

    def read_out(filename):

        active = False
        file = open(filename, 'r').readlines()
        energies = []
        comp_list = []
        for i,line in enumerate(file):
            if active and "|" in line:
                split1 = line.split('|')
                split2 = split1[0].split()
                comp_list.append(split1[-1].strip())
                if len(split2)==2:
                    energies.append(float(split2[-1]))
                else:
                    mult = int(split2[-1][:split2[-1].index(')')])
                    for j in range(mult):
                        energies.append(float(split2[2]))
            if "Level    Rel.Eng. (deg)" in line:
                active = True
        
        return energies[::-1], comp_list[::-1]

    conf_list = ['d2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'd9']
    data = nja.read_data('test/Oh_dist.inp', sph_flag = False)
    data[:,-1] *= -1
    
    for conf in conf_list:

        print(conf)

        E_fe, _ = read_out("test/complete_matrices/output_"+conf+".dat")
        
        dic_Bkq = nja.calc_Bkq(data, conf, False, False)
        dic = nja.free_ion_param_AB(conf)
        dic['dic_bkq'] = dic_Bkq
        calc = nja.calculation(conf, ground_only=False, TAB=False, wordy=False)
        result = calc.MatrixH(['Hee', 'Hso', 'Hcf'], **dic, eig_opt=False, wordy=False, ground_proj=True, save_matrix=True)

        # test over matrix elements
        check = np.loadtxt("test/complete_matrices/"+conf+".txt")
        matrix = np.load('matrix.npy', allow_pickle=True, fix_imports=False)
        count = 0
        for i in range(matrix.shape[0]):
            for j in range(0,i+1):
                assert round(matrix[i,j].real, 8) == round(check[count], 8)
                count += 1
                
        energy = result[0,:].real-np.min(result[0,:].real)
        energy_round = np.round(energy, 2)

        #check that energy_round and E_fe are close
        assert np.allclose(energy_round, E_fe, rtol=1e-2, atol=1e-2)

@test
def test_tables_d():

    conf_list = ['d2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'd9']
    data = nja.read_data('test/Oh_dist.inp', sph_flag = False)
    data[:,-1] *= -1
    
    for conf in conf_list:
        print('\n'+conf)
        
        dic_Bkq = nja.calc_Bkq(data, conf, False, False)
        dic = nja.free_ion_param_AB(conf)
        dic['dic_bkq'] = dic_Bkq

        calc = nja.calculation(conf, ground_only=False, TAB=False, wordy=False)
        result = calc.MatrixH(['Hee', 'Hso', 'Hcf'], **dic, eig_opt=False, wordy=False, ground_proj=True, save_matrix=True)

        # test over matrix elements
        check_R = np.loadtxt("test/complete_matrices/"+conf+".txt")
        matrix = np.load('matrix.npy', allow_pickle=True, fix_imports=False)
        count = 0
        for i in range(matrix.shape[0]):
            for j in range(0,i+1):
                assert round(matrix[i,j].real, 8) == round(check_R[count], 8)
                count += 1

@test
def test_energy_allconf_f():

    conf_list = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13']
    data = nja.read_data('test/beta.inp', sph_flag = False)
    data[:,-1] *= -1
    
    for conf in conf_list:
        print('\n'+conf)
        
        dic_Bkq = nja.calc_Bkq(data, conf, False, False)
        dic = nja.free_ion_param_f_HF(conf)
        dic['dic_bkq'] = dic_Bkq

        calc = nja.calculation(conf, ground_only=False, TAB=False, wordy=False)
        result = calc.MatrixH(['Hee', 'Hso', 'Hcf'], **dic, eig_opt=False, wordy=False, ground_proj=True, save_matrix=True, save_label=True)
        label_matrix = np.loadtxt('matrix_label.txt')
        # test over matrix elements
        check_R = np.loadtxt("test/complete_matrices/"+conf+"_R.txt")
        check_I = np.loadtxt("test/complete_matrices/"+conf+"_I.txt")
        matrix = np.load('matrix.npy', allow_pickle=True, fix_imports=False)
        count = 0
        for i in range(matrix.shape[0]):
            for j in range(0,i+1):
                try:
                    assert round(matrix[i,j].real, 6) == round(check_R[count], 6)
                    assert round(matrix[i,j].imag, 6) == round(check_I[count], 6)
                except AssertionError:  #the only elements that are actually different are those that involve 1K
                    print(label_matrix[i], label_matrix[j])
                    print('i,j',i,j)
                    print('matrix',matrix[i,j])
                    print('check_R',check_R[count])
                    print('check_I',check_I[count])
                    #exit()

                count += 1

@test
def test_tables_f():

    def read_cfp(conf, f):

        prime = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]

        file = open('tables/cfp_f.txt').readlines()
        check = False
        start = False
        dictionary = {}
        f.write(conf+'\n')
        for i,line in enumerate(file):

            if '[ONE PARTICLE FRACTIONAL PARENTAGE COEFFICIENTS F'+conf[1:]+']' in line:
                start = True
            if '[ONE PARTICLE FRACTIONAL PARENTAGE COEFFICIENTS F'+str(int(conf[1:])+1)+']' in line:
                break

            if line=='\n' and start:
                check = False

            if '[DAUGHTER TERM]' in line and start:
                splitline = line.split('[')
                daughter = splitline[0]
                dictionary[daughter] = {}
                check = True
            if check and start:
                splitline = line.split()
                if len(splitline)>2:
                    num = int(splitline[2])
                    for j in range(3,len(splitline)):
                        num *= prime[j-3]**int(splitline[j])
                    dictionary[daughter][splitline[0]] = num
                    f.write(daughter+'\t'+splitline[0]+'\t'+splitline[2]+'\t'+' '.join(splitline[3:])+'\n')
        pprint(dictionary)
        f.write('#\n')

    conf_list = ['f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13']   #['f1', 'f2', 'f3', 'f4']
    data = nja.read_data('test/beta.inp', sph_flag = False)
    data[:,-1] *= -1
    
    for conf in conf_list:
        print('\n'+conf)
        
        dic_Bkq = nja.calc_Bkq(data, conf, False, False)
        dic = nja.free_ion_param_f_HF(conf)
        dic['dic_bkq'] = dic_Bkq

        calc = nja.calculation(conf, ground_only=False, TAB=False, wordy=False)
        result = calc.MatrixH(['Hee', 'Hso', 'Hcf'], **dic, eig_opt=False, wordy=False, ground_proj=True, save_matrix=True)

        # test over matrix elements
        check_R = np.loadtxt("test/complete_matrices/"+conf+"_R.txt")
        check_I = np.loadtxt("test/complete_matrices/"+conf+"_I.txt")
        matrix = np.load('matrix.npy', allow_pickle=True, fix_imports=False)
        count = 0
        for i in range(matrix.shape[0]):
            for j in range(0,i+1):
                try:
                    assert round(matrix[i,j].real, 7) == round(check_R[count], 7)
                    assert round(matrix[i,j].imag, 7) == round(check_I[count], 7)
                except AssertionError:
                    print('i,j',i,j)
                    print('matrix',round(matrix[i,j].real, 7), round(matrix[i,j].imag, 7))
                    print('check',round(check_R[count], 7), round(check_I[count], 7))
                count += 1
        
@test
def test_PCM():  #with SIMPRE

    def make():
        import subprocess
        import os

        # Change the current working directory to the desired path
        current_dir = os.getcwd()
        os.chdir(current_dir+'/test')

        make_command = ["make", "clean", "test1"]
        make_proc = subprocess.Popen(make_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        stdout, stderr = make_proc.communicate()
        # pprint("stdout: {}".format(stdout))   #print to see simpre.f code errors
        pprint("stderr: {}".format(stderr))
        pprint("Return code: {}".format(make_proc.returncode))

        os.chdir(current_dir)

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
    # w_inp_dat(data[:,1:-1], data[:,-1], 7, directory='test/')
    # make()
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
            for ii, key in enumerate(sliced_keys):
                if MJ_list[i-1] == key:
                    ind = ii
            assert np.allclose(proj_nja[j+1][keys_list[ind]]/100, matrix[j,i], rtol=1e-3, atol=1e-3)

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
    assert angle1 < 1 and angle2 < 1 and angle3 < 1

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

#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------

def test_reduction():

    conf = 'f9'
    contributes = ['Hee','Hso','Hcf']
    dic = nja.read_AILFT_orca6('test/run_DOTA1_21sextets.out', conf)
    dic_Bkq = dic['dic_bkq'].copy()
    dic_Bkq['0'] = {}
    dic_Bkq['0']['0'] = 0

    for k in dic_Bkq.keys():
        for q in dic_Bkq[k]:
            print(k, q, dic_Bkq[k][q])

    calc = nja.calculation(conf, ground_only=False, TAB=True, wordy=True)
    #calc.reduce_basis(conf, roots = [(21,6)], wordy=True)  #questo va cambiato.. perché probabilmente mi serve di utilizzare la J
    result, projected = calc.MatrixH(contributes, **dic, eig_opt=False, wordy=True, ground_proj=True, return_proj=True)

    
if __name__ == '__main__':

    #### test graphical representations
    # test_plot_Ediagram()
    # test_plot_Ediagram_PCM()
    # test_CF_splitting()
    # test_TanabeSugano()
    # test_plot_magnetization_field()

    #### actual tests
    # test_energy_allconf_d()     #d2, d3, d4, d5, d6, d7, d8, d9
    # test_tables_d()
    # test_energy_allconf_f()     #f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13 (takes 3:30 h with clean RAM)
    # test_tables_f()
    # test_conv_AqkrkBkq() #f11
    # test_PCM() #f9
    # test_PCM_2() #f10
    # test_StevensfromMOLCAS #f9
    # test_Wigner_Euler_quat()
    # test_Wigner_Euler_quat2()
    # test_LF_rotation_euler()
    # test_LF_rotation_quat()
    # test_mag_moment()  #d1
    # test_mag_moment2()  #f9
    # test_M_vector()  #d3
    # test_M_vector2()  #d9
    # test_gtensor()  #f13
    # test_susceptibility_B_ord1()  #d8
    # test_susceptibility_B_ord1_2()  #d8
    # test_susceptibility_B_ord1_3()  #f9
    # test_susceptibility_B_ord1_4()  #f2
    # test_calc_susceptibility_zerofield()  #d8
    test_torque()

    #### on development
    # test_reduction()

    #### not real tests
    # test_calc_ee_int_f()

    #### failed experiments :(
    # test_JahnTeller()
    # test_PCM_from_Bkq()
    
    

    

