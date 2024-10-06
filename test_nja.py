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
def test_PCM():  

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
    Aqkrk_simpre = cfp_matrix[:,2]
    Bqk_simpre = cfp_matrix[:,3]
    Aqkrk_nja = np.array([dic_Aqkrk[str(int(k))][str(int(q))] for k,q in cfp_matrix[:,0:2]])
    Bqk_nja = np.array([dic_Bqk[str(int(k))][str(int(q))] for k,q in cfp_matrix[:,0:2]])

    assert np.allclose(Aqkrk_simpre, Aqkrk_nja, rtol=1e-5, atol=1e-5)
    assert np.allclose(Bqk_simpre, Bqk_nja, rtol=1e-5, atol=1e-5)

    dic_Bkq2 = nja.from_Aqkrk_to_Bkq(dic_Aqkrk, revers=False)
    dic_Bkq = nja.calc_Bkq(data, conf, False, True)
    Bkq_recalc = np.array([dic_Bkq2[str(int(k))][str(int(q))] for k,q in cfp_matrix[:,0:2]])
    Bkq_calc = np.array([dic_Bkq[str(int(k))][str(int(q))] for k,q in cfp_matrix[:,0:2]])

    assert np.allclose(Bkq_recalc, Bkq_calc, rtol=1e-5, atol=1e-5)

    dic = {'dic_bkq': dic_Bkq2}
    calc = nja.calculation(conf, ground_only=True, TAB=True, wordy=False)
    result, proj_nja = calc.MatrixH(contributes, **dic, eig_opt=False, evaluation=True, return_proj=True, ground_proj=True)

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
def test_plot_Ediagram():

    conf = 'd8'
    calc = nja.calculation(conf, TAB=True, wordy=True)
    basis, dic_LS, basis_l, basis_l_JM = nja.Full_basis(conf)

    dic_orca = nja.read_AILFT_orca6('test/calcsuscenisalfix.out', conf, method='CASSCF', return_V=False, rotangle_V=False, print_orcamatrix=False)

    contributes = ['Hee', 'Hcf', 'Hso']
    theories = ['Hee', 'Hee + Hcf', 'Hee + Hcf + Hso']
    list_contr = []
    E_matrix = []
    proj_LS_dict = {}
    proj_prev_dict = {}
    prev = np.zeros((basis.shape[0]+1,basis.shape[0]), dtype='complex128')
    for i in range(len(contributes)):
        list_contr.append(contributes[i])
        result = calc.MatrixH(list_contr, **dic_orca, field=[0.0,0.0,28.0], evaluation=True, wordy=True, ground_proj=False, return_proj=False)
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

        E_matrix.append([round(result[0,ii].real,3) for ii in range(result.shape[-1])])  #tengo fino alla terza decimale

        prev = result.copy()

    E_matrix = np.array(E_matrix)  # dim = (n. contributi x n. livelli)

    #plot energy levels
    nja.level_fig_tot(E_matrix, theories, proj_LS_dict, proj_prev_dict)

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
    _, projected = calc.MatrixH(contributes, **dic, evaluation=True, ground_proj=True, return_proj=True)
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
    Rot_mat = np.array([[0.696343, 0.027550, -0.717180],[0.216884, 0.944468, 0.246864],[0.684155, -0.327447, 0.651698]])  #non serve il trasposto perché altrimenti a 5K viene l'asse z storto
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
    result, projected = calc.MatrixH(contributes, **dic, eig_opt=False, evaluation=True, wordy=False, ground_proj=True, return_proj=True, save_label=True, save_LF=True)
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
    result, projected = calc.MatrixH(contributes, **dic, eig_opt=False, evaluation=True, wordy=False, ground_proj=True, return_proj=True, save_label=True, save_LF=True)
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
    result = calc.MatrixH(contributes, **dic, eig_opt=False, evaluation=True)
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
    result, projected = calc.MatrixH(contributes, **dic, eig_opt=False, evaluation=True, wordy=False, ground_proj=True, return_proj=True, save_label=True, save_LF=True)
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
    result, projected = calc.MatrixH(contributes, **dic, eig_opt=False, evaluation=True, wordy=False, ground_proj=True, return_proj=True, save_label=True, save_LF=True)
    basis = np.loadtxt('matrix_label.txt')
    LF_matrix = np.load('matrix_LF.npy', allow_pickle=True, fix_imports=False)
    
    chi_B_diff, err_B = nja.susceptibility_B_ord1(np.array([[0.0,0.0,0.0]]), 298., basis, LF_matrix, delta=1)
    
    Magn = nja.Magnetics(calc, ['Hee', 'Hcf', 'Hso','Hz'], dic)

    chi_B = Magn.susceptibility_B_copy(fields=np.array([[0.0,0.0,0.0]]), temp=298., delta = 0.01, evaluation=True)   #provare a mettere + e - differenza invece che + e 0

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
    _, _ = calc.MatrixH(contributes, **dic, eig_opt=False, evaluation=True, wordy=False, ground_proj=True, return_proj=True, save_label=True, save_LF=True)
    basis = np.loadtxt('matrix_label.txt')
    LF_matrix = np.load('matrix_LF.npy', allow_pickle=True, fix_imports=False)
    
    chi_B_diff, err_B = nja.susceptibility_B_ord1(np.array([[0.0,0.0,0.0]]), 298., basis, LF_matrix, delta=1)

    w,v = np.linalg.eig(chi_B_diff)
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
    result, projected = calc.MatrixH(contributes, **dic, eig_opt=False, evaluation=True, wordy=False, ground_proj=True, return_proj=True, save_label=True, save_LF=True)
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

    calc = nja.calculation(conf, ground_only=False, TAB=True, wordy=True)
    calc.reduce_basis(conf, roots = [(21,6)], wordy=True)  #questo va cambiato.. perché probabilmente mi serve di utilizzare la J
    result, projected = calc.MatrixH(contributes, **dic, eig_opt=False, wordy=True, ground_proj=True, return_proj=True, save_label=True, save_LF=True)

    

if __name__ == '__main__':

    #TO DO:
    #one test on ground state composition and energy levels values from f_electron
    #one test on energy levels values from ORCA (on CoTp2)
    #one test on energy levels values from ORCA with basis reduction (on DyDOTA)
    #check energy for all configurations

    #### test figures
    # test_plot_Ediagram()

    #### all passed
    # test_conv_AqkrkBkq() #f11
    # test_PCM() #f9
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
    # test_calc_susceptibility_zerofield()  #d8

    #### on development
    test_reduction()
    

    

