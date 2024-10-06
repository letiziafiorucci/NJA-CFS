
import nja_cfs_v0 as nja
import numpy as np


# TRANSITION METALS 

if 0:
    
    conf = 'd8'
    calc = nja.calculation(conf, TAB=True, wordy=True)
    basis, dic_LS, basis_l, basis_l_JM = nja.Full_basis(conf)

    dic_orca = dev.read_AILFT_orca6('test/calcsuscenisalfix.out', conf, method='CASSCF', return_V=False, rotangle_V=False, print_orcamatrix=False)

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
