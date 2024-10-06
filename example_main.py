
import nja_cfs_v0 as nja
import numpy as np

# TRANSITION METAL COMPLEX 

if 0:
    
    conf = 'd7'  #'d8'
    calc = nja.calculation(conf, TAB=True, wordy=True)
    basis, dic_LS, basis_l, basis_l_JM = nja.Full_basis(conf)
    B0 = [0.0,0.0,28.0]

    # CF parameters from orca file
    dic_orca = nja.read_AILFT_orca6('test/calcsuscenisalfix.out', conf, method='CASSCF', return_V=False, rotangle_V=False, print_orcamatrix=False)
    dic = dic_orca.copy()

    contributes = ['Hee', 'Hcf', 'Hso', 'Hz']
    theories = ['Hee', 'Hee + Hcf', 'Hee + Hcf + Hso', 'Hee + Hcf + Hso + Hz']
    list_contr = []
    E_matrix = []
    proj_LS_dict = {}
    proj_prev_dict = {}
    prev = np.zeros((basis.shape[0]+1,basis.shape[0]), dtype='complex128')
    for i in range(len(contributes)):
        list_contr.append(contributes[i])
        result = calc.MatrixH(list_contr, **dic, field=B0, wordy=True, ground_proj=False, return_proj=False)
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


# LANTHANOID COMPLEX 

# Plot Sievers surface
if 0:
    conf = 'f9'
    ground = nja.ground_term_legend(conf)
    S = (int(ground[0])-1)/2
    L = nja.state_legend(ground[1])
    splitg = ground.split('(')
    J = eval(splitg[-1][:-1])
    M = J
    A2,A4,A6 = nja.coeff_multipole_moments(conf, J, M, L=0, S=0)
    nja.plot_charge_density(A2, A4, A6)

# Ground state only
if 0:
    
    conf = 'f9'
    calc = nja.calculation(conf, ground_only=True, TAB=True, wordy=True)
    basis, dic_LS, basis_l, basis_l_JM = nja.Full_basis(conf)
    B0 = [0.0,0.0,28.0]

    # CF parameters from orca file
    dic_orca = nja.read_AILFT_orca6('test/run_DOTA1_21sextets.out', conf)
    dic = dic_orca.copy()

    # CF parameters from PCM
    # data = nja.read_data('test/beta.inp', sph_flag = False)
    # data[:,-1] *= -1
    # dic_Bqk = nja.calc_Bqk(data, conf, False, True)
    # dic = nja.free_ion_param_f(conf)
    # dic['dic_bkq'] = dic_Bqk

    contributes = ['Hcf', 'Hz']
    theories = ['Hcf', 'Hcf + Hz']
    list_contr = []
    E_matrix = []
    proj_LS_dict = {}
    proj_prev_dict = {}
    prev = np.zeros((basis.shape[0]+1,basis.shape[0]), dtype='complex128')
    for i in range(len(contributes)):
        list_contr.append(contributes[i])
        result = calc.MatrixH(list_contr, **dic, field=B0, wordy=True, ground_proj=False, return_proj=False)
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

# Complete (takes 45 minutes in total with f9)
if 0:
    
    conf = 'f3'
    calc = nja.calculation(conf, ground_only=False, TAB=True, wordy=True)
    basis, dic_LS, basis_l, basis_l_JM = nja.Full_basis(conf)
    B0 = [0.0,0.0,28.0]

    # CF parameters from orca file
    # dic_orca = nja.read_AILFT_orca6('test/run_DOTA1_21sextets.out', conf)
    # dic = dic_orca.copy()

    # CF parameters from PCM
    data = nja.read_data('test/beta.inp', sph_flag = False)
    data[:,-1] *= -1
    dic_Bqk = nja.calc_Bqk(data, conf, False, True)
    dic = nja.free_ion_param_f(conf)
    dic['dic_bkq'] = dic_Bqk

    contributes = ['Hee', 'Hso', 'Hcf', 'Hz']
    theories = ['Hee', 'Hee + Hso', 'Hee + Hso + Hcf', 'Hee + Hso + Hcf + Hz']
    list_contr = []
    E_matrix = []
    proj_LS_dict = {}
    proj_prev_dict = {}
    prev = np.zeros((basis.shape[0]+1,basis.shape[0]), dtype='complex128')
    for i in range(len(contributes)):
        list_contr.append(contributes[i])
        result = calc.MatrixH(list_contr, **dic, field=B0, wordy=True, ground_proj=False, return_proj=False)
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
