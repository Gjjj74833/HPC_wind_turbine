

 
# -*- coding: utf-8 -*-
"""
Betti model implementation with PID controller

@author: Yihan Liu
@version (2023-06-24)
"""
import sys
import numpy as np
import bisect
from multiprocessing import Pool
from datetime import datetime
from gen_wind_imps_n import generate_wind

def process_rotor_performance(input_file = "Cp_Ct.NREL5MW.txt"):
    """
    This function will read the power coefficient surface from a text file generated
    by AeroDyn v15 and store the power coefficient in a 2D list

    Parameters
    ----------
    input_file : String, optional
        The file name of the pwer coefficient

    Returns
    -------
    C_p : 2D list
        The power coefficient. col: pitch angle, row: TSR value
    C_t : 2D list
        The thrust coefficient. col: pitch angle, row: TSR value
    pitch_angles : list
        The pitch angle corresponding to the col of C_p
    TSR_values : list
        The TSR values corresponding to the row of C_p

    """
    
    pitch_angles = []
    TSR_values = []

    with open(input_file, 'r') as file:
        lines = file.readlines()

        # Extract pitch angle vector
        pitch_angles_line = lines[4]
        # Extract TSR value vector
        TSR_values_line = lines[6]
        
        pitch_angles = [float(num_str) for num_str in pitch_angles_line.split()]
        TSR_values = [float(num_str) for num_str in TSR_values_line.split()]
        
        C_p = []
        for i in range(12, 12 + len(TSR_values)):
            Cp_row = [float(num_str) for num_str in lines[i].split()]
            C_p.append(Cp_row)
            
        C_t = []
        for i in range(16 + len(TSR_values), 16 + len(TSR_values) + len(TSR_values)):
            Ct_row = [float(num_str) for num_str in lines[i].split()]
            C_t.append(Ct_row)

    return C_p, C_t, pitch_angles, TSR_values


def CpCtCq(TSR, beta, performance):
    """
    Find the power coefficient based on the given TSR value and pitch angle

    Parameters
    ----------
    TSR : Tip speed ratio
    beta : blade pitch angle
    performance: The rotor performance generated by processing process_rotor_performance()

    Returns
    -------
    C_p: float
        power coefficient
    C_t: float
        thrust coefficient
    """
    beta = np.rad2deg(beta)

    C_p = performance[0] 
    C_t = performance[1]
    pitch_list = performance[2] 
    TSR_list = performance[3]
    
    # Find the closed pitch and TSR value in the list
    pitch_index = bisect.bisect_left(pitch_list, beta)
    TSR_index = bisect.bisect_left(TSR_list, TSR)
    
    # Correct the index if it's out of bounds or if the previous value is closer
    if pitch_index != 0 and (pitch_index == len(pitch_list) or abs(beta - pitch_list[pitch_index - 1]) < abs(beta - pitch_list[pitch_index])):
        pitch_index -= 1
    if TSR_index != 0 and (TSR_index == len(TSR_list) or abs(TSR - TSR_list[TSR_index - 1]) < abs(TSR - TSR_list[TSR_index])):
        TSR_index -= 1
    
    # Get the C_p value at the index 
    return C_p[TSR_index][pitch_index], C_t[TSR_index][pitch_index]


def gen_turbulence(v_bar, L, k_sigma_v, T_s, N_t, white_noise, 
                   delta_omega = 0.002, M = 5000, N = 100):
    """
    Generate turbulencec component for wind speed

    Parameters
    ----------
    v_bar : int
        Average wind speed
    L : int
        Turbulence length
    k_sigma_v : float
        Slope parameter
    T_s : int
        Time step
    N_t : int
        Number of step
    white_noise : np.array
        the white noise with mean = 0 and std = 1, has length 

    Returns
    -------
    Array of wind speed with turbulence

    """
    
    # Step 1: Update the current values of the parameters in 
    # the turbulence component model
    T_F = L / v_bar
    sigma_v = k_sigma_v * v_bar
    K_F = np.sqrt((2 * np.pi * T_F)/(4.20654 * T_s))
    
    # Step 2: Calculate the discrete impulse response of the filter
    delta_omega = 0.002 # Frequency step size
    M = 5000 # Number of frequency points
    N = 100 # Numerical parameters for convolution integration, divide 
            # finite integral from 0 to t to N regions
    
    # Discrete frequency domain P(omega)
    P = np.zeros(M + 1)
    for r in range(M + 1):
        P[r] = np.real(K_F / (1 + 1j * r * delta_omega * T_F)**(5/6))
    
    # Discrete impulse response h(k) === h(T_s*k), k range from 0 to N
    h = np.zeros(N + 1)
    for k in range(N + 1):
        h[k] = T_s * delta_omega * (2/np.pi) * np.sum(P * np.cos(k * np.arange(M + 1) * T_s * delta_omega))
    
    # Step 3: Generate the turbulence component in the interval using convolution
    v_t = np.zeros(N_t + 1)
    
    # Zero-pad the white noise 
    white_noise_padded = np.pad(white_noise, (0, N), 'constant')
    
    for m in range(N_t + 1):
        v_t[m] = T_s * np.sum(h * white_noise_padded[m : m + N + 1])
    
    return v_bar + sigma_v * v_t


def pierson_moskowitz_spectrum(U19_5, zeta, eta, t, random_phases):
    """
    This function generates the Pierson-Moskowitz spectrum for a given wind speed U10 and frequency f.
    
    parameters
    ----------
    U19_5 : float
        the average wind speed at 19.5m above the sea surface
    zeta : float
        the x component to evaluate
    eta : float
        the y component to evaluate. (Note: the coordinate system here is different
                                      from the Betti model. The downward is negative
                                      in this case)
    t: float
        the time to evaluate.
    random_phase : Numpy Array
        the random phase to generate wave. Should be in [0, 2*pi)

    Returns
    -------
    wave_eta : float
        The wave elevation
    [v_x, v_y, a_x, a_y]: list
        The wave velocity and acceleration in x and y direction
    """
    
    g = 9.81  # gravitational constant
    alpha = 0.0081  # Phillips' constant

    f_pm = 0.14*(g/U19_5)  # peak frequency
    
    N = 400
    
    cutof_f = 3*f_pm # Cutoff frequency
    
    f = np.linspace(0.1, cutof_f, N) # Array
    omega = 2*np.pi*f # Array
    delta_f = f[1] - f[0] # Array

    S_pm = (alpha*g**2/((2*np.pi)**4*f**5))*np.exp(-(5/4)*(f_pm/f)**4) # Array
    
    a = np.sqrt(2*S_pm*delta_f)
    k = omega**2/g    
    
    # Generate random phases all at once
    
    
    # Perform the calculations in a vectorized manner
    sin_component = np.sin(omega*t - k*zeta + random_phases)
    cos_component = np.cos(omega*t - k*zeta + random_phases)
    exp_component = np.exp(k*eta)
    
    wave_eta = np.sum(a * sin_component)
    
    v_x = np.sum(omega * a * exp_component * sin_component)
    v_y = np.sum(omega * a * exp_component * cos_component)
    
    a_x = np.sum((omega**2) * a * exp_component * cos_component)
    a_y = -np.sum((omega**2) * a * exp_component * sin_component)

    return wave_eta, [v_x, v_y, a_x, a_y]
    

def structure(x_1, beta, omega_R, t, performance, v_w, random_phases):
    """
    The structure of the Betti model

    Parameters
    ----------
    x_1 : np.array
        The state vector: [zeta v_zeta eta v_eta alpha omega]^T
    beta : float
        The blade pitch angle
    omega_R : double
        Rotor speed
    t : float
        Time
    performance: list
        The rotor performance parameter pass to CpCtCq(TSR, beta, performance)
    v_w: float
        The wind speed with turbulent
    v_aveg: float
        The average wind speed used to compute wave
    random_phase: Numpy Array
        The random parameter used to compute wave

    Returns
    -------
    np.linalg.inv(E) @ F: Numpy Array
        The derivative for the state vector
    v_in : float
        The relative wind speed
    Cp : float
        The power coefficient

    """
    
    zeta = x_1[0] # surge (x) position
    v_zeta = x_1[1] # surge velocity
    eta = x_1[2] # heave (y) position
    v_eta = x_1[3] # heave velocity
    alpha = x_1[4] # pitch position
    omega = x_1[5] # pitch velocity    
    
    g = 9.80665  # (m/s^2) gravity acceleration
    rho_w = 1025  # (kg/m^3) water density

    # Coefficient matrix E
    # Constants and parameters
    M_N = 240000  # (kg) Mass of nacelle
    M_P = 110000  # (kg) Mass of blades and hub
    M_S = 8947870  # (kg) Mass of "structure" (tower and floater)
    m_x = 11127000  # (kg) Added mass in horizontal direction
    m_y = 1504400  # (kg) Added mass in vertical direction

    d_Nh = -1.8  # (m) Horizontal distance between BS and BN
    d_Nv = 126.9003  # (m) Vertical distance between BS and BN
    d_Ph = 5.4305  # (m) Horizontal distance between BS and BP
    d_Pv = 127.5879  # (m) Vertical distance between BS and BP

    J_S = 3.4917*10**9 # (kg*m^2) "Structure" moment of inertia
    J_N = 2607890  # (kg*m^2) Nacelle moment of inertia
    J_P = 50365000  # (kg*m^2) Blades, hub and low speed shaft moment of inertia

    M_X = M_S + m_x + M_N + M_P
    M_Y = M_S + m_y + M_N + M_P
    
    d_N = np.sqrt(d_Nh**2 + d_Nv**2)
    d_P = np.sqrt(d_Ph**2 + d_Pv**2)

    M_d = M_N*d_N + M_P*d_P
    J_TOT = J_S + J_N + J_P + M_N*d_N**2 + M_P*d_P**2

    E = np.array([[1, 0, 0, 0, 0, 0],
         [0, M_X, 0, 0, 0, M_d*np.cos(alpha)],
         [0, 0, 1, 0, 0, 0],
         [0, 0, 0, M_Y, 0, M_d*np.sin(alpha)],
         [0, 0, 0, 0, 1, 0],
         [0, M_d*np.cos(alpha), 0, M_d*np.sin(alpha), 0, J_TOT]]) 

    #####################################################################
    # Force vector F
    
    h = 200  # (m) Depth of water
    h_pt = 47.89  # (m) Height of the floating structure
    r_g = 9  # (m) Radius of floater
    d_Sbott = 10.3397  # (m) Vertical distance between BS and floater bottom
    r_tb = 3  # (m) Maximum radius of the tower
    d_t = 10.3397  # (m) Vertical distance between BS and hooks of tie rods
    l_a = 27  # (m) Distance between the hooks of tie rods
    l_0 = 151.73  # (m) Rest length of tie rods
    
    K_T1 = 2*(1.5/l_0)*10**9  # (N/m) Spring constant of lateral tie rods
    K_T2 = 2*(1.5/l_0)*10**9  # (N/m) Spring constant of lateral tie rods
    K_T3 = 4*(1.5/l_0)*10**9  # (N/m) Spring constant of central tie rod

    d_T = 75.7843 # (m) Vertical distance between BS and BT
    rho = 1.225 # (kg/m^3) Density of air
    C_dN = 1 # (-) Nacelle drag coefficient
    A_N = 9.62 # (m^2) Nacelle area
    C_dT = 1 # (-) tower drag coefficient
    '''
    H_delta = np.array([[-2613.44, 810.13],
                        [810.13, 1744.28]]) # (-) Coefficient for computing deltaFA
    F_delta = np.array([-22790.37, -279533.43]) # (-) Coefficient for computing deltaFA
    C_delta = 10207305.54 # (-) Coefficient for computing deltaFA
    '''
    A = 12469 # (m^2) Rotor area
    n_dg= 2 # （-） Number of floater sub-cylinders
    C_dgper = 1 # (-) Perpendicular cylinder drag coefficient
    C_dgpar = 0.006 # (-) Parallel cylinder drag coefficient
    C_dgb = 1.9 # (-) Floater bottom drag coefficient
    R = 63 # (m) Radius of rotor
    den_l = 116.027 # (kg/m) the mass density of the mooring lines
    dia_l = 0.127 # (m) the diameter of the mooring lines
    h_T = 87.6 # (m) the height of the tower
    D_T = 4.935 # (m) the main diameter of the tower

    # Weight Forces
    Qwe_zeta = 0
    Qwe_eta = (M_N + M_P + M_S)*g
    Qwe_alpha = ((M_N*d_Nv + M_P*d_Pv)*np.sin(alpha) + (M_N*d_Nh + M_P*d_Ph )*np.cos(alpha))*g

    # Buoyancy Forces
    h_wave = pierson_moskowitz_spectrum(v_w, zeta, 0, t, random_phases)[0] + h
    h_p_rg = pierson_moskowitz_spectrum(v_w, zeta + r_g, 0, t, random_phases)[0] + h
    h_n_rg = pierson_moskowitz_spectrum(v_w, zeta - r_g, 0, t, random_phases)[0] + h
    
    h_w = (h_wave + h_p_rg + h_n_rg)/3
    h_sub = min(h_w - h + eta + d_Sbott, h_pt)
    
    d_G = eta - h_sub/2
    V_g = h_sub*np.pi*r_g**2 + max((h_w - h + eta + d_Sbott) - h_pt, 0)*np.pi*r_tb**2

    Qb_zeta = 0
    Qb_eta = -rho_w*V_g*g
    Qb_alpha = -rho_w*V_g*g*d_G*np.sin(alpha)
    
    # Tie Rod Force
    
    D_x = l_a

    l_1 = np.sqrt((h - eta - l_a*np.sin(alpha) - d_t*np.cos(alpha))**2 
                  + (D_x - zeta - l_a*np.cos(alpha) + d_t*np.sin(alpha))**2)
    l_2 = np.sqrt((h - eta + l_a*np.sin(alpha) - d_t*np.cos(alpha))**2 
                  + (D_x + zeta - l_a*np.cos(alpha) - d_t*np.sin(alpha))**2)
    l_3 = np.sqrt((h - eta - d_t*np.cos(alpha))**2 + (zeta - d_t*np.sin(alpha))**2)

    f_1 = max(0, K_T1*(l_1 - l_0))
    f_2 = max(0, K_T2*(l_2 - l_0))
    f_3 = max(0, K_T3*(l_3 - l_0))

    theta_1 = np.arctan((D_x - zeta - l_a*np.cos(alpha) + d_t*np.sin(alpha))
                        /(h - eta - l_a*np.sin(alpha) - d_t*np.cos(alpha)))
    theta_2 = np.arctan((D_x + zeta - l_a*np.cos(alpha) - d_t*np.sin(alpha))
                        /(h - eta + l_a*np.sin(alpha) - d_t*np.cos(alpha)))
    theta_3 = np.arctan((zeta - d_t*np.sin(alpha))/(h - eta - d_t*np.cos(alpha)))

    v_tir = (0.5*dia_l)**2*np.pi
    w_tir = den_l*g
    b_tir = rho_w*g*v_tir
    lambda_tir = w_tir - b_tir

    Qt_zeta = f_1*np.sin(theta_1) - f_2*np.sin(theta_2) - f_3*np.sin(theta_3)
    Qt_eta = f_1*np.cos(theta_1) + f_2*np.cos(theta_2) + f_3*np.cos(theta_3) + 4*lambda_tir*l_0
    Qt_alpha = (f_1*(l_a*np.cos(theta_1 + alpha) - d_t*np.sin(theta_1 + alpha)) 
                - f_2*(l_a*np.cos(theta_2 - alpha) - d_t*np.sin(theta_2 - alpha)) 
                + f_3*d_t*np.sin(theta_3 - alpha) + lambda_tir*l_0
                *(l_a*np.cos(alpha) - d_t*np.sin(alpha)) 
                - lambda_tir*l_0*(l_a*np.cos(alpha) 
                + d_t*np.sin(alpha)) - 2*lambda_tir*l_0*d_t*np.sin(alpha))

    # Wind Force
    v_in = v_w + v_zeta + d_P*omega*np.cos(alpha)

    TSR = (omega_R*R)/v_in

    Cp = 0
    Ct = 0
    
    Cp = CpCtCq(TSR, beta, performance)[0]
    Ct = CpCtCq(TSR, beta, performance)[1]

    
    FA = 0.5*rho*A*Ct*v_in**2
    FAN = 0.5*rho*C_dN*A_N*np.cos(alpha)*(v_w + v_zeta + d_N*omega*np.cos(alpha))**2
    FAT = 0.5*rho*C_dT*h_T*D_T*np.cos(alpha)*(v_w + v_zeta + d_T*omega*np.cos(alpha))**2
    
    Qwi_zeta = -(FA + FAN + FAT)
    Qwi_eta = 0
    Qwi_alpha = (-FA*(d_Pv*np.cos(alpha) - d_Ph*np.sin(alpha))
                 -FAN*(d_Nv*np.cos(alpha) - d_Nh*np.sin(alpha))
                 -FAT*d_T*np.cos(alpha))
    
    # Wave and Drag Forces
    h_pg = np.zeros(n_dg)
    v_per = np.zeros(n_dg) # v_perpendicular relative velocity between water and immersed body
    v_par = np.zeros(n_dg) # v_parallel relative velocity between water and immersed body
    a_per = np.zeros(n_dg) # a_perpendicular acceleration of water
    tempQh_zeta = np.zeros(n_dg)
    tempQh_eta = np.zeros(n_dg)
    tempQwa_zeta = np.zeros(n_dg)
    tempQwa_eta = np.zeros(n_dg)
    Qh_zeta = 0
    Qh_eta = 0
    Qwa_zeta = 0
    Qwa_eta = 0
    Qh_alpha = 0
    Qwa_alpha = 0
    
    v_x = [0, 0]
    v_y = [0, 0]
    a_x = [0, 0]
    a_y = [0, 0]
    height = [0, 0]
    
    for i in range(n_dg):

        h_pg[i] = (i + 1 - 0.5)*h_sub/n_dg
        height[i] = -(h_sub - h_pg[i])
        
        wave = pierson_moskowitz_spectrum(v_w, zeta, height[i], t, random_phases)[1]
        
        v_x[i] = wave[0]
        v_y[i] = wave[1]
        a_x[i] = wave[2]
        a_y[i] = wave[3]
        
        v_per[i] =  ((v_zeta + (h_pg[i] - d_Sbott)*omega*np.cos(alpha) - v_x[i])*np.cos(alpha)
                     + (v_eta + (h_pg[i] - d_Sbott)*omega*np.sin(alpha) - v_y[i])*np.sin(alpha))
        v_par[i] =  ((v_zeta + (h_pg[i] - d_Sbott)*omega*np.cos(alpha) - v_x[i])*np.sin(-alpha)
                    + (v_eta + (h_pg[i] - d_Sbott)*omega*np.sin(alpha) - v_y[i])*np.cos(alpha))
        a_per[i] = a_x[i]*np.cos(alpha) + a_y[i]*np.sin(alpha)
        
        tempQh_zeta[i] = (-0.5*C_dgper*rho_w*2*r_g*(h_sub/n_dg)*  np.abs(v_per[i])*v_per[i]*np.cos(alpha)
                        - 0.5*C_dgpar*rho_w*np.pi*2*r_g*(h_sub/n_dg)*  np.abs(v_par[i])*v_par[i]*np.sin(alpha))
        tempQh_eta[i] = (-0.5*C_dgper*rho_w*2*r_g*(h_sub/n_dg)* np.abs(v_per[i])*v_per[i]*np.sin(alpha)
                         - 0.5*C_dgpar*rho_w*np.pi*2*r_g*(h_sub/n_dg)* np.abs(v_par[i])*v_par[i]*np.cos(alpha))
        tempQwa_zeta[i] = (rho_w*V_g + m_x)*a_per[i]*np.cos(alpha)/n_dg
        tempQwa_eta[i] =  (rho_w*V_g + m_x)*a_per[i]*np.sin(alpha)/n_dg
        
        Qh_zeta += tempQh_zeta[i] 
        Qh_eta += tempQh_eta[i] 
        Qwa_zeta += tempQwa_zeta[i]
        Qwa_eta += tempQwa_eta[i]
        Qh_alpha += (tempQh_zeta[i]*(h_pg[i] - d_Sbott)*np.cos(alpha)
                    + tempQh_eta[i]*(h_pg[i] - d_Sbott)*np.sin(alpha))
        Qwa_alpha += (tempQwa_zeta[i]*(h_pg[i] - d_Sbott)*np.cos(alpha)
                    + tempQwa_eta[i]*(h_pg[i] - d_Sbott)*np.sin(alpha))
    
    Qh_zeta -= 0.5*C_dgb*rho_w*np.pi*r_g**2*np.abs(v_par[0])*v_par[0]*np.sin(alpha)
    Qh_eta -= 0.5*C_dgb*rho_w*np.pi*r_g**2*np.abs(v_par[0])*v_par[0]*np.cos(alpha)

    # net force in x DOF
    Q_zeta = Qwe_zeta + Qb_zeta + Qt_zeta + Qh_zeta + Qwa_zeta + Qwi_zeta + Qh_zeta# 
    # net force in y DOF
    Q_eta = Qwe_eta + Qb_eta + Qt_eta + Qh_eta + Qwa_eta + Qwi_eta + Qh_eta
    # net torque in pitch DOF
    Q_alpha = Qwe_alpha + Qb_alpha + Qt_alpha + Qh_alpha + Qwa_alpha + Qh_alpha + Qwi_alpha

    F = np.array([v_zeta, 
                  Q_zeta + M_d*omega**2*np.sin(alpha), 
                  v_eta, 
                  Q_eta - M_d*omega**2*np.cos(alpha), 
                  omega, 
                  Q_alpha])
    

    return np.linalg.inv(E) @ F, v_in, Cp, h_wave - h, [f_1, f_2, f_3/2]


def WindTurbine(omega_R, v_in, beta, T_E, t, Cp):
    """
    The drivetrain model 

    Parameters
    ----------
    omega_R : float
        The rotor speed
    v_in : float
        The relative wind speed
    beta : float
        The blade pitch angle
    T_E : float
        The generator torque
    t : float
        Time
    Cp : float
        The power coefficient

    Returns
    -------
    domega_R: float
        The derivative of rotor speed

    """
    
    # Constants and parameters
    J_G = 534.116 # (kg*m^2) Total inertia of electric generator and high speed shaft
    J_R = 35444067 # (kg*m^2) Total inertia of blades, hub and low speed shaft
    rho = 1.225 # (kg/m^3) Density of air
    A = 12469 # (m^2) Rotor area
    eta_G = 97 # (-) Speed ratio between high and low speed shafts
    
    tildeJ_R = eta_G**2*J_G + J_R
    tildeT_E = eta_G*T_E
    
    P_wind = 0.5*rho*A*v_in**3

    P_A = P_wind*Cp

    T_A = P_A/omega_R
    domega_R = (1/tildeJ_R)*(T_A - tildeT_E)
    
    return domega_R
    

def Betti(x, t, beta, T_E, performance, v_w, random_phases):
    """
    Combine the WindTurbine model and structure model
    
    Parameters
    ----------
    x : np.array
        the state vector: [zeta, v_zeta, eta, v_eta, alpha, omega, omega_R]^T
    t : float
        time
    beta : float
        blade pitch angle
    T_E : float
        generator torque
    performance: list
        The rotor performance parameter pass to CpCtCq(TSR, beta, performance)
    v_w: float
        The wind speed with turbulent
    v_aveg: float
        The average wind speed used to compute wave
    random_phase: Numpy Array
        The random parameter used to compute wave

    Returns
    -------
    dxdt : Numpy Array
        The derivative of the state vector

    """
    x1 = x[:6]
    omega_R = x[6]
    
    dx1dt, v_in, Cp, h_wave, rope_tension = structure(x1, beta, omega_R, t, performance, v_w, random_phases)
    dx2dt = WindTurbine(omega_R, v_in, beta, T_E, t, Cp)
    dxdt = np.append(dx1dt, dx2dt)
    
    return dxdt, h_wave, rope_tension




def rk4(Betti, x0, t0, tf, dt, beta_0, T_E, performance, v_w, v_wind, seed_wave, T_s1):
    """
    Solve the system of ODEs dx/dt = Betti(x, t) using the fourth-order Runge-Kutta method.

    Parameters:
    Betti : function
        The function to be integrated.
    x0 : np.array
        Initial conditions.
    t0 : float
        Initial time.
    tf : float
        Final time.
    dt : float
        Time step.
    beta : float
        blade pitch angle
    T_E : float
        generator torque
    Cp_type : int
        The mode to compute the power and thrust coefficient. 
        (0: read file; 1: use AeroDyn v15)
    performance: list
        Used when Cp_type = 0. The rotor performance parameter pass to CpCtCq(TSR, beta, performance)
    v_w: float
        The average wind speed
    wind: wind_mutiprocessing
        Used to for simulaton mutiprocessing. Its field containing the wind speed turbulent
        for all simulations
    
    Returns:
    t, x, v_wind[:len(t)], wave_eta
    np.array, np.array, np.array, np.raay
        Time points and corresponding values of state, wind velocities, sea surface elevation
        Each row is a state vector 
    """
    
    d_BS = 37.550 # (m) The position of center of weight of BS (platform and tower)
    
    n = int((tf - t0) / dt) + 1
    t = np.linspace(t0, tf, n)
    x = np.empty((n, len(x0)))
    x[0] = x0

    
    # generate a random seed
    state_before = np.random.get_state()
    #wave_seed = np.random.randint(0, high=10**7)
    np.random.seed(seed_wave)
    random_phases = 2*np.pi*np.random.rand(400)
    np.random.set_state(state_before)
    ###########################################################################
    # PI controller
    integral = 0
    beta = beta_0
    
    
    def PI_blade_pitch_controller(omega_R, dt, beta, integral, error, i, current_region):

        
        eta_G = 97 # (-) Speed ratio between high and low speed shafts
        J_G = 534.116 # (kg*m^2) Total inertia of electric generator and high speed shaft
        J_R = 35444067 # (kg*m^2) Total inertia of blades, hub and low speed shaft
        tildeJ_R = eta_G**2*J_G + J_R
    
        rated_omega_R = 1.26711 # The rated rotor speed is 12.1 rpm
        #rated_omega_R = 1.571
        zeta_phi = 0.7
        omega_phin = 0.6
        beta_k = 0.1099965
        dpdbeta_0 = -25.52*10**6
        
        GK = 1/(1+(beta/beta_k))
        
        K_p = 0.0765*(2*tildeJ_R*rated_omega_R*zeta_phi*omega_phin*GK)/(eta_G*(-dpdbeta_0))
        K_i = 0.013*(tildeJ_R*rated_omega_R*omega_phin**2*GK)/(eta_G*(-dpdbeta_0))
        K_d = 0.187437
   
        error_omega_R = omega_R - rated_omega_R
        error[i] = error_omega_R

        P = K_p*eta_G*error_omega_R
        integral = integral + dt*K_i*eta_G*error_omega_R
        D = (K_d*(error[i] - error[i-1]))/dt

        delta_beta = P + integral + D
        
        # set max change rate in 8 degree per second
        
        if delta_beta > 0 and delta_beta/dt > 0.139626:
            delta_beta = 0.139626*dt
        elif delta_beta < 0 and delta_beta/dt < -0.139626:
            delta_beta = -0.139626*dt
        
        beta += delta_beta
        
        if beta <= 0:
            beta = 0
        elif beta >= np.pi/4:
            beta = np.pi/4
            
        # find proper T_E based on rotor speed
        generator_speed = omega_R * 97 * 9.549297
        
        if current_region == 1:
            new_T_E = 0
            beta = 0
            integral = 0
        elif current_region == 1.5:
            new_T_E = 96.53386 * generator_speed - 64677.68667
            beta = 0
            integral = 0
        elif current_region == 2:
            new_T_E = 0.0255764 * generator_speed**2
            beta = 0
            integral = 0
        elif current_region == 2.5:
            new_T_E = 729.4343 * generator_speed - 813043.45
            beta = 0
            integral = 0
        else:
            new_T_E = 43093.55
        
        if new_T_E < 0:
            new_T_E = 0
        
        max_TE_change_rate = 15000 / dt
        delta_T_E = new_T_E - T_E

        if delta_T_E > max_TE_change_rate:
            new_T_E = T_E + max_TE_change_rate
        elif delta_T_E < -max_TE_change_rate:
            new_T_E = T_E - max_TE_change_rate
        
        return beta, integral, error, new_T_E
    
    def find_region(omega_R, beta, current_region):
      
        generator_speed = omega_R * 97 * 9.549297
        # if in region 1
        if current_region == 1:
            # change to region 1.5 if the generator speed 
            # is greater than 670 rpm
            if generator_speed > 670:
                return 1.5
            # else stay in region 1
            return 1
        
        # if in region 1.5
        if current_region == 1.5:
            # change to region 2 if the generator speed is greater than 871 rpm
            if generator_speed > 871:
                return 2
            # change to region 1 if the generator speed is less than 670 rpm
            if generator_speed < 670:
                return 1
            # else stay in region 1.5
            return 1.5
        
        # if in region 2
        if current_region == 2:
            # change to region 2.5 if the generator speed is greater than 1161.963 rpm
            if generator_speed > 1161.963:
                return 2.5
            # change to region 1.5 if the generator speed is less than 871 rpm
            if generator_speed < 871:
                return 1.5
            return 2
        
        # if in region 2.5
        if current_region == 2.5:
            # change to region 3 if the generator speed is greater than 1173.7 rpm
            if generator_speed > 1173.7:
                return 3
            # change to region 2 if the generator speed is less than 1161.963
            if generator_speed < 1161.963:
                return 2
            return 2.5
        
        # if in region 3
        if current_region == 3:
            if beta == 0:
                global integral
                integral = 0
                return 2.5
            return 3

    ###########################################################################
    error = np.empty(n)
    current_region = 3
    betas = []
    h_waves = []
    T_E_list = []
    P_A_list = []
    rope_tension_list = []
    for i in range(n - 1):
        betas.append(beta)
        #v_average_ml = v_ml[i // int((T_s1 / dt))]
        k1, h_wave, rope_tension = Betti(x[i], t[i], beta, T_E, performance, v_wind[i], random_phases)
        k2 = Betti(x[i] + 0.5 * dt * k1, t[i] + 0.5 * dt, beta, T_E, performance, v_wind[i], random_phases)[0]
        k3 = Betti(x[i] + 0.5 * dt * k2, t[i] + 0.5 * dt, beta, T_E, performance, v_wind[i], random_phases)[0]
        k4 = Betti(x[i] + dt * k3, t[i] + dt, beta, T_E, performance, v_wind[i], random_phases)[0]
        x[i + 1] = x[i] + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
        
        current_region = find_region(x[i][6], beta, current_region)
        beta, integral, error, T_E = PI_blade_pitch_controller(x[i][6], dt, beta, integral, error, i, current_region)
        
        h_waves.append(h_wave)
        T_E_list.append(T_E)
        P_A_list.append(T_E*97*x[i][6])
        rope_tension_list.append(rope_tension)
        
    T_E_list.append(T_E_list[-1])
    P_A_list.append(P_A_list[-1])
    rope_tension_list.append(rope_tension_list[-1])
    
    
    x[:, 4] = -np.rad2deg(x[:, 4])
    x[:, 5] = -np.rad2deg(x[:, 5])
    x[:, 6] = (60 / (2*np.pi))*x[:, 6]
   
    x[:, 0:4] = -x[:, 0:4]
    x[:, 2] += d_BS


   
    steps = int(0.5 / dt)
    # dicard data for first 500s
    discard_steps = int(500 / 0.5)
     

    t_sub = t[::steps]#[discard_steps:]
    x_sub = x[::steps]#[discard_steps:]
    #v_wind_sub = v_wind[:len(t)][::steps]#[discard_steps:]
    h_wave_sub = np.array(h_waves)[::steps]#[discard_steps:]
    betas_sub = betas[::steps]#[discard_steps:]
    T_E_list_sub = T_E_list[::steps]#[discard_steps:]
    P_A_list_sub = P_A_list[::steps]#[discard_steps:]
    rope_tension_list_sub =  rope_tension_list[::steps]
    
    t_final = t_sub-t_sub[0]
    t_free = np.array([t_final[1], t_final[-1]])
    
    return t_free, np.delete(x_sub, [1, 3, 5], axis=1), h_wave_sub, betas_sub, T_E_list_sub, P_A_list_sub, rope_tension_list_sub

def main(end_time, v_w, x0, file_index, wind_speeds_wave_seed, time_step = 0.05, T_s1 = 180):
    """
    Cp computation method

    Parameters
    ----------


    Returns
    -------
    t: np.array
        The time array
    x: 2D array:
        The state at each time.The row of x corresponding to each time step.
        The column is each state [surge, surge_velocity, heave, heave_velocity, pitch, pitch_rate, rotor_speed]
    v_wind: list
        The wind speed at each time step
    wave_eta: list
        The wave elevation at surge = 0 for each time step
    """
    performance = process_rotor_performance()
    
    start_time = 0
    
    wave_seed = wind_speeds_wave_seed[-1]
    wind_speeds = wind_speeds_wave_seed[:-1]

    v_wind = np.repeat(wind_speeds, int(1/time_step))

    # modify this to change run time and step size
    #[Betti, x0 (initial condition), start time, end time, time step, beta, T_E]
    t, x, wave_eta, betas, T_E, P_A, rope_tension = rk4(Betti, x0, start_time, end_time, time_step, 0.32, 43093.55, performance, v_w, v_wind, wave_seed, T_s1)
    
    
    return t, x, wind_speeds, wave_eta, betas, wave_seed, T_E, P_A, rope_tension

def run_simulation(params):
    return main(*params)


def run_simulations_parallel(n_simulations, params):
    
    state = np.array([-2.61426271, 
                 -0.00299848190, 
                 37.5499264, 
                 -0.0558194064,
                 0.00147344971, 
                 -0.000391112846, 
                 1.26855822])

    params.append(state)
    file_index = list(range(0, n_simulations))
    wind_speed_wave_seed_array = np.load(f'./wind_{sys.argv[5].lower()}_ite_{sys.argv[6]}/wind_{sys.argv[1]}.npy')
   
    
    with Pool(int(sys.argv[3])) as p:
        
        all_params = [params + [file_index[i], wind_speed_wave_seed_array[i]] for i in range(n_simulations)]
        
        results = p.map(run_simulation, all_params)

    return results
    
def save_binaryfile(results):
    
    
    t = results[0][0]
    
    # Only take the states part to analyze
    state = np.stack([s[1] for s in results], axis=2)
    wind_speed = np.stack([s[2] for s in results], axis=1)
    wave_eta = np.stack([s[3] for s in results], axis=1)
    betas = np.stack([s[4] for s in results], axis=1)
    seeds = np.stack([s[5] for s in results], axis=1)
    T_E = np.stack([s[6] for s in results], axis=1)
    P_A = np.stack([s[7] for s in results], axis=1)  
    rope_tension = np.stack([s[8] for s in results], axis=2)  
    
    now = datetime.now()
    time = now.strftime('%Y-%m-%d_%H-%M-%S')   

    np.savez(f'./results_{sys.argv[5].lower()}_ite_{sys.argv[6]}/results_{sys.argv[1]}_{time}.npz', t=t,  
                                                            state=state, 
                                                            wind_speed=wind_speed, 
                                                            wave_eta=wave_eta, 
                                                            betas=betas, 
                                                            seeds=seeds, 
                                                            T_E=T_E,
                                                            P_A=P_A,
                                                            rope_tension=rope_tension)
   

###############################################################################
###############################################################################

# How to use
# python Betti_GMM_KDE.py $1={global task ID}, $2={simulation number}, $3={local task ID},
#                         $4={simulation time}, $5={method: kde or gmm}, $6={iteration index},



if __name__ == '__main__':

    v_w = 11
    end_time = int(sys.argv[4])
    n_simulations = int(sys.argv[2])

    params = [end_time, v_w]
    
    results = run_simulations_parallel(n_simulations, params)

    save_binaryfile(results)

 
