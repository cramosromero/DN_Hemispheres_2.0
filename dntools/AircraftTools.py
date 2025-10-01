# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 12:47:29 2022

@author: Asus

SOME FUNCTIONS FOR AIRCRAFT NOISE METRICS AND CALCULATIONS:
    
    SEL_def_calc ->  by deffinition Min_2015
    depropagation -> NASA-subgrup 2
"""
import numpy as np
import dntools.EnvironTools as EnvironTools
import matplotlib.pyplot as plt

# %% SEL Calculation based on ACU data
################################################################
def SEL_def_calc (levels, Fs):
    """
     Returns the array ceneterd on relative time based on the fist event.

     Inputs
     ------
    levels : np.array all event over LMAX-10dB
    Fs: int sample rate

     Returns
     -------
    SEL_def: SEL values all events all channels by deffinition Min_2015
     
     """
    ts = levels.shape[0]             #number of Lp samples (Lp<=Lmax-10db) #time samples
    te = ts/Fs
    
    Leq = 10*np.log10((1/ts)*(np.sum(10**(levels/10))))
 
    SEL_def = Leq + 10*np.log10(te) #by deffinition Min_2015 also in Moreno_2023
    
    return SEL_def, ts

# %% Depropagation, includes: spherical spreading and atmospheric atenuation
################################################################
def depropagator (L_array, heightAGL, rad_to_deprop, band, dx):
    """
     Returns the array ceneterd on relative time based on the fist event.

     Inputs
     ------
    L_array : array-> (events, chanells) all events, all microhpnes 
    rad_to_deprop: int -> depropagation distance from source
    heightAGL: shortest distance from source to central microphone
    band: central frequency, important in the atmospheric absortion
    dx : distance from drone to central position during the flyby in x axis
    
     Returns
     -------
    L_array_depropagated: array-> all events, all microhpnes
    thetas: array of azimut angles
    dist_ground_mics:array microphone distance from central microphone
    phi_dx: polar angles on each dx distance from the mic-line, before and after
     
     Note:
    ------
    n_mics = int number of channels
    theta_ini = 15, initial theta
    theta_jum = 15, jump in theta As is suh¡gested in the NASA SG2
    delta_L_dist_mic: array -> all events, all microphones
    """
    n_mics = L_array.shape[2] #number of channels
    

    (dist_drone_mics, thetas, dist_ground_mics, phi_dx) = EnvironTools.drone_mic_geometry (n_mics, heightAGL,
                                                                                   rad_to_deprop, 0, 15, dx)
    
    # Hansen - spherical spreading
    delta_L_dist_mic = 20*np.log10( rad_to_deprop / dist_drone_mics) 
    
    # by Nathan Burnside - Atmospheric Attenuation of Sound
    dist = dist_drone_mics - rad_to_deprop
    delta_L_ATM = EnvironTools.atmAtten (L_array, band, dist, Tin=25, Psin=29.92, hrin=70)
    
    L_array_depropagated = L_array - delta_L_dist_mic - delta_L_ATM
     
    return L_array_depropagated, thetas, dist_ground_mics, phi_dx

# %% Depropagation, includes: only spherical spreading.
# The atmospheric absortion was considered before for each frecuency and distance
# Based on SPLs
################################################################
def depropagator_L (L_array, dist_drone_mics, rad_to_deprop):
    """
     Returns the array ceneterd on relative time based on the fist event.

     Inputs
     ------
    L_array : array-> (events, chanells) all events, all microhpnes 
    rad_to_deprop: int -> depropagation distance from source
    heightAGL: shortest distance from source to central microphone
    band: central frequency, important in the atmospheric absortion
    dx : distance from drone to central position during the flyby in x axis
    
     Returns
     -------
    L_array_depropagated: array-> all events, all microhpnes
    thetas: array of azimut angles
    dist_ground_mics:array microphone distance from central microphone
    phi_dx: polar angles on each dx distance from the mic-line, before and after
     
     Note:
    ------
    theta_ini = 15, initial theta
    theta_jum = 15, jump in theta As is suh¡gested in the NASA SG2
    delta_L_dist_mic: array -> all events, all microphones
    """
    
    # Hansen - spherical spreading
    delta_L_dist_mic = 20*np.log10( rad_to_deprop / dist_drone_mics) 
    
    # by Nathan Burnside - Atmospheric Attenuation of Sound
    # dist = dist_drone_mics - rad_to_deprop
    # delta_L_ATM = EnvironTools.atmAtten (L_array, band, dist, Tin=25, Psin=29.92, hrin=70)
    
    L_array_depropagated = L_array - delta_L_dist_mic
     
    return L_array_depropagated

# %% Directivity based on a kind of Directivity Index - Marshal
################################################################
def directivity (Levels_array):
    """
     Returns the array of Directivity index DI.

     Inputs
     ------
    Levels_array : array-> (events, chanells) all events, all microhpnes 
    angles: array-> angles for plolt
    
     Returns
     -------
    DI_events: array-> directivity index on each event
     
     Note:
    ------

    """
    DI_events = np.empty(Levels_array.shape)
    
    for eve in range (Levels_array.shape[0]):
        levels = Levels_array[eve,:,:]
        Lm = np.mean(levels)
        DI_events[eve,:] = (levels - Lm) #Fundmentals of acoustics Marshal
    
    return DI_events

 
    
 
    