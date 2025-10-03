# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 17:44:43 2022

@author: Asus

SOME FUNCTIONS FOR ENVIRONMENTAL CONDITIONS to TAKE into acount in DEPROPAGATION :

    
"""
import math

import numpy as np


#%%  Geometrical microhone setup
##  %%%%%%%%%#################################
def drone_mic_geometry (n_mics, AGL, sphr, theta_ini, theta_jum, dx):
    """
     Returns the array centred on relative time based on the fist event.

     Inputs
     ------
    n_mics :int ->number of microphones
    AGL: shortest distance from source to central microphone
    sphr : radius of the sphere to depropagate
    theta_ini : initialg angulo of theta [15º]
    thetha_jum : jump in theta [15º]
    dx : distance from drone to central position during the flyby in x-axis
    
     Returns
     -------
    L_array_depropagated: array-> all events, all microhpnes
     
     Note:
    ------
    delta_L_dist_mic: array -> all events, all microphones
    """

    # n_mics = 3                      # it must be odd, pref 9 or more                            
    # n_cent = round(n_mics/2)        # Central microphone
    # AGL = 50                        # Above the ground level [m] ref. central microphone
    # sphr = 1                        # Sphere radious ofr reconstruction [m] ref. central microphone
    # #mic_line_max_deg = 60          # Maximum agle n_cent-n_mic-drone
    # theta_ini = 15                  # starting theta [deg]
    # theta_jum = 15                  # jumping on theta [deg]
    
    DY = []                           # dy distances mics on ground
    Mic_Drone_Distances = []          # distance mic to drone
    thetas = []                       
    
    for d in range((n_mics//2)+1):
                                                     # dx distances drone to central array position
        dz = AGL                                     # dz distances mics on ground
        
        dy = math.tan(theta_ini  * np.pi / 180)*AGL  # dy distances mics on ground
        DY.append(dy)
        
        dh = np.linalg.norm([dx,dy,dz])              # dh distances drone to mics
        # dh = AGL/math.cos(theta_ini  * np.pi / 180)  # dh distances drone to mics
        Mic_Drone_Distances.append(dh) 
        
        thetas.append(theta_ini) # thetas list
        theta_ini  = theta_ini  + theta_jum                      # Jump in degs
        
        
    lefth_side_DY = list(reversed(DY[1:])) #simetric sides
    lefth_side_MDD = list(reversed(Mic_Drone_Distances[1:]))
    lefth_side_thetas = list(reversed(thetas[1:]))
    
    DY = lefth_side_DY + DY  # both sides dy from central microhpone
    Mic_Drone_Distances = lefth_side_MDD + Mic_Drone_Distances # both sides dh from central microhpone
    
    thetas = list(np.array(lefth_side_thetas)*(-1)) + thetas # both sides thetas from central microhpone
    
    ###Mic_Drone_Distances are the r for noise LEVEL correction for spherical spreading
    
    Dist_Drone_Mics = np.array(Mic_Drone_Distances)
    
    # phi_dx = mpmath.acot(dx/AGL)                         # Phi angle
    if dx==0: dx=0.000001 #protection
    phi_dx = np.arctan(AGL/dx)
    return (Dist_Drone_Mics, thetas, DY, phi_dx)

#%% Atmospheric absortion
######################################

def atmAtten (L_array, band, dist, Tin=14.5, Psin=29.81, hrin=50):
    """
    Returns the correction level because the atmosferic absortion
    % A function to return the atmospheric attenuation of sound due to the vibrational relaxation times of oxygen and nitrogen.
NOTE:  This function does not account for spherical spreading!

Usage: [a] = atmAtten(T,P,RH,d,f)
               a - attenuation of sound for input parameters in dB
               T - temperature in deg C
               P - static pressure in inHg
               RH - relative humidity in %
               d - distance of sound propagation
               f - frequency of sound (may be a vector)

 Nathan Burnside 10/5/04
 AerospaceComputing Inc.
 nburnside@mail.arc.nasa.gov

 References:   Bass, et al., "Journal of Acoustical Society of America", (97) pg 680, January 1995.
               Bass, et al., "Journal of Acoustical Society of America", (99) pg 1259, February 1996.
               Kinsler, et al., "Fundamentals of Acoustics", 4th ed., pg 214, John Wiley & Sons, 2000.
    """
    
    T   = Tin + 273.15      # temp input in K
    To1 = 273.15            # triple point in K
    To  = 293.15            # ref temp in K
    
    Ps = Psin/29.9212598    #static pressure in atm
    Pso = 1                 # reference static pressure
    
    F = band/Ps;            # frequency per atm
    
    # calculate saturation pressure
    
    Psat = 10**(10.79586*(1-(To1/T))-5.02808*np.log10(T/To1)+1.50474e-4*(1-10**(-8.29692*((T/To1)-1)))-4.2873e-4*(1-10**(-4.76955*((To1/T)-1)))-2.2195983)
    h = hrin*Psat/Ps; # calculate the absolute humidity 

    # Scaled relaxation frequency for Nitrogen
    FrN = (To/T)**(1/2)*(9+280*h*math.exp(-4.17*((To/T)**(1/3)-1)))

    # scaled relaxation frequency for Oxygen
    FrO = (24+4.04e4*h*(.02+h)/(.391+h));

    # attenuation coefficient in nepers/m
    alpha = Ps*F**2.*(1.84e-11*(T/To)**(1/2) + (T/To)**(-5/2)*(1.275e-2*math.exp(-2239.1/T)/(FrO+F**2/FrO) + 1.068e-1*math.exp(-3352/T)/(FrN+F**2/FrN)))
    
    #delta_L_ATM = 10*np.log10(math.exp(2*alpha)) *dist 
    # https://uk.mathworks.com/matlabcentral/fileexchange/6000-atmospheric-attenuation-of-sound?#functions_tab
    delta_L_ATM = [10*np.log10(math.exp(2*alpha))*dd for dd in dist]
    return delta_L_ATM

#%% Atmospheric absortion ORIGINAL
# https://uk.mathworks.com/matlabcentral/fileexchange/6000-atmospheric-attenuation-of-sound?#functions_tab
######################################

def atmAtten_AAS ( dist, band, Tin=14.5, Psin=29.81, hrin=50):
    """
    Returns the correction level because the atmosferic absortion
    % A function to return the atmospheric attenuation of sound due to the vibrational relaxation times of oxygen and nitrogen.
NOTE:  This function does not account for spherical spreading!

[atenua_L_ATM]
Usage: [a] = atmAtten(T,P,RH,d,f)
               a - attenuation of sound for input parameters in [dB]
               T - temperature in deg C
               P - static pressure in inHg
               RH - relative humidity in %
               d - distance of sound propagation
               f - frequency of sound (may be a vector)

 Nathan Burnside 10/5/04
 AerospaceComputing Inc.
 nburnside@mail.arc.nasa.gov

 References:   Bass, et al., "Journal of Acoustical Society of America", (97) pg 680, January 1995.
               Bass, et al., "Journal of Acoustical Society of America", (99) pg 1259, February 1996.
               Kinsler, et al., "Fundamentals of Acoustics", 4th ed., pg 214, John Wiley & Sons, 2000.
    """
    atenua_L_ATM = np.zeros((band.shape[0],dist.shape[0]))
    
    T   = Tin + 273.15      # temp input in K
    To1 = 273.15            # triple point in K
    To  = 293.15            # ref temp in K
    
    Ps = Psin/29.9212598    #static pressure in atm
    Pso = 1                 # reference static pressure
        
    for ff in range(band.shape[0]): 
        F = band[ff]/Ps;            # frequency per atm
    
        # calculate saturation pressure
    
        Psat = 10**(10.79586*(1-(To1/T))-5.02808*np.log10(T/To1)+1.50474e-4*(1-10**(-8.29692*((T/To1)-1)))-4.2873e-4*(1-10**(-4.76955*((To1/T)-1)))-2.2195983)
        h = hrin*Psat/Ps; # calculate the absolute humidity 

        # Scaled relaxation frequency for Nitrogen
        FrN = (To/T)**(1/2)*(9+280*h*math.exp(-4.17*((To/T)**(1/3)-1)))

        # scaled relaxation frequency for Oxygen
        FrO = (24+4.04e4*h*(.02+h)/(.391+h));

        # attenuation coefficient in nepers/m
        alpha = Ps*F**2.*(1.84e-11*(T/To)**(1/2) + (T/To)**(-5/2)*(1.275e-2*math.exp(-2239.1/T)/(FrO+F**2/FrO) + 1.068e-1*math.exp(-3352/T)/(FrN+F**2/FrN)))
    
        #delta_L_ATM = 10*np.log10(math.exp(2*alpha)) *dist 
        for dd in range(dist.shape[0]):
            atenua_L_ATM[ff,dd] = (10*np.log10(math.exp(2*alpha)))*dist[dd] 
        
    return atenua_L_ATM

