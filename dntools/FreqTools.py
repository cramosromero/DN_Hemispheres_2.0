# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 17:03:00 2022

@author: Asus
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as ss
from matplotlib.widgets import SpanSelector

import dntools.PyOctaveBand as PyOctaveBand


# %% Calculate the Power Spectral Density
#####################################################
def calc_PSDs(DATA_raw_events,Fs, Ndft, WINDOW):#, Noverlap):
    """
    Returns the array of psd for one event on each microhone.

    Inputs
    ------
    Data_raw_events : array [event, samples(time), channel]
    Fs: int sample rate

    Returns
    -------
    DATA_psd_events : : array [event, samples(freq), channel]
    freq_welch: array frequency vector 
    Notes
    ------

    """
    N_eve = DATA_raw_events.shape[0]
    N_ch = DATA_raw_events.shape[2]
    
    DATA_psd_events = np.empty([N_eve, Ndft//2+1,N_ch])
    DATA_psd_Event = np.empty([Ndft//2+1,N_ch])
    for eve in range (N_eve):
        DATA_raw_eve = DATA_raw_events[eve,:,:]
    
        for ch in range(N_ch):
            freq_welch, DATA_psd_Event[:,ch] = ss.welch(DATA_raw_eve[:,ch], Fs, window=WINDOW, noverlap=Ndft // 4,
                                                        nperseg=Ndft, scaling='density') # noverlap=Noverlap,
            # print(ch)
        DATA_psd_events[eve,:,:] = DATA_psd_Event[:,:]
        
    return DATA_psd_events, freq_welch

# %% Calculate the broad band component from PSD
####################################################
def calc_broadband_PSDs(DATA_PSD_all_events, kernel_size):#, Noverlap):
    """
    Calculates broadband component of PSD using median filtering. This
    technique removes the contribution of tonal peaks.-Fabio-
    
    Returns the array of Broad bandcomponent for one event on each microhone.

    Inputs
    ------
    DATA_PSD_all_events : PSD array [event, samples(time), channel]
    kernel_size: int must be odd

    Returns
    -------
    DATA_psd_broadband_events : : array [event, samples(freq), channel] 
    Notes
    ------

    """
    
    N_eve = DATA_PSD_all_events.shape[0]
    N_samples = DATA_PSD_all_events.shape[1]
    N_ch = DATA_PSD_all_events.shape[2]
    
    DATA_psd_broadband_events = np.empty([N_eve, N_samples,N_ch])
    SmoothedSignal = np.empty([N_samples,N_ch])
    
    for eve in range (N_eve):
        DATA_raw_eve = DATA_PSD_all_events[eve,:,:]
    
        for ch in range(N_ch):
            SmoothedSignal[:,ch] = ss.medfilt(DATA_raw_eve[:,ch], kernel_size) # noverlap=Noverlap,
            # print(ch)
        DATA_psd_broadband_events[eve,:,:] = SmoothedSignal[:,:]
        
    return DATA_psd_broadband_events
    
# %% Calculate the Sound Presure level by band
##############################################################
def calc_SPLs(DATA_raw_events, Fs, fraction, limits, show=0) :
    """
    Returns the array of SPL for one event on each microhone.

    Inputs
    ------
    Data_raw_events : array [event, samples(time), channel]
    Fs: int sample rate
    fraction : {1,3} means 1/1 or 1/3 octave band
    limits: frequency limits : [20, 20000]
    show =0 : show the filter bank on frequency plot

    Returns
    -------
    DATA_overall_spl_events : overall sound pressure levels
    DATA_spl_events : : array [event, spl(freq), channel]
    freq: array frequency vector 
    Notes
    ------

    """
    
    N_eve = DATA_raw_events.shape[0]
    N_ch = DATA_raw_events.shape[2]
    
    if fraction == 3:
        bands = 31
    elif fraction == 1:
        bands = 11
    else:
        print('Error in number of frequency bands')
        
    DATA_spl_events = np.empty([N_eve, bands, N_ch])
    DATA_overall_spl_events = np.empty([N_eve, 1, N_ch])
    for eve in range(N_eve):
        for ch in range (N_ch):
            y = DATA_raw_events[eve,:,ch]
            spl, freq = PyOctaveBand.octavefilter(y, fs=Fs, fraction=fraction, order=6, limits=limits, show=0)
            DATA_spl_events[eve,:,ch] = np.array(spl)
            DATA_overall_spl_events[eve,0,ch] = 10*np.log10(np.sum(10**(np.array(spl)/10)))
    
    return DATA_overall_spl_events, DATA_spl_events, freq

# %% Calculate the Sound Presure level by component
########################################################## 
# def SignalTonalBroad(DATA_raw_events, Fs, Ndft, WINDOW):
#     """
#     Returns two arrays of PSD for each microhone on each event.
#     1 array for tonal component and 
#     1 array  for broad-band component

#     Inputs
#     ------
#     Data_raw_events : array [event, samples(time), channel]
#     Fs: int sample rate
    
#     Returns
#     -------
#     DATA_spl_all_events : : array [event, spl(freq), channel]
#     DATA_spl_broadband_events : : array [event, spl(freq), channel]
#     DATA_spl_tonal_events : : array [event, spl(freq), channel]
#     freq: array frequency vector 
#     Notes
#     ------

#     """
    
#     ## PSD calculation ##
#     ######################

#     df      = Fs/Ndft
#     p_ref   = 20e-6

#     """ Signal PSD"""
#     DATA_PSD_all_events, freq = calc_PSDs(DATA_raw_events,Fs, Ndft, WINDOW)
#     DATA_spl_all_events = 10*np.log10(DATA_PSD_all_events/p_ref**2)
    
#     """ Broadband PSD"""
#     DATA_PSD_broadband_events = calc_broadband_PSDs(DATA_PSD_all_events, kernel_size=31)
#     DATA_spl_broadband_events = 10*np.log10(DATA_PSD_broadband_events/p_ref**2)
   
#     """ Tonal PSD"""
#     DATA_spl_tonal_events = DATA_spl_all_events - DATA_spl_broadband_events
    
#     # fig, axs = plt.subplots()
#     # axs.plot(freq, DATA_spl_all_events[2,:,4],'r', label='Signal')
#     # axs.plot(freq, DATA_spl_broadband_events[2,:,4],'k', label='Broadband')
#     # # axs.plot(freq, DATA_spl_tonal_events[2,:,4],'g', label='Tonal')
#     # axs.set_xscale('log')
#     # plt.legend()
#     # axs.set_title('Components')
    
    
#     return DATA_spl_all_events, DATA_spl_broadband_events, DATA_spl_tonal_events, freq

# %% A_weighting
########################################################## 
from numpy import pi, polymul
from scipy.signal import bilinear, lfilter


def A_weighting(fs):
    """Design of an A-weighting filter.
    b, a = A_weighting(fs) designs a digital A-weighting filter for
    sampling frequency `fs`. Usage: y = scipy.signal.lfilter(b, a, x).
    Warning: `fs` should normally be higher than 20 kHz. For example,
    fs = 48000 yields a class 1-compliant filter.
    References:
       [1] IEC/CD 1672: Electroacoustics-Sound Level Meters, Nov. 1996.
       https://gist.github.com/endolith/148112
    """
    # Definition of analog A-weighting filter according to IEC/CD 1672.
    f1 = 20.598997057568145
    f2 = 107.65264864304628
    f3 = 737.8622307362899
    f4 = 12194.21714799801
    A1000 = 1.9997

    NUMs = [(2*pi * f4)**2 * (10**(A1000/20)), 0, 0, 0, 0]
    DENs = polymul([1, 4*pi * f4, (2*pi * f4)**2],
                   [1, 4*pi * f1, (2*pi * f1)**2])
    DENs = polymul(polymul(DENs, [1, 2*pi * f3]),
                                 [1, 2*pi * f2])

    # Use the bilinear transformation to get the digital filter.
    # (Octave, MATLAB, and PyLab disagree about Fs vs 1/Fs)
    return bilinear(NUMs, DENs, fs)

# %% OCTABE FILTER BANK
## BASED ON ANSI
########################################################## 


