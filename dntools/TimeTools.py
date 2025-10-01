# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 09:53:51 2022

@author: SES271
"""
import numpy as np
import math
import dntools.AircraftTools as AircraftTools
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate

# %% Move te aoustic maximum values in the same T0 FOR FLYOVERS
# ##########################################################################
def segment_time_ADJUST (time_slice, Fs, DATA_raw_events, DATA_acu_events):
    """
    Returns the array ceneterd and sliced on relative time based on the DATA_aco Lmaxs.

    Inputs
    ------
    time_slice : time length segment
    DATA_raw_events : array of raw data
    DATA_raw_events : array of acu data
    Fs: int sample rate

    Returns
    -------
    TIME_r: array time vector zero-centered
    vec_time
    Data_raw_segmented : array of raw data
    Data_acu_segmented : array of raw data
    
    
    Notes
    ------
    first median filter
    second the sement
    """

    samples = round(Fs*time_slice)
    vec_time = np.linspace(0,samples,samples)

    """ medianfilter """

    import scipy.signal as ss
    DATA_acu_events_medianf = np.zeros([DATA_raw_events.shape[0],DATA_raw_events.shape[1]])

    for eve in range(DATA_acu_events.shape[0]):
        data = abs(DATA_acu_events[eve,:,4])
        DATA_acu_events_medianf[eve,:] = ss.medfilt(data,55)
        
    qq_index_max = np.argmax(DATA_acu_events_medianf,axis=1)

    """ segmented """

    Data_raw_segmented = np.empty([DATA_raw_events.shape[0],samples,DATA_raw_events.shape[2]])
    Data_acu_segmented = np.empty([DATA_acu_events.shape[0],samples,DATA_acu_events.shape[2]])

    for eve in range(DATA_raw_events.shape[0]):
        index_c = qq_index_max[eve]
        
        segment_raw = DATA_raw_events[eve,:,:][index_c - samples//2 : index_c + samples//2]
        segment_acu = DATA_acu_events[eve,:,:][index_c - samples//2 : index_c + samples//2]
        
        Data_raw_segmented[eve,:,:] = segment_raw
        Data_acu_segmented[eve,:,:] = segment_acu
        
    # TIME_r = 0
    TIME_r = np.linspace(-samples//2, samples//2, samples)/Fs
    
    return TIME_r,vec_time, Data_raw_segmented, Data_acu_segmented

# %% relatived_time
# ##########################################################################
def relatived_time (DATA_mic,Fs, t_data):
    """
    Returns the array ceneterd on relative time based on the fist event.

    Inputs
    ------
    Data_events : array
    Fs: int sample rate
    t_data: time_window considered for análysis

    Returns
    -------
    relatived_Data_events : array
    TIME_r: array time vector zero-centered
    
    Notes
    ------
    sample_max: sample with maximum aplitude
    offset: differences between sample maxes based on the max(sample max)
    c: center sample in max

    """
    [A,B] = DATA_mic.shape
    sample_max = np.argmax(DATA_mic,axis=0)
    offset = np.max(sample_max)-sample_max

    #print('event-)'+str(np.argmax(sample_max)))
    
    DATA_mic_relative = []
    secs = t_data
    
    for eve in range (DATA_mic.shape[1]):
        
        fr = np.pad(DATA_mic[:,eve],(offset[eve],),'median')
        c = np.argmax(fr)
        fr = fr[c-Fs*secs:c+Fs*secs]
        
        t_r = np.linspace(0, fr.shape[0], fr.shape[0])-offset[eve]
        DATA_mic_relative.append(fr)
    
    
    DATA_mic_relative = np.array(DATA_mic_relative).T
    TIME_r = np.linspace(-Fs*secs,Fs*secs,2*(Fs*secs))
    
    return DATA_mic_relative, TIME_r

# M0 = DATA_mic[:,0]
# M1 = DATA_mic[:,1]
# offset= max_args[0]-max_args[1]
# fp = np.pad(M1,[offset,250-offset],mode='constant',constant_values=0.)
# plt.figure()
# plt.plot(DATA_mic_relative[0])
# plt.plot(DATA_mic_relative[1])
# plt.plot(DATA_mic_relative)
# plt.show()

# %% Descriptive
# ##########################################################################
def descriptive_time (DATA, axx):
    """
    Returns the array ceneterd on relative time based on the fist event.

    Inputs
    ------
    DATA_acu_events: np.array all event, all times, all channels
    Fs: int sample rate

    Returns
    -------
   
   descriptive: array with descriptive statistics
       descriptive=np.array([mean, std, Ymin, Ymax ])
    
    """
    mean = np.mean(DATA, axis=axx)
    std = np.std(DATA, axis=axx)
    Ymin = np.min(DATA, axis=axx)
    Ymax = np.max(DATA, axis=axx)
    
    descriptive=np.array([mean, std, Ymin, Ymax ]).T
    
    return descriptive

# %% SEL calculation event-pressure-mic
# ##########################################################################
def SEL_calc (DATA_acu_events, Fs, r_m, r_new):
    """
    Returns the array ceneterd on relative time based on the fist event.

    Inputs
    ------
    DATA_acu_events: np.array all event, all times, all channels
    Fs: int sample rate
    
    r_m : HAGL during measurements
    r_new : HAGL corrected 

    Returns
    -------
   
   Lmax_array : LMAX values 
   SEL_k_array : SEL values all events all channels 
   SEL_def_array : SEL values all events all channels by deffinition
   SEL_T : seconds for each SEL
   """
   #*************Correction for distance 10m vs 25m haight Mainly Due to Spherical spreading.
    cf = 20*np.log10(r_new/r_m) #correction factor because of Spherical spreading 
    
    SEL_k_array = np.empty([DATA_acu_events.shape[0], DATA_acu_events.shape[2]])
    SEL_def_array = np.empty([DATA_acu_events.shape[0], DATA_acu_events.shape[2]])
    Lmax_array = np.empty([DATA_acu_events.shape[0], DATA_acu_events.shape[2]])
    SEL_T = np.empty([DATA_acu_events.shape[0], DATA_acu_events.shape[2]])
    
    for eve in range(DATA_acu_events.shape[0]):
        M = np.max(DATA_acu_events[eve,:,:], axis=0)
        for chann in range(M.shape[0]):
            LL      =  DATA_acu_events[eve,:,chann] - cf #levels minus the correction
            ref     = LL >= M[chann]-10
            levels  = LL[ref]# values Lmax - 10dB
            #plt.plot(levels)
            #p       = DATA_raw_events[eve,:,chann][ref]
            ts      = levels.shape[0] #time samples
            te      = ts/Fs
            
            SEL_k     = M[chann]+10*np.log10(te) #aproximation KAPOOR_2021
            SEL_k_array [eve,chann] = SEL_k
            
            SEL_def, ts = AircraftTools.SEL_def_calc (levels, Fs)
            SEL_def_array [eve,chann] = SEL_def
            SEL_T [eve,chann] = ts
            
        Lmax_array [eve,:] = M.T
        
    return Lmax_array, SEL_k_array, SEL_def_array, SEL_T

# %% Round
# ##########################################################################
def round_up(n,jump,decimals=-1):
    """
    Returns the nearest number in jupms of jump.
    jump = 3 : 3 6 9...
    jump = 10: 10 20 30...
    """
    multiplier = jump ** decimals
    return math.ceil(n * multiplier) / multiplier

# %% Segment data based on trheshold
# ##########################################################################
def segment_THRESH (THRESH,Fs, Data_raw_segmented, Data_acu_segmented, microphone):
    cha_r = microphone # mic of refference{1,2,3,4,5,6,7,8,9} 
    segment_by_THRESH_events_acu = []
    segment_by_THRESH_events_raw = []
    
    for eve in range(Data_acu_segmented.shape[0]):
        M   = np.max(Data_acu_segmented[eve,:,cha_r-1])
        LL  =  Data_acu_segmented[eve,:,cha_r-1]#levels
        ref = LL >= M-THRESH
        
        levels  = Data_acu_segmented[eve,ref,:]# USUALLY values Lmax - 10dB
        press = Data_raw_segmented[eve,ref,:]# correspondig pressure points
                
        segment_by_THRESH_events_acu.append(levels)
        segment_by_THRESH_events_raw.append(press)
        
    print("next")
        
    #     Data_raw_segmented_trh = Data_raw_segmented[:,ref,:]
    #     Data_acu_segmented_trh = Data_acu_segmented[:,ref,:]
    
    # samples = Data_acu_segmented_trh.shape[1]
    # TIME_r = np.linspace(-samples//2, samples//2, samples)/Fs
    
    # vec_time = np.linspace(0,samples,samples)
    # return TIME_r, vec_time, Data_raw_segmented_trh, Data_acu_segmented_trh 
    return  segment_by_THRESH_events_raw, segment_by_THRESH_events_acu


# %% Leq calculation event-pressure-mic
# ##########################################################################
def LEQ_calc (DATA_acu_events, Fs):
    """
    
    Inputs
    ------
    DATA_acu_events: np.array all event, all times, all channels
    Fs: int sample rate

    Returns
    -------
   
   Leq : Leq values 

    """
    
    Leq_array = np.empty([DATA_acu_events.shape[0], DATA_acu_events.shape[2]])
    
    
    for eve in range(DATA_acu_events.shape[0]):
        for chann in range(DATA_acu_events.shape[2]):
            LL      =  DATA_acu_events[eve,:,chann]
            Leq     =  10*np.log10(np.sum(10**(LL/10))*(1/LL.size))
            Leq_array [eve,chann] = Leq

    return Leq_array 

# %% Dedoppler effect adapted from Eric Grenwood - Penstate University
# ##########################################################################
def THdedopp (time,pressure,a0,ttrack,Xtrack,Ytrack,Ztrack,Xmic,Ymic,Zmic,dfs,**kwargs):
    
    """
    %THdedopp - time history de-Dopplerization of aircraft noise

    THdedop de-Dopplerizes pressure time history signals from stationary bservers
    for a moving source.

    The correction may optionally be applied to transform the signals to those
    measured by virtual observers traveling on the surface of a hemisphere set a
    fixed distance away from the tracking location, including the amplitude
    correction due to sphereical spreading.All inputs and outputs should be provided
    in any consistant set of dimensional coordinates.

    Parameters
    ----------
    time      NxS matrix of time of reception of ground based microphones
    pres      NxS matrix of acoustic pressures associated with time
    a0        speed of sound
    ttrack    vector of time associated with tracking data
    Xtk       tracking data X coordinate
    Ytk       tracking data Y coordinate
    Ztk       tracking data Z coordinate
    Xm        Size N vector of microphone X locations
    Ym        Size N vector of microphone Y locations
    Zm        Size N vector of microphone Z locations
    dfs       Desired sample rate of de-Dopplerized data
    radius    (optional) radius of virtual observers

    Returns
    -------
    dtime     emission or virtual observer reception time
    dpres     de-Dopplerized pressure or observer corrected pressure
    """
    mics = 1
    radius = kwargs.get('radius', None)
    #Calculate uniformly sampled emission time for assumed source location
    dtime = np.arange(ttrack[0],ttrack[-1],1/dfs)
    dpres = np.zeros((mics,dtime.size));
    
    for i in range(mics):
        #Calculate linear propagation distances from assumed source location
        r2 = interpolate.interp1d(ttrack,np.sqrt((Xtrack-Xmic[i])**2 +\
                                                    (Ytrack-Ymic[i])**2 +\
                                                        (Ztrack-Zmic[i])**2),
                                        kind='cubic',fill_value="extrapolate")(dtime)
        #Transform pressure to emission time vector
        dpres = interpolate.interp1d(time,pressure,kind='cubic',fill_value="extrapolate")(dtime+r2/a0)
        #correct for spherical spreading to virtual observers
        if radius:
            dpres = (r2/radius)*dpres
        if radius:
            #Adjust time of emission to time of reception on sphere surface
            dtime = dtime+radius/a0
        
    return dtime, dpres




