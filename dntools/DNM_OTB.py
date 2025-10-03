# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 16:42:40 2023

@author: SES271
Seperation  TONAL, BROADBAND AND OVERALL

"""
import numpy as np
import scipy.signal as ss
import waveform_analysis.weighting_filters as WA
from scipy.signal import find_peaks

import dntools.EnvironTools as EnvironTools
import dntools.FreqTools as FreqTools

# %% DroneNoise - Noise Decomposition
#### Includes of the atmospheric absorption
#########################################################
# ##########################################################################

def separator_OTB (dist_drone_mics, signal, isAw = True, plotfig = False, Fs=50000, fl=50, fu=10000):
    """
    Returns a list of .

    Inputs
    ------
    signal  : raw data in pascals {array 1D}
    fs      : sample rate {int}
    fl      : lower freuquency for OASPL integration
    fu      : upper freuquency for OASPL integration
    isAw    : is the imput Signal including A-weighted {bolean} defoult TRUE
    
    plotfig : if the plot is requiered
              
    Returns
    -------
    SPL_comp : SPL array {Overall SPL, Broadband SPL, Tonal SPL}   
    """
    N_eve = signal.shape[0]
    components = 3
    N_ch = signal.shape[2]
# %% A_weighting
    AW_cond = '' #predifined
    
    if isAw == False:
        signal_A = np.zeros(signal.shape)
        for eve in range(N_eve):
            for ch in range (N_ch):
                signal_A[eve,:,ch] = WA.A_weight(signal[eve,:,ch], Fs)
        signal = signal_A
        
    # %% PSD calculation
    ######################
    Ndft    = 2**16#2**14
    # Ndft = Data_raw_segmented.shape[1]*2
    df      = Fs/Ndft
    p_ref   = 20e-6
    NOVERLAP = 2**12
    WINDOW = 'hann'
    
    """spectrum same microphone, all events"""
    DATA_PSD_all , freq = FreqTools.calc_PSDs(signal, Fs, Ndft, WINDOW)#, NOVERLAP) ##[p^2/Hz]
    freq = np.array(freq)
    
    N_eve = DATA_PSD_all.shape[0]
    components = 3
    N_ch = DATA_PSD_all.shape[2]
    
    DATA_comp_events = np.empty([N_eve, components, N_ch])

    for eve in range(N_eve):
        for ch in range (N_ch):

            """spectrum dB"""
            DATA_PSD = DATA_PSD_all [eve,:,ch] ##[p^2/Hz]
            
            SPL_MIC  = 10*np.log10((DATA_PSD*df)/(p_ref**2)) ######## in [dB] #####
            #### HEre is the inclussion of the atmospheric absorption
            #########################################################
            ##########################################################
            atenua_ATM_all_freqs = EnvironTools.atmAtten_AAS (dist_drone_mics, freq, Tin=14.5, Psin=29.81, hrin=50)
            atenua_L_ATM_f_mic_eve = np.ones(DATA_PSD.shape)*atenua_ATM_all_freqs[:,ch] ##############[dB atmospheric attenuation]############################
            Lp_with_Attmospheric = SPL_MIC + atenua_L_ATM_f_mic_eve # The correctio due to attmospheric attenuation is added.
            
            # %%  SEPARATION
            # Tonal component extraction
        
            P_dB = Lp_with_Attmospheric         # TOTAL component
            P_dB = P_dB * (P_dB>1)              # Only positive SPL
        
           ######### Shaft frequencnny detection
            P_medcurve_sh = ss.medfilt(P_dB, kernel_size=51) # medfilt initial curve
            fl_min = 50
            fp_mask=(freq>=40) & (freq<=100) #Mask for the frequency range for the fist peak - educated guess

            P_dB_shaft = P_dB*fp_mask #Vector of amplitudes where the shaft freq could be allocated- educated guess
            
            PKSHAFT = np.max(P_dB_shaft)
            peaks, _ = find_peaks(P_dB_shaft, height = PKSHAFT-6)
        
            #########Tonal component detection
            P_medcurve = ss.medfilt(P_dB, kernel_size=401) # width medfilt curve
            # selection of levels in the mask over frequency
            #           level higher than midfilter     level at leats 6 dB obove     level over 0   at frequencies up to 1kHz
            P_Tc_dB_mask = (P_dB >= P_medcurve_sh) & ((P_dB - P_medcurve)>=6) & (P_dB>1) & (freq<=2000)

            freq_mask =  freq>=(freq[peaks[0]]-10) # tonal components obove the shaft frec - 6 dB 
            P_Tc_dB_mask = P_Tc_dB_mask * freq_mask # Mask for tonal component
            P_Tc_dB = P_Tc_dB_mask * P_dB # Selection of tonal omponents from the PSD vector

            #########Broadband component selection
            P_Bbc_dB =  (~P_Tc_dB_mask & (P_dB>1)) * P_dB # the remainderf from tonal component detection and only positive SPL
            
            #########Indepentedn componen integration
        
            sf = ((freq>=fl_min) & (freq<=fu))
            sf_bb_lim = ((freq>=1000) & (freq<=fu))#((freq>=fl_min) & (freq<=fu)) 
            sf_tonal_lim = ((freq>=fl_min) & (freq<1000))#((freq>=fl_min) & (freq<=fu)) 
            
            min_dB_value = 0.1
        
            levels_to_overall = P_dB[sf]
            levels_to_overall =levels_to_overall[levels_to_overall >min_dB_value] #only positive SPL
            spl_ov = 10*np.log10(np.sum(10**(levels_to_overall/10))) #logaritmic sum
        
            levels_to_BB = P_Bbc_dB[sf_bb_lim]# P_Bbc_dB[sf]
            levels_to_BB = levels_to_BB[levels_to_BB>min_dB_value] #only positive SPL
            spl_bb = 10*np.log10(np.sum(10**(levels_to_BB/10))) #logaritmic sum
            
            levels_to_tonal = P_Tc_dB[sf_tonal_lim] #P_Tc_dB[sf]
            levels_to_tonal = levels_to_tonal[levels_to_tonal>min_dB_value]
            spl_tn = 10*np.log10(np.sum(10**(levels_to_tonal/10))) #logaritmic sum
        
            SPL_comp = np.around(np.array([spl_ov, spl_bb, spl_tn]),1)
            # print ('event: '+ str(eve+1) + ' mic: '+str(ch+1))
            # print(str(AW_cond)+'-Weighting')
            # print('overall - broadband - tonal')
            # print(SPL_comp )
            
            DATA_comp_events[eve,:,ch] = np.array([spl_ov, spl_bb, spl_tn])
            
            
            if plotfig == True and ch==4:
                import matplotlib.pyplot as plt
                from cycler import cycler

                # import scienceplots #https://www.reddit.com/r/learnpython/comments/ila9xp/nice_plots_for_scientific_papers_theses_and/
                color_cycler    = cycler(color = ['C0', 'C1', 'C5','C2','C3','C4'])
                color_cyc =len(color_cycler)*color_cycler
                lcc = color_cyc()
                
                # plt.style.use(['science', 'grid'])
                fig, (ax0) = plt.subplots(figsize=(10,5))#,gridspec_kw={'height_ratios':[10,1]})
            
                ax0.plot(freq,P_dB,label='Overall SPL',linewidth=2)
                ax0.plot(freq,P_Tc_dB, '^', markersize=3,label='Tonal comp. SPL')
                ax0.plot(freq,P_Bbc_dB, '.', markersize=2.5, label='Broadband comp. SPL')
                ax0.plot(freq,P_medcurve, label='Median-filtered curve',linewidth=0.8,color='r')
                plt.vlines(freq[peaks[0]] , 0, 70,**next(lcc))
                ax0.set_xscale('log')
                ax0.grid(visible=True, which='minor', color='gray', linestyle='--',linewidth=0.5)
                ax0.legend(loc='upper left')
                ax0.set_xlim([10, 10000])
                ax0.set_ylim([1, max(P_dB)])
                ax0.set_title('Signal at microphone'+ str(ch+1))
                ax0.set_xlabel('Frequency $[Hz]$')
                ax0.set_ylabel('Amplitude [dB'+ AW_cond +'] re $(20 uPa)^2$')
                col_labels = ['Overall [dB'+AW_cond+']','Broadband[dB'+AW_cond+']','Tonal [dB'+AW_cond+']']
                table_vals = [[SPL_comp[0], SPL_comp[1], SPL_comp[2]]]
                my_table = plt.table(cellText=table_vals,
                                     colWidths=[0.13] * 3,
                                     colLabels=col_labels,
                                     cellLoc = 'center',
                                     loc='upper right')
                my_table.auto_set_font_size(False)
                my_table.set_fontsize(10)
                plt.tight_layout()
    
    return DATA_comp_events