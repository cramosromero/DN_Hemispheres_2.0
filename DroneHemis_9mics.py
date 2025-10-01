"""
Created on Fri Jan 27 10:55:32 2023
@author: SES271@CR

This script shows the Directivity semiespheres of a Drone flying-by transversaly to
a Ground Plate Microphones line (9 microphones).
References of microphone configuration: "NASA-UNWG/SG2  and ISO-5305:2024
UAM Ground & Flight Test Measurement Protocol""
"""
# %%Libraries"""
import time

start_time = time.time()
import math
import os
import pkgutil

# List all importable modules in the current environment


from dntools import AircraftTools as AircraftTools
from dntools import FileTools as FileTools
from dntools import FreqTools as FreqTools
from dntools import TimeTools as TimeTools
from dntools import Plots_DN_9mics as plots
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import scipy.signal as ss
from cycler import cycler
from matplotlib import colors, rc
from matplotlib.cbook import get_sample_data
from matplotlib.image import imread
from matplotlib.offsetbox import AnnotationBbox, OffsetImage


# Force browser rendering for .py files
pio.renderers.default = 'browser'# "notebook"#

line_cycler     = cycler(linestyle = ['-', '--', '-.', (0, (3, 1, 1, 1)), ':'])
color_cycler    = cycler(color = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5'])
line_color_cycler = (len(line_cycler)*color_cycler
                     + len(color_cycler)*line_cycler)
lc = line_color_cycler()

plt.close('all')

# %%Constants
###########################################
a0              = 331.5 #[m/s]speed of sound
Data_folder     = "Data" # Folder containing the .mat files
workspace_root  = os.getcwd()
n_mics          = 9 #number of microphones
deg_th          = np.array([-90,-75,-60,-45,-30,-15,0,15,30,45,60,75,90]) # theta angles of the microphones
to_bands        =[20, 25, 31, 40, 50, 63, 80, 100, 125, 160,
                  200, 250, 315, 400, 500, 630, 800, 1_000, 1_250, 1_600, 
                  2_000, 2_500, 3_150, 4_000, 5_000, 6_300, 8_000, 10_000, 12_500, 16_000, 20_000] #central 1/3 octave bands

#%% LOADING Data from Mat files.
##########################################
# Measurement name metadata in the filename:
Cases    = [
    # ['EE','T1',25,'F05','N','S','uw', 2],
    ['Ed','M3',10,'F05','Y','W','dw', 2]
    ] # add more cases if needed [starting, wind, payload, droneID, recording]
case = 0 # choose the case to process

PILOT   = Cases[case][0] # pilot ID
DID     = Cases[case][1] # Drone ID
HAGL    = Cases[case][2] # Height Above Ground Level (m)
OPE     = Cases[case][3] # Operation condition 'F15' #{F15, F05, F27}
PYL     = Cases[case][4] # payload (Y/N)
STA     = Cases[case][5] # staring point
D       = '??????' #{hhmmss} Date
WIND    = Cases[case][6] # upwind-down wind
event   = Cases[case][7] # #number of event

identifier, files = FileTools.list_of_files_VS(workspace_root, "Data", PILOT, DID, HAGL, OPE, PYL, STA, D, WIND)
print(files)
"""Output data for saving the results. """
results_folder = os.path.join("Data_out",f"{identifier}_out") #for results OUTPUTS
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# %%Accesing the raw data from .mat files
##########################################

microphone  = 5 # {1,2,3,4,5,6,7,8,9... nMic}
acu_metric  = 'LAFp' # LAFp, LZfp
"""ACCCES DATA_ACU SINGLE EVENT"""
Fs, TT, DATA_raw, DATA_acu = FileTools.data_single_events (files[event-1], n_mics, acu_metric)
"""ACCCES DATA_ACU ALL EVENTS"""
Fs, TT, DATA_raw_events, DATA_acu_events = FileTools.data_all_events (files, DATA_raw, n_mics, acu_metric)

"""PLOTS all microphones, single events"""
fig = plots.plot_mics_time(DATA_raw_events, TT, event, DID, lc)
# fig.savefig(f"{results_folder}\\ev{event}_allmics_raw.svg", format="svg", dpi=300)
"""SIGNAL SEGMENTATION based on LA_max value of the medianfilter signal"""
# This segment could be analized with FFT to obtain the PSD, then the band content
time_slice  = 5 #in seconds
TIME_r, vec_time, Data_raw_segmented, Data_acu_segmented = TimeTools.segment_time_ADJUST(time_slice, Fs,
                                                                                              DATA_raw_events, DATA_acu_events)
"""PLOTS DATA_mic same microphone, all events"""
fig = plots.plot_mic_events_time(TIME_r, Data_acu_segmented, event, microphone, DID, acu_metric, lc)
# fig.savefig(f"{results_folder}\\mic{microphone}_alleve.svg", format="svg", dpi=300)
# %% Dedoppler effect correction
##########################################
"""Geometry for Dedoppler correction"""
DATA_acu_events_dedopp = []
for i_eve in range(DATA_acu_events.shape[0]):
    S_max = np.argmax(DATA_acu_events[event-1,:,4]) # sample where is located tha maximumm SPL. Central mic as refference.
    smp_order_vect = np.linspace(0, TT.shape[0],TT.shape[0])-S_max  # sample order vector
    dx = (smp_order_vect / Fs) * int(OPE[1::]) # distance vector along the flight path
    dz = HAGL # height AGL constan altitude in Z

    dyA = [] # prelocating lateral distance from the center microphone      
    for d in range((n_mics//2)+1): dyA.append( math.tan( d*15  * np.pi / 180)*dz) #from EnvironTools.drone_mic_geometry
    dyB = list(reversed(dyA[1:])) #mirroring the distances from centre mic
    dy = dyB + dyA  # both sides dy from central microhpone

    Xtrack = dx                     # X distance from the microphone line
    Ytrack = np.zeros(dx.size)*0    # lateral distance from the microphone line
    Ztrack = HAGL                   # height AGL constan altitude in Z

    Xmic = np.array([0])            # X distance of the receiver, in this cases is the sam e for all mics   
    Ymic = np.array([0])            # lateral distance of the receiver, in this cases is the sam Ytrack for all mics
    Zmic = np.array([0])            # height of the receiver, in this cases is the sam Ztrack for all mics

    Dedopp_channels = [] # prelocating the dedoppler corrected channels
    Dedopp_time = []     # prelocating the dedoppler corrected time vector
    for mic in range(n_mics):
        pressure = DATA_raw_events[event-1,:,mic-1]
        tt_dedop , Data_raw_mic_dedopp = TimeTools.THdedopp(TT,pressure,a0,TT,Xtrack,Ytrack,Ztrack,Xmic,Ymic,Zmic,Fs)
        Dedopp_channels.append(Data_raw_mic_dedopp[:-int(Fs/2)]) #removing the last 0.5 seconds in samples with NaN values
        Dedopp_time.append(tt_dedop)

    AAA = np.array(Dedopp_channels).T #put dedopp arrays in the same format as DATA_raw_events
    #DATA_raw_mic_dedopp = AAA[np.newaxis, ...] # adding the event dimension
    
    DATA_acu_events_dedopp.append(AAA)  #saving the dedoppler corrected event

DATA_acu_events_dedopp = np.array(DATA_acu_events_dedopp)   #rewriting the array of data with the dedoppler corrected data

_Dedopp =True
if _Dedopp==True:
    DATA_raw_events = DATA_acu_events_dedopp
   
# %%ASpetrogrmas for an specific  microphone and event.
##########################################################
#eve_mic = [event, 5]#{1,2,3,4,5,6,7,8,9} # event and microhphone to plot the spectrogram
ti=2*Fs
tf=25*Fs
press_time_serie = DATA_raw_events[event-1,ti:tf,microphone-1] # acoustic pressure time series
"""PLOTS spectrogram for a given microphone and event"""
fig = plots.plot_spectrogram(press_time_serie, Fs, DID, [event, microphone])
# fig.savefig(f"{results_folder}\\eve{event}mic{microphone}_spect.png", format="png", dpi=300)

# %% SIGNAL SEGMENTATION for Hemisphere calculations and Time fitting based on Lmax threshold
N_plots_dir = {} # For saving the calculated directivity in a Dictionary (partial hemisphere)
SWL_band = {} #For saving the Integrated Sound Power Level in a Dictionary
N_plots_dir_HEMIS = {} # For saving the calculated directivity in a dictionary (Half sphere)
""" Based on time signals lower than the Lmax"""
THRESH = 20 #in dB
segment_by_THRESH_events_raw, segment_by_THRESH_events_acu  = TimeTools.segment_THRESH(THRESH,Fs,Data_raw_segmented, Data_acu_segmented,microphone)

lev_vec = segment_by_THRESH_events_acu[event-1] # size(data,mics) Sund Levels
raw_vec = segment_by_THRESH_events_raw[event-1] # size(data,mics) Passcals

raw_vec = np.reshape(raw_vec,(1,raw_vec.shape[0],raw_vec.shape[1])) #suitable for the SPL by ban definition
t_chunk = lev_vec.shape[0]/Fs
t_vec_segment = np.linspace(0, lev_vec[:,4].shape[0],lev_vec[:,4].shape[0])/Fs
print("seconds of selected chunk:"+str(t_chunk))

"""PLOTS level time series for a given microphone and event that exceeds the THRESHold"""
# fig = plots.plot_level_Max_thresh(lev_vec, t_vec_segment, DID, acu_metric, lc, event)
# %% Back-propagation of the segmented signal based on Lmax - threshold
############################################

# find the sample with the(Lmax) at central microphone
max_Lmax_sam = np.argmax(lev_vec[:,4]) # central microphone
nspls = lev_vec.shape[0] # number of samples in Section
n_ch = 15 #number of chunks eary if it is an odd number
jump = nspls//n_ch

""" MAP of segments in the fligh path"""
si = [0]
sf = [jump-1]
sm = [round(sf[0]/2)+1]

for nw in range(n_ch-1):
    si.append(sf[nw]+1)
    sf.append(sf[nw]+jump)
    sm.append(sm[nw]+jump)
""" Vector dx"""
"estimation based on the central sample"
fly_speed = float(OPE[1:3]) #m/s
dx = []
for dx_i in range(len(sm)):
    dx.append(fly_speed*(abs(max_Lmax_sam-sm[dx_i])/Fs)) ##norm

levs_by_band_chunck = [] #saving the propagated band levels on each chunk

"""SLP of each chuck of data"""
for win_ord in range(len(sm)):
    nsi = si[win_ord]       #   initial sample
    nsf = sf[win_ord]       #   final sample   
    nsc = sm[win_ord]       #   central sample
    """ 1/3 octave band"""
    fraction = 3
    DATA_SPLoverall, DATA_SPLs, freq = FreqTools.calc_SPLs(raw_vec[:,nsi:nsf,:],
                                                            Fs, fraction, limits=[20, 20_000], show=0)
    levs_by_band_chunck.append(DATA_SPLs)
    
""" SPL Back-PROPAGATION """
L_by_band_chunk_BP = [] #saving the propagated band levels on each chunk BAck propagated
DI_PHI = []
rad_to_deprop = 1 #[m]
heightAGL = HAGL #[m]

for win_ord in range(len(levs_by_band_chunck)):
    L_array_all_freqs = levs_by_band_chunck[win_ord]
    L_array_all_freqs_BP = np.ones((1,len(freq), n_mics))*10
    for n_band in range(len(freq)):
        band = freq[n_band]
        L_array = L_array_all_freqs[0,n_band,:]
        L_array_to_BP = np.reshape(L_array,(1,1,L_array.shape[0])) #array suitable for backpropagation
        L_array_depropagated, thetas, dist_ground_mics, phi_dx = AircraftTools.depropagator(L_array_to_BP,heightAGL, rad_to_deprop, band, dx[win_ord])
        L_array_all_freqs_BP[0,n_band,:]=L_array_depropagated
    DI_PHI.append(phi_dx)
    L_by_band_chunk_BP.append(L_array_all_freqs_BP)

#%%  BAND's SPL or OASPL from bands contributions
############################################################
L_by_OASPL_chunk_BP = []
"""OASPL from the contibution of a range of 1/3 octave bands"""
bi = to_bands.index(20) # lower band
bf = to_bands.index(20_000) # upper band

for n_ch in range(len(L_by_band_chunk_BP)):
    L_by_OASPL_chunk_BP.append(10*np.log10(np.sum(10**((L_by_band_chunk_BP[n_ch][0,bi:bf+1,:])/10),axis=0)))

if bi==bf:
    label = f"SPL at band {to_bands[bi]} Hz"
    label_file = f"band {to_bands[bi]} Hz"
else:
    label = f"SPL at bands from {to_bands[bi]} Hz to {to_bands[bf]} Hz"
    label_file = f"bands {to_bands[bi]}-{to_bands[bf]} Hz"

#%% PLOTING and calculating SWL by 1/3 octave band and OASPL
############################################################

"""Output data for saving the Directivity plots. """
polar_plots_folder = os.path.join("Data_out",f"{identifier}_out","DirPlot") #for results OUTPUTS
if not os.path.exists(polar_plots_folder):
    os.makedirs(polar_plots_folder)
"""to_bands    =[20, 25, 31, 40, 50, 63, 80, 100, 125, 160,
                200, 250, 315, 400, 500, 630, 800, 1_000, 1_250, 1_600, 
                2_000, 2_500, 3_150, 4_000, 5_000, 6_300, 8_000, 10_000, 12_500, 16_000,
                20_000, "Overall"]"""
###########


n_plots = [20, 25, 31, 40, 50, 63, 80, 100, 125, 160,
                200, 250, 315, 400, 500, 630, 800, 1_000, 1_250, 1_600, 
                2_000, 2_500, 3_150, 4_000, 5_000, 6_300, 8_000, 10_000, 12_500, 16_000,
                20_000, "Overall"] # for being ploted and saved
# Here FOR for N plots
for i_plot in n_plots:
    
    LEVELS_th_ph_plot = []
    
    if isinstance(i_plot, (int, float)):
        bi = bf = to_bands.index(i_plot) # index band
        for n_ch in range(len(L_by_band_chunk_BP)):
            #it is summ if required an integration across bands.
            LEVELS_th_ph_plot.append(10*np.log10(np.sum(10**((L_by_band_chunk_BP[n_ch][0,bi:bf+1,:])/10),axis=0))) 
    
    elif isinstance(i_plot, (str)):
        bi =  to_bands.index(20)
        bf = to_bands.index(20_000)
        for n_ch in range(len(L_by_band_chunk_BP)):
            #it is summ if required an integration across bands.
                LEVELS_th_ph_plot.append(10*np.log10(np.sum(10**((L_by_band_chunk_BP[n_ch][0,bi:bf+1,:])/10),axis=0)))
    
    LEVELS_th_ph = np.array(LEVELS_th_ph_plot)
    # limits for colorplots  
    plt_Level_min = np.min(LEVELS_th_ph)
    plt_Level_max = np.max(LEVELS_th_ph)
    
    """saving the spl in th edictionary of directivites"""
    N_plots_dir[f'result_{i_plot}'] = LEVELS_th_ph 
        
    """PLOTS PARTIAL-Hemispheres, all events - all mics"""
    ## Coordinates
    ## Azimut angle 
    th_min = 0
    th_max = 180
    ## Polar angle
    ph_min = -90
    ph_max = 90

    j_15 = 1 + n_mics+ (180-(n_mics*15))//15 # Mics angle, jump+ 15 deg.
    th   = np.linspace(th_min*np.pi/180, th_max*np.pi/180, j_15 ) 

    all_phi_deg = 90-(np.array(DI_PHI)*180/np.pi) # Path angle. coordinathes acording with NORAH
    ph_neg = all_phi_deg.argmin()
    ph = np.concatenate([all_phi_deg[0:ph_neg]*-1, all_phi_deg[ph_neg:]]) 
    ph = ph*np.pi/180
    th, ph = np.meshgrid(th,ph)
    
    """Plot Unwrapped directivity from the partial hemisphere"""
    fig = plots.unwrapped_directivity (th, ph, LEVELS_th_ph, plt_Level_min, WIND, i_plot)
    """save the Directivity plots of the Unwrapped partial hemispheres"""
    fig.savefig(f"{polar_plots_folder}\\{i_plot}_{event}_part.svg", format="svg", dpi=300)
    
    #NOW the full HEmisphere likewise HORAH
    ##############################################
    """FILLING THE FULL-Hemispheres, all events - all mics
    adn Allocating in a big matrice for hemisphere and SWL calculation"""
    phi_max = all_phi_deg[0]
    phi_min = -1* all_phi_deg[-1]
    the_max = thetas[-1]
    the_min = thetas[0]
    add_row_up =  round((90-abs(phi_max))/15)
    add_row_down =  round((90-abs(phi_min))/15)
    add_column_left = round((90-abs(the_min))/15)
    add_column_rigth = round((90-abs(the_max))/15)
    #prellocating
    Hemisphere = np.zeros( (add_row_up + LEVELS_th_ph.shape[0] + add_row_down ,
                            add_column_left + LEVELS_th_ph.shape[1] +add_column_rigth))
    
    start_row, start_col = add_row_up, add_column_left
    #Allocating the calculated partial-hemisfhere in the south hemisphere
    Hemisphere[start_row:start_row+LEVELS_th_ph.shape[0], start_col:start_col+LEVELS_th_ph.shape[1]] = LEVELS_th_ph
    
    #Theta extrapolation (VERTICAL ANGLE) estimated for quadcopters
    for the_add in range(add_column_left):
        the_to_fill = the_max + 15* (the_add+1) #angle to fill
        #extrapolation based on the quadcopter directivity in the vertical plane
        G = -0.0011*((the_to_fill)**2) + 0.194 *abs(the_to_fill) - 4.9 # Eq. (1) rom K. Heutschi el at.
        add_coll = abs(the_add-add_column_left+1)#index of the column for adding data
        Hemisphere[add_row_up:LEVELS_th_ph.shape[0]+add_row_up, add_coll] = Hemisphere[add_row_up:LEVELS_th_ph.shape[0]+add_row_up, 
                                                                                add_column_left-the_add] - G
        #symetric behaivour
        Hemisphere[add_row_up:LEVELS_th_ph.shape[0]+add_row_up, LEVELS_th_ph.shape[1]+the_add+2] = Hemisphere[add_row_up:LEVELS_th_ph.shape[0]+add_row_up,
                                                                                                    add_coll]
    
    #PHI extrapolation (HORIZONTAL ANGLE) estimated for argmin NOHRA constant extrapolation
    phi_up_hemisphere = np.tile(Hemisphere[add_row_up,:], (add_row_up,1))
    phi_down_hemisphere = np.tile(Hemisphere[add_row_up + LEVELS_th_ph.shape[0]-1,:], (add_row_down,1))
    
    Hemisphere[0:add_row_up, :] = phi_up_hemisphere
    Hemisphere[add_row_up + LEVELS_th_ph.shape[0]::, :] = phi_down_hemisphere
    Hemisphere = np.round(Hemisphere,1)
    ## Hemisphere #Saving the full hemisphere by BNADS and OASPL
    ############################################################
    N_plots_dir_HEMIS[f'band_{i_plot}'] = Hemisphere 
    ############################################################
    
    """"Full - Hemispheres PLOTS MATPLOTLIB"""
    #Hemispheres MATPLOTLIB
    LEVELS_th_ph = np.copy(Hemisphere)

    # limits for colorplots  
    plt_Level_min = np.min(LEVELS_th_ph)
    plt_Level_max = np.max(LEVELS_th_ph)
    
    th_min = 0
    th_max = 180
    ## Polar angle
    ph_min = -90
    ph_max = 90

    j_15 = 1 + n_mics+ (180-(n_mics*15))//15 # Mics angle, jump+ 15 deg.
    th   = np.linspace(th_min, th_max, j_15 ) 
    th = th*np.pi/180
    
    all_phi_deg_hm = np.concatenate([ np.flip(np.linspace(all_phi_deg[0],90,add_row_up+1)[1:]) ,
                                        all_phi_deg,
                                        np.linspace(all_phi_deg[-1],90,add_row_down+1)[1:] ])#new phi angles for the hemosphere
    
    ph_neg = all_phi_deg_hm.argmin()
    ph = np.concatenate([all_phi_deg_hm[0:ph_neg]*-1, all_phi_deg_hm[ph_neg:]]) 
    ph = ph*np.pi/180
    
    """ Plot Hemisphere 3D - MATPLOTLIB"""
    print(i_plot)
    fig = plots.plot_3D(th, ph, LEVELS_th_ph, i_plot, DID, HAGL, dist_ground_mics, WIND)
    fig.savefig(f"{polar_plots_folder}\\{i_plot}_{event}_hSph.png", format="png", dpi=300)
    
    
    """SWL integration"""
    ### Integrating  by equation in the refference and save by band
    Hemisphere_p2 = (10**(Hemisphere*0.1))*(20e-6)**2 #SPL to p^2
    #Average across the surface http://www.sandv.com/downloads/1407barn.pdf
    p2_aver = np.average(Hemisphere_p2)#sound power
    spl_average = 10*np.log10(p2_aver/(20e-6)**2)
    Surf_hem = (4*np.pi  * rad_to_deprop**2) * 0.5 # Surface of the hemisphere
    SWL = spl_average + 10*np.log10(Surf_hem/1) # Sound Power Level
    SWL_band[f'band_{i_plot}'] = round(SWL,1) #Saving the SWL by band
        
    # Generate theta and phi angles
    """PLOTS SEMIESPHERES, all events - all mics"""
    fig = plots.plot_int3D(th, ph, rad_to_deprop, LEVELS_th_ph, i_plot, DID, WIND)
    fig.write_html(f"{polar_plots_folder}\\{i_plot}_{event}_Sph.html") 
# %% Save dictionary of Noise Directivities and SWL by band and overall sound power level
#########################################################################################
# Output data for saving the results. 
with pd.ExcelWriter(f"{results_folder}\\Hem_{identifier}_{event}.xlsx", engine="xlsxwriter") as writer:
    for sheet_name, content in N_plots_dir_HEMIS.items():
        # Convert content to DataFrame if it isn't already
        if isinstance(content, pd.DataFrame):
            df = content
        else:
            df = pd.DataFrame(content)
        # Write to sheet
        df = pd.DataFrame(content, columns=deg_th[:len(content[0])]) #first raw with THETA angles
        df.insert(0, "Phi", np.round(ph*180/np.pi, 1)) #first columns with PHI angles
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    # Save the SWL results in a separate sheet
    swl_df = pd.DataFrame(list(SWL_band.items()), columns=['Band', 'SWL (dB)'])
    swl_df.to_excel(writer, sheet_name='SWL_Bands', index=False)

print(dist_ground_mics)
print("Process finished --- %s seconds ---" % (time.time() - start_time))      