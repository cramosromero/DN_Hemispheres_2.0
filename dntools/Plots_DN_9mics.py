""" Plots functions for DroneNoise  Hemispheres analysis with 9 microphones"""

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import scipy.signal as ss
from matplotlib import colors


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def plot_mics_time(DATA_raw_events, TT, event, DID, lc):
    """Plot of time series raw data from all microhpnes for a given event

    Args:
        DATA_raw_events: Array of raw data all microphones [events, time, mics]
        TT:Time vector [s]
        DID: Drone ID
        lc: line color iterator

    Returns:
        fig: figure handle
    """
    
    fig, (ax) = plt.subplots()
    for mic_i in range(DATA_raw_events.shape[2]):
        ax.plot(TT, DATA_raw_events[event-1,:,mic_i], label=f'Mic {mic_i+1}',**next(lc))
    ax.set_title(f"{DID} Event {event}")
    ax.legend(loc="best", ncol=3, fontsize="small")
    ax.get_legend().set_title("Microphone")
    ax.set_ylabel('Amplitude [Pa]')
    ax.set_xlabel('Time [s]')
    # plt.show()
    
    return fig

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def plot_mic_events_time(TIME_r, Data_acu_segmented, event, microphone, DID, acu_metric, lc):
    """Plot of time series data from a given microphone for all events

    Args:
        TIME_r: Time vector [s]
        Data_acu_segmented: Array of raw data all microphones [events, time, mics]
        microphone: microphone index {1,2,3,4,5,6,7,8,9... nMic}
        DID: Drone ID
        acu_metric: Acumulation metric 'LAFp' or 'LZfp'
        lc: line color iterator
        
    Returns:
        fig: figure handle
    """
    fig, (ax) = plt.subplots()
    for eve in range(Data_acu_segmented.shape[0]):
        ax.plot(TIME_r, Data_acu_segmented[eve,:,microphone-1],**next(lc), label=f"Event {eve+1}")
    ax.set_title(f"{DID}, Mic{microphone}")    
    ax.legend(loc='upper right', borderpad=0.1,frameon=True)
    ax.get_legend().set_title("Events")
    ax.set_ylabel(f"{acu_metric} [dB]")
    ax.set_xlabel('Time [s]')
    # plt.show()
    return fig
 

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def plot_spectrogram(press_time_serie, Fs, DID, eve_mic):
    """Plot of the spectrogram for a given microphone and event

    Args:
        press_time_serie: acoustic pressure time series [Pa]
        Fs: Sampling frequency [Hz]
        DID: Drone ID
        eve_mic: list with event and microphone index [event, mic]  
    
    Returns:
        fig: figure handle
    """
    nperseg = 10990
    nfft = 2**18
    noverlap = int(nperseg//1.2)
    f, t, Sxx = ss.spectrogram(press_time_serie, nperseg=(2**16)//6, fs=Fs, nfft=2**18, noverlap=noverlap, scaling='density') # spectrogram

    fig, (ax) = plt.subplots(figsize=(6,3))
    pcm = ax.pcolormesh(t, f, 10*np.log10(Sxx/(20e-6)**2), vmin=5, vmax=50, cmap='viridis', shading='auto') # vmin=10,vmax=60 
    ax.set_title(f"event_{eve_mic[0]} mic_{eve_mic[1]} {DID}")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]") 
    y_lim =[30,800] #frequency limits for plotting 
    ax.set_ylim(y_lim[0], y_lim[1])  
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label("SPL [dB/Hz]")
    plt.tight_layout()
    # plt.show()
    
    return fig

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def plot_level_Max_thresh(lev_vec, t_vec_segment, DID, acu_metric, lc, event):
    """Plot of the time series for a given microphone and event

    Args:
        lev_vec: Array of sound levels for all microphones [level, time, mics]
        t_vec_segment: Time vector [s]
        DID: Drone ID
        event: event index {1,2,3,4,5... nEvents}
        
    Returns:
        fig: figure handle
    """
    fig, ax = plt.subplots()
    for mic_i in range(lev_vec.shape[1]):
        ax.plot(t_vec_segment, lev_vec[:,mic_i], **next(lc), label=f"Mic {mic_i+1}")
    ax.set_title(f"{DID}, Mic{mic_i+1}, Event {event}")    
    ax.legend(loc='upper right', borderpad=0.1,frameon=True)
    ax.get_legend().set_title("Events")
    ax.set_ylabel(f"{acu_metric} [dB]")
    ax.set_xlabel('Time [s]')
    plt.tight_layout()
    # plt.show()
    
    return fig

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def unwrapped_directivity (th, ph, LEVELS_th_ph, plt_Level_min, WIND, i_plot):
    """
    Definition of Unwrapped directivity plot
    Unwrapped directivity plot of the SPL directivity pattern

    Args:
        th : thetha angles in horizontal angle theta [-60 to 60]
        ph : path angles in azimuth angle phi based on the path definition
        LEVELS_th_ph: amplitude in dB at each (th,ph) point by frequency band [phi, theta]
        plt_Level_min: minimum level for plotting [dB]
        WIND : wind direction 'uw' or 'dw'
        i_plot : index of plot by freuency band or overall spl

    Returns:
        fig: figure handle
    """

    # PRELOCATION of the LEVELS on the SEMIESPHERE
    DL_theta_phi = np.ones(th.shape)*(plt_Level_min-6) # It is just for prelocationg low residual noise
    # LEVELS LOCATION on the SEMIESPHERE
    for i_phi in range(LEVELS_th_ph.shape[0]):
        DL_theta_phi[i_phi,2:11] = LEVELS_th_ph[i_phi,:]

    mask = np.ones(th.shape) #zones without information
    masked = np.ma.masked_where(DL_theta_phi>plt_Level_min-1, mask)#zones without information

    ## FROM CONTROL ROOM POINT OF VIEW
    strength = np.flip(np.flip(DL_theta_phi,axis=0),axis=1) #For visualize the amplitude as a strength-wise amplitude color 
    # strength = np.copy(DL_theta_phi)
    ## "unwrapped"-directivity
    #th_ticks_label = ['','','1','2','3','4','5','6','7','8','9','','']
    deg_th = np.array([-90,-75,-60,-45,-30,-15,0,15,30,45,60,75,90])
    deg_ph = -1*ph[:,0]*180/np.pi

    x = deg_th
    y = deg_ph

    X,Y = np.meshgrid(x,y)
    Z = DL_theta_phi
    
    if WIND =='dw' : #flip the aray if the drone flyes up-wind (for the order in the Sphere)
        LEVELS_th_ph = np.flip(LEVELS_th_ph, 1)
        #th_ticks_label = ['','','9','8','7','6','5','4','3','4','1','','']

    cmap = plt.colormaps["viridis"]
    norm = colors.Normalize(vmin = np.min(strength), vmax = np.max(strength), clip = False)
    
    fig,ax = plt.subplots(figsize=(5,4))

    surf3 = ax.contourf(X, Y, Z, 20, origin='lower',cmap=cmap, alpha=1)
    surf = ax.contour(X, Y, Z, 20, colors='black', linewidths=0.5, linestyles='dashed')
    surf2 = ax.clabel(surf, inline=True, fontsize=9, fmt='%1.1f')

    ax.set_title(f"Band {i_plot}", fontsize=12)
    ax.set_xlabel('$\Theta^{\circ}$', fontsize=12)
    ax.set_ylabel('$\phi^{\circ}$', fontsize=13)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    plt.tight_layout()
    ax.grid(False)
    
    return fig
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def plot_3D(th, ph, LEVELS_th_ph, i_plot, DID, HAGL, dist_ground_mics, WIND):
    """Definition of 3D Hemisphere plot
    3D Hemisphere plot of the SPL directivity pattern
    Args:
        th : mic resolution angles in horizontal angle theta [-90 to 90]
        ph=: path resolution angle in azimuth angle phi [-90 to 90]
        LEVELS_th_ph: amplitude in dB at each (th,ph) point by frequency band
        i_plot: index of plot by freuency band or overall spl   
        DID: Drone ID
        HAGL: Height above ground level [m]
        dist_ground_mics: distance between ground mics [m]
        WIND: wind direction 'uw' or 'dw'

    Returns:
        fig: figure handle
    """
    
    th_all, ph_all = np.meshgrid(th,ph)
    x = HAGL * np.cos(th_all)
    y = -HAGL * np.sin(th_all) * np.sin(ph_all)
    z = HAGL - (HAGL *np.sin(th_all) * np.cos(ph_all))

    strength = np.copy(LEVELS_th_ph)  #For visualize the amplitude as a strength-wise amplitude color

    # if WIND =='dw' : #flip the aray if the drone flyes up-wind (for the order in the Sphere)
    #     LEVELS_th_ph = np.flip(LEVELS_th_ph, 1)

    cmap = plt.colormaps["viridis"]  #plt.colormaps["Greys"]#plt.colormaps["viridis"] 
    norm = colors.Normalize(vmin = np.min(strength), vmax = np.max(strength), clip = False)
    # Figure
    fig = plt.figure(figsize=(8,8))
    plt.figaspect(1)
    ax = fig.add_subplot(111, projection='3d')
    
    ## axis directivity
    ec = ax.plot_surface(x, y, z, edgecolor='none',facecolors = cmap(norm(strength)),
                            cmap=cmap, antialiased=False)
    ax.set_box_aspect([1,1,1])
    #Mic-line as a reference
    for nm in range(1, len(dist_ground_mics), 1):
        xcor = dist_ground_mics[nm]*HAGL / dist_ground_mics[0]
        if nm > len(dist_ground_mics)//2:
            xcor= xcor*-1
            
        ax.plot([xcor],[0],[-HAGL],marker='.', color='k')
    if WIND =='dw' :
        ax.plot([dist_ground_mics[0]*HAGL / dist_ground_mics[0]],[0],[-HAGL],marker='P', color='r')
    else:
        ax.plot([-1*dist_ground_mics[0]*HAGL / dist_ground_mics[0]],[0],[-HAGL],marker='P', color='r')
        
    #Figure axis
    ax.set_axis_off()
    
    ax.set_title(f"SPL Hemispheres Band {i_plot} \n {DID}", fontsize=12)
    ax.view_init(28,137) # i like this view
    ax.view_init(28,45) # i like this view
    plt.tight_layout()
    
    return fig

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def plot_int3D(th, ph, rad_to_deprop, LEVELS_th_ph, i_plot, DID, WIND):
    
    th_mx, ph_mx = np.meshgrid(th, ph)
    
    # NORAH coordinates
    # rad_to_deprop = 10
    x = rad_to_deprop * np.cos(th_mx)
    y = -rad_to_deprop * np.sin(th_mx) * np.sin(ph_mx)
    z = rad_to_deprop - (rad_to_deprop * np.sin(th_mx) * np.cos(ph_mx))
    
    strength = np.flip(LEVELS_th_ph, axis=0)
    # strength = np.flip(np.flip(LEVELS_th_ph, axis=0), axis=1)
    
    if WIND == 'dw':
        strength = np.flip(strength, axis=1)
    
    # Plotly figure
    fig = go.Figure()
    
    # Add 3D surface
    fig.add_trace(go.Surface(x=x, y=y, z=z, surfacecolor=strength,
                                colorscale='Viridis', colorbar=dict(title='SPL [dB]'),
                                opacity=0.9, cmin=np.min(strength), cmax=np.max(strength)))
    # Add arrow from West (-X) to East (+X)
    fig.add_trace(go.Cone(
        x=[0],  # starting at the center
        y=[0],
        z=[1],
        u=[0],  # arrow direction (x=+1 → East)
        v=[1],  # arrow direction (y=+1 → North)
        w=[0],
        sizemode="absolute",
        sizeref=0.1,  # adjust size
        anchor="tail",  # tail at (0,0,0)
        colorscale=[[0, "red"], [1, "black"]],  # solid red arrow
        showscale=False
    ))
    fig.update_layout(scene=dict(
                        xaxis_title='X (N-S)',
                        yaxis_title='Y (W-E)',
                        zaxis_title='Z (Up)',
                        aspectmode='data'
                        ),
                        title=f"SPL Hemispheres {DID} {i_plot}",
                        margin=dict(l=0, r=0, t=50, b=0))
    
    #fig.show()
    return fig
