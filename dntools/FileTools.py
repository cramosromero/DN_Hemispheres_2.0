# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 17:19:58 2022

@author: SES271@CR
"""
import numpy as np
import glob # STRINGS
import os
from scipy import io #READ MAT FILES


# %% DroneNoise Measurement filenames
# ##########################################################################
def list_of_files_VS(workspace_root,Data_folder, PILOT, DID, HAGL, OPE, PYL, STA, D, WIND):

    subfolder = f"{PILOT}_{DID}_{HAGL}_{OPE}_{PYL}_{STA}_{D}_{WIND}"
    root_folder = f"{PILOT}_{DID}_{HAGL}_{OPE}_{PYL}_{STA}_{D}_"
    # List all files in that folder
    data_subfolder = os.path.join(workspace_root, Data_folder, subfolder, root_folder)

    # Find matching folders
    files = glob.glob(data_subfolder+"?"+".mat") # ? file in order numbered
    print(files)
    
    identifier = f"{PILOT}_{DID}_{HAGL}_{OPE}_{PYL}_{STA}_{WIND}"
    # Convert to relative paths (relative to workspace root)
    return identifier, files

def list_files (PILOT,DroneID,HAGL,OPE,PYL,STA,DATE, COUNT):
    """
    Returns a list of files for Drones' measurements .mat for a nominal identifiers.

    Inputs
    ------
    PILOT : {'Ed','others'}
        Ed = Edimbrough Drone Company
    
    DroneID: {'M3','Fp', '3p', 'Yn'}
        M3 = DJI Matrice 300
        Fp = DJI FPV
        3p = DJI Mini 3 Pro
        Yn = Yunnec Typhoon
    
    HAGL: {10} Drone Height above ground level
        10 = meters
        
    OPE: {'F05','F15','H00','THL'}
        F05 = Flyover 5m/s
        F15 = Flyover 15m/s
        H00 = Hovering
        THL = Take-over/Hovel/Landing
        
   PYL: {'Y','N'}
        Y = Yes
        N = No
       
   STA: {'E','W','N','S'} Starting point of fly-bys
       E = East
       W = West
       N = North
       S = Sputh
       
   DATE: {aammdd} date format
   
   COUNT: {'uw', 'dw'}
   

    Returns
    -------
    filename : list
        List containing filename list at given fy conditions  
    """
    # Relative path of the folder containing matfiles 
    folder = "../Data" # put folder  with the .mat files to process
    """ HP """
    # folder = "C:/Users/ses271/OneDrive - University of Salford/Documents/ARC_Salford/DroneNoiseMeas/Dewe_exports/" # put folder  with the h5 to process
    """ ASUS """
    # folder = "C:/Users/Asus/OneDrive/OneDrive - University of Salford/Documents/ARC_Salford/DroneNoiseMeas/Dewe_exports/"
    
    
    subfolder = PILOT+'_'+DroneID+'_'+str(HAGL)+'_'+OPE+'_'+PYL+'_'+STA+'_'+DATE+'_'+COUNT+'/'
    
    root_file = PILOT+'_'+DroneID+'_'+str(HAGL)+'_'+OPE+'_'+PYL+'_'+STA+'_'+DATE+'_'
    
    FF = folder + subfolder + root_file
    files = glob.glob(FF+"?"+".mat") # ? file in order numbered
    identifier = subfolder[0:-10]+COUNT

    return identifier, files

# %% Read and split data from single event .mat file
# ##########################################################################
def data_single_events (file_str, n_mics, acu_metric):
    """
    Returns two arrays of data from a single even file .mat
    
    Inputs
    ------
    file_str: str dirctory of the recorded event
           
    
    Returns
    -----
    Fs: int
    DATA_raw: array
    DATA_acu: array
    
    Notes
    -----
    kr: KEY name based on the raw data 
    ka: KEY name based on the acoustic data
    In both casses the data is appended in lists, then trnasformed into transposed arrays
    
    DATA_acu = DATA_acu - 6 #correction because of the reflective plate
    DATA_raw = DATA_raw / 2 #correction because of the reflective plate
    
    """
    DATA = io.loadmat(file_str) # Read the .mat file
    
    Fs = int(DATA['Sample_rate'][0][0])
    ##### LIST of DATA dictionary

    DATA_raw = []
    DATA_acu = []
    
    for m in range(n_mics):
        kr = 'Data1_Mic_'+str(m+1)                    # acces to dictionary key row
        ka = 'Data1_Mic_'+str(m+1)+'_'+acu_metric     # acces to dictionary key acu
        
        m_raw_data = np.squeeze(DATA[kr])  # acces to data ad squeeze
        m_acu_data = np.squeeze(DATA[ka])  # acces to data ad squeeze
        
        DATA_raw.append(m_raw_data)       # concat data in collumns
        DATA_acu.append(m_acu_data)
        print(m)
    DATA_raw = np.array(DATA_raw).T # concat data in collumns
    DATA_acu = np.array(DATA_acu).T # concat data in collumns
    
    DATA_acu = DATA_acu - 6  #correction because of the reflective plate -6
    DATA_raw = DATA_raw /2  #correction because of the reflective plate /2
    
    TT = (np.linspace(0, DATA_raw.shape[0], DATA_raw.shape[0]))/Fs #time vector
    
    return Fs, TT, DATA_raw, DATA_acu

# %% Read and split data from ALL event .mat file
# ##########################################################################
def data_all_events (files, DATA_raw, n_mics, acu_metric):
    """
    Returns two arrays of data from a single even file .mat
    
    Inputs
    ------
    file_str: str dirctory of the recorded event
    flybys : str{o, e, b} odd , even, both
    DATA_raw : array all raw data
    n_mics : int number of mice
    acu_metric: str acoustic metris last name
    
    Returns
    -----
    Fs: int
    DATA_raw_events: 3d array [[event], [time], [mic]]
    DATA_acu_events: 3D array [[event], [time], [mic]]
    
    Notes
    -----
    kr: KEY name based on the raw data 
    ka: KEY name based on the acoustic data
    In both casses the data is appended in lists, then trnasformed into transposed arrays
    
    DATA_acu_events = DATA_acu_events - 6 #correction because of the reflective plate
    DATA_raw_events = DATA_raw_events / 2 #correction because of the reflective plate
    """
    
    DATA_raw_events = np.empty([len(files),DATA_raw.shape[0],DATA_raw.shape[1]])
    DATA_acu_events = np.empty([len(files),DATA_raw.shape[0],DATA_raw.shape[1]])
    
    for eve in range(len(files)):   
        DATA = io.loadmat(files[eve]) #Read the .mat file
        
        Fs = int(DATA['Sample_rate'][0][0])
        ##### LIST of DATA dictionary

        DATA_raw = []
        DATA_acu = []
        
        for m in range(n_mics):
            kr = 'Data1_Mic_'+str(m+1)                    #acces to dictionary key row
            ka = 'Data1_Mic_'+str(m+1)+'_'+acu_metric     #acces to dictionary key acu
            
            m_raw_data = np.squeeze(DATA[kr])  #acces to data ad squeeze
            m_acu_data = np.squeeze(DATA[ka])  #acces to data ad squeeze
            
            DATA_raw.append(m_raw_data)       # concat data in collumns
            DATA_acu.append(m_acu_data)
            print(m)
        DATA_raw = np.array(DATA_raw).T     #concat data in collumns
        DATA_acu = np.array(DATA_acu).T     #concat data in collumns
    
        DATA_raw_events[eve,:,:]=DATA_raw
        DATA_acu_events[eve,:,:]=DATA_acu   #concat data in 3D
        print('Event '+ str( eve))
    
    DATA_acu_events = DATA_acu_events -6   #correction because of the reflective plate Rasmussen -6
    DATA_raw_events = DATA_raw_events /2  #correction because of the reflective plate Rasmussen /2
    
    TT = (np.linspace(0, DATA_raw.shape[0], DATA_raw.shape[0]))/Fs #time vector
    
    return Fs,TT, DATA_raw_events, DATA_acu_events
                        



