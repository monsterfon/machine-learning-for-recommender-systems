# Orignalna koda: dzisandy
# https://github.com/dzisandy

# 2024-07 : ZBIRKA VSEH RUTIN IZ DZS Final Project
#


import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from scipy import signal




# 07-29 DODANO LOADING

def load_subj_data(directory, subject_id, sampling_ts, data_chest=1, data_wrist=1):
    d_subj = [] #pd.DataFrame()  
    d_T = []
    d_ts = []
    
    if data_wrist==1:
        d_w = pd.read_csv(directory + '\\S' + str(subject_id) + '_W.csv')
        d_T = d_w['Target']
        d_ts = d_w['Unnamed: 0'] * sampling_ts     
        d_w.drop(['Unnamed: 0'], axis = 1, inplace= True)
        d_w.drop(['Target'], axis = 1, inplace= True)        
        
        d_subj = d_w   #pd.concat([d_subj, d_w])
        d_subj.describe()
        
    if data_chest==1:
        d_c = pd.read_csv(directory + '\\S'  + str(subject_id) + '_C.csv')
        if len(d_T)==0:
            d_T = d_c['Target']
            d_ts = d_c['Unnamed: 0'] * sampling_ts     
        d_c.drop(['Unnamed: 0'], axis = 1, inplace= True)
        d_c.drop(['Target'], axis = 1, inplace= True)
        if len(d_subj)==0:
            d_subj = d_c
        else:
            d_subj = d_subj.join(d_c)
    
    d_subj = d_subj.join(d_ts)
    d_subj['ID'] = subject_id

    d_subj = d_subj.join(d_T)
    d_subj.rename(columns={'Unnamed: 0': "Time"}, inplace = True)
        
    return d_subj    


def data_filter_labels(data_1, param_class = 0):
        
    # ID of the respective study protocol condition, sampled at 700 Hz. The following IDs 
    # are provided: 0 = not defined / transient, 1 = baseline, 2 = stress, 3 = amusement, 
    # 4 = meditation, 5/6/7 = should be ignored in this dataset 

    data_labels = []

    if param_class == 0:
        data_1['OrigTarget'] = data_1['Target']
        # Nov klas 1 naj bo amusement, meditation
        # 0 vrze ven
        # 2 stres bo postal klas 0    
        data_1['Target'].replace(3,1, inplace = True)
        data_1['Target'].replace(4,1, inplace = True)
        data_1.drop(data_1.loc[data_1['Target'] == 0].index, axis=0, inplace=True)
        data_1.drop(data_1.loc[data_1['Target'] > 2].index, axis=0, inplace=True)
        data_1['Target'].replace(2,0, inplace = True)
        data_1.index = range(0, data_1.shape[0])
        data_labels = ['Stres', 'Amuzem & Medit']

    if param_class == 1:
        # Clanek: class 0 non-stress: baseline, amusement
        # class 1: stres
        data_1['OrigTarget'] = data_1['Target']
        data_1.drop(data_1.loc[data_1['Target'] == 0].index, axis = 0, inplace=True)
        data_1.drop(data_1.loc[data_1['Target'] >= 4].index, axis=0, inplace = True)
        data_1['Target'].replace(1,0, inplace = True)
        data_1['Target'].replace(3,0, inplace = True)
        data_1['Target'].replace(2,1, inplace = True)
        data_labels = ['Baseline,Amusement', 'Stres']
    
    if param_class == 2:
        data_1['OrigTarget'] = data_1['Target']
        # Clanek: 3 razredi 1,2,3
        data_1.drop(data_1.loc[data_1['Target'] == 0].index, axis = 0, inplace=True)
        data_1.drop(data_1.loc[data_1['Target'] >= 4].index, axis=0, inplace = True)    
        data_labels = ['Baseline', 'Stres', 'Amusement' ]
    
    if param_class == 3:
        # Originalni razredi 1..4
        # 
        data_1['OrigTarget'] = data_1['Target']
        data_1.drop(data_1.loc[data_1['Target'] == 0].index, axis = 0, inplace=True)
        data_1.drop(data_1.loc[data_1['Target'] > 4].index, axis=0, inplace = True)
        data_labels = ['Baseline', 'Stres', 'Amusement', 'Meditation' ]
    
    if param_class == 4:
        # Originalni razredi 0 .. 7
        #  
        data_1['OrigTarget'] = data_1['Target']
        
        
        data_labels = ['0', '1', '2', '3', '4', '5', '6', '7' ]
    
    if param_class == 5:
        # 0 baseline, 1 stres
        # 
        data_1['OrigTarget'] = data_1['Target']
        data_1.drop(data_1.loc[data_1['Target'] == 0].index, axis = 0, inplace=True)
        data_1.drop(data_1.loc[data_1['Target'] > 2].index, axis=0, inplace = True)
        data_1['Target'].replace(1,0, inplace = True)
        data_1['Target'].replace(2,1, inplace = True)

        data_labels = ['Baseline', 'Stres' ]    
    
    return data_1, data_labels


# classes_type
# 0: klas 1 naj bo amusement, meditation, 0 stres
# 1: 0 non-stress: baseline, amusement, class 1: stres
# 

def generate_dataframe (datas, selected_features, classes_type = 1):

    data_list = []
    dt2_labels = []
    # Izloci labele
    for dt in datas:
        dt2, dt2_labels = data_filter_labels(dt, classes_type)
        data_list.append(dt2)
        print('  Data samples: ', dt2.shape)

    #  VSI SUBJEKTI ZDRUZENI
    data = pd.concat(data_list, axis=0, ignore_index=True)

    # FILTRIRANJE NAPAK
    # Najdi vrstice z inf vrednostmi
    print(data.index[np.isinf(data).any(axis=1)])
    data.drop(data.index[np.isinf(data).any(axis=1)], inplace=True)

    # IZLOCI STOLPCE FEATURES KI NISO ZAHTEVANI
    if 'ACC_C' not in selected_features:
        data.drop(['ACC_X_mean_C', 'ACC_Y_mean_C', 'ACC_Z_mean_C', 'ACC_X_std_C', 'ACC_Y_std_C', 
                   'ACC_Z_std_C', 'ACC_X_max_C', 'ACC_Y_max_C', 'ACC_Z_max_C', 'ACC_X_iabs_C', 'ACC_Y_iabs_C', 
                   'ACC_Z_iabs_C', 'ACC_3D_mean_C', 'ACC_3D_std_C', 'ACC_3D_iabs_C'], axis=1, inplace=True)
    if 'Temp_C' not in selected_features:
        data.drop(['Temp_C_mean', 'Temp_C_std', 'Temp_C_dynamic_range', 'Temp_C_slope'], axis=1, inplace=True)
    if 'EDA_C' not in selected_features:
        data.drop(['EDA_C_mean', 'EDA_C_std', 'EDA_C_dynamic_range', 'EDA_C_slope'], axis=1, inplace=True)
    if 'SCR_C' not in selected_features:
        data.drop(['SCR_mean_C', 'SCL_mean_C', 'SCR_std_C', 'SCL_std_C', 'SCR_peaks_C'], axis=1, inplace=True)
    if 'HR_C' not in selected_features:
        data.drop(['HR_mean_ECG_C', 'HR_std_ECG_C', 'HR_max_ECG_C', 'HR_min_ECG_C', 'HR_NN50_ECG_C', 
                   'HR_pNN50_ECG_C', 'HR_rmssd_ECG_C', 'HR_rr_mean_ECG_C', 'HR_rr_std_ECG_C', 'HR_ULF_ECG_C', 
                   'HR_HF_ECG_C', 'HR_LF_ECG_C', 'HR_UHF_ECG_C', 'HR_rate_L_H_ECG_C'], axis=1, inplace=True)
    if 'EMG_C' not in selected_features:
        data.drop(['EMG_mean_C', 'EMG_std_C', 'EMG_dynamic_range_C'], axis=1, inplace=True)
    if 'Resp_C' not in selected_features:
        data.drop(['resp_I_mean_dur_C', 'resp_E_mean_dur_C', 'resp_I_std_dur_C', 'resp_E_std_dur_C', 'resp_IE_ratio_C',
                   'resp_I_ampl_max_C', 'resp_E_ampl_max_C', 'resp_volume_C', 'resp_rate_mean_C', 'resp_rate_std_C', 
                   'resp_I_mean_C', 'resp_E_mean_C', 'resp_I_std_C', 'resp_E_std_C'], axis=1, inplace=True)

    if 'ACC_W' not in selected_features:
        data.drop(['ACC_X_mean_W', 'ACC_Y_mean_W', 'ACC_Z_mean_W', 'ACC_X_std_W', 'ACC_Y_std_W', 'ACC_Z_std_W',
                   'ACC_X_max_W', 'ACC_Y_max_W', 'ACC_Z_max_W', 'ACC_X_iabs_W', 'ACC_Y_iabs_W', 'ACC_Z_iabs_W', 
                   'ACC_3D_mean_W', 'ACC_3D_std_W', 'ACC_3D_iabs_W'], axis=1, inplace=True)
    if 'Temp_W' not in selected_features:
        data.drop(['Temp_W_mean', 'Temp_W_std', 'Temp_W_dynamic_range', 'Temp_W_slope'], axis=1, inplace=True)
    if 'EDA_W' not in selected_features:
        data.drop(['EDA_W_mean', 'EDA_W_std', 'EDA_W_dynamic_range', 'EDA_W_slope'], axis=1, inplace=True)
    if 'SCR_W' not in selected_features:
        data.drop(['SCR_mean_W', 'SCL_mean_W', 'SCR_std_W', 'SCL_std_W', 'SCR_peaks_W'], axis=1, inplace=True)
    if 'HR_W' not in selected_features:
        data.drop(['HR_mean_BVP_W', 'HR_std_BVP_W', 'HR_max_BVP_W', 'HR_min_BVP_W', 'HR_NN50_BVP_W', 
                   'HR_pNN50_BVP_W', 'HR_rmssd_BVP_W', 'HR_rr_mean_BVP_W', 'HR_rr_std_BVP_W', 'HR_ULF_BVP_W', 
                   'HR_HF_BVP_W', 'HR_LF_BVP_W', 'HR_UHF_BVP_W', 'HR_rate_L_H_BVP_W'], axis=1, inplace=True)


    column_names = ['ACC_X_mean_W', 'ACC_Y_mean_W', 'ACC_Z_mean_W', 'ACC_X_std_W', 'ACC_Y_std_W', 'ACC_Z_std_W', 'ACC_X_max_W', 'ACC_Y_max_W', 'ACC_Z_max_W', 'ACC_X_iabs_W', 'ACC_Y_iabs_W', 'ACC_Z_iabs_W', 'ACC_3D_mean_W', 'ACC_3D_std_W', 'ACC_3D_iabs_W', 
                'EDA_W_mean', 'EDA_W_std', 'EDA_W_dynamic_range', 'EDA_W_slope', 
                'Temp_W_mean', 'Temp_W_std', 'Temp_W_dynamic_range', 'Temp_W_slope', 
                'SCR_mean_W', 'SCL_mean_W', 'SCR_std_W', 'SCL_std_W', 'SCR_peaks_W', 
                'HR_mean_BVP_W', 'HR_std_BVP_W', 'HR_max_BVP_W', 'HR_min_BVP_W', 'HR_NN50_BVP_W', 'HR_pNN50_BVP_W', 'HR_rmssd_BVP_W', 'HR_rr_mean_BVP_W', 'HR_rr_std_BVP_W', 'HR_ULF_BVP_W', 'HR_HF_BVP_W', 'HR_LF_BVP_W', 'HR_UHF_BVP_W', 'HR_rate_L_H_BVP_W', 
                'ACC_X_mean_C', 'ACC_Y_mean_C', 'ACC_Z_mean_C', 'ACC_X_std_C', 'ACC_Y_std_C', 'ACC_Z_std_C', 'ACC_X_max_C', 'ACC_Y_max_C', 'ACC_Z_max_C', 'ACC_X_iabs_C', 'ACC_Y_iabs_C', 'ACC_Z_iabs_C', 'ACC_3D_mean_C', 'ACC_3D_std_C', 'ACC_3D_iabs_C', 
                'HR_mean_ECG_C', 'HR_std_ECG_C', 'HR_max_ECG_C', 'HR_min_ECG_C', 'HR_NN50_ECG_C', 'HR_pNN50_ECG_C', 'HR_rmssd_ECG_C', 'HR_rr_mean_ECG_C', 'HR_rr_std_ECG_C', 'HR_ULF_ECG_C', 'HR_HF_ECG_C', 'HR_LF_ECG_C', 'HR_UHF_ECG_C', 'HR_rate_L_H_ECG_C', 
                'resp_I_mean_dur_C', 'resp_E_mean_dur_C', 'resp_I_std_dur_C', 'resp_E_std_dur_C', 'resp_IE_ratio_C', 'resp_I_ampl_max_C', 'resp_E_ampl_max_C', 'resp_volume_C', 'resp_rate_mean_C', 'resp_rate_std_C', 'resp_I_mean_C', 'resp_E_mean_C', 'resp_I_std_C', 'resp_E_std_C', 
                'Temp_C_mean', 'Temp_C_std', 'Temp_C_dynamic_range', 'Temp_C_slope', 
                'EDA_C_mean', 'EDA_C_std', 'EDA_C_dynamic_range', 'EDA_C_slope', 
                'SCR_mean_C', 'SCL_mean_C', 'SCR_std_C', 'SCL_std_C', 'SCR_peaks_C', 
                'EMG_mean_C', 'EMG_std_C', 'EMG_dynamic_range_C', 
                'Time', 'ID', 'Target']

    data_stats = data.describe()

    #  max vrednosti po stolpcih
    maxdf = data_stats.iloc[7]
    maxdf.describe()

    print(np.isinf(data_stats).any())
    # Filtriraj
    a = data_stats.columns.to_series()[np.isinf(data_stats).any()]
    print('Ali so stolpci z inf ? ', a)




    #  data.to_csv(data_set+FEATURESDIR+'Data01.csv')
    # data.describe()
    return data, dt2_labels




# ORIGINALNE IN POPRAVLJENE METODE 
# GENERIRANJE FEATURES

def MSRS(data, name, f):
    """ Osnovne stat lastnosti signala v oknu: mean, stdev, slope, dyn range
    Args:
        data (_type_): _description_
        name (_type_): _description_
        f (_type_): _description_

    Returns:
        _type_: _description_
    """
    i_025 = int(f*0.25)
    i_60 = int(f*60)
    i = 0
    start = 0
    end = i_60

    n_samp = int((len(data) - i_60)/i_025)

    mean = np.empty(int((len(data) - i_60)/i_025))
    std = np.empty(int((len(data) - i_60)/i_025))
    dynamic_range = np.empty(int((len(data) - i_60)/i_025))
    slope = np.empty(int((len(data) - i_60)/i_025))
    
    print('MSRS: Lenght : ', len(data), 'nsampl : ', n_samp)
    while i < n_samp:
        
    #while i*i_025 + i_60 < int(len(data)):
        #print('    i: ', i, ' st:', start, ' end:', end)
        mean[i] = data[start:end+1].mean()
        #mean[i] = data[start:end].mean()
        std[i] = data[start:end+1].std()
        #std[i] = data[start:end].std()
        dynamic_range[i] = 20*np.log10(data[start:end+1].max()/data[start:end+1].min())
        slope[i] = (float(data[end]) - float(data[start]))/i_60
        i += 1
        start = i*i_025
        end = i*i_025 + i_60
    return {'{}_mean'.format(name): mean,'{}_std'.format(name):std,'{}_dynamic_range'.format(name):dynamic_range,'{}_slope'.format(name):slope}
    #return {'mean_{}'.format(name): mean,'std_{}'.format(name):std,'slope_{}'.format(name):slope}

# ************************************************************************************
def N2_MSRS(data, name, frequency, window_ts, wind_shift_ts):
    i_025 = int(frequency*wind_shift_ts)
    i_60 = int(frequency*window_ts)
    i = 0
    start = 0
    end = i_60
    
    mean = np.empty(int((len(data) - i_60)/i_025))
    std = np.empty(int((len(data) - i_60)/i_025))
    dynamic_range = np.empty(int((len(data) - i_60)/i_025))
    slope = np.empty(int((len(data) - i_60)/i_025))
    
    n_samp = int((len(data) - i_60)/i_025)
    while i < n_samp:
    #while i*i_025 + i_60 < int(len(data)):
        mean[i] = data[start:end+1].mean()
        std[i] = data[start:end+1].std()
        dynamic_range[i] = 20*np.log10(data[start:end+1].max()/data[start:end+1].min())
        slope[i] = (float(data[end]) - float(data[start]))/i_60
        i += 1
        start = i*i_025
        end = i*i_025 + i_60
    #return {'mean_{}'.format(name): mean,'std_{}'.format(name):std,'dynamic_range_{}'.format(name):dynamic_range,'slope_{}'.format(name):slope}
    return {'{}_mean'.format(name): mean, '{}_std'.format(name):std, '{}_dynamic_range'.format(name):dynamic_range,'{}_slope'.format(name):slope}


# ************************************************************************************

def decompose_eda(eda):
    scr_list = []
    b, a = signal.butter(4, 0.5/2)
    gsr_filt = signal.filtfilt(b, a, eda, axis=0)
    b, a = signal.butter(4,0.5/2,'highpass')
    scr = signal.filtfilt(b,a,gsr_filt,axis=0)
    scl = [float(x-y) for x,y in zip(gsr_filt,scr)]
    for i in range(len(scr)):
        scr_list.append(scr[i][0])
    return scr_list, scl


def SCRL(scr,scl,f):
    scr = np.array(scr)
    scl = np.array(scl)
    i_025 = int(f*0.25)
    i_60 = int(f*60)
    i = 0
    start = 0
    end = i_60
    mean_l = np.empty(int((len(scr) - i_60)/i_025))
    mean_r = np.empty(int((len(scr) - i_60)/i_025))
    std_l = np.empty(int((len(scr) - i_60)/i_025))
    std_r = np.empty(int((len(scr) - i_60)/i_025))
    peaks = np.empty(int((len(scr) - i_60)/i_025))
    peak = np.empty(int((len(scr) - i_60)/i_025))
    out = {}
    while i*i_025 + i_60 < int(len(scr)):
        mean_r[i] = scr[start:end+1].mean()
        std_r[i] = scr[start:end+1].std()
        mean_l[i] = scl[start:end+1].mean()
        std_l[i] = scl[start:end+1].std()
        peaks[i] = len(signal.find_peaks(scr[start:end+1],height = 0 ,distance=5)[0])
        #if i % 100 ==0: 
        #    print(i)
        i += 1
        start = i*i_025
        end = i*i_025 + i_60
    return {'SCR_mean': mean_r, 'SCL_mean': mean_l, 'SCR_std':std_r, 'SCL_std':std_l, 'SCR_peaks':peaks}
    #return {'mean_r': mean_r,'mean_l': mean_l,'std_r':std_r,'std_l':std_l,'peaks':peaks}


# ************************************************************************************

def N2_SCRL(scr, scl, frequency, window_ts, wind_shift_ts, label):
    
    i_025 = int(frequency*wind_shift_ts)
    i_60 = int(frequency*window_ts)
    
    scr = np.array(scr)
    scl = np.array(scl)
    #i_025 = int(f*0.25)
    #i_60 = int(f*60)
    
    i = 0
    start = 0
    end = i_60
    mean_l = np.empty(int((len(scr) - i_60)/i_025))
    mean_r = np.empty(int((len(scr) - i_60)/i_025))
    std_l = np.empty(int((len(scr) - i_60)/i_025))
    std_r = np.empty(int((len(scr) - i_60)/i_025))
    peaks = np.empty(int((len(scr) - i_60)/i_025))
    peak = np.empty(int((len(scr) - i_60)/i_025))
    out = {}
    n_samp = int((len(scr) - i_60)/i_025)
    while i < n_samp:    
    #while i*i_025 + i_60 < int(len(scr)):
        mean_r[i] = scr[start:end+1].mean()
        std_r[i] = scr[start:end+1].std()
        mean_l[i] = scl[start:end+1].mean()
        std_l[i] = scl[start:end+1].std()
        peaks[i] = len(signal.find_peaks(scr[start:end+1],height = 0 ,distance=5)[0])
        i += 1
        start = i*i_025
        end = i*i_025 + i_60
    return {'SCR_mean'+ label: mean_r, 'SCL_mean'+ label: mean_l, 'SCR_std'+ label:std_r, 'SCL_std'+ label:std_l, 'SCR_peaks'+ label:peaks}


def EMG(data,f):
    i_025 = int(f*0.25)
    i_5 = int(f*5)
    i = 0
    start = 0
    end = i_5
    dynamic_range = np.empty(int((len(data) - i_5)/i_025))
    mean = np.empty(int((len(data) - i_5)/i_025))
    std = np.empty(int((len(data) - i_5)/i_025))
    while i*i_025 + i_5 < int(len(data)):
        mean[i] = data[start:end+1].mean()
        std[i] = data[start:end+1].std()
        #print(data[start:end+1].max(), data[start:end+1].min())
        dynamic_range[i] = 20*np.log(abs(data[start:end+1].max())/abs(data[start:end+1].min()))
        #if i % 100 ==0: 
        #    print(i)
        i += 1
        start = i*i_025
        end = i*i_025 + i_5
    return {'EMG_mean': mean, 'EMG_std':std, 'EMG_dynamic_range': dynamic_range}
    #return {'mean': mean, 'std':std, 'dynamic_range': dynamic_range}


# ************************************************************************************

def N2_EMG(data, frequency, window_ts, wind_shift_ts, label):
    
    i_025 = int(frequency*wind_shift_ts)
    i_5 = int(frequency*window_ts)
    
    #i_025 = int(f*0.25)
    #i_5 = int(f*5)
    
    i = 0
    start = 0
    end = i_5
    dynamic_range = np.empty(int((len(data) - i_5)/i_025))
    mean = np.empty(int((len(data) - i_5)/i_025))
    std = np.empty(int((len(data) - i_5)/i_025))
    
    n_samp = int((len(data) - i_5)/i_025)
    while i < n_samp: 
            
    #while i*i_025 + i_5 < int(len(data)):
        mean[i] = data[start:end+1].mean()
        std[i] = data[start:end+1].std()
        #print(data[start:end+1].max(), data[start:end+1].min())
        dynamic_range[i] = 20*np.log(abs(data[start:end+1].max())/abs(data[start:end+1].min()))
        #if i % 100 ==0: 
        #    print(i)
        i += 1
        start = i*i_025
        end = i*i_025 + i_5
    return {'EMG_mean'+ label: mean, 'EMG_std'+ label:std, 'EMG_dynamic_range'+ label: dynamic_range}
    #return {'mean': mean, 'std':std, 'dynamic_range': dynamic_range}


def intgr(data, timestep):
    return (data[:len(data) - 1].sum() + data[1:].sum())/2*timestep

# ************************************************************************************

def ACC_features(data, window_size, num_0_25_sec, timestep_data, label):
    ACC_X_mean = []
    ACC_Y_mean = []
    ACC_Z_mean = []
    ACC_X_std = []
    ACC_Y_std = []
    ACC_Z_std = []
    ACC_X_max = []
    ACC_Y_max = []
    ACC_Z_max = []
    ACC_3D_mean =[]
    ACC_3D_std =[]
    ACC_X_iabs = [] 
    ACC_Y_iabs = [] 
    ACC_Z_iabs = [] 
    ACC_3D_iabs =[]
    
    for i in range(window_size, len(data), num_0_25_sec):
        ACC_X_mean.append( data[:,0][i - window_size:i].mean())
        ACC_X_std.append(data[:,0][i - window_size:i].std())
        ACC_X_max.append(np.fabs(data[:,0][i - window_size:i]).max())
        ACC_X_iabs.append(intgr(np.fabs(data[:,0][i - window_size:i]),timestep_data))
        ACC_Y_mean.append(data[:,1][i - window_size:i].mean())
        ACC_Y_std.append(data[:,1][i - window_size:i].std())
        ACC_Y_max.append(np.fabs(data[:,1][i - window_size:i]).max())
        ACC_Y_iabs.append(intgr(np.fabs(data[:,1][i - window_size:i]),timestep_data))
        ACC_Z_mean.append(data[:,2][i - window_size:i].mean())
        ACC_Z_std.append(data[:,2][i - window_size:i].std())
        ACC_Z_max.append(np.fabs(data[:,2][i - window_size:i]).max())
        ACC_Z_iabs.append(intgr(np.fabs(data[:,2][i - window_size:i]),timestep_data))
        ACC_3D = np.sqrt(data[:,0][i - window_size:i]**2 + data[:,1][i - window_size:i]**2 +data[:,2][i - window_size:i]**2)
        ACC_3D_mean.append(ACC_3D.mean())
        ACC_3D_std.append( ACC_3D.std())
        ACC_3D_iabs.append(intgr(ACC_3D,timestep_data))
    return {'ACC_X_mean' + label :ACC_X_mean,
        'ACC_Y_mean'+ label:ACC_Y_mean,
        'ACC_Z_mean'+ label:ACC_Z_mean,
        'ACC_X_std' + label: ACC_X_std,
        'ACC_Y_std' + label: ACC_Y_std,
        'ACC_Z_std' + label: ACC_Z_std,
            'ACC_X_max'+ label: ACC_X_max,
            'ACC_Y_max'+ label: ACC_Y_max,
            'ACC_Z_max'+ label: ACC_Z_max,
            'ACC_X_iabs'+ label: ACC_X_iabs,
            'ACC_Y_iabs'+ label: ACC_Y_iabs,
            'ACC_Z_iabs'+ label: ACC_Z_iabs,
            'ACC_3D_mean'+ label:ACC_3D_mean,
            'ACC_3D_std'+ label:ACC_3D_std,
            'ACC_3D_iabs'+ label:ACC_3D_iabs  
        }




def f_fr_n(freq, max_freq, l ):
    if freq < max_freq:
        return int(freq * l/max_freq)
    else:
        return l - 1
    
    
# ************************************************************************************

def Detect_peaks_ECG(data, window_size, num_0_25_sec, timestep_data, distance, label): #150 - respiban
    HR_mean = []
    HR_std = []
    HR_max = []
    HR_min = []
    N_HRV_50 = []
    P_HRV_50 = []
    rmssd = []
    rr_mean = []
    rr_std = []
    ULF = []
    HF = []
    LF = []
    UHF = []
    rate_L_H = [] 
    
    for i in range(window_size, len(data), num_0_25_sec):
        
        f_p = find_peaks(data[i - window_size:i ], height = 0.4, distance = distance)
        #time features
        f_p_diff = np.diff(f_p[0]) * timestep_data
                
        # heart rate mean std min max 
        HR_mean.append((60/f_p_diff).mean())  
        HR_std.append((60/f_p_diff).std()) 
        HR_max.append((60/f_p_diff).max())
        HR_min.append((60/f_p_diff).max())
        #NN50
        #pNN50
        NN50 = sum(np.abs(np.diff(f_p_diff)) > 0.050)
        N_HRV_50.append(NN50)
        P_HRV_50.append(NN50/len(f_p_diff))
        #rr_features
        rmssd.append(np.sqrt(np.mean(np.square(np.diff(f_p_diff)))))
        rr_mean.append(f_p_diff.mean())
        rr_std.append(f_p_diff.std())
        # freq features
        f_p_diff_fft = savgol_filter(np.diff(f_p_diff), 5,2)
        
        T = window_size * timestep_data
        k = np.arange(len(f_p_diff_fft))
        freqs = k/T
        m = freqs.max()/2
        l = int(len(freqs)/2)
        ffts = abs(np.fft.fft(f_p_diff_fft)*np.hamming(len(k)))**2
        ULF.append(sum( ffts[ f_fr_n(0.01,m,l):f_fr_n(0.04,m,l) ] ) )
        HF.append(sum( ffts[ f_fr_n(0.15,m,l):f_fr_n(0.4,m,l) ] ) )
        LF.append(sum( ffts[ f_fr_n(0.04,m,l):f_fr_n(0.15,m,l) ] ) )
        UHF.append(sum( ffts[ f_fr_n(0.4,m,l):f_fr_n(1,m,l) ] ) )
        
        rate_L_H.append(LF[-1]/HF[-1])
        
    return {'HR_mean' + label: np.array(HR_mean),
            'HR_std' + label: np.array(HR_std),
            'HR_max'+ label: np.array(HR_max),
            'HR_min'+ label : np.array(HR_min),
            'HR_NN50' + label: np.array(N_HRV_50),
           'HR_pNN50' + label: np.array(P_HRV_50),
           'HR_rmssd' + label: np.array(rmssd),
           'HR_rr_mean' + label: np.array(rr_mean),
           'HR_rr_std' + label: np.array(rr_std),
           'HR_ULF' + label: np.array(ULF),
           'HR_HF'+ label:np.array(HF),
           'HR_LF'+ label:np.array(LF),
           'HR_UHF'+ label:np.array(UHF),
           'HR_rate_L_H'+ label:np.array(rate_L_H)}
        
        



def find_duration(resp_max,resp_min,resp_max_ampl,resp_min_ampl):
    resp_mean_I = []
    resp_mean_E = []
    resp_ampl_I = []
    resp_ampl_E = []
    iterator_I = 0
    iterator_E = 0 
    shift = 0
    for i_max in range(len(resp_max)):
        for i_min in range(shift,len(resp_min)):
            if resp_min[i_min] > resp_max[i_max]:
                shift = i_min 
                if shift > 0:
                    resp_mean_I.append(resp_max[i_max] - resp_min[i_min - 1])
                    resp_ampl_I.append(resp_max_ampl[i_max] - resp_min_ampl[i_min - 1])
                    break;
                if shift == 0:
                    break;
    shift = 0 
    for i_min in range(len(resp_min)):
        for i_max  in range(shift,len(resp_max)):
            if resp_max[i_max] > resp_min[i_min]:
                shift = i_max
                if shift > 0:
                    resp_mean_E.append( resp_min[i_min] - resp_max[i_max - 1]  )
                    resp_ampl_E.append( resp_max_ampl[i_max - 1] - resp_min_ampl[i_min] )
                    break;
                if shift == 0:
                    break;
    return [np.array(resp_mean_I), np.array(resp_mean_E), np.array(resp_ampl_I), np.array(resp_ampl_E)]
                    
    
    
    
def intgr(data, timestep):
    return (data[:len(data) - 1].sum() + data[1:].sum())/2*timestep 
    
    

# ************************************************************************************
            
def Resp_features( data, window_size, num_0_25_sec, timestep_data, label):
    data = savgol_filter(data,15,1)
    resp_I_mean_dr = []
    resp_E_mean_dr = []
    resp_E_std_dr = []
    resp_I_std_dr = [] 
    resp_I_mean = []
    resp_E_mean = []
    resp_E_std = []
    resp_I_std = []
    resp_IE_ratio = [] 
    resp_I_ampl_max = []
    resp_E_ampl_max = []
    volume = []
    resp_rate_mean = []
    resp_rate_std = []
    for i in range(window_size, len(data), num_0_25_sec):
        resp_peak_max = find_peaks(data[i - window_size:i ], height = -1, distance = 200,prominence=True)
        resp_peak_min = find_peaks(- data[i - window_size:i ], height = -1, distance = 200,prominence=True)
        
        #features
        resp_I_E = find_duration(resp_peak_max[0], resp_peak_min[0], 
                                 resp_peak_max[1]['peak_heights'],  - resp_peak_min[1]['peak_heights'])
        resp_I_mean_dr.append(resp_I_E[0].mean() * timestep_data)
        resp_E_mean_dr.append(resp_I_E[1].mean() * timestep_data)      
        resp_I_std_dr.append(resp_I_E[0].std() * timestep_data) 
        resp_E_std_dr.append(resp_I_E[1].std() * timestep_data) 
        resp_IE_ratio.append(resp_I_mean_dr[-1]/resp_E_mean_dr[-1])
        resp_I_ampl_max.append(resp_I_E[2].max())
        resp_E_ampl_max.append(resp_I_E[3].max())
        resp_I_mean.append(resp_I_E[2].mean())
        resp_E_mean.append(resp_I_E[3].mean())
        resp_E_std.append(resp_I_E[2].std())
        resp_I_std.append(resp_I_E[3].std())
        
        
        
        volume.append(intgr(np.fabs(data[i - window_size:i ]),timestep_data))
        resp_rate_mean.append( np.hstack((resp_peak_max[0],resp_peak_min[0])).mean() * timestep_data) 
        resp_rate_std.append( np.hstack((resp_peak_max[0],resp_peak_min[0])).std()* timestep_data) 
        
    return {'resp_I_mean_dur'+ label: resp_I_mean_dr,
           'resp_E_mean_dur'+ label: resp_E_mean_dr,
           'resp_I_std_dur'+ label: resp_I_std_dr,
           'resp_E_std_dur'+ label: resp_E_std_dr,
           'resp_IE_ratio'+ label : resp_IE_ratio,
           'resp_I_ampl_max'+ label : resp_I_ampl_max,
           'resp_E_ampl_max'+ label : resp_E_ampl_max,
           'resp_volume'+ label:volume,
           'resp_rate_mean'+ label:resp_rate_mean,
           'resp_rate_std'+ label:resp_rate_std,
           'resp_I_mean'+ label: resp_I_mean,
           'resp_E_mean'+ label: resp_E_mean,
            'resp_I_std'+ label: resp_I_std,
            'resp_E_std'+ label: resp_E_std,
           }







def make_target(data_new):
    target = []
    for i in range(175, len(data_new[b'label']),175 ):
        target.append(int(data_new[b'label'][i - 175:i].mean()))
    return np.array(target)




def data_collection(data_new):
    numb_of_measures_4_HZ = data_new[b'signal'][b'wrist'][b'EDA'].shape[0]

    experiment_time = numb_of_measures_4_HZ * 0.25

    # ACC_Chest
    ACC = data_new[b'signal'][b'chest'][b'ACC']
    window_size_ts = 5
    number_o_in_0_25_sec = int(data_new[b'signal'][b'chest'][b'ECG'].shape[0]/data_new[b'signal'][b'wrist'][b'EDA'].shape[0])
    window_size_o =  number_o_in_0_25_sec * 4 * window_size_ts 
    timestep_re = numb_of_measures_4_HZ * 0.25/data_new[b'signal'][b'chest'][b'EDA'].shape[0]

    ACC_data_chest = pd.DataFrame(ACC_features(ACC, window_size_o,number_o_in_0_25_sec,timestep_re,'_chest'))

    # ECG_chest
    window_size_ts = 60
    number_o_in_0_25_sec = int(data_new[b'signal'][b'chest'][b'ECG'].shape[0]/data_new[b'signal'][b'wrist'][b'EDA'].shape[0])
    window_size_o =  number_o_in_0_25_sec * 4 * window_size_ts 
    timestep_re = numb_of_measures_4_HZ * 0.25/data_new[b'signal'][b'chest'][b'EDA'].shape[0]

    ECG = data_new[b'signal'][b'chest'][b'ECG']
    ECG = ECG.reshape((ECG.shape[0],))
    ECG_data_chest = pd.DataFrame(Detect_peaks_ECG(ECG,window_size_o,number_o_in_0_25_sec,timestep_re, distance = 150,label = '_ECG'))
    # Resp_chest
    resp = data_new[b'signal'][b'chest'][b'Resp']
    resp = resp.reshape((resp.shape[0],))
    resp_data = pd.DataFrame(Resp_features(resp,window_size_o,number_o_in_0_25_sec,timestep_re))
    
    
    
    # TEMP_chest
    Temp_chest = pd.DataFrame(MSRS(data_new[b'signal'][b'chest'][b'Temp'],'Temp_chest',700))
    
    #EDA_chest
    EDA_chest = pd.DataFrame(MSRS(data_new[b'signal'][b'chest'][b'EDA'],'EDA_chest',700))
    
    #SRCL_chest
    scr_chest, scl_chest = decompose_eda(data_new[b'signal'][b'chest'][b'EDA'])
    SCRL_chest = pd.DataFrame( SCRL(scr_chest,scl_chest,700))
    
    #EMG_chest
    EMG_chest = pd.DataFrame(EMG(data_new[b'signal'][b'chest'][b'EMG'], 700))
    #all data chest
    data_chest = pd.concat([ACC_data_chest.iloc[:EDA_chest.shape[0]],ECG_data_chest,resp_data,Temp_chest,
                        EDA_chest,SCRL_chest,EMG_chest.iloc[:EDA_chest.shape[0]]] , axis = 1)


    # BVP_wrist 
    
    BVP = data_new[b'signal'][b'wrist'][b'BVP']
    BVP = BVP.reshape((BVP.shape[0],))
    window_size_ts = 60
    number_o_in_0_25_sec_wr = int(data_new[b'signal'][b'wrist'][b'BVP'].shape[0]/data_new[b'signal'][b'wrist'][b'EDA'].shape[0])
    window_size_o_wr = 4 * window_size_ts * number_o_in_0_25_sec_wr
    timestep_wr = numb_of_measures_4_HZ * 0.25/data_new[b'signal'][b'wrist'][b'BVP'].shape[0]
    
    
    BVP_feature = pd.DataFrame(Detect_peaks_ECG(BVP, window_size_o_wr ,number_o_in_0_25_sec_wr,timestep_wr, distance = 25, label = '_BVP'))
    print(BVP_feature.shape)
    
    #ACC_wrist
    ACC = data_new[b'signal'][b'wrist'][b'ACC']
    window_size_ts = 5
    number_o_in_0_25_sec_wr = int(data_new[b'signal'][b'wrist'][b'ACC'].shape[0]/data_new[b'signal'][b'wrist'][b'EDA'].shape[0])
    window_size_o_wr = 4 * window_size_ts * number_o_in_0_25_sec_wr
    timestep_wr = numb_of_measures_4_HZ * 0.25/data_new[b'signal'][b'wrist'][b'ACC'].shape[0]

    ACC_feature = pd.DataFrame(ACC_features(ACC, window_size_o_wr ,number_o_in_0_25_sec_wr,timestep_wr, label = '_wrist'))
    print(ACC_feature.shape)
    #TEMP_WRIST
    Temp_wrist = pd.DataFrame(MSRS(data_new[b'signal'][b'wrist'][b'TEMP'],'Temp_wrist',4))
    #EDA_wrist
    EDA_wrist = pd.DataFrame(MSRS(data_new[b'signal'][b'wrist'][b'EDA'],'EDA_wrist',4))
    #SCRL_wrist
    scr_wrist, scl_wrist = decompose_eda(data_new[b'signal'][b'wrist'][b'EDA'])
    SCRL_wrist = pd.DataFrame(SCRL(scr_wrist,scl_wrist,4))
    data_wrist = pd.concat([ACC_feature[:EDA_wrist.shape[0]],EDA_wrist,Temp_wrist,SCRL_wrist,BVP_feature],axis = 1)


    return [data_chest , data_wrist]




from sklearn.metrics import (r2_score, mean_squared_error, accuracy_score, 
                             f1_score, roc_auc_score, classification_report,
                             confusion_matrix, average_precision_score,
                             precision_recall_curve)

from sklearn.metrics import precision_score, recall_score, average_precision_score
import seaborn as sns
import time
import os
import numpy as np
import matplotlib.pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK']='True'
#%matplotlib inline





def print_results(name, y_te, prob_te, pred_te, params, time):
    roc_te = roc_auc_score(y_te, prob_te)
    ac_te = accuracy_score(y_te, pred_te)
    f1_te = f1_score(y_te, pred_te)
    print('{} with best params: {}'.format(name, params[name]))
    print('\t                 Value')
    print('\tAccuracy:        {:.3f}'.format(ac_te))
    print('\tF1 score:        {:.3f}'.format(f1_te))
    print('\tROC_AUC:         {:.3f}'.format(roc_te))
    print('\tFit duration:    {:.3f}'.format(time))
    

    
def draw_pr_curve(recall, precision, average_precision):
    
    plt.figure(figsize=(8,4))
    plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve: AP={0:0.2f}'.format(
              average_precision))
    plt.show()
    
def print_stats(y_te, y_pred, y_prob, num_classes = 2):
    
    print(classification_report(y_te, y_pred, digits = 3))
    if num_classes <= 2:
        print('roc_auc \t {0:0.3f}'.format(roc_auc_score(y_te, y_prob, multi_class='ovo')))
    
    
    #if num_classes <= 2:
        # Draw precision-recall curve
    #    precision, recall, _ = precision_recall_curve(y_te, y_prob)

        # originalno klici funkcijo, sedaj kopirano sem
    #    draw_pr_curve(recall, precision, average_precision_score(y_te, y_pred))
        


    plt.figure(figsize=(10,3))
    ax1 = plt.subplot(1,2,1)

    if num_classes <= 2:
        average_precision = average_precision_score(y_te, y_pred, average='weighted')
        precision, recall, _ = precision_recall_curve(y_te, y_prob)
        
        ax1.step(recall, precision, color='b', alpha=0.2,
                where='post')
        ax1.fill_between(recall, precision, step='post', alpha=0.2,
                            color='b')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall curve: AP={0:0.2f}'.format(
                    average_precision))

    ax2 = plt.subplot(1,2,2)
    sns.heatmap(confusion_matrix(y_te, y_pred, normalize='true'), annot=True, fmt="1.3f", cmap="PiYG")

    # Draw confusion matrix
    #sns.heatmap(confusion_matrix(y_te, y_pred), annot=True, fmt="d")
    #sns.heatmap(confusion_matrix(y_te, y_pred, normalize='true'), annot=True, fmt="1.3f", cmap="PiYG")
    plt.show()
    
    
    
from sklearn.metrics import precision_score, recall_score, average_precision_score, balanced_accuracy_score, PrecisionRecallDisplay, RocCurveDisplay, ConfusionMatrixDisplay
    
def print_stats_2 (classifier, model_name, x_test, y_test, y_pred, num_classes = 2, data_labels = []):
    
    print(classification_report(y_test, y_pred, digits = 3))
    
    #if num_classes <= 2:
    #    print('roc_auc \t {0:0.3f}'.format(roc_auc_score(y_test, y_prob, multi_class='ovo')))
    
    
    #if num_classes <= 2:
        # Draw precision-recall curve
    #    precision, recall, _ = precision_recall_curve(y_te, y_prob)

        # originalno klici funkcijo, sedaj kopirano sem
    #    draw_pr_curve(recall, precision, average_precision_score(y_te, y_pred))

    
    d_lab = data_labels
    if len(data_labels) <= 0:
        d_lab = None
        
    plt.figure(figsize=(12,3))
    ax1 = plt.subplot(1,3,1)

    if num_classes <= 2:
        
        display = PrecisionRecallDisplay.from_estimator(
            classifier, x_test, y_test, name=model_name, plot_chance_level=True, ax=ax1
        )
        _ = display.ax_.set_title("2-class Precision-Recall curve")
        
        #average_precision = average_precision_score(y_te, y_pred, average='weighted')
        #precision, recall, _ = precision_recall_curve(y_te, y_prob)
        
        #ax1.step(recall, precision, color='b', alpha=0.2,
        #        where='post')
        #ax1.fill_between(recall, precision, step='post', alpha=0.2,
        #                    color='b')

        #plt.xlabel('Recall')
        #plt.ylabel('Precision')
        #plt.title('Precision-Recall curve: AP={0:0.2f}'.format(
        #            average_precision))

    ax3 = plt.subplot(1,3,2)

    if num_classes <= 2:
        svc_disp = RocCurveDisplay.from_estimator(classifier, x_test, y_test, ax=ax3)    
    else:
        sns.heatmap(confusion_matrix(y_test, y_pred, normalize='pred'), annot=True, fmt="1.3f", cmap="PiYG", yticklabels=d_lab, xticklabels=d_lab)
        ax3.set_xlabel('Precision')


    ax2 = plt.subplot(1,3,3)
    
    # NOVA VERZIJA confusion matrix 
    #disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, normalize='true', cmap = "PiYG", ax=ax2, display_labels=d_lab)

    sns.heatmap(confusion_matrix(y_test, y_pred, normalize='true'), annot=True, fmt="1.3f", cmap="PiYG", yticklabels=d_lab, xticklabels=d_lab)
    ax2.set_xlabel('Recall')
    # Draw confusion matrix
    #sns.heatmap(confusion_matrix(y_te, y_pred), annot=True, fmt="d")
    #sns.heatmap(confusion_matrix(y_te, y_pred, normalize='true'), annot=True, fmt="1.3f", cmap="PiYG")
    plt.show()    
    
    
def print_stats_threshold(y_te, y_prob, thr):
    y_res = y_prob > thr
    
    print(classification_report(y_te, y_res))
    print('roc_auc \t {0:0.3f}'.format(roc_auc_score(y_te, y_prob)))
    
    # Draw precision-recall curve
    precision, recall, _ = precision_recall_curve(y_te, y_prob)
    draw_pr_curve(recall, precision, average_precision_score(y_te, y_res))
    
    # Draw confusion matrix
    sns.heatmap(confusion_matrix(y_te, y_res), annot=True, fmt="d")
    plt.show()
    
    
    
# FUNKCIJA ZA SINHRONO SHRANJEVANJE NOTEBOOKA TAKOJ

import time
from IPython.display import display, Javascript
import hashlib

def save_notebook(file_path):
    ''' Save notebook
    '''
    start_md5 = hashlib.md5(open(file_path,'rb').read()).hexdigest()
    display(Javascript('IPython.notebook.save_checkpoint();'))
    current_md5 = start_md5
    
    while start_md5 == current_md5:
        time.sleep(1)
        current_md5 = hashlib.md5(open(file_path,'rb').read()).hexdigest()
        
   

import subprocess
import os
        
def save_html(notebook_fn, html_fn, options = ' --no-input'):
    ''' Save notebook
    '''
    command1 = f'jupyter-nbconvert {notebook_fn} --output {html_fn} --to html' + options
    print(command1)
    return_code = subprocess.run(command1, capture_output=True)
    print('Saving to HTML Result code: ', return_code.stderr)


## Konverzija vseh v html
# !!jupyter nbconvert *.ipynb
# !!jupyter nbconvert '2024_TestML_03e.ipynb' --to html --output 'test_1.html'

from scipy.io import savemat
import numpy as np

def save_matlab(data, out_folder, feat_id = '001', group_id = '000', id4 = '000'):

    data_cols = data.columns.to_list()

    IDs = data['ID'].unique()
    #print(IDs)

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    for id in IDs:

        id_str =f'{id:03d}'

        data_id = data.loc[(data.loc[:,'ID'] == id)]
        
        data_id_feats = data_id.columns.to_list()[0:-4]
                
        data_id['Label1'] = data_id['Target']
        data_id['Label2'] = data_id['OrigTarget']
        
        data_id['Frame1'] = data_id['Time']*10.0
        data_id['Frame2'] = data_id['Frame1']
        
        data_id_cols = ['Frame1', 'Frame2', 'Time', 'Label1', 'Label2'] + data_id_feats
        
        data_id = data_id.reindex(columns= data_id_cols)
        
        #print(data_id.columns.to_list())

        fn = os.path.join(out_folder, feat_id + '-'+id_str+'-'+group_id + '-' + id4 )
        
        data_id.to_csv(fn+'.csv', header=False, index=False)
        data_id.to_csv(fn+'.pandas.csv', header=True, index=False)
        
        mlab_dict = {'variable': data_id.to_numpy()}
        
        savemat(fn+'.mat', mlab_dict)
        
        

def temporal_filter(data, filter_type = 1, median_size = 7 ):
    # V OrigTarget so napovedane labele, po filtriranju pa filtrirane 
    IDs = data['ID'].unique()

    for id in IDs:

        data_id = data.loc[(data.loc[:,'ID'] == id)]
        
        data_id_feats = data_id.columns.to_list()[0:-4]
                
        #data_id['OrigTarget'] = data_id['OrigTarget'].rolling(window = median_size, center=False, min_periods=1).median()
        
        data_id['OrigTarget'] = data_id['OrigTarget'].rolling(window = median_size, min_periods = 1, center = True).median()
        data_id['OrigTarget'] = data_id['OrigTarget'].astype(int)
        
        
        data.loc[(data.loc[:,'ID'] == id)] = data_id
        
        
    return data

def probability_filter(data, filter_type = 1, prob_threshold = 0.5, median_size = 7 ):
    # V OrigTarget so napovedane labele, po filtriranju pa filtrirane 
    IDs = data['ID'].unique()

    for id in IDs:

        data_id = data.loc[(data.loc[:,'ID'] == id)]
        
        data_id_feats = data_id.columns.to_list()[0:-4]
                
        data_median = data_id['OrigTarget'].rolling(window = median_size, min_periods = 1, center = False).median()
        repl_index = data_id['ProbDiff'] < prob_threshold

        data_id['OrigTarget'][repl_index] = data_median[repl_index].astype(int)
                
        #data_id['OrigTarget'] = data_id['OrigTarget'].rolling(window = median_size, center=False, min_periods=1).median()
        #data_id['OrigTarget'] = data_id['OrigTarget'].rolling(window = median_size, min_periods = 1, center = True).median()
        #data_id['OrigTarget'] = data_id['OrigTarget'].astype(int)
        data.loc[(data.loc[:,'ID'] == id)] = data_id
        
    return data
  
  
        
from sklearn import preprocessing
import numpy as np

def data_scaling(data, scaling_type, NCOLS = 4, COLS_DROP = ['Target', 'ID', 'Time', 'OrigTarget']):

    #NCOLS = 4
    #COLS_DROP = ['Target', 'ID', 'Time', 'OrigTarget']

    # 1. STANDARDIZACIJA PODATKOV

    if scaling_type == 1:

        scaler = preprocessing.StandardScaler()

        X = data.drop(COLS_DROP, axis=1)
        
        scaler.fit(X)

        print(scaler.mean_)
        print(len(scaler.mean_))

        X_scaled = scaler.transform(X)

        print(X_scaled.shape)

        # ZAPISI STANDARDIZIRANE PODATKE NAZAJ V data
        data.iloc[:,0:X_scaled.shape[1]] = X_scaled

    if scaling_type == 2:

        scaler = preprocessing.StandardScaler()

        X = data.drop(COLS_DROP, axis=1)
        
        X_scaled = X.copy()
        
        Ids = data.ID.unique()
        
        for id in Ids:
            
            row_ind = data.ID == id 

            pod = data[row_ind]
            print(pod.shape)
            #print(pod.iloc[:, 0:pod.shape[1]-3])
                    
            scaler.fit(pod.iloc[:, 0:pod.shape[1]-NCOLS])    
            
            X_scaled[data.ID == id] = scaler.transform(pod.iloc[:, 0:pod.shape[1]-NCOLS])
        
        print(X_scaled.shape)

        # ZAPISI STANDARDIZIRANE PODATKE NAZAJ V data
        data.iloc[:,0:X_scaled.shape[1]] = X_scaled

    if scaling_type == 3:
    # individualno, normiranje na ne-stres: baseline + adjustment

        scaler = preprocessing.StandardScaler()

        X = data.drop(COLS_DROP, axis=1)
        X_scaled = X.copy()
        
        Ids = data.ID.unique()
        
        for id in Ids:
            
            pod_all_labels = data.loc[(data.loc[:,'ID'] == id)]         
            # ne-stres : to je sedaj target 0
            
            pod = data.loc[(data.loc[:,'ID'] == id) & (data.loc[:,'Target']==0)]
            print(f'Vsi podatki id={id} :', f'{pod_all_labels.shape},  baseline vzorci: {pod.shape}')
            #print(pod.iloc[:, 0:pod.shape[1]-3])
                    
            scaler.fit(pod.iloc[:, 0:pod.shape[1]-NCOLS])    
            
            X_scaled[data.ID == id] = scaler.transform(pod_all_labels.iloc[:, 0:pod.shape[1]-NCOLS])
        #X_scaled = scaler.transform(X)

        print(X_scaled.shape)

        # ZAPISI STANDARDIZIRANE PODATKE NAZAJ V data
        data.iloc[:,0:X_scaled.shape[1]] = X_scaled

    if scaling_type == 4:
    # Normalizacija features med [0, 1] min max
        
        scaler = preprocessing.MinMaxScaler()

        X = data.drop(COLS_DROP, axis=1)
        scaler.fit(X)

        #print(scaler.mean_)
        #print(len(scaler.mean_))

        X_scaled = scaler.transform(X)

        print(X_scaled.shape)

        # ZAPISI STANDARDIZIRANE PODATKE NAZAJ V data
        data.iloc[:,0:X_scaled.shape[1]] = X_scaled

    if scaling_type == 5:
    # individualno, normiranje na baseline

        scaler = preprocessing.StandardScaler()

        
        X = data.drop(COLS_DROP, axis=1)
        X_scaled = X.copy()
        
        Ids = data.ID.unique()
        
        for id in Ids:
            
            pod_all_labels = data.loc[(data.loc[:,'ID'] == id)]         
            # ne-stres : to je sedaj target 0
            
            pod = data.loc[(data.loc[:,'ID'] == id) & (data.loc[:,'OrigTarget']==1)]
            print(f'Vsi podatki id={id} :', f'{pod_all_labels.shape},  baseline vzorci: {pod.shape}')
            #print(pod.iloc[:, 0:pod.shape[1]-3])
                    
            scaler.fit(pod.iloc[:, 0:pod.shape[1]-NCOLS])    
            
            X_scaled[data.ID == id] = scaler.transform(pod_all_labels.iloc[:, 0:pod.shape[1]-NCOLS])
        #X_scaled = scaler.transform(X)

        print(X_scaled.shape)

        # ZAPISI STANDARDIZIRANE PODATKE NAZAJ V data
        data.iloc[:,0:X_scaled.shape[1]] = X_scaled
        
        


    if scaling_type == 6:
    # QUANTILE TRANSFORM

        X = data.drop(COLS_DROP, axis=1)
        X_scaled = X.copy()
        
        Ids = data.ID.unique()
        
        for id in Ids:
            
            row_ind = data.ID == id 

            pod = data[row_ind]
            print(pod.shape)
            
            
            X_scaled[data.ID == id] = preprocessing.quantile_transform(pod.iloc[:, 0:pod.shape[1]-NCOLS])

        print(X_scaled.shape)
        
        # ZAPISI STANDARDIZIRANE PODATKE NAZAJ V data
        data.iloc[:,0:X_scaled.shape[1]] = X_scaled    


    if scaling_type == 7:

        scaler = preprocessing.RobustScaler()

        X = data.drop(COLS_DROP, axis=1)
        
        X_scaled = X.copy()
        
        Ids = data.ID.unique()
        
        for id in Ids:
            
            row_ind = data.ID == id 

            pod = data[row_ind]
            print(pod.shape)
            #print(pod.iloc[:, 0:pod.shape[1]-3])
                    
            scaler.fit(pod.iloc[:, 0:pod.shape[1]-NCOLS])    
            
            X_scaled[data.ID == id] = scaler.transform(pod.iloc[:, 0:pod.shape[1]-NCOLS])
        
        print(X_scaled.shape)

        # ZAPISI STANDARDIZIRANE PODATKE NAZAJ V data
        data.iloc[:,0:X_scaled.shape[1]] = X_scaled


    if scaling_type == 8:
    # individualno, normiranje na baseline, Robust Scaler

        scaler = preprocessing.RobustScaler()

        
        X = data.drop(COLS_DROP, axis=1)
        X_scaled = X.copy()
        
        Ids = data.ID.unique()
        
        for id in Ids:
            
            pod_all_labels = data.loc[(data.loc[:,'ID'] == id)]         
            # ne-stres : to je sedaj target 0
            
            pod = data.loc[(data.loc[:,'ID'] == id) & (data.loc[:,'OrigTarget']==1)]
            print(f'Vsi podatki id={id} :', f'{pod_all_labels.shape},  baseline vzorci: {pod.shape}')
            #print(pod.iloc[:, 0:pod.shape[1]-3])
                    
            scaler.fit(pod.iloc[:, 0:pod.shape[1]-NCOLS])    
            
            X_scaled[data.ID == id] = scaler.transform(pod_all_labels.iloc[:, 0:pod.shape[1]-NCOLS])

        print(X_scaled.shape)

        # ZAPISI STANDARDIZIRANE PODATKE NAZAJ V data
        data.iloc[:,0:X_scaled.shape[1]] = X_scaled


    if scaling_type == 9:
    # QUANTILE TRANSFORM, NORMAL

        X = data.drop(COLS_DROP, axis=1)
        X_scaled = X.copy()
        
        Ids = data.ID.unique()
        
        for id in Ids:
            
            row_ind = data.ID == id 

            pod = data[row_ind]
            print(pod.shape)
            
            
            X_scaled[data.ID == id] = preprocessing.quantile_transform(pod.iloc[:, 0:pod.shape[1]-NCOLS], output_distribution='normal')

        print(X_scaled.shape)
        
        # ZAPISI STANDARDIZIRANE PODATKE NAZAJ V data
        data.iloc[:,0:X_scaled.shape[1]] = X_scaled    

    if scaling_type == 10:
    # QUANTILE TRANSFORM, NORMAL, ALL

        X = data.drop(COLS_DROP, axis=1)

        X_scaled = preprocessing.quantile_transform(X, output_distribution='normal')

        # ZAPISI STANDARDIZIRANE PODATKE NAZAJ V data
        data.iloc[:,0:X_scaled.shape[1]] = X_scaled


    return data