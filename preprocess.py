import os
import pandas as pd
import mne
from scipy.signal import resample
import pyarrow.parquet as pq
import pywt
import torch
import numpy as np

def prep(df, originrate, samplerate, output_dir, lowpass_freq):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    for signals_path in df['signals_path'].unique():
        
        file_name = os.path.basename(signals_path)
        output_file_path = os.path.join(output_dir, file_name)
        
        if os.path.exists(output_file_path):
            print(f"File {output_file_path} already processed, skipping.")
            df.loc[df['signals_path'] == signals_path, 'signals_path'] = output_file_path
            continue

        signal_df = pd.read_parquet(signals_path)
        ch_names = list(signal_df.columns)
        ch_types = ['eeg'] * len(ch_names) 
        print(ch_names)
        print(signal_df.head())
        info = mne.create_info(ch_names=ch_names, sfreq=originrate, ch_types=ch_types)
        raw = mne.io.RawArray(signal_df.values.T, info)

        picks = mne.pick_types(raw.info, eeg=True, meg=False, exclude=[])

        raw.filter(l_freq=None, h_freq=lowpass_freq, picks = picks, n_jobs=-1)

        raw.resample(sfreq=samplerate, n_jobs=-1)
        processed_data = raw.get_data().T 

        processed_df = pd.DataFrame(processed_data, columns=signal_df.columns)
        processed_df.to_parquet(output_file_path)
        df.loc[df['signals_path'] == signals_path, 'signals_path'] = output_file_path

    return df


def filter_patients(df, min_duration=18):
  
    df['duration'] = df['end_time'] - df['start_time']
    for patient_id in df.index.get_level_values('patient').unique():
        patient_df = df.xs(patient_id, level='patient', drop_level=False)
        
        label_1_df = patient_df[patient_df['label'] == 1] 
        if not (any(label_1_df['duration'] > 18) ):
            df = df.drop(patient_id, level='patient')
    
    return df

def remove_short_segments(df, min_duration):
 
    df['duration'] = df['end_time'] - df['start_time']
    
    filtered_df = df[df['duration'] == min_duration]
    
    return filtered_df


def filter_shortpatient(df, min=12):
    df['duration'] = df['end_time'] - df['start_time']
      
    for session_id in df.index.get_level_values('session').unique():
        session_df = df.xs(session_id, level='session', drop_level=False)
        
        label_1_df = session_df[session_df['label'] == 1]
            
        if len(label_1_df) < 2:
                df = df.drop(session_id, level='session')
    
    return df



def filter_patients_label(df):
    for patient_id in df.index.get_level_values('session').unique():
        patient_df = df.xs(patient_id, level='session', drop_level=False)
        
        if not all(patient_df['label'].isin([0, 3])):
            df = df.drop(patient_id, level='session')
            continue
        
    return df



def filter_patientsi(df, min_duration=12):
  
    df['duration'] = df['end_time'] - df['start_time']
    
    for patient_id in df.index.get_level_values('patient').unique():
        patient_df = df.xs(patient_id, level='patient', drop_level=False)
        
        if (patient_df['label'] == 0).all():
            df = df.drop(patient_id, level='patient')
            continue
  
        non_zero_df = patient_df[patient_df['label'] != 0]
        if not any(non_zero_df['duration'] > min_duration):
            df = df.drop(patient_id, level='patient')
    
    return df



def filter_patients3(df, min_duration=12):
  
    df['duration'] = df['end_time'] - df['start_time']
    
    for patient_id in df.index.get_level_values('patient').unique():
        patient_df = df.xs(patient_id, level='patient', drop_level=False)
        
        if (patient_df['label'] == 0).all():
            df = df.drop(patient_id, level='patient')
            continue
  
        non_zero_df = patient_df[patient_df['label'] == 3]
        if not any(non_zero_df['duration'] > min_duration):
            df = df.drop(patient_id, level='patient')
    
    return df





def apply_wavelet_transform(data, wavelet='db4', level=4):
    # Perform a multi-level discrete wavelet transform
    coeffs = pywt.wavedec(data.cpu().numpy(), wavelet, level=level)
    # Concatenate all wavelet coefficients into one array
    transformed_data = torch.tensor(np.concatenate(coeffs)).float().to(data.device)
    return transformed_data


def apply_wavelet_transform_multichannel(data, wavelet='db4', level=4):

    channels = data.size(0)
    transformed_channels = []
    

    for ch in range(channels):
        transformed_channel = apply_wavelet_transform(data[ch], wavelet, level)
        transformed_channels.append(transformed_channel)
    
    return torch.stack(transformed_channels)


def balance_data(features, labels, num_classes=2):
    label_indices = [np.where(labels == i)[0] for i in range(num_classes)]
    
    if any(len(indices) == 0 for indices in label_indices):
        return features, labels 

    max_class_count = max(len(indices) for indices in label_indices)
    
    balanced_features = []
    balanced_labels = []

    for i, indices in enumerate(label_indices):
        upsampled_indices = np.random.choice(indices, size=max_class_count, replace=True)
        balanced_features.append(features[upsampled_indices])
        balanced_labels.append(np.full(max_class_count, i))

    balanced_features = np.concatenate(balanced_features)
    balanced_labels = np.concatenate(balanced_labels)

    return balanced_features, balanced_labels






