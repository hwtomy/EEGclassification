import os
import pandas as pd
import mne
from scipy.signal import resample
import pyarrow.parquet as pq

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
        
        if not any(label_1_df['duration'] > 18):
            df = df.drop(patient_id, level='patient')
    
    return df

def remove_short_segments(df, min_duration):
 
    df['duration'] = df['end_time'] - df['start_time']
    
    filtered_df = df[df['duration'] == min_duration]
    
    return filtered_df


