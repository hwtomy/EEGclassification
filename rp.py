import numpy as np
import pandas as pd
from seiz_eeg.clips import make_clips
from preprocess import prep, filter_patients, remove_short_segments




def clips(df, sampling_rate, target_sampling_rate, output_dir, lowpass_freq, clip_length, clip_stride):

    print(df.head())
    df = prep(df, sampling_rate, target_sampling_rate, output_dir, lowpass_freq)
    #df = filter_patients(df)
    df['sampling_rate'] = df['sampling_rate'].replace(250, 100)
    print(df.head())
    all_clips_df = pd.DataFrame()
    all_clips_list = []
    count = 0
    sfre = 100
    max_duration_samples = 20 * 60 * sfre
    # Loop through each unique signals_path and create clips
    for signals_path in df['signals_path'].unique():
        signal_df = df[df['signals_path'] == signals_path]
        # count += 1
        # if count > 10:
        #     break
        mapping = {  
        "0": 0,
        "1": 1,
        "2": 1,
        "3": 1,
        "4": 1,
        }
        # patient_id = signal_df.index.get_level_values('session')[0]

        # # patient_df = df.loc[patient_id]
        # # if (patient_df['label'] == '0').all():
        # #     continue

        # patient_df = df.xs(patient_id, level='session', drop_level=False)

        # if (patient_df['label'] == 0).all():
        #     continue

        if signal_df.empty:
            print(f"Warning: {signals_path} No signal")
            continue  

        signal_duration = signal_df['end_time'] - signal_df['start_time']
        if signal_duration.min() < 18:
            print(f"Warning: {signals_path} No enough time")
            continue

        print(f"Process signal: {signals_path}")
        signal_duration = signal_df['end_time'] - signal_df['start_time']
        total_samples = signal_duration * 100

        if total_samples.max() > max_duration_samples:
            #print(f"Warning: {signals_path} exceeds 20 minutes, trimming signal.")
            signal_df = signal_df.iloc[:max_duration_samples] 
        #print(signal_df[['start_time', 'end_time', 'label']].head())
        #print(signal_df.head())
        # count += 1
        # if count > 30:
        #     break
        clips_df = make_clips(
            segments_df=signal_df,
            clip_length=clip_length,
            clip_stride=clip_stride,
            overlap_action="ignore",
        )
        
        if clips_df is None or clips_df.empty:
            print(f"Warning: {signals_path} No new clips")
            continue 
        clips_df['label'] = clips_df['label'].replace(mapping)
        print(clips_df.head())
        #exit ()
        #num_rows = clips_df.shape[0]
        #print(f"Number of clips: {num_rows}")
        # Concatenate clips to the main DataFrame
        all_clips_list.append(clips_df)
        #all_clips_df = pd.concat([all_clips_df, clips_df], ignore_index=False)
    # print(all_clips_list.head())
    all_clips_df = pd.concat(all_clips_list, ignore_index=False)

    all_clips_df = remove_short_segments(all_clips_df, min_duration=6)
    return all_clips_df

