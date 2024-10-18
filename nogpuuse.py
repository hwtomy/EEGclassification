
import pandas as pd
import numpy as np
from rp import clips
import os
import shutil
from clean import clear_directory
from loss import RelativePositioningLossm
from pretext import RPDataset, collate_fn, split_dataset, LabelDataset, taset, collate_fnt, balance_dataframe, RPDataset3
from preprocess import remove_short_segments, filter_shortpatient, filter_patients_label, filter_patientsi, filter_patients3




df = pd.read_parquet("/datasets2/epilepsy/TUSZ/processed/train/segments.parquet")
output_dir = "./data/train"
sampling_rate=250
target_sampling_rate=100
lowpass_freq=50
sfre = 100
clip_length = 6
clip_stride = 6
df = filter_patients3(df)
# print("show")
# print(df.head())
all_clips_df = clips(df, sampling_rate, target_sampling_rate, output_dir, lowpass_freq,clip_length, clip_stride)
all_clips_df.to_parquet('./data/processed_train3.parquet', engine='pyarrow')


df = pd.read_parquet("/datasets2/epilepsy/TUSZ/processed/dev/segments.parquet")
output_dir = "./data/test"
sampling_rate=250
target_sampling_rate=100
lowpass_freq=50
sfre = 100
clip_length = 6
clip_stride = 6
df = filter_patients3(df)
all_clips_df = clips(df, sampling_rate, target_sampling_rate, output_dir, lowpass_freq,clip_length, clip_stride)
all_clips_df.to_parquet('./data/processed_test3.parquet', engine='pyarrow')