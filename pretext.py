from torch.utils.data import DataLoader, Dataset
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import torch
import random
import pandas as pd
import numpy as np
from torch.utils.data._utils.collate import default_collate
import torch
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data
import os
import sys
import pickle

class RPDataset(Dataset):
    def __init__(self, clips_df, tau_pos, tau_neg, sampling_rate=100, num_pairs=400):
        self.clips_df = clips_df
        self.tau_pos = tau_pos
        self.tau_neg = tau_neg
        self.sampling_rate = sampling_rate
        self.num_pairs = num_pairs

    
        self.sessions = list(self.clips_df.groupby('session'))

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, index):
    
        session_id, session_df = self.sessions[index]
        

        pos_pairs = self.generate_pairs(session_df, pair_type="positive", num_pairs=300)
        neg_pairs = self.generate_pairs(session_df, pair_type="negative", num_pairs=300)

        if len(pos_pairs) == 0 and len(neg_pairs) == 0:
            #print(f"Skipping session {session_id}, not enough pairs")
            return []
        #print(f"Generated {len(pos_pairs)} positive pairs and {len(neg_pairs)} negative pairs")
        if (len(pos_pairs)+len(neg_pairs)) > 400:
            if (len(neg_pairs) > 200 and len(pos_pairs) < 200):
                neg_pairs = random.sample(neg_pairs, 400-len(pos_pairs))
            elif (len(neg_pairs) > 200 and len(pos_pairs) > 200):
                pos_pairs = random.sample(pos_pairs, 200)
                neg_pairs = random.sample(neg_pairs, 200)
            else:
                pos_pairs = random.sample(pos_pairs, 400-len(neg_pairs))
                
        all_pairs = pos_pairs + neg_pairs
        #random.shuffle(all_pairs)

        return all_pairs

    def generate_pairs(self, session_df, pair_type, num_pairs):
        pairs = []
        att = 0
        maxa = 400
        while len(pairs) < num_pairs and att < maxa:
            anchor_window = session_df.sample(1).iloc[0]

            if pair_type == "positive":
                paired_window = self.get_positive_window(anchor_window, session_df)
                label = 1 
            else:
                paired_window = self.get_negative_window(anchor_window, session_df)
                label = -1  

            if paired_window is None:
                att += 1
                continue

            anchor_data = self.read_signal(anchor_window)
            paired_data = self.read_signal(paired_window)


            pairs.append((anchor_data, paired_data, label))
            att += 1

        return pairs

    def get_positive_window(self, anchor_window, session_df):
        pos_mask = (session_df['start_time'] >= anchor_window['start_time'] - self.tau_pos) & \
                   (session_df['start_time'] <= anchor_window['start_time'] + self.tau_pos)
        pos_samples = session_df[pos_mask]
        
        if len(pos_samples) == 0:
            return None


        return pos_samples.sample(1).iloc[0]

    def get_negative_window(self, anchor_window, session_df):
        neg_mask = (session_df['start_time'] < anchor_window['start_time'] - self.tau_neg) | \
                   (session_df['start_time'] > anchor_window['start_time'] + self.tau_neg)
        neg_samples = session_df[neg_mask]

        if len(neg_samples) == 0:
            return None

        return neg_samples.sample(1).iloc[0]

    # def read_signal(self, window):
    #     signal_df = pd.read_parquet(window['signals_path'])
    #     start_sample = int(window['start_time'] * self.sampling_rate)
    #     end_sample = int(window['end_time'] * self.sampling_rate)
    #     return signal_df.values[start_sample:end_sample, :].T
    def read_signal(self, window):
        if not hasattr(self, 'signal_cache'):
            self.signal_cache = {}
        signal_path = window['signals_path']
        if signal_path not in self.signal_cache:
            self.signal_cache[signal_path] = pd.read_parquet(signal_path)
        signal_df = self.signal_cache[signal_path]
        start_sample = int(window['start_time'] * self.sampling_rate)
        end_sample = int(window['end_time'] * self.sampling_rate)
        return signal_df.values[start_sample:end_sample, :].T




    # def read_signal(self, window):
    #     signal_df = pd.read_parquet(window['signals_path'])
    #     total_samples = signal_df.shape[1] 
    #     start_sample = max(0, int(window['start_time'] * self.sampling_rate))
    #     end_sample = min(total_samples, int(window['end_time'] * self.sampling_rate))
    #     return signal_df.values[:, start_sample:end_sample].T 


# def collate_fn(batch):
#     anchor_pos, anchor_neg = [], []
#     for pos_pair, neg_pair in batch:
#         anchor_pos.append(pos_pair)
#         anchor_neg.append(neg_pair)
#     return anchor_pos, anchor_neg



# def split_dataset(clips_df, test_size=0.2, random_state=42):
#     session_ids = clips_df['session'].unique()

#     train_sessions, test_sessions = train_test_split(session_ids, test_size=test_size, random_state=random_state)
#     train_df = clips_df[clips_df['session'].isin(train_sessions)]
#     test_df = clips_df[clips_df['session'].isin(test_sessions)]

#     return train_df, test_df

def split_dataset(clips_df, test_size=0.2, random_state=42):
    clips_df = clips_df.reset_index(level=['session'])
    
    session_ids = clips_df['session'].unique()

    train_sessions, test_sessions = train_test_split(session_ids, test_size=test_size, random_state=random_state)

    train_df = clips_df[clips_df['session'].isin(train_sessions)]
    test_df = clips_df[clips_df['session'].isin(test_sessions)]

    return train_df, test_df




def split_dataset1(clips_df, test_size=0.2, random_state=42):

    session_ids = clips_df.index


    train_sessions, test_sessions = train_test_split(session_ids, test_size=test_size, random_state=random_state)


    train_df = clips_df.loc[train_sessions]
    test_df = clips_df.loc[test_sessions]

    return train_df, test_df




class LabelDataset(Dataset):
    def __init__(self, clips_df, sampling_rate=100):
        self.clips_df = clips_df
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.clips_df)

    def __getitem__(self, index):
        window = self.clips_df.iloc[index]
        label = window['label'] 
        signal_data = self.read_signal(window)
        # if signal_data is None or signal_data.size == 0:
        #     return None, None

        if signal_data is not None and signal_data.size > 0:
            return signal_data, label 
            
        index = (index + 1) % len(self.clips_df)
        

    def read_signal(self, window):
        signal_df = pd.read_parquet(window['signals_path'])
        start_sample = int(window['start_time'] * self.sampling_rate)
        end_sample = int(window['end_time'] * self.sampling_rate)
        signal_data = signal_df.values[:, start_sample:end_sample]
        if signal_data.size == 0:
            return None
    
        return signal_data


def pad_to_fixed_len(data, fixed_len):
    if data.shape[1] < fixed_len:
        pad_len = fixed_len - data.shape[1]

        return torch.cat([data, torch.zeros(pad_len, data.shape[0])], dim=0)
 
    return data[:,:fixed_len]  


def collate_fnB(batch):
    fixed_len = 600 

    anchor_data_list = []
    paired_data_list = []
    labels_list = []
    alabels_list = []

    for pairs in batch:
        for anchor_data, paired_data, label, alabels in pairs:
            anchor_data = torch.tensor(anchor_data)
            paired_data = torch.tensor(paired_data)

            
            anchor_data_padded = pad_to_fixed_len(anchor_data, fixed_len)
            paired_data_padded = pad_to_fixed_len(paired_data, fixed_len)

            anchor_data_list.append(anchor_data_padded)
            paired_data_list.append(paired_data_padded)
            labels_list.append(label)
            alabels_list.append(alabels)


        return list(zip(anchor_data_list, paired_data_list, labels_list, alabels_list))



def collate_fn(batch):
    fixed_len = 600 

    anchor_data_list = []
    paired_data_list = []
    labels_list = []
 

    for pairs in batch:
        for anchor_data, paired_data, label in pairs:
            anchor_data = torch.tensor(anchor_data)
            paired_data = torch.tensor(paired_data)

            
            anchor_data_padded = pad_to_fixed_len(anchor_data, fixed_len)
            paired_data_padded = pad_to_fixed_len(paired_data, fixed_len)

            anchor_data_list.append(anchor_data_padded)
            paired_data_list.append(paired_data_padded)
            labels_list.append(label)
  


    return list(zip(anchor_data_list, paired_data_list, labels_list))



def collate_fnt(batch):
    fixed_len = 600 

    anchor_data_list = []
    labels_list = []

    for pairs in batch:
        for anchor_data, label in pairs:
            anchor_data = torch.tensor(anchor_data)
            anchor_data_padded = pad_to_fixed_len(anchor_data, fixed_len)
            anchor_data_list.append(anchor_data_padded)
            labels_list.append(label)


    return list(zip(anchor_data_list, labels_list))


class taset(Dataset):
    def __init__(self, clips_df, sampling_rate=100):
        self.clips_df = clips_df
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.clips_df)

    def __getitem__(self, index):
        pairs = []
        window = self.clips_df.iloc[index]
        label = window['label'] 

        signal_data = self.read_signal(window)
        
        if len(signal_data) == 0:
            return []
        else:
            pairs.append((signal_data, label))

        return pairs



    def read_signal(self, window):
        if not hasattr(self, 'signal_cache'):
            self.signal_cache = {}
        signal_path = window['signals_path']
        if signal_path not in self.signal_cache:
            self.signal_cache[signal_path] = pd.read_parquet(signal_path)
        signal_df = self.signal_cache[signal_path]
        start_sample = int(window['start_time'] * self.sampling_rate)
        end_sample = int(window['end_time'] * self.sampling_rate)
        return signal_df.values[start_sample:end_sample, :].T



# def balance_dataframe(clips_df):

#     label_0_df = clips_df[clips_df['label'] == 0]
#     label_1_df = clips_df[clips_df['label'] == 1]

#     if len(label_1_df) == 0 or len(label_0_df) == 0:
#         return clips_df  

#     label_1_upsampled = label_1_df.sample(len(label_0_df), replace=True)

#     balanced_df = pd.concat([label_0_df, label_1_upsampled]).reset_index(drop=True)

#     return balanced_df

def balance_dataframe(clips_df):
    label_0_df = clips_df[clips_df['label'] == 0]
    label_1_df = clips_df[clips_df['label'] == 1]

    num_to_sample = len(label_0_df) - len(label_1_df)

    if num_to_sample > 0:
        label_1_upsampled = label_1_df.sample(num_to_sample, replace=True)
        label_1_df = pd.concat([label_1_df, label_1_upsampled])

    balanced_df = pd.concat([label_0_df, label_1_df]).reset_index(drop=True)

    return balanced_df





class CPCDataset(Dataset):
    def __init__(self, clips_df, N_c, N_p, N_b, sampling_rate=100, fixed_len=600):
        """
        :param N_c: Number of context windows (past).
        :param N_p: Number of future windows to predict.
        :param N_b: Number of negative windows per future window to sample.
        :param sampling_rate: The sampling rate of the signals.
        :param fixed_len: The fixed length of the windows (for padding or truncating).
        """
        self.clips_df = clips_df
        self.N_c = N_c
        self.N_p = N_p
        self.N_b = N_b
        self.sampling_rate = sampling_rate
        self.fixed_len = fixed_len
        self.sessions = list(self.clips_df.groupby('session'))  # Group by session

    def __len__(self):
        return len(self.sessions) 

    def __getitem__(self, index):

        session_id, session_df = self.sessions[index]
        num_windows_in_session = len(session_df)
        if num_windows_in_session <= self.N_c + self.N_p:
   
            return None

        random_idx = np.random.randint(self.N_c, num_windows_in_session - self.N_p)

        context_windows = session_df.iloc[random_idx - self.N_c:random_idx]
        future_windows = session_df.iloc[random_idx:random_idx + self.N_p]


        negative_windows = self.sample_negative_windows(session_df, num_neg_windows=self.N_p * self.N_b, exclude=pd.concat([context_windows, future_windows]))

        context_data = [self.read_signal(window) for idx, window in context_windows.iterrows()]
        future_data = [self.read_signal(window) for idx, window in future_windows.iterrows()]

        negative_data = []
        for i in range(self.N_p):
            neg_samples_for_future = [self.read_signal(window) for idx, window in negative_windows.iloc[i * self.N_b:(i + 1) * self.N_b].iterrows()]
            negative_data.append(torch.stack(neg_samples_for_future))


        context_data = torch.stack(context_data)  
        future_data = torch.stack(future_data)    
        negative_data = torch.stack(negative_data)  

        # context_data = context_data.unsqueeze(0)  
        # future_data = future_data.unsqueeze(0)    
        # negative_data = negative_data.unsqueeze(0) 

        return context_data, future_data, negative_data


    def sample_negative_windows(self, session_df, num_neg_windows, exclude):

        exclude_start_times = exclude['start_time'].values
        negative_mask = ~session_df['start_time'].isin(exclude_start_times)
        negative_windows = session_df[negative_mask].sample(num_neg_windows, replace=True)
        return negative_windows

    def read_signal(self, window):

        if not hasattr(self, 'signal_cache'):
            self.signal_cache = {}

        # Load the signal if not already cached
        signal_path = window['signals_path']
        if signal_path not in self.signal_cache:
            self.signal_cache[signal_path] = pd.read_parquet(signal_path)
        signal_df = self.signal_cache[signal_path]

        # Extract the signal between start_time and end_time
        start_sample = int(window['start_time'] * self.sampling_rate)
        end_sample = int(window['end_time'] * self.sampling_rate)
        signal_data = signal_df.values[start_sample:end_sample, :].T

        # Pad or truncate the signal to the fixed length
        if signal_data.shape[1] < self.fixed_len:
            pad_len = self.fixed_len - signal_data.shape[1]
            signal_data = np.pad(signal_data, ((0, 0), (0, pad_len)), mode='constant')
        else:
            signal_data = signal_data[:, :self.fixed_len]

        return torch.tensor(signal_data, dtype=torch.float32)

def collate_fnc(batch):
    batch = [data for data in batch if data is not None]
    
    if len(batch) == 0:
        return None
    
    return default_collate(batch)




class RPDataset1(Dataset):
    def __init__(self, clips_df, tau_pos, tau_neg, sampling_rate=100, num_pairs=400):
        self.clips_df = clips_df
        self.tau_pos = tau_pos
        self.tau_neg = tau_neg
        self.sampling_rate = sampling_rate
        self.num_pairs = num_pairs

    
        self.sessions = list(self.clips_df.groupby('session'))

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, index):
    
        session_id, session_df = self.sessions[index]
        

        pos_pairs = self.generate_pairs(session_df, pair_type="positive", num_pairs=400)
        neg_pairs = self.generate_pairs(session_df, pair_type="negative", num_pairs=300)

        if len(pos_pairs) == 0 and len(neg_pairs) == 0:
            #print(f"Skipping session {session_id}, not enough pairs")
            return []
        #print(f"Generated {len(pos_pairs)} positive pairs and {len(neg_pairs)} negative pairs")
        if (len(pos_pairs)+len(neg_pairs)) > 400:
            pos_pairs = random.sample(pos_pairs, 400-len(neg_pairs))
        all_pairs = pos_pairs + neg_pairs
        #random.shuffle(all_pairs)

        return all_pairs

    def generate_pairs(self, session_df, pair_type, num_pairs):
        pairs = []
        att = 0
        maxa = 400
        while len(pairs) < num_pairs and att < maxa:
            anchor_window = session_df.sample(1).iloc[0]

            if pair_type == "positive":
                paired_window = self.get_positive_window(anchor_window, session_df)
                label = 1 
            else:
                paired_window = self.get_negative_window(anchor_window, session_df)
                label = -1  

            if paired_window is None:
                att += 1
                continue

            anchor_data = self.read_signal(anchor_window)
            paired_data = self.read_signal(paired_window)


            pairs.append((anchor_data, paired_data, label))
            att += 1

        return pairs

    def get_positive_window(self, anchor_window, session_df):
        if anchor_window['label'] == 1:
            pos_mask = (session_df['start_time'] >= anchor_window['start_time'] - self.tau_pos) & \
                    (session_df['start_time'] <= anchor_window['start_time'] + self.tau_pos)
            pos_samples = session_df[pos_mask]

            if len(pos_samples) == 0:
                return None

            prev_window = session_df[session_df['start_time'] < anchor_window['start_time']].tail(1)
            next_window = session_df[session_df['start_time'] > anchor_window['start_time']].head(1)

            if not prev_window.empty and prev_window.iloc[0]['label'] == 0:
                pos_mask_after = (session_df['start_time'] > anchor_window['start_time']) & \
                                (session_df['start_time'] <= anchor_window['start_time'] + self.tau_pos)
                pos_samples_after = session_df[pos_mask_after]
                if len(pos_samples_after) > 0:
                    return pos_samples_after.sample(1).iloc[0]
            
            if not next_window.empty and next_window.iloc[0]['label'] == 0:
                pos_mask_before = (session_df['start_time'] < anchor_window['start_time']) & \
                                (session_df['start_time'] >= anchor_window['start_time'] - self.tau_pos)
                pos_samples_before = session_df[pos_mask_before]
                if len(pos_samples_before) > 0:
                    return pos_samples_before.sample(1).iloc[0]

        pos_mask = (session_df['start_time'] >= anchor_window['start_time'] - self.tau_pos) & \
                (session_df['start_time'] <= anchor_window['start_time'] + self.tau_pos)
        pos_samples = session_df[pos_mask]

        if len(pos_samples) == 0:
            return None

        return pos_samples.sample(1).iloc[0]


    def get_negative_window(self, anchor_window, session_df):
        neg_mask = (session_df['start_time'] < anchor_window['start_time'] - self.tau_neg) | \
                   (session_df['start_time'] > anchor_window['start_time'] + self.tau_neg)
        neg_samples = session_df[neg_mask]

        if len(neg_samples) == 0:
            return None

        return neg_samples.sample(1).iloc[0]


    def read_signal(self, window):
        if not hasattr(self, 'signal_cache'):
            self.signal_cache = {}
        signal_path = window['signals_path']
        if signal_path not in self.signal_cache:
            self.signal_cache[signal_path] = pd.read_parquet(signal_path)
        signal_df = self.signal_cache[signal_path]
        start_sample = int(window['start_time'] * self.sampling_rate)
        end_sample = int(window['end_time'] * self.sampling_rate)
        return signal_df.values[start_sample:end_sample, :].T





class RPDataset1(Dataset):
    def __init__(self, clips_df, tau_pos, tau_neg, sampling_rate=100, num_pairs=400):
        self.clips_df = clips_df
        self.tau_pos = tau_pos
        self.tau_neg = tau_neg
        self.sampling_rate = sampling_rate
        self.num_pairs = num_pairs

        self.sessions = list(self.clips_df.groupby('session'))

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, index):
        session_id, session_df = self.sessions[index]
        
        pos_pairs = self.generate_pairs(session_df, pair_type="positive", num_pairs=400)
        neg_pairs = self.generate_pairs(session_df, pair_type="negative", num_pairs=300)

        if len(pos_pairs) == 0 and len(neg_pairs) == 0:
            return []
        
        if (len(pos_pairs) + len(neg_pairs)) > 400:
            pos_pairs = random.sample(pos_pairs, 400 - len(neg_pairs))
        all_pairs = pos_pairs + neg_pairs

        return all_pairs

    def generate_pairs(self, session_df, pair_type, num_pairs):
        pairs = []
        att = 0
        maxa = 400
        while len(pairs) < num_pairs and att < maxa:
            anchor_window = session_df.sample(1).iloc[0]

            if pair_type == "positive":
                paired_window = self.get_positive_window(anchor_window, session_df)
                label = 1
            else:
                paired_window = self.get_negative_window(anchor_window, session_df)
                label = -1

            if paired_window is None:
                att += 1
                continue

            anchor_data = self.read_signal(anchor_window)
            paired_data = self.read_signal(paired_window)

            pairs.append((anchor_data, paired_data, label))
            att += 1

        return pairs

    def get_positive_window(self, anchor_window, session_df):
        if anchor_window['label'] == 1:
        
            prev_window = session_df[session_df['start_time'] < anchor_window['start_time']].tail(1)
            next_window = session_df[session_df['start_time'] > anchor_window['start_time']].head(1)

            if not prev_window.empty and prev_window.iloc[0]['label'] == 0:
                pos_mask_after = (session_df['start_time'] > anchor_window['start_time']) & \
                                 (session_df['label'] == 1) & \
                                 (session_df['start_time'] <= anchor_window['start_time'] + self.tau_pos)
                pos_samples_after = session_df[pos_mask_after]
                if len(pos_samples_after) > 0:
                    return pos_samples_after.sample(1).iloc[0]

        pos_mask = (session_df['start_time'] >= anchor_window['start_time'] - self.tau_pos) & \
                   (session_df['start_time'] <= anchor_window['start_time'] + self.tau_pos) & \
                   (session_df['label'] == 1)
        pos_samples = session_df[pos_mask]

        if len(pos_samples) == 0:
            return None

        return pos_samples.sample(1).iloc[0]

    def get_negative_window(self, anchor_window, session_df):
        if anchor_window['label'] == 0:
            prev_window = session_df[session_df['start_time'] < anchor_window['start_time']].tail(1)
            if not prev_window.empty and prev_window.iloc[0]['label'] == 1:
                neg_mask_before = (session_df['start_time'] < anchor_window['start_time']) & \
                                  (session_df['label'] == 1) & \
                                  (session_df['start_time'] >= anchor_window['start_time'] - self.tau_neg)
                neg_samples_before = session_df[neg_mask_before]
                if len(neg_samples_before) > 0:
                    return neg_samples_before.sample(1).iloc[0]

        neg_mask = (session_df['start_time'] < anchor_window['start_time'] - self.tau_neg) | \
                   (session_df['start_time'] > anchor_window['start_time'] + self.tau_neg)
        neg_samples = session_df[neg_mask]

        if len(neg_samples) == 0:
            return None

        return neg_samples.sample(1).iloc[0]


    def read_signal(self, window):
        if not hasattr(self, 'signal_cache'):
            self.signal_cache = {}
        signal_path = window['signals_path']
        if signal_path not in self.signal_cache:
            self.signal_cache[signal_path] = pd.read_parquet(signal_path)
        signal_df = self.signal_cache[signal_path]
        start_sample = int(window['start_time'] * self.sampling_rate)
        end_sample = int(window['end_time'] * self.sampling_rate)
        return signal_df.values[start_sample:end_sample, :].T



class RPDataset2(Dataset):
    def __init__(self, clips_df, tau_pos, tau_neg, sampling_rate=100, num_pairs=400):
        self.clips_df = clips_df
        self.tau_pos = tau_pos
        self.tau_neg = tau_neg
        self.sampling_rate = sampling_rate
        self.num_pairs = num_pairs

        self.sessions = list(self.clips_df.groupby('session'))

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, index):
        session_id, session_df = self.sessions[index]
        
        pos_pairs = self.generate_pairs(session_df, pair_type="positive", num_pairs=400)
        neg_pairs = self.generate_pairs(session_df, pair_type="negative", num_pairs=200)

        if len(pos_pairs) == 0 and len(neg_pairs) == 0:
            return []
        
        if (len(pos_pairs) + len(neg_pairs)) > 400:
            pos_pairs = random.sample(pos_pairs, 400 - len(neg_pairs))
        all_pairs = pos_pairs + neg_pairs

        return all_pairs

    def generate_pairs(self, session_df, pair_type, num_pairs):
        pairs = []
        att = 0
        maxa = 400
        while len(pairs) < num_pairs and att < maxa:
            anchor_window = session_df.sample(1).iloc[0]

            if pair_type == "positive":
                paired_window = self.get_positive_window(anchor_window, session_df)
                label = 1
            else:
                paired_window = self.get_negative_window(anchor_window, session_df)
                label = -1

            if paired_window is None:
                att += 1
                continue

            anchor_data = self.read_signal(anchor_window)
            paired_data = self.read_signal(paired_window)

            pairs.append((anchor_data, paired_data, label))
            att += 1

        return pairs

    def get_positive_window(self, anchor_window, session_df):
        if anchor_window['label'] == 0:
            prev_window = session_df[session_df['start_time'] < anchor_window['start_time']].tail(1)
            next_window = session_df[session_df['start_time'] > anchor_window['start_time']].head(1)

    
            if not prev_window.empty and not next_window.empty and next_window.iloc[0]['label'] == 1:
                neg_mask_before = (session_df['start_time'] < anchor_window['start_time']) & \
                                  (session_df['label'] == 0) & \
                                  (session_df['start_time'] >= anchor_window['start_time'] - self.tau_neg)
                neg_samples_before = session_df[neg_mask_before]
                if len(neg_samples_before) > 0:
                    return neg_samples_before.sample(1).iloc[0]

        pos_mask = (session_df['start_time'] >= anchor_window['start_time'] - self.tau_pos) & \
                   (session_df['start_time'] <= anchor_window['start_time'] + self.tau_pos) & \
                   (session_df['label'] == 1)
        pos_samples = session_df[pos_mask]

        if len(pos_samples) == 0:
            return None

        return pos_samples.sample(1).iloc[0]

    def get_negative_window(self, anchor_window, session_df):
        if anchor_window['label'] == 1:
            prev_window = session_df[session_df['start_time'] < anchor_window['start_time']].tail(1)
            next_window = session_df[session_df['start_time'] > anchor_window['start_time']].head(1)

            if not prev_window.empty and  not next_window.empty and next_window.iloc[0]['label'] == 0:
                pos_mask_after = (session_df['start_time'] > anchor_window['start_time']) & \
                                 (session_df['label'] == 0) & \
                                 (session_df['start_time'] <= anchor_window['start_time'] + self.tau_pos)
                pos_samples_after = session_df[pos_mask_after]
                if len(pos_samples_after) > 0:
                    return pos_samples_after.sample(1).iloc[0]

        neg_mask = (session_df['start_time'] < anchor_window['start_time'] - self.tau_neg) | \
                   (session_df['start_time'] > anchor_window['start_time'] + self.tau_neg)
        neg_samples = session_df[neg_mask]

        if len(neg_samples) == 0:
            return None

        return neg_samples.sample(1).iloc[0]

    def read_signal(self, window):
        if not hasattr(self, 'signal_cache'):
            self.signal_cache = {}
        signal_path = window['signals_path']
        if signal_path not in self.signal_cache:
            self.signal_cache[signal_path] = pd.read_parquet(signal_path)
        signal_df = self.signal_cache[signal_path]
        start_sample = int(window['start_time'] * self.sampling_rate)
        end_sample = int(window['end_time'] * self.sampling_rate)
        return signal_df.values[start_sample:end_sample, :].T



class RPDataset3(Dataset):
    def __init__(self, clips_df, tau_pos, tau_neg, sampling_rate=100, num_pairs=400):
        self.clips_df = clips_df
        self.tau_pos = tau_pos
        self.tau_neg = tau_neg
        self.sampling_rate = sampling_rate
        self.num_pairs = num_pairs

        self.sessions = list(self.clips_df.groupby('session'))

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, index):
        session_id, session_df = self.sessions[index]
        
        pos_pairs = self.generate_pairs(session_df, pair_type="positive", num_pairs=400)
        neg_pairs = self.generate_pairs(session_df, pair_type="negative", num_pairs=200)

        if len(pos_pairs) == 0 and len(neg_pairs) == 0:
            return []
        
        if (len(pos_pairs) + len(neg_pairs)) > 400:
            pos_pairs = random.sample(pos_pairs, 400 - len(neg_pairs))
        all_pairs = pos_pairs + neg_pairs

        return all_pairs

    def generate_pairs(self, session_df, pair_type, num_pairs):
        pairs = []
        att = 0
        maxa = 400
        while len(pairs) < num_pairs and att < maxa:
            anchor_window = session_df.sample(1).iloc[0]

            if pair_type == "positive":
                paired_window = self.get_positive_window(anchor_window, session_df)
                label = 1
            else:
                paired_window = self.get_negative_window(anchor_window, session_df)
                label = -1

            if paired_window is None:
                att += 1
                continue

            anchor_data = self.read_signal(anchor_window)
            paired_data = self.read_signal(paired_window)

            pairs.append((anchor_data, paired_data, label))
            att += 1

        return pairs

    def get_positive_window(self, anchor_window, session_df):
 
        if anchor_window['label'] == 1:
            pos_mask = (session_df['start_time'] >= anchor_window['start_time'] - self.tau_pos) & \
                       (session_df['start_time'] <= anchor_window['start_time'] + self.tau_pos) & \
                       (session_df['label'] == 1)
        else:
            pos_mask = (session_df['start_time'] >= anchor_window['start_time'] - self.tau_pos) & \
                       (session_df['start_time'] <= anchor_window['start_time'] + self.tau_pos) & \
                       (session_df['label'] == 0)

        pos_samples = session_df[pos_mask]

        if len(pos_samples) == 0:
            return None

        return pos_samples.sample(1).iloc[0]

    def get_negative_window(self, anchor_window, session_df):
        if anchor_window['label'] == 1:
            neg_mask = (session_df['start_time'] >= anchor_window['start_time'] - self.tau_neg) & \
                       (session_df['start_time'] <= anchor_window['start_time'] + self.tau_neg) & \
                       (session_df['label'] == 0)
        else:
            neg_mask = (session_df['start_time'] >= anchor_window['start_time'] - self.tau_neg) & \
                       (session_df['start_time'] <= anchor_window['start_time'] + self.tau_neg) & \
                       (session_df['label'] == 1)

        neg_samples = session_df[neg_mask]

        if len(neg_samples) == 0:
            return None

        return neg_samples.sample(1).iloc[0]

    def read_signal(self, window):
        if not hasattr(self, 'signal_cache'):
            self.signal_cache = {}
        signal_path = window['signals_path']
        if signal_path not in self.signal_cache:
            self.signal_cache[signal_path] = pd.read_parquet(signal_path)
        signal_df = self.signal_cache[signal_path]
        start_sample = int(window['start_time'] * self.sampling_rate)
        end_sample = int(window['end_time'] * self.sampling_rate)
        return signal_df.values[start_sample:end_sample, :].T


class RPDataset4(Dataset):
    def __init__(self, clips_df, tau_pos, tau_neg, sampling_rate=100, num_pairs=400):
        self.clips_df = clips_df
        self.sampling_rate = sampling_rate
        self.num_pairs = num_pairs
        self.sessions = list(self.clips_df.groupby('session'))

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, index):
        session_id, session_df = self.sessions[index]
        
        pos_pairs = self.generate_pairs(session_df, pair_type="positive", num_pairs=400)
        neg_pairs = self.generate_pairs(session_df, pair_type="negative", num_pairs=200)

        if len(pos_pairs) == 0 and len(neg_pairs) == 0:
            return []
        
        if (len(pos_pairs) + len(neg_pairs)) > 400:
            pos_pairs = random.sample(pos_pairs, 400 - len(neg_pairs))
        all_pairs = pos_pairs + neg_pairs

        return all_pairs

    def generate_pairs(self, session_df, pair_type, num_pairs):
        pairs = []
        att = 0
        maxa = 400
        while len(pairs) < num_pairs and att < maxa:
            anchor_window = session_df.sample(1).iloc[0]

            if pair_type == "positive":
                paired_window = self.get_positive_window(anchor_window, session_df)
                label = 1
            else:
                paired_window = self.get_negative_window(anchor_window, session_df)
                label = -1

            if paired_window is None:
                att += 1
                continue

            anchor_data = self.read_signal(anchor_window)
            paired_data = self.read_signal(paired_window)

            pairs.append((anchor_data, paired_data, label))
            att += 1

        return pairs

    def get_positive_window(self, anchor_window, session_df):

        if anchor_window['label'] == 1:
            pos_mask = session_df['label'] == 1

        else:
            pos_mask = session_df['label'] == 0

        pos_samples = session_df[pos_mask]

        if len(pos_samples) == 0:
            return None

        return pos_samples.sample(1).iloc[0]

    def get_negative_window(self, anchor_window, session_df):

        if anchor_window['label'] == 1:
            neg_mask = session_df['label'] == 0
        else:
            neg_mask = session_df['label'] == 1

        neg_samples = session_df[neg_mask]

        if len(neg_samples) == 0:
            return None

        return neg_samples.sample(1).iloc[0]

    def read_signal(self, window):
        if not hasattr(self, 'signal_cache'):
            self.signal_cache = {}
        signal_path = window['signals_path']
        if signal_path not in self.signal_cache:
            self.signal_cache[signal_path] = pd.read_parquet(signal_path)
        signal_df = self.signal_cache[signal_path]
        start_sample = int(window['start_time'] * self.sampling_rate)
        end_sample = int(window['end_time'] * self.sampling_rate)
        return signal_df.values[start_sample:end_sample, :].T




class RPDataset5(Dataset):
    def __init__(self, clips_df, tau_pos, tau_neg, sampling_rate=100):
        self.clips_df = clips_df
        self.tau_pos = tau_pos
        self.tau_neg = tau_neg
        self.sampling_rate = sampling_rate

        self.sessions = list(self.clips_df.groupby('session'))

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, index):
        session_id, session_df = self.sessions[index]
        all_pairs = []
        anchor_label_0_pairs = []
        anchor_label_1_pairs = []

 
        for _, anchor_window in session_df.iterrows():
            pos_pair = self.generate_pair(session_df, anchor_window, pair_type="positive")
            neg_pair = self.generate_pair(session_df, anchor_window, pair_type="negative")

 
            if pos_pair is None or neg_pair is None:
                continue
            
   
            anchor_label = anchor_window['label']
            all_pairs.append((*pos_pair, anchor_label)) 
            all_pairs.append((*neg_pair, anchor_label))  

  
            if anchor_label == 0:
                anchor_label_0_pairs.append((*pos_pair, anchor_label))
                anchor_label_0_pairs.append((*neg_pair, anchor_label))
            else:
                anchor_label_1_pairs.append((*pos_pair, anchor_label))
                anchor_label_1_pairs.append((*neg_pair, anchor_label))

   
        if len(anchor_label_0_pairs) > len(anchor_label_1_pairs):
            diff = len(anchor_label_0_pairs) - len(anchor_label_1_pairs)

            sampled_pairs = random.choices(anchor_label_1_pairs, k=diff)
            all_pairs.extend(sampled_pairs)
  
        return all_pairs if len(all_pairs) > 0 else []

    def generate_pair(self, session_df, anchor_window, pair_type):
        if pair_type == "positive":
            paired_window = self.get_positive_window(anchor_window, session_df)
            label = 1
        else:
            paired_window = self.get_negative_window(anchor_window, session_df)
            label = -1

        if paired_window is not None:
            anchor_data = self.read_signal(anchor_window)
            paired_data = self.read_signal(paired_window)
            return (anchor_data, paired_data, label)
        return None

    def get_positive_window(self, anchor_window, session_df):
        if anchor_window['label'] == 1:
            pos_mask = (session_df['start_time'] >= anchor_window['start_time'] - self.tau_pos) & \
                       (session_df['start_time'] <= anchor_window['start_time'] + self.tau_pos) & \
                       (session_df['label'] == 1)
        else:
            pos_mask = (session_df['start_time'] >= anchor_window['start_time'] - self.tau_pos) & \
                       (session_df['start_time'] <= anchor_window['start_time'] + self.tau_pos) & \
                       (session_df['label'] == 0)

        pos_samples = session_df[pos_mask]

        if len(pos_samples) == 0:
            return None

        return pos_samples.sample(1).iloc[0]

    def get_negative_window(self, anchor_window, session_df):
        if anchor_window['label'] == 1:
            neg_mask = (session_df['start_time'] >= anchor_window['start_time'] - self.tau_neg) & \
                       (session_df['start_time'] <= anchor_window['start_time'] + self.tau_neg) & \
                       (session_df['label'] == 0)
        else:
            neg_mask = (session_df['start_time'] >= anchor_window['start_time'] - self.tau_neg) & \
                       (session_df['start_time'] <= anchor_window['start_time'] + self.tau_neg) & \
                       (session_df['label'] == 1)

        neg_samples = session_df[neg_mask]

        if len(neg_samples) == 0:
            return None

        return neg_samples.sample(1).iloc[0]

    def read_signal(self, window):
        if not hasattr(self, 'signal_cache'):
            self.signal_cache = {}
        signal_path = window['signals_path']
        if signal_path not in self.signal_cache:
            self.signal_cache[signal_path] = pd.read_parquet(signal_path)
        signal_df = self.signal_cache[signal_path]
        start_sample = int(window['start_time'] * self.sampling_rate)
        end_sample = int(window['end_time'] * self.sampling_rate)
        return signal_df.values[start_sample:end_sample, :].T



class RPDataset6(Dataset):
    def __init__(self, clips_df, tau_pos, tau_neg, sampling_rate=100, num_pairs=400):
        self.clips_df = clips_df
        self.tau_pos = tau_pos
        self.tau_neg = tau_neg
        self.sampling_rate = sampling_rate
        self.num_pairs = num_pairs

        self.sessions = list(self.clips_df.groupby('session'))

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, index):
        session_id, session_df = self.sessions[index]
        
        pos_pairs = self.generate_pairs(session_df, pair_type="positive", num_pairs=400)
        neg_pairs = self.generate_pairs(session_df, pair_type="negative", num_pairs=200)

        if len(pos_pairs) == 0 and len(neg_pairs) == 0:
            return []
        
        if (len(pos_pairs) + len(neg_pairs)) > 400:
            pos_pairs = random.sample(pos_pairs, 400 - len(neg_pairs))
        all_pairs = pos_pairs + neg_pairs

        return all_pairs

    def generate_pairs(self, session_df, pair_type, num_pairs):
        pairs = []
        att = 0
        maxa = 400
        while len(pairs) < num_pairs and att < maxa:
            anchor_window = session_df.sample(1).iloc[0]

            if pair_type == "positive":
                paired_window = self.get_positive_window(anchor_window, session_df)
                label = 1
            else:
                paired_window = self.get_negative_window(anchor_window, session_df)
                label = -1

            if paired_window is None:
                att += 1
                continue

            anchor_data = self.read_signal(anchor_window)
            paired_data = self.read_signal(paired_window)
            alabels = anchor_window['label']

            pairs.append((anchor_data, paired_data, label, alabels))
            att += 1

        return pairs

    def get_positive_window(self, anchor_window, session_df):
 
        if anchor_window['label'] == 1:
            pos_mask = (session_df['start_time'] >= anchor_window['start_time'] - self.tau_pos) & \
                       (session_df['start_time'] <= anchor_window['start_time'] + self.tau_pos) & \
                       (session_df['label'] == 1)
        else:
            pos_mask = (session_df['start_time'] >= anchor_window['start_time'] - self.tau_pos) & \
                       (session_df['start_time'] <= anchor_window['start_time'] + self.tau_pos) & \
                       (session_df['label'] == 0)

        pos_samples = session_df[pos_mask]

        if len(pos_samples) == 0:
            return None

        return pos_samples.sample(1).iloc[0]

    def get_negative_window(self, anchor_window, session_df):
        if anchor_window['label'] == 1:
            neg_mask = (session_df['start_time'] >= anchor_window['start_time'] - self.tau_neg) & \
                       (session_df['start_time'] <= anchor_window['start_time'] + self.tau_neg) & \
                       (session_df['label'] == 0)
        else:
            neg_mask = (session_df['start_time'] >= anchor_window['start_time'] - self.tau_neg) & \
                       (session_df['start_time'] <= anchor_window['start_time'] + self.tau_neg) & \
                       (session_df['label'] == 1)

        neg_samples = session_df[neg_mask]

        if len(neg_samples) == 0:
            return None

        return neg_samples.sample(1).iloc[0]

    def read_signal(self, window):
        if not hasattr(self, 'signal_cache'):
            self.signal_cache = {}
        signal_path = window['signals_path']
        if signal_path not in self.signal_cache:
            self.signal_cache[signal_path] = pd.read_parquet(signal_path)
        signal_df = self.signal_cache[signal_path]
        start_sample = int(window['start_time'] * self.sampling_rate)
        end_sample = int(window['end_time'] * self.sampling_rate)
        return signal_df.values[start_sample:end_sample, :].T


class tifrc(Dataset):
    def __init__(self, clips_df):
        self.clips_df = clips_df
        self.sessions = list(self.clips_df.groupby('session'))


    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, index):
        for _, anchor_window in session_df.iterrows():
            pos = self.read_signal(anchor_window)
            pos = abs(np.fft.fft(pos))
            neg_pairs = self.get_positive_window(anchor_window, session_df)
            paired_data = self.read_signal(paired_window)

            paired_data = self.read_signal(paired_window)

            pairs.append((anchor_data, pos, paired_data))

        return pairs

    def get_negative_window(self, anchor_window, session_df):
        neg_mask = (session_df['start_time'] < anchor_window['start_time'] - self.tau_neg) | \
                   (session_df['start_time'] > anchor_window['start_time'] + self.tau_neg)
        neg_samples = session_df[neg_mask]

        if len(neg_samples) == 0:
            return None

        return neg_samples.sample(1).iloc[0]

    def read_signal(self, window):
        if not hasattr(self, 'signal_cache'):
            self.signal_cache = {}
        signal_path = window['signals_path']
        if signal_path not in self.signal_cache:
            self.signal_cache[signal_path] = pd.read_parquet(signal_path)
        signal_df = self.signal_cache[signal_path]
        start_sample = int(window['start_time'] * self.sampling_rate)
        end_sample = int(window['end_time'] * self.sampling_rate)
        return signal_df.values[start_sample:end_sample, :].T


def process_pairs(pairs, fft_points=400, sampling_rate=100, freq_range=(0.5, 40), target_length=600):

    with open('./data/adj_mx_3d.pkl', 'rb') as pf:
        adj_mx_data = pickle.load(pf)

    adj_mx = adj_mx_data[2] 
    adj_mx = torch.tensor(adj_mx, dtype=torch.float32)

    edge_index, edge_weight = dense_to_sparse(adj_mx)
    freq_res = sampling_rate / fft_points
    freq_start = int(freq_range[0] / freq_res)
    freq_end = int(freq_range[1] / freq_res)
    
    data_list = []
    data_list1 = []
    for data in pairs:
        anchor_data, paired_data, label = zip(*data)
        anchor_data = np.array(anchor_data)
        paired_data = np.array(paired_data)

        if anchor_data.shape[1] < target_length:
            pad_size = target_length - anchor_data.shape[1]
            anchor_data = torch.nn.functional.pad(anchor_data, (0, pad_size))
        if paired_data.shape[1] < target_length:
            pad_size = target_length - paired_data.shape[1]
            paired_data = torch.nn.functional.pad(paired_data, (0, pad_size))
        

        anchor_fft = torch.fft.fft(anchor_data, n=fft_points, dim=-1)
        paired_fft = torch.fft.fft(paired_data, n=fft_points, dim=-1)

        anchor_fft = torch.abs(anchor_fft)[:, freq_start:freq_end]
        paired_fft = torch.abs(paired_fft)[:, freq_start:freq_end]

        anchor_graph = Data(
            x=anchor_fft,
            edge_index=edge_index,
            edge_attr=edge_weight,
            y=torch.tensor([label])
        )
        
        paired_graph = Data(
            x=paired_fft,
            edge_index=edge_index,
            edge_attr=edge_weight,
            y=torch.tensor([label])
        )
        
        data_list.append(anchor_graph)
        data_list1.append(paired_graph)
    
    return data_list, data_list1

def process_sin(pairs, fft_points=400, sampling_rate=100, freq_range=(0.5, 40), target_length=600):
    with open('./data/adj_mx_3d.pkl', 'rb') as pf:
        adj_mx_data = pickle.load(pf)

    adj_mx = adj_mx_data[2] 
    adj_mx = torch.tensor(adj_mx, dtype=torch.float32)

    edge_index, edge_weight = dense_to_sparse(adj_mx)
    freq_res = sampling_rate / fft_points
    freq_start = int(freq_range[0] / freq_res)
    freq_end = int(freq_range[1] / freq_res)
    
    data_list = []
    for window, label in pairs:
        signal_data = window
        if signal_data.shape[1] < target_length:
            pad_size = target_length - signal_data.shape[1]
            signal_data = torch.nn.functional.pad(signal_data, (0, pad_size)) 

        signal_fft = torch.fft.fft(signal_data, n=fft_points, dim=-1)
        signal_fft = torch.abs(signal_fft)[:, freq_start:freq_end]

        graph = Data(
            x=signal_fft,
            edge_index=edge_index,
            edge_attr=edge_weight,
            y=torch.tensor([label])
        )
        
        data_list.append(graph)
    return data_list

class APN(Dataset):
    def __init__(self, clips_df, tau_pos, tau_neg, sampling_rate=100, num_pairs=400):
        self.clips_df = clips_df
        self.tau_pos = tau_pos
        self.tau_neg = tau_neg
        self.sampling_rate = sampling_rate
        self.num_pairs = num_pairs

    
        self.sessions = list(self.clips_df.groupby('session'))

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, index):
    
        session_id, session_df = self.sessions[index]
        

        pos_pairs = self.generate_pairs(session_df, num_pairs=200)
  

        if len(pos_pairs) <200 or len(neg_pairs) < 200:
            #print(f"Skipping session {session_id}, not enough pairs")
            return []
        #print(f"Generated {len(pos_pairs)} positive pairs and {len(neg_pairs)} negative pairs")
        # if (len(pos_pairs)+len(neg_pairs)) > 400:
        #     if (len(neg_pairs) > 200 and len(pos_pairs) < 200):
        #         neg_pairs = random.sample(neg_pairs, 400-len(pos_pairs))
        #     elif (len(neg_pairs) > 200 and len(pos_pairs) > 200):
        #         pos_pairs = random.sample(pos_pairs, 200)
        #         neg_pairs = random.sample(neg_pairs, 200)
        #     else:
        #         pos_pairs = random.sample(pos_pairs, 400-len(neg_pairs))
                
        all_pairs = pos_pairs + neg_pairs
        #random.shuffle(all_pairs)

        return all_pairs

    def generate_pairs(self, session_df, num_pairs):
        pairs = []
        att = 0
        maxa = 400
        while len(pairs) < num_pairs and att < maxa:
            anchor_window = session_df.sample(1).iloc[0]


            pos_window = self.get_positive_window(anchor_window, session_df)
            label = 1 
            neg_window = self.get_negative_window(anchor_window, session_df)
            label1 = -1  

            if paired_window is None:
                att += 1
                continue

            anchor_data = self.read_signal(anchor_window)
            pos_data = self.read_signal(pos_window)
            neg_data = self.read_signal(neg_window)

            pairs.append((anchor_data, pos_data, label, neg_data, label1))
            att += 1

        return pairs

    def get_positive_window(self, anchor_window, session_df):
        pos_mask = (session_df['start_time'] >= anchor_window['start_time'] - self.tau_pos) & \
                   (session_df['start_time'] <= anchor_window['start_time'] + self.tau_pos)
        pos_samples = session_df[pos_mask]
        
        if len(pos_samples) == 0:
            return None


        return pos_samples.sample(1).iloc[0]

    def get_negative_window(self, anchor_window, session_df):
        neg_mask = (session_df['start_time'] < anchor_window['start_time'] - self.tau_neg) | \
                   (session_df['start_time'] > anchor_window['start_time'] + self.tau_neg)
        neg_samples = session_df[neg_mask]

        if len(neg_samples) == 0:
            return None

        return neg_samples.sample(1).iloc[0]


    def read_signal(self, window):
        if not hasattr(self, 'signal_cache'):
            self.signal_cache = {}
        signal_path = window['signals_path']
        if signal_path not in self.signal_cache:
            self.signal_cache[signal_path] = pd.read_parquet(signal_path)
        signal_df = self.signal_cache[signal_path]
        start_sample = int(window['start_time'] * self.sampling_rate)
        end_sample = int(window['end_time'] * self.sampling_rate)
        return signal_df.values[start_sample:end_sample, :].T

def collate_fnD(batch):
    fixed_len = 600 

    anchor_data_list = []
    paired_data_list = []
    neg_data_list = []
    label_list = []
    label1_list = []
 

    for pairs in batch:
        for anchor_data, pos_data, label, neg_data, label1 in pairs:
            anchor_data = torch.tensor(anchor_data)
            pos_data = torch.tensor(pos_data)
            neg_data = torch.tensor(neg_data)
            
            anchor_data_padded = pad_to_fixed_len(anchor_data, fixed_len)
            pos_data_padded = pad_to_fixed_len(pos_data, fixed_len)
            neg_data_padded = pad_to_fixed_len(neg_data, fixed_len)

            anchor_data_list.append(anchor_data_padded)
            paired_data_list.append(paired_data_padded)
            neg_data_list.append(neg_data_padded)

            label_list.append(label)
            label1_list.append(label1)
  


        return list(zip(anchor_data_list, pos_data_list, label_list, neg_data_list, label1_list))