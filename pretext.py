from torch.utils.data import DataLoader, Dataset
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import torch
import random
import pandas as pd
import numpy as np
from torch.utils.data._utils.collate import default_collate

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



def balance_dataframe(clips_df):

    label_0_df = clips_df[clips_df['label'] == 0]
    label_1_df = clips_df[clips_df['label'] == 1]

    if len(label_1_df) == 0 or len(label_0_df) == 0:
        return clips_df  

    label_1_upsampled = label_1_df.sample(len(label_0_df), replace=True)

    balanced_df = pd.concat([label_0_df, label_1_upsampled]).reset_index(drop=True)

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




