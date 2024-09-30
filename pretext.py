from torch.utils.data import DataLoader, Dataset
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import torch
import random
import pandas as pd

class RPDataset(Dataset):
    def __init__(self, clips_df, tau_pos, tau_neg, sampling_rate=100, num_pairs=200):
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




class CPCDataset(Dataset):
    def __init__(self, clips_df, N_c, N_p, N_b, sampling_rate=100):
        """
        :param N_c: Number of context windows.
        :param N_p: Number of future windows to predict.
        :param N_b: Number of negative windows to sample
        """
        self.clips_df = clips_df
        self.N_c = N_c
        self.N_p = N_p
        self.N_b = N_b
        self.sampling_rate = sampling_rate
        self.sessions = list(self.clips_df.groupby('session'))  # Group by session

    def __len__(self):
        return len(self.sessions)  # Each session is treated as a sequence

    def __getitem__(self, index):
        session_id, session_df = self.sessions[index]
        
        # Sample context windows
        context_windows = self.sample_windows(session_df, num_windows=self.N_c)

        # Sample future windows (those that follow the context)
        future_windows = self.sample_windows(session_df, num_windows=self.N_p, after=context_windows[-1])

        # Sample negative windows (random windows not related to the context or future)
        negative_windows = self.sample_negative_windows(session_df, num_windows=self.N_b, exclude=context_windows + future_windows)

        # Get the corresponding signal data for each sampled window
        context_data = [self.read_signal(window) for window in context_windows]
        future_data = [self.read_signal(window) for window in future_windows]
        negative_data = [self.read_signal(window) for window in negative_windows]

        return torch.stack(context_data), torch.stack(future_data), torch.stack(negative_data)

    def sample_windows(self, session_df, num_windows, after=None):
        if after is not None:
            pos_mask = session_df['start_time'] > after['end_time']
            valid_windows = session_df[pos_mask]
        else:
            valid_windows = session_df
        
        sampled_windows = valid_windows.sample(num_windows, replace=False)
        return sampled_windows

    def sample_negative_windows(self, session_df, num_windows, exclude):
        exclude_start_times = [window['start_time'] for window in exclude]
        neg_mask = ~session_df['start_time'].isin(exclude_start_times)
        negative_windows = session_df[neg_mask].sample(num_windows, replace=False)
        return negative_windows

    def read_signal(self, window):
        signal_df = pd.read_parquet(window['signals_path'])
        start_sample = int(window['start_time'] * self.sampling_rate)
        end_sample = int(window['end_time'] * self.sampling_rate)


        if signal_data.shape[0] < self.fixed_len:
            pad_len = self.fixed_len - signal_data.shape[0]
            signal_data = torch.cat([torch.tensor(signal_data), torch.zeros(pad_len, signal_data.shape[1])], dim=0)
        else:
            signal_data = torch.tensor(signal_data[:self.fixed_len, :]) 
        return signal_df.values[start_sample:end_sample, :].T

