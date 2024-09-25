from torch.utils.data import DataLoader, Dataset
import pandas as pd
import random


import random
import pandas as pd

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
        neg_pairs = self.generate_pairs(session_df, pair_type="negative", num_pairs=400)

        if len(pos_pairs) == 0 or len(neg_pairs) == 0:
            return []

        all_pairs = pos_pairs + neg_pairs
        random.shuffle(all_pairs)

        return all_pairs

    def generate_pairs(self, session_df, pair_type, num_pairs):
        pairs = []
        att = 0
        maxa = 800
        while len(pairs) < num_pairs and att < maxa:
            anchor_window = session_df.sample(1).iloc[0]

            if pair_type == "positive":
                paired_window = self.get_positive_window(anchor_window, session_df)
                label = 1 
            else:
                paired_window = self.get_negative_window(anchor_window, session_df)
                label = 0  

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

    def read_signal(self, window):
        signal_df = pd.read_parquet(window['signals_path'])
        start_sample = int(window['start_time'] * self.sampling_rate)
        end_sample = int(window['end_time'] * self.sampling_rate)
        return signal_df.values[:, start_sample:end_sample]



    # def read_signal(self, window):
    #     signal_df = pd.read_parquet(window['signals_path'])
    #     total_samples = signal_df.shape[1] 
    #     start_sample = max(0, int(window['start_time'] * self.sampling_rate))
    #     end_sample = min(total_samples, int(window['end_time'] * self.sampling_rate))
    #     return signal_df.values[:, start_sample:end_sample].T 


def collate_fn(batch):
    anchor_pos, anchor_neg = [], []
    for pos_pair, neg_pair in batch:
        anchor_pos.append(pos_pair)
        anchor_neg.append(neg_pair)
    return anchor_pos, anchor_neg



def split_dataset_by_session(clips_df, test_size=0.2, random_state=42):
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
        return signal_data, label

    def read_signal(self, window):
        signal_df = pd.read_parquet(window['signals_path'])
        start_sample = int(window['start_time'] * self.sampling_rate)
        end_sample = int(window['end_time'] * self.sampling_rate)
        return signal_df.values[:, start_sample:end_sample]