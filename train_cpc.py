import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from rp import clips
import os
import shutil
from tqdm import tqdm
from clean import clear_directory
from shallow import Shallow, ContrastiveNet
from loss import RelativePositioningLossm
from pretext import RPDataset, collate_fn, split_dataset, LabelDataset, taset, collate_fnt, CPCDataset
from preprocess import remove_short_segments



def train_cpc_model(train_loader, test_loader, model, optimizer, criterion, epochs, threshold):
    model.train()
    device = 'cuda:2'
    floss = float('inf')
    loss_history = []
    tloss = []
    for epoch in range(epochs):
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (context_seq, target_seq) in enumerate(progress_bar):
            optimizer.zero_grad()
            context_seq = context_seq.to(device)
            target_seq = target_seq.to(device)
            
            optimizer.zero_grad()

            output, target_encoding = model(context_seq, target_seq)

            loss = criterion(output, target_encoding)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

            closs = running_loss / len(train_loader)
            loss_history.append(closs)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {closs}")

        closs = running_loss / len(train_loader)
        loss_history.append(closs)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {closs}")

        test_loss = evaluate_model(test_loader, model)
        tloss.append(test_loss)
        print(f"Test Loss: {test_loss}")

        # if abs(closs-floss) < threshold:
        #     print(f"Early stopping at epoch {epoch + 1} with test loss {test}")
        #     break
        floss = min(closs, floss)
        torch.save(model.state_dict(), os.path.join("./result/", f"model_epoch_{epoch + 1}.pt"))

    df = pd.DataFrame({
        "Epoch": range(1, len(loss_history) + 1),
        "Loss": loss_history
    })
    df.to_excel('./results/loss_history.xlsx', index=False)


    df = pd.DataFrame({
        "Epoch": range(1, len(tloss) + 1),
        "Loss": loss_history
    })
    df.to_excel('./results/tloss_history.xlsx', index=False)



def evaluate_model(test_loader, model):
    device = 'cuda:2'
    model.eval()  
    tbar = tqdm(test_loader, desc="Evaluation")  
    total_correct = 0  
    total_samples = 0  
    all_preds = []  
    all_labels = []  
    
    with torch.no_grad(): 
        for batch_idx, batch_data in enumerate(tbar):
            x1_batch, labels = zip(*batch_data)
            # x1_batch = torch.stack([torch.tensor(x).float() for x in x1_batch]).to(device)
            # labels = torch.tensor(labels).float().to(device)

            x1_batch = torch.stack([x.clone().detach().float() for x in x1_batch]).to(device)
            labels = torch.tensor(labels)
            labels = labels.clone().detach().float().to(device)

            x1_batch = x1_batch.unsqueeze(1) 

            outputs = model.emb_net(x1_batch)

            preds = (outputs >= 0).float()

            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = total_correct / total_samples
    
    print(f"Test Accuracy: {accuracy}")
    
    return accuracy




# folder_to_clear = './data/train'
# clear_directory(folder_to_clear)
CUDA_VISIBLE_DEVICES=2
#df = pd.read_parquet("/datasets2/epilepsy/TUSZ/processed/train/segments.parquet")
output_dir = "./data/train"
sampling_rate=250
target_sampling_rate=100
lowpass_freq=50
sfre = 100
clip_length = 6
clip_stride = 6
#all_clips_df = clips(df, sampling_rate, target_sampling_rate, output_dir, lowpass_freq,clip_length, clip_stride)
#all_clips_df.to_parquet('./data/processed.parquet', engine='pyarrow')
all_clips_df = pd.read_parquet('./data/processed.parquet')
all_clips_df = remove_short_segments(all_clips_df, 6)
df = all_clips_df
label_1_count = df[df['label'] == 1].shape[0]
label_0_count = df[df['label'] == 0].shape[0]
print(f"Label 1 count: {label_1_count}")
print(f"Label 0 count: {label_0_count}")
#exit()

device = 'cuda:2'

emb_size = 100
emb = Shallow(1, 40)
model = CPCModel(emb_size=100, num_layers=1).to(device)
#criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
criterion = RelativePositioningLossm(emb_size).to(device)
#print(df.head())
train_df, test_df = split_dataset(all_clips_df)
session_count = train_df['session'].nunique()
#print(f"Number of unique sessions: {session_count}")
train_dataset = CPCDataset(train_df, N_c, N_p,N_b)
#print(train_dataset[0])
test_dataset = taset(test_df)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,collate_fn=collate_fn, num_workers=8)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,collate_fn = collate_fnt, num_workers=0)

train_model(train_loader, test_loader, model, optimizer, criterion, epochs=500, threshold=0.01)
model = CPCModel(emb_size=100, num_layers=1).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)