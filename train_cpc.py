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
from cpc_shallow import Shallow, CPCModel
from loss import RelativePositioningLossm
from pretext import RPDataset, collate_fn, split_dataset, LabelDataset, taset, collate_fnt, CPCDataset, collate_fnc
from preprocess import remove_short_segments
from loss import SimpleCPCLoss



def train_cpc_model(train_loader, test_loader, model, optimizer, criterion, epochs, threshold):
    model.train()
    device = 'cuda:2'
    floss = float('inf')
    loss_history = []
    tloss = []

    for epoch in range(epochs):
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

        for batch_idx, (context_seq, target_seq, negative_seq) in enumerate(progress_bar):
            context_seq = context_seq.to(device)
            target_seq = target_seq.to(device)
            negative_seq = negative_seq.to(device)

            optimizer.zero_grad()

            # context_seq = context_seq.unsqueeze(0) 
            # target_seq =  target_seq.unsqueeze(0)
            # negative_seq =  negative_seq.unsqueeze(0)

            # Forward pass through the model
            contrastive_output = model(context_seq, target_seq, negative_seq)

            # Calculate CPC contrastive loss
            loss = criterion(contrastive_output, target_seq, negative_seq)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

        closs = running_loss / len(train_loader)
        loss_history.append(closs)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {closs}")

        test_loss = evaluate_model(test_loader, model)
        tloss.append(test_loss)
        print(f"Test Loss: {test_loss}")

        #floss = min(closs, floss)

    # Save the model and losses
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_folder = os.path.join("./result/CPC/", timestamp)
    os.makedirs(result_folder, exist_ok=True)

    model_save_path = os.path.join(result_folder, f"model_RP.pt")
    torch.save(model.state_dict(), model_save_path)

    df = pd.DataFrame({
        "Epoch": range(1, len(loss_history) + 1),
        "Loss": loss_history
    })
    loss_file_path = os.path.join(result_folder, 'trainloss.xlsx')
    df.to_excel(loss_file_path, index=False)

    df = pd.DataFrame({
        "Epoch": range(1, len(tloss) + 1),
        "Loss": tloss
    })
    loss_file_path = os.path.join(result_folder, 'testloss.xlsx')
    df.to_excel(loss_file_path, index=False)




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



Nc = 5
Np = 1
Nb = 10

device = 'cuda:2'

emb_size = 100
emb = Shallow(1, 40)
model = CPCModel(encoder=emb, emb_size=100, num_layers=1).to(device)
#criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
criterion = SimpleCPCLoss(Np, Nb).to(device)
#print(df.head())
train_df, test_df = split_dataset(all_clips_df)
session_count = train_df['session'].nunique()
#print(f"Number of unique sessions: {session_count}")
train_dataset = CPCDataset(train_df, Nc, Np, Nb)
#print(train_dataset[0])
test_dataset = taset(test_df)
train_loader = DataLoader(train_dataset, batch_size=16, collate_fn=collate_fnc, shuffle=True, num_workers=8)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

train_cpc_model(train_loader, test_loader, model, optimizer, criterion, epochs=500, threshold=0.01)
#criterion = nn.CrossEntropyLoss()
