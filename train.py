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
from loss import RelativePositioningLoss
from pretext import RPDataset, collate_fn, split_dataset, LabelDataset




# def train_model(train_loader, test_loader, model, optimizer, criterion, epochs, threshold):
#     model.train()
#     for epoch in range(epochs):
#         running_loss = 0.0
#         progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
#         for batch_idx, (anchor_pos_batch, anchor_neg_batch) in enumerate(progress_bar):
            
#             anchor_pos_x1, anchor_pos_x2, pos_labels = zip(*anchor_pos_batch)
#             anchor_neg_x1, anchor_neg_x2, neg_labels = zip(*anchor_neg_batch)
            
            
#             anchor_pos_x1 = [torch.tensor(x) if not isinstance(x, torch.Tensor) else x for x in anchor_pos_x1]
#             anchor_neg_x1 = [torch.tensor(x) if not isinstance(x, torch.Tensor) else x for x in anchor_neg_x1]
#             anchor_pos_x2 = [torch.tensor(x) if not isinstance(x, torch.Tensor) else x for x in anchor_pos_x2]
#             anchor_neg_x2 = [torch.tensor(x) if not isinstance(x, torch.Tensor) else x for x in anchor_neg_x2]
#             pos_labels = [torch.tensor(x) if not isinstance(x, torch.Tensor) else x for x in pos_labels]
#             neg_labels = [torch.tensor(x) if not isinstance(x, torch.Tensor) else x for x in neg_labels]

           
#             x1 = torch.cat([torch.stack(anchor_pos_x1), torch.stack(anchor_neg_x1)]).float().cuda(2)
#             x2 = torch.cat([torch.stack(anchor_pos_x2), torch.stack(anchor_neg_x2)]).float().cuda(2)
#             labels = torch.cat([torch.stack(pos_labels), torch.stack(neg_labels)]).float().cuda(2)


#             optimizer.zero_grad()
#             loss = criterion(x1, x2, labels)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()

        
#         print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}")

        
#         test_loss = evaluate_model(test_loader, model, criterion)
#         print(f"Test Loss: {test_loss}")

        
#         if test_loss < threshold:
#             print(f"Early stopping at epoch {epoch + 1} with test loss {test_loss}")
#             break

        
#         torch.save(model.state_dict(), os.path.join("result", f"model_epoch_{epoch + 1}.pt"))


# def evaluate_model(test_loader, model, criterion):
#     model.eval()  
#     test_loss = 0.0
#     tbar = tqdm(test_loader, desc="Evaluation") 
#     total_correct = 0 
#     total_samples = 0  
#     all_preds = [] 
#     all_labels = [] 
    
#     with torch.no_grad():  
#         for batch_idx, (anchor_pos_batch, anchor_neg_batch) in enumerate(tbar):
            

#             anchor_pos_x1, anchor_pos_x2, pos_labels = zip(*anchor_pos_batch)
#             anchor_neg_x1, anchor_neg_x2, neg_labels = zip(*anchor_neg_batch)

#             anchor_pos_x1 = torch.stack([torch.tensor(x).float() for x in anchor_pos_x1]).to('cuda')
#             anchor_pos_x2 = torch.stack([torch.tensor(x).float() for x in anchor_pos_x2]).to('cuda')
#             anchor_neg_x1 = torch.stack([torch.tensor(x).float() for x in anchor_neg_x1]).to('cuda')
#             anchor_neg_x2 = torch.stack([torch.tensor(x).float() for x in anchor_neg_x2]).to('cuda')

#             pos_labels = torch.tensor(pos_labels).float().to('cuda')
#             neg_labels = torch.tensor(neg_labels).float().to('cuda')

#             pos_outputs = model(anchor_pos_x1, anchor_pos_x2)
#             neg_outputs = model(anchor_neg_x1, anchor_neg_x2)

#             outputs = torch.cat([pos_outputs, neg_outputs], dim=0).squeeze()
#             labels = torch.cat([pos_labels, neg_labels], dim=0)

#             loss = criterion(outputs, labels)
#             test_loss += loss.item()

#             preds = (outputs >= 0.5).float()

#             total_correct += (preds == labels).sum().item()
#             total_samples += labels.size(0)


#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())
    

#     accuracy = total_correct / total_samples
#     avg_test_loss = test_loss / len(test_loader)
    
#     print(f"Test Loss: {avg_test_loss}, Accuracy: {accuracy}")
    
#     return avg_test_loss, accuracy


def train_model(train_loader, test_loader, model, optimizer, criterion, epochs, threshold): 
    model.train()
    device = 'cuda:2'
    for epoch in range(epochs):
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, batch_data in enumerate(progress_bar):
            anchor_data, paired_data, labels = zip(*batch_data)

            anchor_data = torch.stack([torch.tensor(x) if not isinstance(x, torch.Tensor) else x for x in anchor_data]).float().to(device)
            paired_data = torch.stack([torch.tensor(x) if not isinstance(x, torch.Tensor) else x for x in paired_data]).float().to(device)
            labels = torch.tensor(labels).float().to(device)
            
            optimizer.zero_grad()

            loss = criterion(anchor_data, paired_data, labels)
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}")

        test_loss = evaluate_model(test_loader, model, criterion)
        print(f"Test Loss: {test_loss}")

        if test_loss < threshold:
            print(f"Early stopping at epoch {epoch + 1} with test loss {test_loss}")
            break

        torch.save(model.state_dict(), os.path.join("result", f"model_epoch_{epoch + 1}.pt"))




def evaluate_model(test_loader, model):
    device = 'cuda:2'
    model.eval()  
    tbar = tqdm(test_loader, desc="Evaluation")  
    total_correct = 0  
    total_samples = 0  
    all_preds = []  
    all_labels = []  
    
    with torch.no_grad(): 
        for batch_idx, (x1_batch, labels) in enumerate(tbar):
 
            x1_batch = torch.stack([torch.tensor(x).float() for x in x1_batch]).to(device)
            labels = torch.tensor(labels).float().to(device)

            outputs = model.emb_net(x1_batch)

            preds = (torch.sigmoid(outputs).squeeze() >= 0.5).float()

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
df = all_clips_df
label_1_count = df[df['label'] == 1].shape[0]
label_0_count = df[df['label'] == 0].shape[0]
print(f"Label 1 count: {label_1_count}")
print(f"Label 0 count: {label_0_count}")
#exit()

device = 'cuda:2'

emb_size = 100
emb = Shallow(1, 40)
model = ContrastiveNet(emb, emb_size).to(device)
optimizer = Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
criterion = RelativePositioningLoss(emb_size).to(device)

train_df, test_df = split_dataset(all_clips_df)
train_dataset = RPDataset(train_df, tau_pos=18, tau_neg=600)
print(train_dataset[0])
test_dataset = LabelDataset(test_df)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn, num_workers=4)

train_model(train_loader, test_loader, model, optimizer, criterion, epochs=100, threshold=0.01)
# folder_to_clear = './data/train'
# clear_directory(folder_to_clear)



