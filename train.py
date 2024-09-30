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
from pretext import RPDataset, collate_fn, split_dataset, LabelDataset, taset, collate_fnt
from preprocess import remove_short_segments
from datetime import datetime




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
class RelativePositioningLoss(torch.nn.Module):
    def __init__(self, emb_size, w0=0.0):
        super(RelativePositioningLoss, self).__init__()
        self.w = torch.nn.Parameter(torch.randn(emb_size))  
        self.w0 = w0  

    def forward(self, x1, x2, y):
    
        h_x1 = model(x1)  
        h_x2 = model(x2)  

        g_RP = torch.abs(h_x1 - h_x2)

        score = torch.dot(self.w, g_RP.T) + self.w0
        loss = torch.log(1 + torch.exp(-y * score))

        return loss.mean()

def train_model(train_loader, test_loader, model, optimizer, criterion, epochs, threshold): 
    model.train()
    device = 'cuda:2'
    floss = float('inf')
    loss_history = []
    tloss = []
    for epoch in range(epochs):
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, batch_data in enumerate(progress_bar):
            #print(f"Batch data: {batch_data}")
            optimizer.zero_grad()
            batch_loss = 0.0 
            anchor_data, paired_data, labels = zip(*batch_data)
            # print(f"Anchor data list type: {type(anchor_data)}")
            # print(f"First element type: {type(anchor_data[0])}")
            # exit()
            #anchor_data, paired_data, labels = batch_data
            # print(f"Batch {batch_idx} - anchor_data shape: {anchor_data[0].shape}, paired_data shape: {paired_data[0].shape}")
            # print(len(anchor_data))

            anchor_data = torch.stack([torch.tensor(x) if not isinstance(x, torch.Tensor) else x for x in anchor_data]).float().to(device)
            paired_data = torch.stack([torch.tensor(x) if not isinstance(x, torch.Tensor) else x for x in paired_data]).float().to(device)
            labels = torch.tensor(labels).float().to(device)
            # for i in range(len(anchor_data)):
            #     anchor_tensor = torch.tensor(anchor_data[i]).float().to(device)  
            #     paired_tensor = torch.tensor(paired_data[i]).float().to(device)  
            #     label_tensor = torch.tensor(labels[i]).float().to(device)
    
            anchor_data = anchor_data.unsqueeze(1) 
            paired_data = paired_data.unsqueeze(1)


            # output = model(anchor_data, paired_data)
            
            # loss = criterion(output, labels)

            output = model(anchor_data, paired_data)  
            loss = criterion(output, labels) 
            loss.backward()
            running_loss += loss.item()
            optimizer.step()



            # for i in range(len(batch_data)): 
            #     anchor_datass = anchor_data[i]
                
  
            #     paired_datass = paired_data[i]
            #     labelss = labels[i]


            #     anchor_datass = anchor_datass.unsqueeze(0).float().to(device)
            #     paired_datass = paired_datass.unsqueeze(0).float().to(device)
            #     #labelss = torch.tensor([labelss]).unsqueeze(0).float().to(device)

            #     optimizer.zero_grad()
            #     output = model(anchor_datass, paired_datass)

            #     #print(output)
            #     #print(labelss)
            #     labelss = torch.tensor([labelss]).float().to(device).view_as(output)
            #     loss = criterion(output, labelss)
            #     batch_loss += loss

            # batch_loss.backward() 
            # optimizer.step()       
            # running_loss += batch_loss.item()

                
    
        closs = running_loss / len(train_loader)
        loss_history.append(closs)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {closs}")

        test_loss = evaluate_model(test_loader, model)
        tloss.append(test_loss)
        #print(f"Test Loss: {test_loss}")

        # if abs(closs-floss) < threshold:
        #     print(f"Early stopping at epoch {epoch + 1} with test loss {test}")
        #     break
        floss = min(closs, floss)
  

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_folder = os.path.join("./result", timestamp)
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
            #print(f"Preds: {preds}")
            total_correct += (preds == labels).sum().item()
            #print(labels.size(0))
            total_samples += preds.numel()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    #print(f"Total correct: {total_correct}")
    #print(f"Total samples: {total_samples}")
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
#torch.cuda.empty_cache()
emb_size = 100
emb = Shallow(1, 40)
model = ContrastiveNet(emb, emb_size).to(device)
optimizer = Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
criterion = RelativePositioningLossm(emb_size).to(device)
#print(df.head())
train_df, test_df = split_dataset(all_clips_df)
session_count = train_df['session'].nunique()
#print(f"Number of unique sessions: {session_count}")
train_dataset = RPDataset(train_df, tau_pos=18, tau_neg=90)
#print(train_dataset[0])
test_dataset = taset(test_df)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,collate_fn=collate_fn, num_workers=16)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,collate_fn = collate_fnt, num_workers=0)

train_model(train_loader, test_loader, model, optimizer, criterion, epochs=150, threshold=0.01)
# folder_to_clear = './data/train'
# clear_directory(folder_to_clear)



