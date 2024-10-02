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
from pretext import RPDataset1, collate_fn, split_dataset, LabelDataset, taset, collate_fnt, balance_dataframe
from preprocess import remove_short_segments, filter_shortpatient
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score,  f1_score
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import pickle
import wandb
import random


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

def train_model(train_loader, model, optimizer, criterion, epochs, threshold): 
    model.train()
    device = 'cuda:3'
    floss = float('inf')
    loss_history = []
    tloss = []
    for epoch in range(epochs):
        running_loss = 0.0
        pastloss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        count = 0
        for batch_idx, batch_data in enumerate(progress_bar):
            optimizer.zero_grad()
            batch_loss = 0.0 
            anchor_data, paired_data, labels = zip(*batch_data)

            anchor_data = torch.stack([torch.tensor(x) if not isinstance(x, torch.Tensor) else x for x in anchor_data]).float().to(device)
            paired_data = torch.stack([torch.tensor(x) if not isinstance(x, torch.Tensor) else x for x in paired_data]).float().to(device)
            labels = torch.tensor(labels).float().to(device)
    
    
            anchor_data = anchor_data.unsqueeze(1) 
            paired_data = paired_data.unsqueeze(1)

            output = model(anchor_data, paired_data)  
            loss = criterion(output, labels) 
            loss.backward()
            running_loss += loss.item()
            optimizer.step()

                
    
        closs = running_loss / len(train_loader)
        wandb.log({"loss": closs})
        loss_history.append(closs)
        if (abs(pastloss - closs) <= 0.03):
            count += 1
        else:
            count = 0
        pastloss = closs
        
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {closs}")
        if count == 10:
            return model
        floss = min(closs, floss)
  

 
    return model


def extract_features(model, data_loader, device='cuda:3'):
    model.eval()
    extracted_features = []
    labels = []

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(data_loader, desc="Extracting Features")):
            anchor_data, batch_labels = zip(*batch_data)
            anchor_data = torch.stack([torch.tensor(x) if not isinstance(x, torch.Tensor) else x for x in anchor_data]).float().to(device)
            anchor_data = anchor_data.unsqueeze(1)
            
            feature_embeddings = model.emb_net(anchor_data).cpu().numpy()  
            extracted_features.extend(feature_embeddings)
            labels.extend(batch_labels)

    extracted_features = np.array(extracted_features)
    labels = np.array(labels)
    return extracted_features, labels


def balance_data(features, labels, num_classes=2):
    label_indices = [np.where(labels == i)[0] for i in range(num_classes)]
    
    if any(len(indices) == 0 for indices in label_indices):
        return features, labels 

    max_class_count = max(len(indices) for indices in label_indices)
    
    balanced_features = []
    balanced_labels = []

    for i, indices in enumerate(label_indices):
        upsampled_indices = np.random.choice(indices, size=max_class_count, replace=True)
        balanced_features.append(features[upsampled_indices])
        balanced_labels.append(np.full(max_class_count, i))

    balanced_features = np.concatenate(balanced_features)
    balanced_labels = np.concatenate(balanced_labels)

    return balanced_features, balanced_labels


def train_logistic_regression(train_features, train_labels, test_features, test_labels):
    logistic_regression = LogisticRegression(C=1, max_iter=100)
    
    logistic_regression.fit(train_features, train_labels)

    y_pred = logistic_regression.predict(test_features)
    
    print(classification_report(test_labels, y_pred, zero_division=0))
    
    print(f"Validation Accuracy: {accuracy_score(test_labels, y_pred)}")

    return logistic_regression



def evaluate_on_test_set_with_shallow(test_loader, model, logistic_model, device='cuda:3'):
   
    test_features, test_labels = extract_features( model, test_loader, device)
    test_features, test_labels = balance_data(test_features, test_labels)
    
    y_test_pred = logistic_model.predict(test_features)
    
    print("Test Set Performance:")
    print(classification_report(test_labels, y_test_pred, zero_division=0))

    print(f"Test Accuracy: {accuracy_score(test_labels, y_test_pred)}")


    accuracy = accuracy_score(test_labels, y_test_pred)

    f1_macro = f1_score(test_labels, y_test_pred, average='macro', zero_division=0)

    f1_micro = f1_score(test_labels, y_test_pred, average='micro', zero_division=0)

    return accuracy, f1_macro, f1_micro

def save_results_to_txt(file_path, accuracy, f1_macro, f1_micro):
    with open(file_path, 'w') as file:
        file.write(f"Test Accuracy: {accuracy}\n")
        file.write(f"F1 Macro: {f1_macro}\n")
        file.write(f"F1 Micro: {f1_micro}\n")




def evaluate_on_test_set(test_loader, model, logistic_model, scaler, device):

    test_features, test_labels = extract_features(model, test_loader, device)

    test_features = scaler.transform(test_features)

    y_test_pred = logistic_model.predict(test_features)

    accuracy = accuracy_score(test_labels, y_test_pred)

    print(f"Accuracy: {accuracy}")

    return accuracy











wandb.init(
    # set the wandb project where this run will be logged
    project="SSLCONTRASTIVE",

    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.001,
    "architecture": "SHallow",
    "dataset": "THU-seizure",
    "epochs":150,
    }
)
# folder_to_clear = './data/train'
# clear_directory(folder_to_clear)
CUDA_VISIBLE_DEVICES=3
#df = pd.read_parquet("/datasets2/epilepsy/TUSZ/processed/train/segments.parquet")
output_dir = "./data/train"
sampling_rate=250
target_sampling_rate=100
lowpass_freq=50
sfre = 100
clip_length = 6
clip_stride = 6
##all_clips_df = clips(df, sampling_rate, target_sampling_rate, output_dir, lowpass_freq,clip_length, clip_stride)
#all_clips_df.to_parquet('./data/processed_train.parquet', engine='pyarrow')
all_clips_df = pd.read_parquet('./data/processed_train.parquet')
all_clips_df = remove_short_segments(all_clips_df, 6)
df = all_clips_df
label_1_count = df[df['label'] == 1].shape[0]
label_0_count = df[df['label'] == 0].shape[0]
print(f"Label 1 count: {label_1_count}")
print(f"Label 0 count: {label_0_count}")
#exit()

device = 'cuda:3'
#torch.cuda.empty_cache()
emb_size = 100
emb = Shallow(1, 40)
model = ContrastiveNet(emb, emb_size).to(device)
optimizer = Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
criterion = RelativePositioningLossm(emb_size).to(device)
train_df, test_df = split_dataset(all_clips_df)
session_count = train_df['session'].nunique()
train_dataset = RPDataset1(train_df, tau_pos=18, tau_neg=60)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,collate_fn=collate_fn, num_workers=8)



trained_model=train_model(train_loader, model, optimizer, criterion, epochs=150, threshold=0.01)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
result_folder = os.path.join("./result/RP/", timestamp)

os.makedirs(result_folder, exist_ok=True)

model_save_path = os.path.join(result_folder, f"shallow_RP.pt")
torch.save(model.state_dict(), model_save_path)

train_df = balance_dataframe(train_df)
print("success")
train_dataset = taset(train_df)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,collate_fn=collate_fnt, num_workers=0)
test_df = balance_dataframe(test_df)
test_dataset = taset(test_df)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,collate_fn = collate_fnt, num_workers=0)






train_features, train_labels = extract_features(trained_model, train_loader, device)
test_features, test_labels = extract_features(trained_model, test_loader, device)

logistic_model = train_logistic_regression(train_features,train_labels, test_features, test_labels)


os.makedirs(result_folder, exist_ok=True)
model_path = os.path.join(result_folder, 'logistic.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(logistic_model, f)



print("Evaluation result")
all_clips_df = pd.read_parquet('./data/processed.parquet')
all_clips_df = remove_short_segments(all_clips_df, 6)
df = all_clips_df
df = balance_dataframe(df)
test_dataset = taset(df)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True,collate_fn=collate_fnt, num_workers=8)
accuracy, f1_macro, f1_micro = evaluate_on_test_set_with_shallow(test_loader, trained_model, logistic_model)

result_file_path = os.path.join(result_folder, 'result.txt')
save_results_to_txt(result_file_path, accuracy, f1_macro, f1_micro)



