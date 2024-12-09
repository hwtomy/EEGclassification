import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import torch
from torch.utils.data import  Dataset
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from rp import clips
import os
import shutil
from tqdm import tqdm
from clean import clear_directory
from shallow import Shallow, ContrastiveNet, ContrastiveNet_deep, Shallow_deep_with_attention, Shallow_deep_with_selfattention, Shallow_deep_with_linformer, mulconv
from shallow import Shallow_cwt_with_attention, Shallow_wt_with_attentionc, Shallow_wt_with_attentionc, FFT_GNN
from loss import RelativePositioningLoss
from pretext import RPDataset, collate_fn, split_dataset, LabelDataset, taset, collate_fnt, balance_dataframe, RPDataset3, process_pairs, process_sin
from preprocess import remove_short_segments, filter_shortpatient
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score,  f1_score
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import pickle
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from torch.cuda.amp import autocast, GradScaler
from mlp import MLPBinaryClassifier, train_logistic_regression
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch





def train_model(anchor_loader, pair_loader, model, optimizer, criterion, scheduler, epochs, threshold): 
    torch.autograd.set_detect_anomaly(True)
    model.train()
    device = 'cuda:2'
    floss = float('inf')
    loss_history = []
    tloss = []
    pastloss = 0.0

    for epoch in range(epochs):
        running_loss = 0.0
        #pastloss = 0.0
        progress_bar = tqdm(zip(anchor_loader, pair_loader), desc=f"Epoch {epoch+1}/{epochs}")
        count = 0
        for anchor_batch, pair_batch in progress_bar:
            optimizer.zero_grad()
            batch_loss = 0.0 
            
            # Move batches to device
            anchor_batch = anchor_batch.to(device)
            pair_batch = pair_batch.to(device)
            
            # Forward pass
            output = model(anchor_batch.x, anchor_batch.edge_index, anchor_batch.edge_attr, anchor_batch.batch,
                           pair_batch.x, pair_batch.edge_index, pair_batch.edge_attr, pair_batch.batch)
            
            # Loss computation
            loss = criterion(output, anchor_batch.y)
            print(loss)
            exit()
            loss.backward()
            
            running_loss += loss.item()
            optimizer.step()
        scheduler.step()


                
    
        closs = running_loss / 32
        current_lr = scheduler.get_last_lr()[0]
        wandb.log({"loss": closs, "lr": current_lr})
        loss_history.append(closs)
        if (abs(pastloss - closs) <= 0.001):
            count += 1
        else:
            count = 0
        pastloss = closs
        
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {closs}")
        if count == 10:
            return model
        floss = min(closs, floss)
  

 
    return model







def extract_features(model, data_loader, device='cuda:2'):
    model.eval()
    extracted_features = []
    labels = []
    progress_bar = tqdm(zip(data_loader), desc=f"Epoch {epoch+1}/{epochs}")
    with torch.no_grad():
         for anchor_batch in progress_bar:
            anchor_batch = anchor_batch.to(device)
            
            feature_embeddings = model.emb_net(anchor_batch.x, anchor_batch.edge_index, anchor_batch.edge_attr, anchor_batch.batch).cpu().numpy()  
            extracted_features.extend(feature_embeddings)
            labels.extend(anchor_batch.y)

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

def evaluate_on_test_set_with_shallow(test_loader, model, logistic_model, device='cuda:2'):
   
    test_features, test_labels = extract_features( model, test_loader, device)
    #test_features, test_labels = balance_data(test_features, test_labels)
    
    y_test_pred = logistic_model.predict(test_features)
    
    print("Test Set Performance:")
    print(classification_report(test_labels, y_test_pred, zero_division=0))

    print(f"Test Accuracy: {accuracy_score(test_labels, y_test_pred)}")


    accuracy = accuracy_score(test_labels, y_test_pred)
    # print(f"Test Accuracy: {accuracy}")

    f1_macro = f1_score(test_labels, y_test_pred, average='macro', zero_division=0)
    # print(f"F1-score (Macro): {f1_macro}")

    f1_micro = f1_score(test_labels, y_test_pred, average='micro', zero_division=0)
    # print(f"F1-score (Micro): {f1_micro}")

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



def save_results_to_txt(file_path, accuracy, f1_macro, f1_micro, mlpt):
    with open(file_path, 'w') as file:
        file.write(f"Test Accuracy: {accuracy}\n")
        file.write(f"F1 Macro: {f1_macro}\n")
        file.write(f"F1 Micro: {f1_micro}\n")
        file.write(f"Model: {mlpt}\n")




CUDA_VISIBLE_DEVICES=2


wandb.init(
    # set the wandb project where this run will be logged
    project="SSLCONTRASTIVE3",

    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.0005,
    "architecture": "SHallow-deepsupervision-mulconv",
    "dataset": "THU-seizure",
    "epochs":1500,
    }
)




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
all_clips_df = filter_shortpatient(all_clips_df, 2)
df = all_clips_df
label_1_count = df[df['label'] == 1].shape[0]
label_0_count = df[df['label'] == 0].shape[0]
print(f"Label 1 count: {label_1_count}")
print(f"Label 0 count: {label_0_count}")
#exit()

device = 'cuda:2'
#torch.cuda.empty_cache()
emb_size = 16
# emb = Shallow(1, 40)
emb = FFT_GNN(1, 40)
# model = ContrastiveNet_deep(emb, emb_size).to(device)
model = ContrastiveNet(emb, emb_size).to(device)
optimizer = Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.999), weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=2000, eta_min=0.00001)
criterion = RelativePositioningLoss(emb_size).to(device)
# criterion = RelativePositioningLoss_deep(emb_size).to(device)
#print(df.head())
train_df, test_df = split_dataset(all_clips_df)
train_df.to_parquet('./data/processed_trains.parquet', engine='pyarrow')
test_df.to_parquet('./data/processed_tests.parquet', engine='pyarrow')
session_count = train_df['session'].nunique()
#print(f"Number of unique sessions: {session_count}")
train_dataset = RPDataset(train_df, tau_pos=18, tau_neg=60)
#print(train_dataset[0])
train_dataset, train1_dataset= process_pairs(train_dataset)
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,collate_fn=collate_fn, num_workers=16)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=8)
train1_loader = DataLoader(train1_dataset, batch_size=32, shuffle=False,num_workers=8)

trained_model=train_model(train_loader, train1_loader, model, optimizer, criterion,scheduler, epochs=2000, threshold=0.01)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
result_folder = os.path.join("./result/RP/", timestamp)

os.makedirs(result_folder, exist_ok=True)

model_save_path = os.path.join(result_folder, f"shallow_RP.pt")
torch.save(trained_model.state_dict(), model_save_path)

train_df = balance_dataframe(train_df)
print("success")
train_dataset = taset(train_df)
train_dataset = process_sin(train_dataset)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
test_df = balance_dataframe(test_df)
test_dataset = taset(test_df)
test_dataset = process_sin(test_dataset)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)






train_features, train_labels = extract_features(trained_model, train_loader, device)
test_features, test_labels = extract_features(trained_model, test_loader, device)

# balanced_train_features, balanced_train_labels = balance_data(train_features, train_labels)

logistic_model = train_logistic_regression(train_features,train_labels, test_features, test_labels)


os.makedirs(result_folder, exist_ok=True)
model_path = os.path.join(result_folder, 'logistic.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(logistic_model, f)

# mlp_model = train_mlp(train_features,train_labels, test_features, test_labels)


# os.makedirs(result_folder, exist_ok=True)
# model_save_path = os.path.join(result_folder, f"mlp_RP.pt")
# torch.save(mlp_model.state_dict(), model_save_path)



#logistic_model = online_train_logistic_regression(train_loader, test_loader, trained_model, device, save_path=result_folder)





print("Evaluation result")
all_clips_df = pd.read_parquet('./data/processed_test.parquet')
all_clips_df = all_clips_df.groupby(level='session').filter(lambda x: (x['label'] != 0).any())
all_clips_df = remove_short_segments(all_clips_df, 6)
df = all_clips_df
df['label'] = df['label'].replace(3, 1)
df['label'] = df['label'].replace(2, 1)
df['label'] = df['label'].replace(4, 1)
df = balance_dataframe(df)
test_dataset = taset(df)
test_datr = process_sin(test_dataset)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=8)
accuracy, f1_macro, f1_micro = evaluate_on_test_set_with_shallow(test_loader, trained_model, logistic_model)
wandb.log({"accuracy": accuracy, "f1_macro": f1_macro, "f1_micro": f1_micro})

result_file_path = os.path.join(result_folder, 'result.txt')
save_results_to_txt(result_file_path, accuracy, f1_macro, f1_micro, "logistic")


wandb.finish()



