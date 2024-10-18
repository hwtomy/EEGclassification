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
from shallow import Shallow, ContrastiveNet, ContrastiveNet_deep, Shallow_deep_with_attention, Shallow_deep_with_selfattention, Shallow_deep_with_linformer
from loss import RelativePositioningLoss
from pretext import RPDataset, collate_fn, split_dataset, LabelDataset, taset, collate_fnt, balance_dataframe, RPDataset3
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





def train_model(train_loader, model, optimizer, criterion, scheduler, epochs, threshold): 
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
        scheduler.step()


                
    
        closs = running_loss / len(train_loader)
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


def train_mlp(train_features, train_labels, test_features, test_labels, device='cuda:2'):
    X_train = torch.tensor(train_features, dtype=torch.float32).to(device)
    y_train = torch.tensor(train_labels, dtype=torch.float32).unsqueeze(1).to(device)
    X_test = torch.tensor(test_features, dtype=torch.float32).to(device)
    y_test = torch.tensor(test_labels, dtype=torch.float32).unsqueeze(1).to(device)
    
    mlp_model = MLPBinaryClassifier(input_size=train_features.shape[1]).to(device)
    criterion = nn.BCELoss()  
    optimizer = Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    scheduler = CosineAnnealingLR(optimizer, T_max=150, eta_min=0.0001)
    
    epochs = 150
    for epoch in tqdm(range(epochs), desc="Training MLP", unit="epoch"):
        mlp_model.train()
        optimizer.zero_grad()
        outputs = mlp_model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
        wandb.log({"MLP_loss": loss})
    mlp_model.eval()
    with torch.no_grad():
        y_pred = mlp_model(X_test)
        y_pred_class = (y_pred > 0.5).float()
        y_pred_class = y_pred_class.squeeze().cpu().numpy()

        #print(classification_report(test_labels, y_pred_class, zero_division=0))
        print(f"Validation Accuracy: {accuracy_score(test_labels, y_pred_class):.4f}")

    return mlp_model


# def evaluate_on_test_set_with_shallow(test_loader, model, mlp_model, device='cuda:2'):
 
#     test_features, test_labels = extract_features(model, test_loader, device)
#     test_features, test_labels = balance_data(test_features, test_labels)
    

#     test_features = torch.tensor(test_features, dtype=torch.float32).to(device)
#     test_labels = torch.tensor(test_labels, dtype=torch.float32).to(device)
    

#     mlp_model.eval()
    
#     with torch.no_grad():

#         y_test_pred = mlp_model(test_features)
#         y_test_pred = (y_test_pred > 0.5).float()  
#         y_test_pred = y_test_pred.cpu().numpy()  
    
#     test_labels = test_labels.cpu().numpy()
    
#     print("Test Set Performance:")
#     print(classification_report(test_labels, y_test_pred, zero_division=0))
    
#     accuracy = accuracy_score(test_labels, y_test_pred)
#     print(f"Test Accuracy: {accuracy}")
    
#     f1_macro = f1_score(test_labels, y_test_pred, average='macro', zero_division=0)
#     f1_micro = f1_score(test_labels, y_test_pred, average='micro', zero_division=0)

#     return accuracy, f1_macro, f1_micro


def save_results_to_txt(file_path, accuracy, f1_macro, f1_micro, mlpt):
    with open(file_path, 'w') as file:
        file.write(f"Test Accuracy: {accuracy}\n")
        file.write(f"F1 Macro: {f1_macro}\n")
        file.write(f"F1 Micro: {f1_micro}\n")
        file.write(f"Model: {mlpt}\n")




CUDA_VISIBLE_DEVICES=2


wandb.init(
    # set the wandb project where this run will be logged
    project="SSLCONTRASTIVE_improve",

    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.0005,
    "architecture": "SHallow-deepsupervision",
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
emb_size = 100
# emb = Shallow(1, 40)
emb = Shallow_deep_with_attention(1, 40)
# model = ContrastiveNet_deep(emb, emb_size).to(device)
model = ContrastiveNet(emb, emb_size).to(device)
optimizer = Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.999), weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=3000, eta_min=0.00001)
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

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,collate_fn=collate_fn, num_workers=16)



trained_model=train_model(train_loader, model, optimizer, criterion,scheduler, epochs=3000, threshold=0.01)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
result_folder = os.path.join("./result/RP/", timestamp)

os.makedirs(result_folder, exist_ok=True)

model_save_path = os.path.join(result_folder, f"shallow_RP.pt")
torch.save(trained_model.state_dict(), model_save_path)

train_df = balance_dataframe(train_df)
print("success")
train_dataset = taset(train_df)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,collate_fn=collate_fnt, num_workers=0)
test_df = balance_dataframe(test_df)
test_dataset = taset(test_df)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,collate_fn = collate_fnt, num_workers=0)






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
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True,collate_fn=collate_fnt, num_workers=8)
accuracy, f1_macro, f1_micro = evaluate_on_test_set_with_shallow(test_loader, trained_model, logistic_model)
wandb.log({"accuracy": accuracy, "f1_macro": f1_macro, "f1_micro": f1_micro})

result_file_path = os.path.join(result_folder, 'result.txt')
save_results_to_txt(result_file_path, accuracy, f1_macro, f1_micro, "logistic")


wandb.finish()



