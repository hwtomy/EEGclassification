import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from torch.optim import Adam
import torch.optim as optim
from torch import optim

from rp import clips
import os
import shutil
from tqdm import tqdm
from clean import clear_directory
from shallow import Shallow, ContrastiveNet, ContrastiveNet_deep, Shallow_deep_with_attention, Shallow_deep_with_selfattention, Shallow_deep_with_linformer
from shallow import ContrastiveNetB
from loss import RelativePositioningLoss
from pretext import RPDataset, collate_fn, split_dataset, LabelDataset, taset, collate_fnt, balance_dataframe, RPDataset3, RPDataset5, collate_fnB, RPDataset6
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





def train_model(train_loader, test_loader, model, optimizer, criterion, scheduler, epochs, threshold, model_save_path, gradient_clipping=0.5): 
    model.train()
    device = 'cuda:2'
    floss = float('inf')
    loss_history = []
    tloss = []
    pastloss = 0.0
    for epoch in range(epochs):
        running_loss = 0.0
        ctl = 0
        cal = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        count = 0
        for batch_idx, batch_data in enumerate(progress_bar):
            optimizer.zero_grad()
            count+=1
            batch_loss = 0.0 
            anchor_data, paired_data, clabels, alabels = zip(*batch_data)

            anchor_data = torch.stack([torch.tensor(x) if not isinstance(x, torch.Tensor) else x for x in anchor_data]).float().to(device)
            paired_data = torch.stack([torch.tensor(x) if not isinstance(x, torch.Tensor) else x for x in paired_data]).float().to(device)
            clabels = torch.tensor(clabels).float().to(device)
            alabels = torch.tensor(alabels).long().to(device)
            # print(alabels.min(), alabels.max())


            anchor_data = anchor_data.unsqueeze(1) 
            paired_data = paired_data.unsqueeze(1)

            output, caloss = model(anchor_data, paired_data, alabels)  
            ctloss = criterion(output, clabels) 
            loss = 1.5*ctloss + 0.5*caloss
            loss.backward()
            running_loss += loss.item()
            ctl+=ctloss.item()
            cal+=caloss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=2)
            optimizer.step()
            scheduler.step()




        closs = running_loss / len(train_loader)
        consloss = ctl / len(train_loader)
        claloss = cal / len(train_loader)
        loss_history.append(closs)
        current_lr = scheduler.get_last_lr()[0]
        test_accuracy, test_f1 = validation(model, test_loader, device)
        wandb.log({"loss": closs, "contrasitive loss": consloss, "classification loss": claloss, "lr": current_lr, 
                    "test_accuracy": test_accuracy, "test_f1": test_f1})
        # scheduler.step(test_accuracy)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {closs}")
        if count == 10:
            return model
    if epoch%100==0:
        torch.save(model.state_dict(), model_save_path)
    return model



def validation(model, data_loader, device='cuda:2'):
    model.eval()
    extracted_features = []
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(data_loader, desc="validation")):
            anchor_data, batch_labels = zip(*batch_data)
            anchor_data = torch.stack([torch.tensor(x) if not isinstance(x, torch.Tensor) else x for x in anchor_data]).float().to(device)
            anchor_data = anchor_data.unsqueeze(1)
            
            z1 = model.emb_net(anchor_data)
            output = model.mlp_classifier(z1)
            preds = torch.argmax(output, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_labels)

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    return accuracy, f1









def save_results_to_txt(file_path, accuracy, f1_macro, f1_micro):
    with open(file_path, 'w') as file:
        file.write(f"Test Accuracy: {accuracy}\n")
        file.write(f"F1 Macro: {f1_macro}\n")
        file.write(f"F1 Micro: {f1_micro}\n")



CUDA_VISIBLE_DEVICES=2


wandb.init(
    # set the wandb project where this run will be logged
    project="SSLCONTRASTIVE_improve",

    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.0005,
    "architecture": "SHallow-deepsupervision-connet change",
    "dataset": "THU-seizure",
    "epochs":3000,
    }
)





output_dir = "./data/train"
all_clips_df = pd.read_parquet('./data/processed_train.parquet')
all_clips_df = remove_short_segments(all_clips_df, 6)
all_clips_df = filter_shortpatient(all_clips_df, 2)
all_clips_df['label'] = all_clips_df['label'].replace(3, 1)
all_clips_df['label'] = all_clips_df['label'].replace(2, 1)
all_clips_df['label'] = all_clips_df['label'].replace(4, 1)
df = all_clips_df
label_1_count = df[df['label'] == 1].shape[0]
label_0_count = df[df['label'] == 0].shape[0]
print(f"Label 1 count: {label_1_count}")
print(f"Label 0 count: {label_0_count}")
#exit()

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
result_folder = os.path.join("./result/RP/", timestamp)
os.makedirs(result_folder, exist_ok=True)
model_save_path = os.path.join(result_folder, f"shallow_RP.pt")

device = 'cuda:2'
#torch.cuda.empty_cache()
emb_size = 100
# emb = Shallow(1, 40)
emb = Shallow_deep_with_attention(1, 40)

model = ContrastiveNetB(emb, emb_size).to(device)
optimizer = Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.999), weight_decay=1e-5)
scheduler = CosineAnnealingLR(optimizer, T_max=3000, eta_min=0.00001)
# scheduler =  optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=20)
criterion = RelativePositioningLoss(emb_size).to(device)

train_df, test_df = split_dataset(all_clips_df)
train_df = balance_dataframe(train_df)
train_dataset = RPDataset(train_df, tau_pos=18, tau_neg=60)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,collate_fn=collate_fn, num_workers=16)

test_df = balance_dataframe(test_df)
test_dataset = taset(test_df)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,collate_fn = collate_fnt, num_workers=0)

trained_model=train_model(train_loader,test_loader, model, optimizer, criterion,scheduler, epochs=3000, threshold=0.01, model_save_path=model_save_path)
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# result_folder = os.path.join("./result/RP/", timestamp)

# os.makedirs(result_folder, exist_ok=True)

# model_save_path = os.path.join(result_folder, f"shallow_RPB.pt")
torch.save(trained_model.state_dict(), model_save_path)






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
accuracy, f1_macro, f1_micro = validation(trained_model,test_loader)
wandb.log({"accuracy": accuracy, "f1_macro": f1_macro, "f1_micro": f1_micro})

result_file_path = os.path.join(result_folder, 'result.txt')
save_results_to_txt(result_file_path, accuracy, f1_macro, f1_micro, "logistic")


wandb.finish()



