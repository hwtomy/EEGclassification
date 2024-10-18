from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import pandas as pd
from shallow import Shallow, ContrastiveNet, Shallow_deep_with_selfattention, Shallow_deep_with_attention, Shallow_deep_with_linformer
from pretext import taset, collate_fnt, balance_dataframe, split_dataset, split_dataset1 
from torch.optim import Adam
from rp import clips
import os
import shutil
from tqdm import tqdm  
from preprocess import remove_short_segments, filter_shortpatient
from shallow import Shallow, ContrastiveNet
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime
from mlp import MLPBinaryClassifier, train_logistic_regression, train_svm_rbf, train_random_forest, train_xgboost,  balancebag, smote_adaboost,  ResNetClassifier
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.preprocessing import StandardScaler
import wandb
from sklearn.metrics import classification_report, accuracy_score,  f1_score
from preprocess import balance_data
import numpy as np
import pickle



# def balance_data(features, labels, num_classes=2):
#     label_indices = [np.where(labels == i)[0] for i in range(num_classes)]
    
#     if any(len(indices) == 0 for indices in label_indices):
#         return features, labels 

#     max_class_count = max(len(indices) for indices in label_indices)
    
#     balanced_features = []
#     balanced_labels = []

#     for i, indices in enumerate(label_indices):
#         upsampled_indices = np.random.choice(indices, size=max_class_count, replace=True)
#         balanced_features.append(features[upsampled_indices])
#         balanced_labels.append(np.full(max_class_count, i))

#     balanced_features = np.concatenate(balanced_features)
#     balanced_labels = np.concatenate(balanced_labels)

#     return balanced_features, balanced_labels

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

    extracted_features = np.stack(extracted_features)
    labels = np.stack(labels)
    return extracted_features, labels




# def evaluate_on_test_set_with_shallow(test_loader, model, mlp_model, device='cuda:3'):
 
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
    
#     # print("Test Set Performance:")
#     # print(classification_report(test_labels, y_test_pred, zero_division=0))
    
#     accuracy = accuracy_score(test_labels, y_test_pred)
#     print(f"Test Accuracy: {accuracy}")
#     f1 = f1_score(test_labels, y_test_pred)
#     print(f"Test F1 Score: {f1:.4f}")
    
#     f1_macro = f1_score(test_labels, y_test_pred, average='macro', zero_division=0)
#     f1_micro = f1_score(test_labels, y_test_pred, average='micro', zero_division=0)

#     return accuracy, f1_macro, f1_micro



def evaluate_on_test_set_with_shallow(test_loader, model, logistic_model, device='cuda:3'):
   
    test_features, test_labels = extract_features( model, test_loader, device)
    
    y_test_pred = logistic_model.predict(test_features)
    

    accuracy = accuracy_score(test_labels, y_test_pred)
    print(f"Test Accuracy: {accuracy}")
    # print("Unique values in test_labels:", np.unique(test_labels))
    # print("Unique values in y_test_pred:", np.unique(y_test_pred))

    f1 = f1_score(test_labels, y_test_pred)
    print(f"Test F1 Score: {f1:.4f}")

    report = classification_report(test_labels, y_test_pred)
    print("Classification Report:")
    print(report)

    f1_macro = f1_score(test_labels, y_test_pred, average='macro', zero_division=0)

    f1_micro = f1_score(test_labels, y_test_pred, average='micro', zero_division=0)

    return accuracy, f1_macro, f1_micro

def save_results_to_txt(file_path, accuracy, f1_macro, f1_micro):
    with open(file_path, 'w') as file:
        file.write(f"Test Accuracy: {accuracy}\n")
        file.write(f"F1 Macro: {f1_macro}\n")
        file.write(f"F1 Micro: {f1_micro}\n")



def train_mlp(train_features, train_labels, test_features, test_labels, device='cuda:3'):

    scaler = StandardScaler()
    X_train = torch.tensor(train_features, dtype=torch.float32).to(device)
    y_train = torch.tensor(train_labels, dtype=torch.float32).unsqueeze(1).to(device)
    X_test = torch.tensor(test_features, dtype=torch.float32).to(device)
    y_test = torch.tensor(test_labels, dtype=torch.float32).unsqueeze(1).to(device)
    
    mlp_model = ResNetClassifier().to(device)
    criterion = nn.BCELoss()  
    optimizer = Adam(mlp_model.parameters(), lr=0.0005, betas=(0.9, 0.999), weight_decay=1e-6)
    scheduler = CosineAnnealingLR(optimizer, T_max=300, eta_min=0.00005)
    
    epochs =300
    for epoch in tqdm(range(epochs), desc="Training MLP", unit="epoch"):
        mlp_model.train()
        optimizer.zero_grad()
        outputs = mlp_model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if (epoch+1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    mlp_model.eval()
    with torch.no_grad():
        y_pred = mlp_model(X_test)
        y_pred_class = (y_pred > 0.5).float()
        y_pred_class = y_pred_class.squeeze().cpu().numpy()

        #print(classification_report(test_labels, y_pred_class, zero_division=0))
        print(f"Validation Accuracy: {accuracy_score(test_labels, y_pred_class):.4f}")
        f1 = f1_score(test_labels, y_pred_class)
        print(f"Validation F1 Score: {f1:.4f}")


    with torch.no_grad():
        y_pred = mlp_model(X_train)
        y_pred_class = (y_pred > 0.5).float()
        y_pred_class = y_pred_class.squeeze().cpu().numpy()

        #print(classification_report(test_labels, y_pred_class, zero_division=0))
        print(f"train Accuracy: {accuracy_score(train_labels, y_pred_class):.4f}")
        f1 = f1_score(train_labels, y_pred_class)
        print(f"train F1 Score: {f1:.4f}")

    return mlp_model


def save_results_to_txt(file_path, accuracy, f1_macro, f1_micro, mlpt):
    with open(file_path, 'w') as file:
        file.write(f"Test Accuracy: {accuracy}\n")
        file.write(f"F1 Macro: {f1_macro}\n")
        file.write(f"F1 Micro: {f1_micro}\n")
        file.write(f"Model: {mlpt}\n")





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
all_clips_df = all_clips_df.groupby(level='session').filter(lambda x: (x['label'] != 0).any())
all_clips_df = remove_short_segments(all_clips_df, 6)

df = all_clips_df

all_clips_df['label'] = all_clips_df['label'].replace(3, 1)
all_clips_df['label'] = all_clips_df['label'].replace(2, 1)
all_clips_df['label'] = all_clips_df['label'].replace(4, 1)
all_clips_df = filter_shortpatient(all_clips_df, 2)
df = all_clips_df
# label_4_count = df[df['label'] == 4].shape[0]
# label_3_count = df[df['label'] == 3].shape[0]
# label_2_count = df[df['label'] == 2].shape[0]
# label_1_count = df[df['label'] == 1].shape[0]
# label_0_count = df[df['label'] == 0].shape[0]
# print(f"Label 4 count: {label_4_count}")
# print(f"Label 3 count: {label_3_count}")
# print(f"Label 2 count: {label_2_count}")
# print(f"Label 1 count: {label_1_count}")
# print(f"Label 0 count: {label_0_count}")
device = 'cuda:3'

emb_size = 100
emb = Shallow_deep_with_selfattention(1, 40)
model = ContrastiveNet(emb, emb_size).to(device)

model_path = './result/RP/20241017_151145/shallow_RP.pt'
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
trained_model = model
#all_clips_df = balance_dataframe(all_clips_df)
#print(all_clips_df.head())
train_df, test_df = split_dataset(all_clips_df)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
result_folder = os.path.join("./result/class/", timestamp)

os.makedirs(result_folder, exist_ok=True)


train_df = balance_dataframe(train_df)
print("success")
train_dataset = taset(train_df)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,collate_fn=collate_fnt, num_workers=0)
test_df = balance_dataframe(test_df)
test_dataset = taset(test_df)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,collate_fn = collate_fnt, num_workers=0)






train_features, train_labels = extract_features(trained_model, train_loader, device)
test_features, test_labels = extract_features(trained_model, test_loader, device)

logistic_model = train_svm_rbf(train_features,train_labels, test_features, test_labels)


os.makedirs(result_folder, exist_ok=True)
model_path = os.path.join(result_folder, 'logistic.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(logistic_model, f)

# mlp_model = train_mlp(train_features,train_labels, test_features, test_labels)


# os.makedirs(result_folder, exist_ok=True)
# model_save_path = os.path.join(result_folder, f"mlp_RP.pt")
# torch.save(mlp_model.state_dict(), model_save_path)





print("Evaluation result")
all_clips_df = pd.read_parquet('./data/processed_test.parquet')
# all_zero_sessions = all_clips_df.groupby(level='session').filter(lambda x: (x['label'] == 0).all())

# num_sessions_all_zero = all_zero_sessions.index.get_level_values('session').nunique()

# print(f"Number of sessions where all labels are 0: {num_sessions_all_zero}")
all_clips_df = all_clips_df.groupby(level='session').filter(lambda x: (x['label'] != 0).any())
all_clips_df = remove_short_segments(all_clips_df, 6)
df = all_clips_df

df['label'] = df['label'].replace(3, 1)
df['label'] = df['label'].replace(2, 1)
df['label'] = df['label'].replace(4, 1)
all_clips_df = filter_shortpatient(all_clips_df, 2)
# label_4_count = df[df['label'] == 4].shape[0]
# label_3_count = df[df['label'] == 3].shape[0]
# label_2_count = df[df['label'] == 2].shape[0]
# label_1_count = df[df['label'] == 1].shape[0]
# label_0_count = df[df['label'] == 0].shape[0]
# print(f"Label 4 count: {label_4_count}")
# print(f"Label 3 count: {label_3_count}")
# print(f"Label 2 count: {label_2_count}")
# print(f"Label 1 count: {label_1_count}")
# print(f"Label 0 count: {label_0_count}")
df = balance_dataframe(df)
test_dataset = taset(df)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True,collate_fn=collate_fnt, num_workers=8)
accuracy, f1_macro, f1_micro = evaluate_on_test_set_with_shallow(test_loader, trained_model, logistic_model)
# wandb.log({"accuracy": accuracy, "f1_macro": f1_macro, "f1_micro": f1_micro})

result_file_path = os.path.join(result_folder, 'result.txt')
save_results_to_txt(result_file_path, accuracy, f1_macro, f1_micro, "mlp")



