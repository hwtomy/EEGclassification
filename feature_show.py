from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import pandas as pd
from shallow import Shallow, ContrastiveNet, Shallow_deep_with_selfattention, Shallow_deep_with_attention
from pretext import taset, collate_fnt, balance_dataframe        
from torch.optim import Adam
from rp import clips
import os
import shutil
from tqdm import tqdm  
from preprocess import remove_short_segments
from shallow import Shallow, ContrastiveNet
import numpy as np


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



device = 'cuda:3'
emb_size = 100
emb = Shallow_deep_with_selfattention(1, 40)
model = ContrastiveNet(emb, emb_size).to(device)

model_path = './result/RP/20241017_151145/shallow_RP.pt'
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
all_clips_df = pd.read_parquet('./data/processed_train.parquet')
all_clips_df = remove_short_segments(all_clips_df, 6)
df = all_clips_df
df = balance_dataframe(df)
label_1_count = df[df['label'] == 1].shape[0]
label_0_count = df[df['label'] == 0].shape[0]
print(f"Label 1 count: {label_1_count}")
print(f"Label 0 count: {label_0_count}")
test_dataset = taset(df)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True,collate_fn=collate_fnt, num_workers=0)
features, labels = extract_features(model, test_loader)



tsne = TSNE(n_components=2, random_state=42)
features_2d = tsne.fit_transform(features)

plt.figure(figsize=(8, 6))

scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='viridis', s=5)
plt.colorbar(scatter)
save_dir = './result/RP'
plt.title("t-SNE Feature Distribution with Labels")
save_path = os.path.join(save_dir, 'tsne_features_10171.png')
plt.savefig(save_path)





all_clips_df = pd.read_parquet('./data/processed_test.parquet')
all_clips_df = remove_short_segments(all_clips_df, 6)
df = all_clips_df
df['label'] = df['label'].replace(3, 1)
df['label'] = df['label'].replace(2, 1)
df['label'] = df['label'].replace(4, 1)
df = balance_dataframe(df)
label_1_count = df[df['label'] == 1].shape[0]
label_0_count = df[df['label'] == 0].shape[0]
print(f"Label 1 count: {label_1_count}")
print(f"Label 0 count: {label_0_count}")
test_dataset = taset(df)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True,collate_fn=collate_fnt, num_workers=0)
features, labels = extract_features(model, test_loader)



tsne = TSNE(n_components=2, random_state=42)
features_2d = tsne.fit_transform(features)

plt.figure(figsize=(8, 6))

scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='viridis', s=5)
plt.colorbar(scatter)
save_dir = './result/RP'
plt.title("t-SNE Feature Distribution with Labels")
save_path = os.path.join(save_dir, 'tsne_features_10172.png')
plt.savefig(save_path)


