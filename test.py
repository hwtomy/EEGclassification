import torch
from torch.utils.data import DataLoader
import pandas as pd
from shallow import Shallow, ContrastiveNet 
from pretext import taset, collate_fnt         
from torch.optim import Adam
from rp import clips
import os
import shutil
from tqdm import tqdm      
from pretext import RPDataset, collate_fn, split_dataset, LabelDataset, taset, collate_fnt
from preprocess import remove_short_segments


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
device = 'cuda:2'



CUDA_VISIBLE_DEVICES=2
df = pd.read_parquet("/datasets2/epilepsy/TUSZ/processed/dev/segments.parquet")
output_dir = "./data/test"
sampling_rate=250
target_sampling_rate=100
lowpass_freq=50
sfre = 100
clip_length = 6
clip_stride = 6
all_clips_df = clips(df, sampling_rate, target_sampling_rate, output_dir, lowpass_freq,clip_length, clip_stride)
all_clips_df.to_parquet('./data/processed.parquet', engine='pyarrow')
#all_clips_df = pd.read_parquet('./data/processed.parquet')
all_clips_df = remove_short_segments(all_clips_df, 6)
df = all_clips_df
label_1_count = df[df['label'] == 1].shape[0]
label_0_count = df[df['label'] == 0].shape[0]
print(f"Label 1 count: {label_1_count}")
print(f"Label 0 count: {label_0_count}")


emb_size = 100
emb = Shallow(1, 40)
model = ContrastiveNet(emb, emb_size).to(device)


model_path = './result/20240928_204648/model_RP.pt'  
model.load_state_dict(torch.load(model_path, map_location=device))


#new_test_df = pd.read_parquet('./data/new_test_data.parquet')

new_test_dataset = taset(df)
new_test_loader = DataLoader(new_test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fnt, num_workers=0)

new_test_accuracy = evaluate_model(new_test_loader, model)
print(f"New Test Accuracy: {new_test_accuracy}")
