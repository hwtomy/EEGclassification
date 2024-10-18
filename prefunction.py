def online_train_logistic_regression(train_loader, test_loader, model, device, num_classes=2, save_path='./features/'):
    model.eval()
    scaler = StandardScaler()  
    logistic_model = SGDClassifier(loss='log_loss', max_iter=1000,  penalty="l2", class_weight="balanced")
    all_classes_initialized = False
    first_batch = True  
    epoch = 0
   
    for batch_idx, batch_data in enumerate(train_loader):
        with torch.no_grad():
            anchor_data, labels = zip(*batch_data)
            anchor_data = torch.stack([torch.tensor(x) if not isinstance(x, torch.Tensor) else x for x in anchor_data]).float().to(device)

            anchor_data = anchor_data.unsqueeze(1)
            features = model.emb_net(anchor_data).cpu().numpy() 
        #balanced_features, balanced_labels = balance_data(features, np.array(labels))
            labels = np.array(labels)

        if first_batch or not all_classes_initialized:
            features = scaler.fit_transform(features)
            logistic_model.partial_fit(features, labels, classes=np.arange(num_classes)) 
            all_classes_initialized = True
            first_batch = False
        else:
            balanced_features = scaler.transform(features)
            logistic_model.partial_fit(features, labels)

        print(f"Processed batch {batch_idx + 1}")

        if (batch_idx + 1) % len(train_loader) == 0: 
            accuracy = evaluate_on_test_set(test_loader, model, logistic_model, scaler, device)
            print(f"Test - Accuracy: {accuracy}")

    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, 'regression_model.pkl'), 'wb') as f:
        pickle.dump(logistic_model, f)

    return logistic_model



# def extract_half(model, data_loader, device='cuda:2'):
#     model.eval()
#     extracted_features = []
#     labels = []
#     with torch.no_grad():
#         for batch_idx, batch_data in enumerate(tqdm(data_loader, desc="Extracting Features")):
#             anchor_data, batch_labels = zip(*batch_data)
#             anchor_data = torch.stack([torch.tensor(x) if not isinstance(x, torch.Tensor) else x for x in anchor_data]).float().to(device)
#             anchor_data = anchor_data.unsqueeze(1)
#             with autocast():
#                 feature_embeddings = model.emb_net(anchor_data).cpu().numpy()  
#             extracted_features.extend(feature_embeddings)
#             labels.extend(batch_labels)

#     extracted_features = np.array(extracted_features)
#     labels = np.array(labels)
#     return extracted_features, labels





# def train_half(train_loader, model, optimizer, criterion, scheduler, epochs, threshold): 
#     model.train()
#     device = 'cuda:2'
#     floss = float('inf')
#     loss_history = []
#     tloss = []
#     scaler = GradScaler()
#     for epoch in range(epochs):
#         running_loss = 0.0
#         pastloss = 0.0
#         progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
#         count = 0
#         for batch_idx, batch_data in enumerate(progress_bar):
#             #print(f"Batch data: {batch_data}")
#             optimizer.zero_grad()
#             batch_loss = 0.0 
#             anchor_data, paired_data, labels = zip(*batch_data)

#             anchor_data = torch.stack([torch.tensor(x) if not isinstance(x, torch.Tensor) else x for x in anchor_data]).float().to(device)
#             paired_data = torch.stack([torch.tensor(x) if not isinstance(x, torch.Tensor) else x for x in paired_data]).float().to(device)
#             labels = torch.tensor(labels).float().to(device)

#             anchor_data = anchor_data.unsqueeze(1) 
#             paired_data = paired_data.unsqueeze(1)
#             with autocast():
#                 output = model(anchor_data, paired_data)  
#                 loss = criterion(output, labels)
   

#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()
#             running_loss += loss.item()
#             optimizer.step()
#             scheduler.step()


  
#         closs = running_loss / len(train_loader)
#         loss_history.append(closs)
#         if (abs(pastloss - closs) <= 0.006):
#             count += 1
#         else:
#             count = 0
#         pastloss = closs
#         current_lr = scheduler.get_last_lr()[0]
#         wandb.log({"loss": closs})
#         wandb.log({"learning_rate": current_lr})
        
#         print(f"Epoch {epoch + 1}/{epochs}, Loss: {closs}")
#         if count == 10:
#             return model
  
#     return model




# def evaluate_model(test_loader, model):
#     device = 'cuda:2'
#     model.eval()  
#     tbar = tqdm(test_loader, desc="Evaluation")  
#     total_correct = 0  
#     total_samples = 0  
#     all_preds = []  
#     all_labels = []  
    
#     with torch.no_grad(): 
#         for batch_idx, batch_data in enumerate(tbar):
#             x1_batch, labels = zip(*batch_data)
#             # x1_batch = torch.stack([torch.tensor(x).float() for x in x1_batch]).to(device)
#             # labels = torch.tensor(labels).float().to(device)

#             x1_batch = torch.stack([x.clone().detach().float() for x in x1_batch]).to(device)
#             labels = torch.tensor(labels)
#             labels = labels.clone().detach().float().to(device)

#             x1_batch = x1_batch.unsqueeze(1) 

#             outputs = model.emb_net(x1_batch)

#             preds = (outputs >= 0).float()
#             #print(f"Preds: {preds}")
#             total_correct += (preds == labels).sum().item()
#             #print(labels.size(0))
#             total_samples += preds.numel()

#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())
#     #print(f"Total correct: {total_correct}")
#     #print(f"Total samples: {total_samples}")
#     accuracy = total_correct / total_samples
    
#     print(f"Test Accuracy: {accuracy}")
    
#     return accuracy
