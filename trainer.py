import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
import math
import gc
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, average_precision_score, matthews_corrcoef, 
                             confusion_matrix, roc_curve, precision_recall_curve)
from scipy.interpolate import interp1d
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from numpy import linspace, mean

from models import CombinedClassifier
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CombinedDataset(Dataset):
    def __init__(self, combined_features, labels):
        self.combined_features = torch.FloatTensor(combined_features)
        self.labels = torch.FloatTensor(labels).unsqueeze(1) 
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        combined_feature = self.combined_features[idx]
        label = self.labels[idx]
        return combined_feature, label

def build_features(edges, protrek_look_up, prot_t5_gat_look_up, original_graph):
    combined_features_list = []
    labels = []
    
    valid_edges = []
    for u, v in edges:
        if u in protrek_look_up and v in protrek_look_up and \
           u in prot_t5_gat_look_up and v in prot_t5_gat_look_up:
            valid_edges.append((u, v))
        
    single_prot_dim = next(iter(protrek_look_up.values())).shape[0]
    single_prot_t5_gat_dim = next(iter(prot_t5_gat_look_up.values())).shape[0] 
    prot_dim = single_prot_dim * 2 
    prot_t5_gat_dim = single_prot_t5_gat_dim * 2

    for u, v in tqdm(valid_edges, desc="Engineering features"):
        vec_protrek = np.concatenate((protrek_look_up[u], protrek_look_up[v]))
        vec_prot_t5_gat = np.concatenate((prot_t5_gat_look_up[u], prot_t5_gat_look_up[v]))
        combined_feature = np.concatenate((vec_protrek, vec_prot_t5_gat))        
        combined_features_list.append(combined_feature)        
        if original_graph.has_edge(u, v) or original_graph.has_edge(v, u):
            labels.append(1)
        else:
            labels.append(0) 
    combined_features_array = np.array(combined_features_list)    
    return combined_features_array, np.array(labels), prot_dim, prot_t5_gat_dim

def generate_neg_edges(G0, edges_num, seed):
    random.seed(seed)
    neg_edges = random.sample(list(G0.edges), edges_num)
    return neg_edges

def PPIPrediction(original_graph, G_train, G0, test_pos_edges, seed, training_pos_edges, protrek_look_up_all, prot_t5_gat_look_up_all):
    random.seed(seed)
    np.random.seed(seed)
    
    train_neg_edges = generate_neg_edges(G0, len(training_pos_edges), seed)
    all_train_edges = list(training_pos_edges) + train_neg_edges
    random.shuffle(all_train_edges)
    
    folds = 5
    labels = np.array([1 if original_graph.has_edge(u, v) or original_graph.has_edge(v, u) else 0 for u, v in all_train_edges])
    kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    
    fold_results = []
    metrics_dict = {
        'accuracy': [], 'precision': [], 'recall': [], 'specificity': [],
        'f1': [], 'auc': [], 'prc': [], 'mcc': []
    }

    mean_fpr = linspace(0, 1, 100) 
    tprs = [] 
    mean_recall = linspace(0, 1, 100)
    precisions = []

    for i, (train_index, val_index) in enumerate(kf.split(all_train_edges, labels)):
        print(f"开始第 {i+1}/{folds} 折...")
        
        train_edges_for_fold = [all_train_edges[j] for j in train_index]
        val_edges = [all_train_edges[j] for j in val_index]
        combined_train, y_train, prot_dim, prot_t5_gat_dim = build_features(train_edges_for_fold, protrek_look_up_all, prot_t5_gat_look_up_all, original_graph)
        combined_val, y_val, _, _ = build_features(val_edges, protrek_look_up_all, prot_t5_gat_look_up_all, original_graph)

        scaler = StandardScaler()
        scaler.fit(combined_train)
        combined_train_scaled = scaler.transform(combined_train)
        combined_val_scaled = scaler.transform(combined_val)
        
        combined_dim = combined_train_scaled.shape[1]
        hidden_dim = 128 

        train_loader = DataLoader(CombinedDataset(combined_train_scaled, y_train), batch_size=32, shuffle=True)
        val_loader = DataLoader(CombinedDataset(combined_val_scaled, y_val), batch_size=32, shuffle=False)
        
        epochs = 30
        model = CombinedClassifier(combined_dim=combined_dim, 
                                   prot_dim=prot_dim,
                                   prot_t5_gat_dim=prot_t5_gat_dim,
                                   hidden_dim=hidden_dim).to(device)
        criterion = nn.BCEWithLogitsLoss()
        
        total_steps = epochs * len(train_loader)
        warmup_steps = int(0.1 * total_steps)
        
        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=3e-5)
        scheduler = LambdaLR(optimizer, lr_lambda)

        best_auc = 0.0
        patience_counter = 0

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for combined_inputs, labels_batch in train_loader:
                combined_inputs, labels_batch = combined_inputs.to(device), labels_batch.to(device)
                optimizer.zero_grad()
                outputs = model(combined_inputs)
                loss = criterion(outputs, labels_batch)
                loss.backward()
                optimizer.step()
                scheduler.step()
                running_loss += loss.item()
            
            model.eval()
            y_pred_list = []
            y_val_list = []
            with torch.no_grad():
                for combined_inputs, labels_batch in val_loader:
                    combined_inputs, labels_batch = combined_inputs.to(device), labels_batch.to(device) 
                    outputs = model(combined_inputs)
                    y_pred_list.append(torch.sigmoid(outputs).cpu().numpy()) 
                    y_val_list.append(labels_batch.cpu().numpy())
            
            y_pred_proba = np.vstack(y_pred_list).flatten()
            y_val_epoch = np.vstack(y_val_list).flatten()
            fold_auc = roc_auc_score(y_val_epoch, y_pred_proba) if len(np.unique(y_val_epoch)) > 1 else 0.5

            if fold_auc > best_auc:
                best_auc = fold_auc
                torch.save(model.state_dict(), f'best_model_fold_{i+1}.pt')
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= 5: break
        if os.path.exists(f'best_model_fold_{i+1}.pt'):
            model.load_state_dict(torch.load(f'best_model_fold_{i+1}.pt'))
            try: os.remove(f'best_model_fold_{i+1}.pt')
            except: pass

        model.eval()
        y_pred_list, y_val_list = [], []
        with torch.no_grad():
            for combined_inputs, labels_batch in val_loader:
                combined_inputs, labels_batch = combined_inputs.to(device), labels_batch.to(device)
                outputs = model(combined_inputs)
                y_pred_list.append(torch.sigmoid(outputs).cpu().numpy())
                y_val_list.append(labels_batch.cpu().numpy())

        y_pred_proba = np.vstack(y_pred_list).flatten()
        y_val_final = np.vstack(y_val_list).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)

        if len(np.unique(y_val_final)) > 1:
            fpr, tpr, _ = roc_curve(y_val_final, y_pred_proba)
            tpr_interp = interp1d(fpr, tpr, kind='linear')
            tprs.append(tpr_interp(mean_fpr)) 
            
            precision, recall, _ = precision_recall_curve(y_val_final, y_pred_proba)
            unique_recall, unique_indices = np.unique(recall[::-1], return_index=True)
            precision_interp = interp1d(unique_recall, precision[::-1][unique_indices], kind='linear', fill_value='extrapolate')
            precisions.append(precision_interp(mean_recall))
            
            fold_auc = roc_auc_score(y_val_final, y_pred_proba)
            fold_prc = average_precision_score(y_val_final, y_pred_proba)
        else:
            fold_auc = fold_prc = 0.5

        tn, fp, fn, tp = confusion_matrix(y_val_final, y_pred).ravel()
        specificity = tn / (tn + fp + 1e-6)
        
        metrics_dict['accuracy'].append(accuracy_score(y_val_final, y_pred))
        metrics_dict['precision'].append(precision_score(y_val_final, y_pred, zero_division=0))
        metrics_dict['recall'].append(recall_score(y_val_final, y_pred, zero_division=0))
        metrics_dict['specificity'].append(specificity)
        metrics_dict['f1'].append(f1_score(y_val_final, y_pred, zero_division=0))
        metrics_dict['auc'].append(fold_auc)
        metrics_dict['prc'].append(fold_prc)
        metrics_dict['mcc'].append(matthews_corrcoef(y_val_final, y_pred))

        fold_results.append({
            'metric': f'fold_{i+1}',
            'accuracy': metrics_dict['accuracy'][-1],
            'precision': metrics_dict['precision'][-1],
            'recall': metrics_dict['recall'][-1],
            'specificity': metrics_dict['specificity'][-1],
            'f1': metrics_dict['f1'][-1],
            'auc': metrics_dict['auc'][-1],
            'prc': metrics_dict['prc'][-1],
            'mcc': metrics_dict['mcc'][-1]
        })

        del model, combined_train, combined_val
        gc.collect()
        torch.cuda.empty_cache()

    # 保存结果逻辑
    all_results = fold_results + [
        {'metric': 'mean', **{k: np.mean(v) for k, v in metrics_dict.items()}},
        {'metric': 'std', **{k: np.std(v) for k, v in metrics_dict.items()}}
    ]
    pd.DataFrame(all_results).to_csv('human.csv', index=False)