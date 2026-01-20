import os
import gc
import csv
import torch
import random
import copy
import numpy as np
import networkx as nx
import itertools
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
from Bio import pairwise2
from models import Readout, DGI_Discriminator, GAT_OUTPUT_DIM, PROTT5_DIM, GAT_HIDDEN_DIM, GAT_HEADS, GATForPortT5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def DGI_Pretrainer(model, x, edge_index, edge_attr, cache_path, epochs, lr):
    # DGI self-supervised pre-training
    print(f"\n--- Start DGI Self-supervised Pre-training (Epochs: {epochs}, LR: {lr}) ---")
    feat_dim = GAT_OUTPUT_DIM 
    readout = Readout().to(device)
    discriminator = DGI_Discriminator(feat_dim, feat_dim).to(device)
    optimizer = optim.AdamW(list(model.parameters()) + list(discriminator.parameters()) + list(readout.parameters()), lr=lr)
    bce_loss = nn.BCEWithLogitsLoss()
    pos_labels = torch.ones(x.size(0), dtype=torch.float).to(device) 
    neg_labels = torch.zeros(x.size(0), dtype=torch.float).to(device)
    best_loss = float('inf')

    for epoch in range(1, epochs + 1):
        model.train()
        discriminator.train()
        readout.train()
        optimizer.zero_grad()
        h_pos = model(x, edge_index, edge_attr)
        c = readout(h_pos) 
        perm = torch.randperm(x.size(0))
        x_neg = x[perm]
        h_neg = model(x_neg, edge_index, edge_attr)
        pos_scores = discriminator(h_pos, c)
        neg_scores = discriminator(h_neg, c)
        loss = bce_loss(pos_scores, pos_labels) + bce_loss(neg_scores, neg_labels)
        loss.backward()
        optimizer.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), cache_path)
        if epoch % 10 == 0 or epoch == epochs:
             print(f"  DGI Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}")
    del readout, discriminator, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return model

def split_train_test_graph(input_edgelist, fu_edgelist, seed):
    # Graph data preparation and splitting
    G1 = nx.read_edgelist(input_edgelist)
    G0 = nx.read_weighted_edgelist(fu_edgelist)
    G_all = copy.deepcopy(G1) 
    G_all.remove_nodes_from(list(nx.isolates(G_all)))
    train_graph_filename = 'graph_train.edgelist'
    nx.write_edgelist(G_all, train_graph_filename, data=False)
    L1 = list(G_all.nodes())
    L0 = list(G0.nodes())
    for i in range(len(L0)):
        if L0[i] not in L1:
            G0.remove_node(L0[i])
    G0.remove_nodes_from(list(nx.isolates(G0)))
    num = len(G_all.edges) - len(G0.edges)
    G = nx.Graph()
    G.add_edges_from(itertools.combinations(L1, 2))
    G.remove_edges_from(G_all.edges())
    G.remove_edges_from(G0.edges())
    random.seed(seed)
    for edge in list(G.edges):
        node_u, node_v = edge
        if G.degree(node_u) > 10 and G.degree(node_v) > 10:
            G.remove_edge(node_u, node_v)
    neg_edges = random.sample(list(G.edges), num)
    G0.add_edges_from(neg_edges)
    return G1, G_all, [], train_graph_filename, G0, list(G_all.edges())

def calculate_similarity_score(seq1, seq2):
    # Calculate Smith-Waterman similarity score
    alignments = pairwise2.align.localms(seq1, seq2, 2, -1, -0.5, -0.1)
    if alignments:
        score = alignments[0].score
        seq_norm = seq1 if len(seq1) < len(seq2) else seq2
        max_alignments = pairwise2.align.localms(seq_norm, seq_norm, 2, -1, -0.5, -0.1)      
        max_possible_score = max_alignments[0].score if max_alignments else 0.0
        return score / max_possible_score if max_possible_score > 0 and score > 0 else (0.0 if score <= 0 else score)
    return 0.0

def get_similarity_edge_features(G_full_pos, all_sequences):
    # Compute edge features based on sequence similarity
    edge_scores = {}
    edges_to_compute = list(G_full_pos.edges())
    for u, v in tqdm(edges_to_compute, desc="Calculating Smith-Waterman scores"):
        seq_u, seq_v = all_sequences.get(u), all_sequences.get(v)
        key = tuple(sorted((u, v)))
        if seq_u and seq_v:
            edge_scores[key] = calculate_similarity_score(seq_u, seq_v)
        else:
            edge_scores[key] = 0.0
    np.save(cache_file_name, edge_scores)
    return edge_scores

def prepare_gat_inputs(G_full_pos, prot_t5_look_up_all, similarity_edge_scores):
    # Prepare tensors for GAT input
    all_proteins = list(G_full_pos.nodes())
    node_to_idx = {node: i for i, node in enumerate(all_proteins)}
    prot_t5_features = [prot_t5_look_up_all[p] for p in all_proteins]
    x = torch.tensor(np.stack(prot_t5_features), dtype=torch.float).to(device)
    edge_list = list(G_full_pos.edges())
    bi_directional_edges = edge_list + [(v, u) for u, v in edge_list]
    edge_index_list, edge_attr_list = [], [] 
    for u, v in bi_directional_edges:
        if u in node_to_idx and v in node_to_idx:
            edge_index_list.append((node_to_idx[u], node_to_idx[v]))
            key = tuple(sorted((u, v))) 
            score = similarity_edge_scores.get(key, 1.0) if similarity_edge_scores else 1.0
            edge_attr_list.append([score]) 
    row = torch.tensor([r for r, c in edge_index_list], dtype=torch.long)
    col = torch.tensor([c for r, c in edge_index_list], dtype=torch.long)
    edge_index = torch.stack([row, col], dim=0).to(device)
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float).to(device) 
    return x, edge_index, edge_attr, all_proteins

def load_sequences_for_ids(file_path, protein_ids):
    # Load protein sequences from CSV file
    sequences = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2 and row[0] in protein_ids:
                sequences[row[0]] = row[1].upper().replace(' ', '')
    return sequences