import os
import torch
import numpy as np
import gc
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from utils import split_train_test_graph, load_sequences_for_ids, get_similarity_edge_features, prepare_gat_inputs, DGI_Pretrainer, device
from feature_extraction import get_protrek_sequence_features_for_ids, get_t5_features_for_ids, get_gat_prot_t5_features, LOCAL_PROT_T5_PATH
from models import GATForPortT5, PROTT5_DIM, GAT_HIDDEN_DIM, GAT_OUTPUT_DIM, GAT_HEADS, DGI_PRETRAIN_EPOCHS, DGI_PRETRAIN_LR
from trainer import PPIPrediction

def main(args):
    G1, G_train, _, train_graph_filename, G0, training_pos_edges = split_train_test_graph(args.input1, args.input2, args.seed)
    all_proteins = set(G1.nodes()) | set(G0.nodes())
    all_sequences = load_sequences_for_ids('data/human/human.csv', all_proteins)
    
    protrek_look_up = get_protrek_sequence_features_for_ids({p: all_sequences[p] for p in all_proteins if p in all_sequences})

    prot_t5_look_up = get_t5_features_for_ids({p: all_sequences[p] for p in all_proteins if p in all_sequences})

    similarity_scores = get_similarity_edge_features(G1, all_sequences) 
    
    x, ei, ea, _ = prepare_gat_inputs(G1, prot_t5_look_up, similarity_scores)
    model = GATForPortT5(PROTT5_DIM, GAT_HIDDEN_DIM, GAT_OUTPUT_DIM, GAT_HEADS).to(device)
    DGI_Pretrainer(model, x, ei, ea, dgi_cache, DGI_PRETRAIN_EPOCHS, DGI_PRETRAIN_LR)
    del x, ei, ea, model

    prot_t5_gat_look_up = get_gat_prot_t5_features(G1, prot_t5_look_up, similarity_scores, dgi_cache)

    PPIPrediction(
        G1,                
        G_train,           
        G0,               
        [],                
        args.seed,           
        training_pos_edges,  
        protrek_look_up,     
        prot_t5_gat_look_up 
    )
    if os.path.exists(train_graph_filename): os.remove(train_graph_filename)

if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input1', default="data/human/human_pos.edgelist")
    parser.add_argument('--input2', default="data/human/human_neg.edgelist")
    parser.add_argument('--seed', default=42, type=int)
    main(parser.parse_known_args()[0])