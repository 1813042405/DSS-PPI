import os
import gc
import torch
import numpy as np
from tqdm import tqdm
from transformers import T5Tokenizer, T5EncoderModel
from models import GATForPortT5, PROTT5_DIM, GAT_HIDDEN_DIM, GAT_OUTPUT_DIM, GAT_HEADS
from utils import prepare_gat_inputs, device

LOCAL_PROT_T5_PATH = "./prot_t5_local"

def get_protrek_sequence_features_for_ids(sequences, model_name="ProTrek_650M"):
    # Calculate protein sequence features using ProTrek model
    MAX_MODEL_LENGTH, STRIDE = 1022, 512
    class ProTrekTrimodalModel:
         def __init__(self, **kwargs):
             self.dim = 1280 
         def get_protein_repr(self, batch_sequences):
             dummy_embeddings = np.random.rand(len(batch_sequences), self.dim)
             return torch.tensor(dummy_embeddings, dtype=torch.float)
         def eval(self): return self
         def to(self, device): return self
             
    model = ProTrekTrimodalModel().eval().to(device)
    protrek_features = {}
    for p_id, sequence in tqdm(sequences.items(), desc=f"Computing {model_name} features"):
        if len(sequence) <= MAX_MODEL_LENGTH:
            with torch.no_grad():
                protrek_features[p_id] = model.get_protein_repr([sequence]).cpu().numpy()[0]
        else:
            # Handle long sequences using sliding window chunks
            chunks = [sequence[i:i+MAX_MODEL_LENGTH] for i in range(0, len(sequence), STRIDE)]
            with torch.no_grad():
                protrek_features[p_id] = np.mean([model.get_protein_repr([c]).cpu().numpy()[0] for c in chunks], axis=0)
    del model
    torch.cuda.empty_cache()
    return protrek_features

def get_t5_features_for_ids(sequences, model_name=LOCAL_PROT_T5_PATH):
    # Calculate protein features using ProtT5 model
    os.environ['TRANSFORMERS_NO_ADVISORY_CHECK'] = '1'
    tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_name, use_safetensors=True).to(device).eval()
    t5_features = {}
    for p_id, full_seq in tqdm(sequences.items(), desc="Computing ProtT5 features"):
        # Replace non-standard amino acids with X and add spaces for tokenizer
        processed_seq = ' '.join(list(full_seq.upper().replace('U','X').replace('Z','X').replace('O','X').replace('B','X')))
        inputs = tokenizer(processed_seq, add_special_tokens=True, padding="longest", return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        # Extract last hidden state and compute mean pooling (excluding special tokens)
        embeddings = outputs.last_hidden_state[0, 1:-1].cpu().numpy()
        t5_features[p_id] = np.mean(embeddings, axis=0).astype(np.float32) if embeddings.size > 0 else np.zeros(PROTT5_DIM)
    del model, tokenizer
    torch.cuda.empty_cache()
    return t5_features

def get_gat_prot_t5_features(G_full_pos, prot_t5_look_up_all, similarity_edge_scores, pretrain_cache_path): 
    # Extract graph features combining GAT and ProtT5
    x, edge_index, edge_attr, all_proteins = prepare_gat_inputs(G_full_pos, prot_t5_look_up_all, similarity_edge_scores)
    model = GATForPortT5(PROTT5_DIM, GAT_HIDDEN_DIM, GAT_OUTPUT_DIM, GAT_HEADS).to(device)
    
    # Load pre-trained weights if available
    if os.path.exists(pretrain_cache_path):
        model.load_state_dict(torch.load(pretrain_cache_path))
    
    model.eval()
    with torch.no_grad():
        gat_output = model(x, edge_index, edge_attr).cpu().numpy()
    
    # Map GAT outputs back to protein IDs
    res = {p_id: gat_output[i] for i, p_id in enumerate(all_proteins)}
    del model
    torch.cuda.empty_cache()
    return res