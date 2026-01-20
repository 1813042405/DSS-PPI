import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.utils import softmax

GAT_HEADS = 8          
GAT_HIDDEN_DIM = 512   
GAT_OUTPUT_DIM = 256   
PROTT5_DIM = 1024   
PROTREK_DIM = 1280      
DGI_PRETRAIN_EPOCHS = 50 
DGI_PRETRAIN_LR = 1e-4   

def initialize_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Parameter):
        if len(m.shape) >= 2: 
             nn.init.xavier_uniform_(m)
        else:
             nn.init.uniform_(m)

class GATWithEdgeFeatures(nn.Module):
    def __init__(self, in_channels, out_channels, heads, concat=True, dropout=0.6):
        super(GATWithEdgeFeatures, self).__init__()
        self.heads = heads
        self.out_channels = out_channels
        self.concat = concat
        self.dropout = dropout

        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.lin_edge = nn.Linear(1, heads * out_channels, bias=False)
        self.att = nn.Parameter(torch.Tensor(1, heads, 3 * out_channels))
        self.reset_parameters()
        self.dropout_layer = nn.Dropout(p=self.dropout)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.lin_edge.weight)
        nn.init.xavier_uniform_(self.att)

    def forward(self, x, edge_index, edge_attr):
        x = self.lin(x).view(-1, self.heads, self.out_channels)
        edge_attr = self.lin_edge(edge_attr.view(-1, 1)).view(-1, self.heads, self.out_channels)  
        row, col = edge_index
        x_i = x[row]
        x_j = x[col]
        z = torch.cat([x_i, x_j, edge_attr], dim=-1)
        alpha = (self.att * z).sum(dim=-1)
        alpha = F.leaky_relu(alpha, negative_slope=0.2)
        alpha = self.dropout_layer(alpha)
        alpha = softmax(alpha, row)
        out = alpha.unsqueeze(-1) * x_j
        out = scatter_add(out, row, dim=0, dim_size=x.size(0))
        return self.update(out)

    def update(self, aggr_out):
        if self.concat:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)
        return aggr_out

class GATForPortT5(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads, dropout_rate=0.1):  
        super().__init__()
        self.gat1 = GATWithEdgeFeatures(input_dim, input_dim // heads, heads=heads)
        self.gat2 = GATWithEdgeFeatures(hidden_dim, hidden_dim // heads, heads=heads)
        self.gat3 = GATWithEdgeFeatures(output_dim, output_dim, heads=1)
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.ln_final = nn.LayerNorm(output_dim)
        self.linear1 = nn.Linear(input_dim, input_dim)
        self.linear2 = nn.Linear(input_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, output_dim)
        self.skip1 = nn.Identity()
        self.skip2 = nn.Linear(input_dim, hidden_dim)
        self.skip3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            initialize_weight(m)

    def forward(self, x, edge_index, edge_attr):
        identity = x
        x = self.linear1(x)
        x = self.gat1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = x + self.skip1(identity)

        identity = x
        x = self.linear2(x)
        x = self.gat2(x, edge_index, edge_attr)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = x + self.skip2(identity)

        identity = x
        x = self.linear3(x)
        x = self.gat3(x, edge_index, edge_attr)
        x = self.ln_final(x)
        x = self.relu(x)
        x = x + self.skip3(identity)
        return x

class DGI_Discriminator(nn.Module):
    def __init__(self, n_feature, m_feature):
        super(DGI_Discriminator, self).__init__()
        self.lin = nn.Linear(n_feature, m_feature)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        self.lin.bias.data.fill_(0)

    def forward(self, h, c, h_neg=None):
        pos_score = (h * c).sum(dim=-1)
        if h_neg is not None:
            neg_score = (h_neg * c).sum(dim=-1)
            return pos_score, neg_score
        return pos_score

class Readout(nn.Module):
    def __init__(self):
        super(Readout, self).__init__()
    def forward(self, h):
        return h.mean(dim=0).unsqueeze(0)

class CombinedClassifier(nn.Module):
    def __init__(self, combined_dim, prot_dim, prot_t5_gat_dim, hidden_dim=128): 
        super(CombinedClassifier, self).__init__()
        self.prot_dim = prot_dim           
        self.prot_t5_gat_dim = prot_t5_gat_dim 
        self.total_seq_dim = prot_dim + prot_t5_gat_dim 
        self.combined_dim = combined_dim   

        self.prot_embed = nn.Sequential(
            nn.Linear(self.prot_dim, hidden_dim * 8), 
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 8),
            nn.Dropout(0.3), 
            nn.Linear(hidden_dim * 8, hidden_dim * 4), 
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.Dropout(0.3)
        )

        self.prot_t5_embed = nn.Sequential(
            nn.Linear(self.prot_t5_gat_dim, hidden_dim * 2), 
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Dropout(0.3), 
            nn.Linear(hidden_dim * 2, hidden_dim * 4), 
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.Dropout(0.3)
        )
        
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 4 * 2, hidden_dim * 4),
            nn.Sigmoid()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1) 
        )

    def forward(self, combined_inputs):
        prot_inputs = combined_inputs[:, :self.prot_dim]
        prot_t5_inputs = combined_inputs[:, self.prot_dim:self.total_seq_dim]
        prot_embedded = self.prot_embed(prot_inputs)
        prot_t5_embedded = self.prot_t5_embed(prot_t5_inputs)
        gate_input = torch.cat([prot_embedded, prot_t5_embedded], dim=1)
        g = self.gate(gate_input)
        fused_features = g * prot_embedded + (1 - g) * prot_t5_embedded
        x = self.classifier(fused_features)
        return x