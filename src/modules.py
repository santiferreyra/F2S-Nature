import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv
from torch_geometric.nn.pool import global_mean_pool


class F2S(nn.Module):
    """Features to Signatures model.
    
    Predicts side effects frequencies from SMILES by learning deep
    representations of molecules.
    """
    def __init__(self, molecule_embedding,
                        side_effect_embedding,
                        global_bias=None):
        super(F2S, self).__init__()
        self.molecule_embedding = molecule_embedding
        self.side_effect_embedding = side_effect_embedding

        if global_bias is None:
            self.global_bias = nn.Parameter(torch.randn(1))
        else:
            self.global_bias = global_bias
        
        self.side_bias = None
        self.mol_bias = None
    
    def forward(self, batch_data, side_effect_ids=torch.arange(994), send_embs=False):
        # Unpack batch
        x = batch_data.x  # [num_nodes, in_channels]
        edge_index = batch_data.edge_index
        batch = batch_data.batch
        
        idx = batch_data.idx
        
        mol_bias, mol_embed = self.molecule_embedding(x, edge_index, batch)  # shape [M, D]
        
        side_bias, side_embed = self.side_effect_embedding(side_effect_ids)  # shape [S, D]

        self.side_bias = side_bias
        self.mol_bias = mol_bias.unsqueeze(-1)
        
        scores = mol_embed @ side_embed.T + self.global_bias + side_bias + mol_bias.unsqueeze(-1)               # shape [M, S]
        
        if send_embs:
            return scores, idx, mol_embed, side_embed
        
        return scores, idx  # return scores and indices of drugs in the batch  


class MessagePassingEncoderBias(nn.Module):
    def __init__(self, in_channels=72, hidden_channels=64, out_channels=16,
                predictor_depth=1, num_message_passes=3, dropout=0.1):
        super().__init__()        
        self.W_input = nn.Linear(in_channels, hidden_channels)
        self.W_hidden = nn.Linear(hidden_channels, hidden_channels)
        self.W_output = nn.Linear(in_channels + hidden_channels, hidden_channels)
        self.out_channels = out_channels
        
        self.aggregation = global_mean_pool
        self.batch_norm = nn.BatchNorm1d(hidden_channels)
        self.predictor_depth = predictor_depth
        self.num_message_passes = num_message_passes
        self.dropout = nn.Dropout(dropout)
        
        self.predictor_layers = nn.ModuleList()
        
        layers_left = predictor_depth
        while layers_left > 1:
            self.predictor_layers.append(nn.Linear(hidden_channels, hidden_channels))
            self.predictor_layers.append(nn.ReLU())
            layers_left -= 1
        
        self.predictor_layers.append(nn.Linear(hidden_channels, out_channels + 1))
        
        self.predictor = nn.Sequential(*self.predictor_layers)
    
    def get_reverse_edge_indices(self, edge_index: torch.Tensor) -> torch.Tensor:
        edge_tuples = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))
        edge_to_idx = {edge: i for i, edge in enumerate(edge_tuples)}
        
        reversed_edges = [(t, s) for s, t in edge_tuples]
        rev_edge_indices = [edge_to_idx[rev] for rev in reversed_edges]
        
        return torch.tensor(rev_edge_indices, dtype=torch.long)
    
    def aggregate_by_index(self, H, edge_index):
        index_torch = edge_index[1].unsqueeze(1).repeat(1, H.shape[1])
        M = torch.zeros(H.shape[0], H.shape[1], dtype=H.dtype, device=H.device).scatter_reduce_(
            0, index_torch, H, reduce='sum', include_self=False
        )[edge_index[0]]
        
        return M
    
    def message(self, H, edge_index, rev_edge_index):
        M_all = self.aggregate_by_index(H, edge_index)
        M_rev = H[rev_edge_index]
        
        return M_all - M_rev
    
    def update(self, M, H):
        H_t = self.W_hidden(M)
        H_t = F.relu(H + H_t)
        H_t = self.dropout(H_t)
        
        return H_t
    
    def message_passing(self, feats, edge_index):
        rev_edge_index = self.get_reverse_edge_indices(edge_index)
        
        H_0 = self.W_input(feats)
        H = F.relu(H_0)
        
        for _ in range(self.num_message_passes - 1):
            M = self.message(H, edge_index, rev_edge_index)
            H = self.update(M, H_0)
        
        H = self.W_output(torch.cat([feats, H], dim=1))
        H = F.relu(H)
        H = self.dropout(H)
        
        return H
    
    def forward(self, feats, edge_index, batch):
        representations = self.message_passing(feats, edge_index)
        representations = self.aggregation(representations, batch)
        representations = self.batch_norm(representations)
        
        preds = self.predictor(representations)

        bias, representation = preds[:, 0], preds[:, 1:]
        representation = F.relu(representation)
        
        return bias, representation

class GATEncoderBias(nn.Module):
    def __init__(self, in_channels=72, hidden_channels=64, out_channels=16,
                 dropout=0.1, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout)
        self.conv3 = GATConv(hidden_channels * heads, hidden_channels, dropout=dropout)
        
        self.aggregation = global_mean_pool
        self.batch_norm = nn.BatchNorm1d(hidden_channels)
        self.dropout = nn.Dropout(dropout)
        self.out_channels = out_channels
        
        # Predictor for bias (+1 outputs) and representation
        self.predictor = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels + 1)
        )
        
    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        
        representations = self.aggregation(x, batch)
        representations = self.batch_norm(representations)
        
        preds = self.predictor(representations)

        bias, representation = preds[:, 0], preds[:, 1:]
        representation = F.relu(representation)
        
        return bias, representation

class SideEffectBertEmbeddingBias(nn.Module):
    def __init__(self, bert_data, num_side_effects=994, embedding_size=16, 
                    negative_slope=0.01):
        super(SideEffectBertEmbeddingBias, self).__init__()
        self.bert_data = bert_data
        self.num_side_effects = num_side_effects
        self.embedding_size = embedding_size
        self.negative_slope = negative_slope
        
        self.embedding = nn.Sequential(
            nn.Linear(bert_data.shape[1], 128),
            nn.LeakyReLU(negative_slope=self.negative_slope),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=self.negative_slope),
            nn.Linear(64, embedding_size + 1)
        )
    
    def forward(self, side_effect):
        data = self.bert_data[side_effect,:]
        preds = self.embedding(data)

        bias, representation = preds[:, 0], preds[:, 1:]
        representation = F.relu(representation)
        
        return bias, representation

class GAT3(torch.nn.Module):
    def __init__(self, input_dim=109, input_dim_e=243, output_dim=200, output_dim_e=64, dropout=0.2, heads=10):
        super(GAT3, self).__init__()

        # graph layers : drug
        self.gcn1 = GATConv(input_dim, 128, heads=heads, dropout=dropout)
        self.gcn2 = GATConv(128 * heads, output_dim, heads=heads, dropout=dropout)
        self.gcn5 = GATConv(output_dim * heads, output_dim, dropout=dropout)
        self.fc_g1 = nn.Linear(output_dim, output_dim)
        self.fc_g2 = nn.Linear(output_dim, output_dim)

        # # graph layers : sideEffect
        self.gcn3 = GATConv(input_dim_e, 128, heads=heads, dropout=dropout)
        self.gcn4 = GATConv(128 * heads, output_dim, heads=heads, dropout=dropout)
        self.gcn6 = GATConv(output_dim * heads, output_dim, dropout=dropout)
        self.fc_g3 = nn.Linear(output_dim, output_dim)
        self.fc_g4 = nn.Linear(output_dim, output_dim)

        # activation and regularization
        self.relu = nn.ReLU()

    def forward(self, data_e, not_FC=True):
        x_e, edge_index_e = data_e.x, data_e.edge_index

        # 副作用
        x_e = self.relu(self.gcn3(x_e, edge_index_e))
        x_e = self.relu(self.gcn4(x_e, edge_index_e))
        x_e = self.relu(self.gcn6(x_e, edge_index_e))

        if not not_FC:
            x_e = self.relu(self.fc_g3(x_e))
            x_e = F.dropout(x_e, p=0.5, training=self.training)
            x_e = self.fc_g4(x_e)

        return x_e

class SideEffectBertEmbeddingBiasDSGAT(nn.Module):
    def __init__(self, bert_data, se_graph, num_side_effects=994, embedding_size=16, 
                    negative_slope=0.01):
        super(SideEffectBertEmbeddingBiasDSGAT, self).__init__()
        self.bert_data = bert_data
        self.num_side_effects = num_side_effects
        self.embedding_size = embedding_size
        self.negative_slope = negative_slope

        input_dim_e = se_graph.x.shape[1]
        self.gat = GAT3(output_dim=embedding_size + 1, input_dim_e=input_dim_e)
        self.se_graph = se_graph
        
        # self.embedding = nn.Sequential(
        #     nn.Linear(bert_data.shape[1] + 200, 128),
        #     nn.LeakyReLU(negative_slope=self.negative_slope),
        #     nn.Linear(128, 64),
        #     nn.LeakyReLU(negative_slope=self.negative_slope),
        #     nn.Linear(64, embedding_size + 1)
        # )
    
    def forward(self, side_effect):
        preds = self.gat(self.se_graph)

        bias, representation = preds[:, 0], preds[:, 1:]
        representation = F.relu(representation)
        
        return bias, representation