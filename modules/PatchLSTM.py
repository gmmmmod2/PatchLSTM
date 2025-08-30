import torch
import torch.nn as nn
from modules.Componet import RiskEmbedding, GCNApproxFusion, RiskExposure, TimePatchLSTM, POIBERTEncoder

class TransformerEncoder(nn.Module):
    def __init__(self, dim, num_heads, num_layers, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        return self.encoder(x)

class AttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, x):
        # x: (B, N, C)
        weights = torch.softmax(self.attn(x), dim=1)  # (B, N, 1)
        return (x * weights).sum(dim=1)               # (B, C)
    
class PatchLSTM(nn.Module):
    def __init__(
        self,*, 
        vocab_size: int = 700,
        input_dim: int = 768, 
        hidden_dim: int = 512, 
        patch_sizes: list = [10, 20, 40],
        num_layers: int = 1,
        num_heads: int = 8,
        n_mlp: int = 3,
        num_layers_intra: int = 2,
        num_layers_inter: int = 2,
        dropout: float = 0.1,
        is_double: bool = True,
        use_neighbors: bool = True
        ):
        super().__init__()
        self.num_scales = len(patch_sizes)
        self.patch_sizes = patch_sizes
        self.is_double = is_double
        
        self.encoder = POIBERTEncoder(vocab_size, input_dim)
        self.riskEmbedding = RiskEmbedding(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout, is_double=is_double)
        self.gcn = GCNApproxFusion(in_dim=hidden_dim, out_dim=hidden_dim, use_neighbors=use_neighbors)
        self.exp = RiskExposure(in_dim=hidden_dim, hidden_dim=hidden_dim, dropout=dropout)
        
        self.patch_extractors = nn.ModuleList([
            TimePatchLSTM(patch_size=p, input_dim=hidden_dim, hidden_dim=hidden_dim, num_layers=num_layers)
            for p in patch_sizes
        ])
        
        # Stage 1: Intra-Scale Transformer
        self.intra_transformer = TransformerEncoder(
            dim=hidden_dim, num_heads=num_heads, num_layers=num_layers_intra
        )
        
        # Stage 2: Inter-Scale Transformer
        self.scale_emb = nn.Parameter(torch.randn(self.num_scales, hidden_dim))  # 可学习尺度编码
        self.inter_transformer = TransformerEncoder(
            dim=hidden_dim, num_heads=num_heads, num_layers=num_layers_inter
        )
        
        self.mlp = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for _ in range(n_mlp)
        ])
        
        self.pool = AttentionPool(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 2)
        
    def forward(self, data: dict):
        poi_embeddings, neighbor_embeddings = self.encoder(data['poi_ids'], data['neighbor_ids'])
        
        pointRisk, pointIncidentPred = self.riskEmbedding(poi_embeddings, data['hour'])           # [B,L,H], [B,L,1]
        neighborRisk, neighborIncidentPred = self.riskEmbedding(neighbor_embeddings, data['hour'])# [B,L,K,H], [B,L,K,1]
        
        x = self.gcn(pointRisk, neighborRisk)
        x = self.exp(x, data['stay_duration'])
        
        scale_outputs = []
        for i, patch_model in enumerate(self.patch_extractors):
            patch_tokens = patch_model(x)  # (B, N_i, C)
            patch_tokens = self.intra_transformer(patch_tokens)  # 尺度内时序建模
            scale_outputs.append(patch_tokens)
        
        all_tokens = []
        for i, tokens in enumerate(scale_outputs):
            scale_code = self.scale_emb[i].unsqueeze(0).unsqueeze(0)  # (1,1,C)
            tokens = tokens + scale_code  # 加尺度信息
            all_tokens.append(tokens)
        all_tokens = torch.cat(all_tokens, dim=1)  # (B, sum(N_i), C)
        
        fused_tokens = self.inter_transformer(all_tokens)
        pooled = self.pool(fused_tokens)  # (B, C)
        
        for layer in self.mlp:
            pooled = layer(pooled)
        logits = self.classifier(pooled)   # (B, 2) — 输出两类logits: [normal, anomaly]
        
        return {
            "logits": logits,                              # 主任务：异常/风险分类
            "incident_pred_center": pointIncidentPred,     # [B, L, 1]
            "incident_pred_neighbors": neighborIncidentPred# [B, L, K, 1]
        }