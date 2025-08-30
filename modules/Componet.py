import torch
import torch.nn as nn
import math

class PeriodicEmbedding(nn.Module):
    def __init__(self, T: int, embedding_dim: int, sigma: float = 1.0):
        """
        T: 编码维度（需为偶数）
        sigma: 初始化频率参数的正态分布标准差
        """
        super().__init__()
        assert T % 2 == 0, "T must be even"
        self.T = T
        self.c = nn.Parameter(torch.normal(0, sigma, size=(1, T//2)))
        self.linear = nn.Linear(T, embedding_dim)
    
    def forward(self, x: torch) -> torch.Tensor:
        # 生成相位: [batch_size, 1] -> [batch_size, T//2]
        phases = 2 * math.pi * self.c * x  
        
        # 分解: [batch_size, T//2] -> [batch_size, T]
        sin_enc = torch.sin(phases)
        cos_enc = torch.cos(phases)
        v = torch.cat([sin_enc, cos_enc], dim=-1)  
        
        return self.linear(v) # [batch_size, T//2] -> [batch_size, embedding_dim]

class RiskEmbedding(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, dropout=0.1, is_double=False):
        super().__init__()
        self.is_double = is_double
        self.text_proj = nn.Sequential(
            nn.Linear(input_dim, input_dim//2), 
            nn.ReLU(),
            nn.Linear(input_dim//2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)) # 投影到风险空间里面
        
        self.time_proj = PeriodicEmbedding(T=hidden_dim, embedding_dim=hidden_dim)
        
        self.fusion = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout))
        
        self.incident_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, text_seq, time_seq):
        if len(text_seq.shape) == 4:
            B, S, N, I = text_seq.shape
            time_seq = time_seq.unsqueeze(2).expand(-1, -1, N, -1)
    
        text_feat = self.text_proj(text_seq) # [B,S,I] -> [B,S,H] or [B,S,N,I] -> [B,S,N,H]
        time_feat = self.time_proj(time_seq) # [B,S,1] -> [B,S,H] or [B,S,N,1] -> [B,S,N,H]

        combined = torch.cat((text_feat, time_feat), dim=-1) # [B,S,H] -> [B,S,H*2] or [B,S,N,H] -> [B,S,N,H*2]
        risk_embed = self.fusion(combined) # [B,S,H*2] -> [B,S,H] or [B,S,N,H*2] -> [B,S,N,H]
        if self.is_double:
            incident_pred = self.incident_head(risk_embed)  # [B,S,1] 或 [B,S,N,1]
        else:
            incident_pred = None
        return risk_embed, incident_pred

class GCNApproxFusion(nn.Module):
    def __init__(self, in_dim=512, out_dim=512, use_neighbors=True):
        super().__init__()
        self.linear_self = nn.Linear(in_dim, out_dim)
        self.linear_neigh = nn.Linear(in_dim, out_dim)
        self.activation = nn.ReLU()  # 或 GELU，按需替换
        self.use_neighbors = use_neighbors

    def forward(self, self_feat, neighbor_feat):
        """
        self_feat:     [B, T, D]          主节点特征
        neighbor_feat: [B, T, K, D]       邻居特征
        return:        [B, T, D]          融合后的特征
        """
        if self.use_neighbors and neighbor_feat is not None:
            neighbor_mean = neighbor_feat.mean(dim=2)  # [B, T, D]
            fused = self.linear_self(self_feat) + self.linear_neigh(neighbor_mean)
            return self.activation(fused)
        else:
            # 不使用邻居：直接返回 self 分支（可保留同维线性映射，参数共享）
            return self_feat
        

class RiskExposure(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.time_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.Dropout(dropout))

        self.fusion = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim))
        
        self.bilstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,  # 因为是双向LSTM，输出维度会乘2
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
    def forward(self, risk, staytime):
        staytime = self.time_proj(staytime)
        combined = torch.cat((risk, staytime), dim=-1)
        ExposurEmbed = self.fusion(combined)
        ExposurEmbed, _ = self.bilstm(ExposurEmbed)
        return ExposurEmbed
    
class TimePatchLSTM(nn.Module):
    def __init__(self, patch_size: int, input_dim: int, hidden_dim: int, num_layers=1):
        super().__init__()
        self.patch_size = patch_size
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # x: (B, L, C)
        B, L, C = x.shape
        p = self.patch_size
        assert L % p == 0, f"L={L} 不能被 patch_size={p} 整除"
        N = L // p

        x = x.reshape(B, N, p, C).reshape(B * N, p, C)
        _, (h_n, _) = self.lstm(x)
        patch_repr = self.proj(h_n[-1])
        return patch_repr.view(B, N, C)  # (B, N, C)

class POIBERTEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, padding_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=padding_idx)

    def forward(self, poi_ids, neighbor_ids):
        poi_emb = self.embedding(poi_ids)
        poi_mask = (poi_ids != 0).unsqueeze(-1)  # mask padding
        poi_emb = poi_emb * poi_mask.float()
        poi_sum = poi_emb.sum(dim=2)
        poi_count = poi_mask.sum(dim=2).clamp(min=1)
        poi_emb = poi_sum / poi_count  # (B, seq_len, emb_dim)
    
        neighbor_emb = self.embedding(neighbor_ids)  # (B, seq_len, num_neighbors, max_seq, emb_dim)
        neighbor_mask = (neighbor_ids != 0).unsqueeze(-1)
        neighbor_emb = neighbor_emb * neighbor_mask.float()
        neighbor_sum = neighbor_emb.sum(dim=3)
        neighbor_count = neighbor_mask.sum(dim=3).clamp(min=1)
        neighbor_emb = neighbor_sum / neighbor_count  # (B, seq_len, num_neighbors, emb_dim)
        
        return poi_emb, neighbor_emb