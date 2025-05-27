import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
import numpy as np
class LayerNorm(nn.LayerNorm):
    """LayerNorm but with an optional bias."""
    def __init__(self, ndim, bias=True):
        super().__init__(ndim, elementwise_affine=bias)

class MultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.dropout_module = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, embed_dim = x.size()
        
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if attn_mask is not None:
            attn_weights = attn_weights.masked_fill(attn_mask == 0, float("-inf"))
            
        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
            
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout_module(attn_weights)
        
        attn = torch.matmul(attn_weights, v)
        attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        out = self.out_proj(attn)
        
        if need_weights:
            return out, attn_weights
        return out, None

class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_drop,
        )
        self.norm2 = LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            drop=drop,
        )
        self.drop_path = nn.Identity() if drop_path == 0.0 else nn.Dropout(drop_path)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x))[0])
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class IMUEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        num_blocks: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.7,
    ):
        super().__init__()
        
        # IMU stem
        self.imu_stem = nn.Sequential(
            nn.Linear(48, embed_dim, bias=False),
            LayerNorm(embed_dim),
        )
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, 251, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_blocks)]
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
            )
            for i in range(num_blocks)
        ])
        
        # Output head
        self.head = nn.Sequential(
            LayerNorm(embed_dim),
            nn.Dropout(0.5),
            nn.Linear(embed_dim, 1024, bias=False),
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, channels, sequence_length]
        batch_size = x.shape[0]
        
        # Process through IMU stem
        x = x.transpose(1, 2)  # [batch_size, sequence_length, channels]
        x = x.reshape(batch_size, -1, 48)  # Reshape to match expected input
        x = self.imu_stem(x)
        
        # Add position embeddings
        x = x + self.pos_embed[:, :x.size(1), :]
        
        # Add CLS token
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        # Process through transformer blocks
        x = self.blocks(x)
        
        # Get CLS token output
        x = x[:, 0]
        
        # Process through head
        x = self.head(x)
        
        return x

def process_imu_data(imu_data: torch.Tensor) -> torch.Tensor:
    """Process IMU data into the correct format for the encoder.
    
    Args:
        imu_data: Raw IMU data of shape [sequence_length, channels]
        
    Returns:
        Processed tensor of shape [1, channels, sequence_length]
    """
    if isinstance(imu_data, np.ndarray):
        imu_data = torch.from_numpy(imu_data)
    return imu_data.transpose(0, 1).unsqueeze(0).float()

class IMUEncoderQuick(IMUEncoder):
    """A quick version of the IMU encoder that loads pretrained weights."""
    def __init__(self, pretrained_path: str = ".checkpoints/imu_encoder.pth"):
        super().__init__()
        self.load_state_dict(torch.load(pretrained_path))
        self.eval() 
        
if __name__ == "__main__":
    imu_encoder = IMUEncoderQuick()
    imu_data = np.random.randn(2000, 6)
    imu_encoder.forward(process_imu_data(imu_data))