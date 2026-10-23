import torch.nn as nn
from models.ReLUExeprt_v2 import TimeMoE
from einops.layers.torch import Rearrange

def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class FeedForwardModule(nn.Module):
    def __init__(self, dim, mult=4, dropout=0):
        super().__init__()
        self.ffm = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * mult),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ffm(x)


class ConformerConvModule(nn.Module):
    def __init__(self, dim, expansion_factor=2, kernel_size=31, dropout=0.):
        super().__init__()
        inner_dim = dim * expansion_factor
        self.ccm = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n c -> b c n'),
            nn.Conv1d(dim, inner_dim * 2, 1),
            nn.GLU(dim=1),
            nn.Conv1d(inner_dim, inner_dim, kernel_size=kernel_size,
                      padding=get_padding(kernel_size), groups=inner_dim),
            nn.BatchNorm1d(inner_dim),
            nn.SiLU(),
            nn.Conv1d(inner_dim, dim, 1),
            Rearrange('b c n -> b n c'),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ccm(x)

class AttentionModule(nn.Module):
    def __init__(self, dim, n_head=8, dropout=0.):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_head, dropout=dropout, batch_first=True)
        self.layernorm = nn.LayerNorm(dim)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        x_norm = self.layernorm(x)
        out, _ = self.attn(x_norm, x_norm, x_norm,
                            attn_mask=attn_mask,
                            key_padding_mask=key_padding_mask)
        return out

class ConformerBlock(nn.Module):
    def __init__(self, dim, gate, n_head=4, ffm_mult=4, ccm_expansion_factor=2, ccm_kernel_size=31,
                 ffm_dropout=0., attn_dropout=0., ccm_dropout=0.,
                 num_bands=2, num_experts=4, moe_k=2, window_size=128):
        super().__init__()
        self.ffm1 = FeedForwardModule(dim, ffm_mult, dropout=ffm_dropout)
        self.attn = AttentionModule(dim, n_head, dropout=attn_dropout)
        self.ccm = ConformerConvModule(dim, ccm_expansion_factor, ccm_kernel_size, dropout=ccm_dropout)
        # replace second FFN with MoE
        self.gate = gate
        if self.gate == "freq":
            self.ffm2 =  FeedForwardModule(dim, ffm_mult, dropout=ffm_dropout)
        elif self.gate == "time":
            self.ffm2 = TimeMoE(
                d_model=dim, hidden_dim=2*dim, rank=4,
                num_experts=4, window_size=128, hop_size=64,
                top_k=2, l1_coeff=1e-3,
                router_ln=True,
                expert_dropout=0.1, expert_activation="silu",
                expert_residual_scale=0.9)
        self.post_norm = nn.LayerNorm(dim)
        self.balance_loss = None

    def forward(self, x):
        # FFN1
        x = x + 0.5*self.ffm1(x)
        # Attention
        x = x + self.attn(x)
        # Convolution
        x = x + self.ccm(x)
        x = x + 0.5*self.ffm2(x)
        # MoE FFN2

        #lb_loss = getattr(self.ffm2, 'balance_loss', None),
        #if lb_loss is not None:
        #    self.balance_loss = lb_loss
        
        # final norm
        x = self.post_norm(x)
        return x
