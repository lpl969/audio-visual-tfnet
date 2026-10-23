import torch
import torch.nn as nn
from models.MoeComformer_FFN import ConformerBlock
import torch.nn.functional as F


class DenseBlock(nn.Module):
    def __init__(self, channels, kernel_size=(2, 3), depth=4):
        super().__init__()
        self.channels = channels
        self.depth = depth
        self.layers = nn.ModuleList([])
        for i in range(depth):
            dilation = 2 ** i
            pad_length = dilation
            self.layers.append(
                nn.Sequential(
                    nn.ConstantPad2d((1, 1, pad_length, 0), value=0.0),
                    nn.Conv2d(channels * (i + 1), channels, kernel_size, dilation=(dilation, 1)),
                    nn.InstanceNorm2d(channels, affine=True),
                    nn.PReLU(channels),
                )
            )

    def forward(self, x):
        skip = x
        for layer in self.layers:
            x = layer(skip)
            skip = torch.cat([x, skip], dim=1)
        return x


class DownsampleBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels),
        )

    def forward(self, x):
        return self.block(x)


class DenseEncoder(nn.Module):
    def __init__(self, h, in_channel, num_levels: int = 2):
        super().__init__()
        self.base_channels = h.dense_channel
        self.num_levels = num_levels

        self.stem = nn.Sequential(
            nn.Conv2d(in_channel, self.base_channels, kernel_size=(1, 1)),
            nn.InstanceNorm2d(self.base_channels, affine=True),
            nn.PReLU(self.base_channels),
        )

        self.dense_blocks = nn.ModuleList(
            DenseBlock(self.base_channels, depth=4) for _ in range(num_levels)
        )
        self.downsample_blocks = nn.ModuleList(
            DownsampleBlock(self.base_channels) for _ in range(num_levels)
        )

        self.bottleneck = DenseBlock(self.base_channels, depth=4)

    def forward(self, x):
        """
        返回
        ----
        latent:
            最深层表示，形状约为 (B, C, T, F / 2**num_levels)。
        """
        # skips = []
        x = self.stem(x)
        for dense_block, down_block in zip(self.dense_blocks, self.downsample_blocks):
            x = dense_block(x)
            # skips.append(x)
            x = down_block(x)
        latent = self.bottleneck(x)
        return latent


class ConvergeAttention(nn.Module):
    def __init__(self, dim=192, head=6, proj_drop=0.1):
        super().__init__()
        self.heads = head
        head_dim = dim // head
        self.scale = head_dim ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, 2 * dim, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(proj_drop)

    def forward(self, t, f):
        residual_x = t
        B, N, C = t.shape

        f = self.norm(f)
        t = self.norm(t)

        q = self.q(t).reshape(B, N, self.heads, C // self.heads).permute(0, 2, 1, 3).contiguous()
        k, v = self.kv(f).chunk(2, dim=-1)
        k = k.reshape(B, N, self.heads, C // self.heads).permute(0, 2, 1, 3).contiguous()
        v = v.reshape(B, N, self.heads, C // self.heads).permute(0, 2, 1, 3).contiguous()

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1).contiguous()
        x = self.proj(x)
        x = self.attn_drop(x)

        x = x + residual_x
        return x

class HybridFusion(nn.Module):
    def __init__(self, C, reduction=16, k=3, alpha_init=0.5):
        super().__init__()
        # — 全局门控路径（SE‐Style） —
        self.global_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),           # (B,C,1,1)
            nn.Conv2d(C, C//reduction, 1),
            nn.ReLU(),
            nn.Conv2d(C//reduction, C, 1),
            nn.Sigmoid()                       # g_glob ∈ [0,1]^{B×C×1×1}
        )
        # — 局部门控路径（DW‐Conv + Mish） —
        self.local_reduce = nn.Conv2d(2*C, C, 1)  # 先拼接时/频再降维
        self.local_dw   = nn.Conv2d(C, C, kernel_size=k, padding=k//2, groups=C)
        self.local_act  = nn.Mish()
        self.local_gate = nn.Sequential(
            nn.Conv2d(C, C, 1),
            nn.Sigmoid()                         # g_loc ∈ [0,1]^{B×C×T×F}
        )
        # — 融合权重 α —
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

    def forward(self, Ft, Ff):
        # Ff, Ft: (B, C, T, F)
        # 1) 全局门控
        g_glob = self.global_gate(Ff)                       # (B,C,1,1)

        # 2) 局部门控
        z = torch.cat([Ff, Ft], dim=1)                      # (B,2C,T,F)
        z = self.local_reduce(z)                            # (B, C, T, F)
        z = self.local_act(self.local_dw(z))                # (B, C, T, F)
        g_loc = self.local_gate(z)                          # (B, C, T, F)

        # 3) 融合两种门控
        # A) 乘法融合
        g = g_glob * g_loc                                   # 广播 g_glob 到 (B,C,T,F)
        # 或 B) 加权和融合
        # g = self.alpha * g_glob + (1-self.alpha) * g_loc

        # 4) 加权融合特征
        out = g * Ff + (1-g) * Ft
        return out

class PatchEmbed(nn.Module):
    def __init__(self, patch_size=(1, 1), stride=(1, 1), in_channel=1, embed_dim=192, normal=True):
        super().__init__()
        self.split_patch = nn.Conv2d(in_channel, embed_dim, kernel_size=patch_size, stride=stride)
        self.normal = nn.BatchNorm2d(embed_dim) if normal else nn.Identity()

    def forward(self, x):  # x: (B, C, T, F)
        x = self.split_patch(x)          # (B, embed_dim, T', F')
        x = self.normal(x)
        return x

class TSConformerBlockSeg4(nn.Module):

    def __init__(self, h, attn_heads=8, proj_drop=0.1):
        super().__init__()
        C = h.dense_channel
        patch_size = getattr(h, "patch_size", (1, 1))
        patch_stride = getattr(h, "patch_stride", patch_size)

        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            stride=patch_stride,
            in_channel=C,
            embed_dim=C,
            normal=True,
        )

        self.time_transformer = ConformerBlock(dim=C, gate='time', n_head=4)
        self.freq_transformer = ConformerBlock(dim=C, gate='freq', n_head=4)
        self.hybrid_fusion = HybridFusion(C, reduction=16, k=3)

        self.cross_attn_pre = ConvergeAttention(dim=C, head=attn_heads, proj_drop=proj_drop)
        self.cross_attn_fuse = ConvergeAttention(dim=C, head=attn_heads, proj_drop=proj_drop)

        self.proj_map = nn.Sequential(
            nn.Conv2d(C, C, kernel_size=1, bias=False),
            nn.BatchNorm2d(C),
            nn.SiLU()
        )
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def _map_to_tokens(self, m):
        return m.flatten(2).transpose(1, 2).contiguous()

    def _tokens_to_map(self, tok, T, F):
        B, N, C = tok.shape
        return tok.transpose(1, 2).contiguous().view(B, C, T, F)

    def forward(self, x, return_intermediate=False):
        B, C, T, Freq = x.shape
        res = x

        y1 = self.patch_embed(x)
        T1, F1 = y1.shape[2], y1.shape[3]

        y1_tok = self._map_to_tokens(y1)
        y2_tok = self.cross_attn_pre(y1_tok, y1_tok)
        y2_map = self._tokens_to_map(y2_tok, T1, F1)

        x_time = y1.permute(0, 3, 2, 1).contiguous().view(B * F1, T1, C)
        time = self.time_transformer(x_time)
        x_time = (time + x_time).view(B, F1, T1, C).permute(0, 3, 2, 1)

        x_freq = x_time.permute(0, 2, 3, 1).contiguous().view(B * T1, F1, C)
        freq = self.freq_transformer(x_freq)
        x_freq = (freq + x_freq).view(B, T1, F1, C).permute(0, 3, 1, 2).contiguous()

        out_hybrid = self.hybrid_fusion(x_time, x_freq)

        out_tok = self._map_to_tokens(out_hybrid)
        y2_tok = self._map_to_tokens(y2_map)
        fused_tok = self.cross_attn_fuse(out_tok, y2_tok)
        fused_map = self._tokens_to_map(fused_tok, T1, F1)

        fused_map = self.proj_map(fused_map)
        fused_map = F.interpolate(fused_map, size=(T, Freq), mode="bilinear", align_corners=False)

        out = self.alpha * out_hybrid + (1.0 - self.alpha) * fused_map
        out = F.interpolate(out, size=(T, Freq), mode="bilinear", align_corners=False)
        final = res + out

        if return_intermediate:
            return final, out
        return final
