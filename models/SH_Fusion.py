import torch
import torch.nn as nn
import torch.nn.functional as F

from models import model_scp as mscp
from models.MoeComformer_FFN import FeedForwardModule


def auto_pad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class DarkConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, auto_pad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.LeakyReLU(0.01, inplace=True) if act else nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DarkResidualBlock(nn.Module):
    def __init__(self, c1, shortcut=True):
        super().__init__()
        c2 = c1 // 2
        self.shortcut = shortcut
        self.layer1 = DarkConv(c1, c2, p=0)
        self.layer2 = DarkConv(c2, c1, k=3)

    def forward(self, x):
        residual = x
        out = self.layer1(x)
        out = self.layer2(out)
        out = out + residual
        if self.shortcut:
            out = out + residual
        return out


class DarkNetImageEncoder(nn.Module):
    def __init__(self, embed_dim=64, out_hw=(14, 14), init_weight=True):
        super().__init__()
        self.out_hw = out_hw
        self.features = nn.Sequential(
            DarkConv(3, 32, 3),
            DarkConv(32, 64, 3, 2),
            *self._make_layer(64, num_blocks=1),
            DarkConv(64, 128, 3, 2),
            *self._make_layer(128, num_blocks=2),
            DarkConv(128, 256, 3, 2),
            *self._make_layer(256, num_blocks=8),
            DarkConv(256, 512, 3, 2),
            *self._make_layer(512, num_blocks=8),
            DarkConv(512, 1024, 3, 2),
            *self._make_layer(1024, num_blocks=4),
        )
        self.proj = nn.Conv2d(1024, embed_dim, kernel_size=1, bias=True)

        if init_weight:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        feat = self.proj(feat)
        feat = F.interpolate(feat, size=self.out_hw, mode="bilinear", align_corners=False)
        tokens = feat.flatten(2).transpose(1, 2).contiguous()
        return tokens

    @staticmethod
    def _make_layer(in_channels, num_blocks):
        return nn.Sequential(*[DarkResidualBlock(in_channels) for _ in range(num_blocks)])

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="leaky_relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)


class AdaptiveImageConditioner(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.image_norm = nn.LayerNorm(embed_dim)
        self.audio_norm = nn.LayerNorm(embed_dim)
        self.affine = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, embed_dim * 2),
        )
        self.gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid(),
        )

    def forward(self, image_tokens, audio_tokens):
        audio_context = self.audio_norm(audio_tokens.mean(dim=1))
        scale, shift = self.affine(audio_context).chunk(2, dim=-1)
        gate = self.gate(audio_context).unsqueeze(1)

        conditioned = self.image_norm(image_tokens)
        conditioned = conditioned * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return image_tokens + gate * conditioned


class HyperbolicProjector(nn.Module):
    def __init__(self, in_dim, out_dim, init_c=1.0):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )
        self.log_c = nn.Parameter(torch.log(torch.tensor(float(init_c))))

    def curvature(self):
        return F.softplus(self.log_c) + 1e-6

    def expmap0(self, x):
        c = self.curvature().to(dtype=x.dtype, device=x.device)
        sqrt_c = torch.sqrt(c)
        norm = torch.norm(x, dim=-1, keepdim=True).clamp_min(1e-6)
        scaled = torch.tanh(sqrt_c * norm) * x / (sqrt_c * norm)
        max_norm = (1.0 - 1e-4) / sqrt_c
        point_norm = torch.norm(scaled, dim=-1, keepdim=True).clamp_min(1e-6)
        factor = torch.clamp(max_norm / point_norm, max=1.0)
        return scaled * factor

    def logmap0(self, y):
        c = self.curvature().to(dtype=y.dtype, device=y.device)
        sqrt_c = torch.sqrt(c)
        norm = torch.norm(y, dim=-1, keepdim=True).clamp_min(1e-6)
        return torch.atanh(sqrt_c * norm) * y / (sqrt_c * norm)

    def dist(self, x, y):
        c = self.curvature().to(dtype=x.dtype, device=x.device)
        sqrt_c = torch.sqrt(c)
        x2 = torch.sum(x * x, dim=-1, keepdim=True)
        y2 = torch.sum(y * y, dim=-1, keepdim=True)
        diff2 = torch.sum((x - y) * (x - y), dim=-1, keepdim=True)
        denom = (1.0 - c * x2) * (1.0 - c * y2)
        z = 1.0 + 2.0 * c * diff2 / denom.clamp_min(1e-6)
        return torch.acosh(z.clamp_min(1.0 + 1e-6)) / sqrt_c

    def forward(self, x):
        tangent = self.proj(x)
        hyper = self.expmap0(tangent)
        return tangent, hyper


class StableGatedSHFusion(nn.Module):
    def __init__(self, in_ch_mel=4, in_ch_sh=25, hidden_ch=32, reduction=4, gate_min=0.0):
        super().__init__()
        self.mel_enc = nn.Sequential(
            nn.Conv2d(in_ch_mel, hidden_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.SiLU(inplace=True),
        )
        self.sh_enc = nn.Sequential(
            nn.Conv2d(in_ch_sh, hidden_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.SiLU(inplace=True),
        )

        mid = max(hidden_ch // reduction, 4)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_ch, mid, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, hidden_ch, 1),
        )

        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, -2.0)

        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.gate_min = float(gate_min)

        self.fuse = nn.Sequential(
            nn.Conv2d(hidden_ch, hidden_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.SiLU(inplace=True),
        )
        self.proj = nn.Conv2d(hidden_ch, in_ch_mel, kernel_size=1, bias=True)

    def forward(self, mel, shc):
        em = self.mel_enc(mel)
        es = self.sh_enc(shc)

        g = torch.sigmoid(self.gate(es))
        if self.gate_min > 0:
            g = self.gate_min + (1 - self.gate_min) * g

        e = em + self.alpha * (g * es)
        e = self.fuse(e)
        return self.proj(e)


def stft_mag_pha(x, n_fft=512, hop_length=128, win_length=512, center=True):
    B, C, T = x.shape
    window = torch.hann_window(win_length, device=x.device)
    x2 = x.reshape(B * C, T)
    spec = torch.stft(
        x2,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        return_complex=True
    )
    mag = spec.abs().reshape(B, C, spec.shape[-2], spec.shape[-1])
    pha = torch.angle(spec).reshape(B, C, spec.shape[-2], spec.shape[-1])
    return mag, pha


def dct_1d(x, norm="ortho"):
    T = x.shape[-1]
    device, dtype = x.device, x.dtype
    n = torch.arange(T, dtype=dtype, device=device)
    k = n.unsqueeze(1)
    mat = torch.cos(torch.pi * k * (2 * n + 1) / (2 * T))
    if norm == "ortho":
        mat[0] *= 1.0 / torch.sqrt(torch.tensor(2.0, dtype=dtype, device=device))
        mat *= torch.sqrt(torch.tensor(2.0 / T, dtype=dtype, device=device))
    else:
        mat *= torch.sqrt(torch.tensor(2.0 / T, dtype=dtype, device=device))
    x2 = x.reshape(-1, T)
    y2 = x2 @ mat.t()
    return y2.reshape(x.shape)


def idct_1d(x, norm="ortho"):
    T = x.shape[-1]
    device, dtype = x.device, x.dtype
    n = torch.arange(T, dtype=dtype, device=device)
    k = n.unsqueeze(1)
    mat = torch.cos(torch.pi * (2 * k + 1) * n / (2 * T))
    if norm == "ortho":
        mat[:, 0] *= 1.0 / torch.sqrt(torch.tensor(2.0, dtype=dtype, device=device))
        mat *= torch.sqrt(torch.tensor(2.0 / T, dtype=dtype, device=device))
    else:
        mat *= torch.sqrt(torch.tensor(2.0 / T, dtype=dtype, device=device))
    x2 = x.reshape(-1, T)
    y2 = x2 @ mat.t()
    return y2.reshape(x.shape)


def _ck_local(name, t):
    if not torch.isfinite(t).all():
        raise RuntimeError(f"[NaN/Inf@SCP] {name}")


def compute_scp_features(mag, num_bands=4, n_mfcc=13):
    B, Freq, T = mag.shape
    band_size = Freq // num_bands
    scp_list = []
    _ck_local("mag_in", mag)
    for i in range(num_bands):
        s = i * band_size
        e = (i + 1) * band_size if i < num_bands - 1 else Freq
        band_mag = mag[:, s:e, :]
        band_mag = band_mag.clamp_min(1e-8)
        log_energy = torch.log(band_mag)
        band_energy = log_energy.mean(dim=1, keepdim=True)

        x = band_energy.squeeze(1)
        c = dct_1d(x, norm="ortho")
        if n_mfcc < T:
            c[:, n_mfcc:] = 0
        x_smooth = idct_1d(c, norm="ortho")
        scp_list.append(x_smooth.unsqueeze(1))

    scp = torch.cat(scp_list, dim=1)
    if scp.shape[1] != Freq:
        scp = scp.permute(0, 2, 1).unsqueeze(1)
        scp = F.interpolate(scp, size=(T, Freq), mode="bilinear", align_corners=False)
        scp = scp.squeeze(1).permute(0, 2, 1)

    scp = (scp - scp.mean(dim=(1, 2), keepdim=True)) / (scp.std(dim=(1, 2), keepdim=True) + 1e-8)
    scp = scp * mag.std(dim=(1, 2), keepdim=True) + mag.mean(dim=(1, 2), keepdim=True)
    return scp


class _HParams:
    def __init__(self, dense_channel):
        self.dense_channel = dense_channel


class TFNet(nn.Module):
    def __init__(
        self,
        num_cls=5,
        dense_channel=64,
        num_tsblocks=4,
        n_fft=512,
        hop_length=128,
        win_length=512,
        out_freq=224,
        out_time=16,
        scp_num_bands=4,
        scp_n_mfcc=13,
        img_out_hw=(14, 14),
    ):
        super().__init__()
        self.num_class = num_cls
        self.h = _HParams(dense_channel=dense_channel)

        self.n_fft = n_fft
        self.hop = hop_length
        self.win = win_length
        self.out_freq = out_freq
        self.out_time = out_time

        self.scp_num_bands = scp_num_bands
        self.scp_n_mfcc = scp_n_mfcc

        self.sh_fusion = StableGatedSHFusion(in_ch_mel=4, in_ch_sh=25)

        self.encoder = mscp.DenseEncoder(self.h, in_channel=9)
        self.blocks = nn.ModuleList([mscp.TSConformerBlockSeg4(self.h) for _ in range(num_tsblocks)])

        # 图像分支
        self.image_encoder = DarkNetImageEncoder(
            embed_dim=dense_channel,
            out_hw=img_out_hw,
            init_weight=True,
        )

        self.image_conditioner = AdaptiveImageConditioner(dense_channel)
        self.image_ffn = FeedForwardModule(dense_channel, mult=4, dropout=0.1)
        self.av_attention = mscp.ConvergeAttention(dim=dense_channel, head=4, proj_drop=0.1)
        self.av_ffn = FeedForwardModule(dense_channel, mult=4, dropout=0.1)
        self.hyperbolic_projector = HyperbolicProjector(dense_channel, dense_channel)
        self.similarity_scale = nn.Parameter(torch.tensor(1.0))

        self.img_out_hw = img_out_hw

        head_in_dim = dense_channel * 3 + 1
        self.cls_head1 = nn.Linear(head_in_dim, 512)
        self.cls_head2 = nn.Linear(512, num_cls)
        self.pos_head1 = nn.Linear(head_in_dim, 512)
        self.pos_head2 = nn.Linear(512, 3)

    def _map_to_tokens(self, x):
        return x.flatten(2).transpose(1, 2).contiguous()

    def forward(self, audio, shc, image):
        """
        audio: (B,4,T)
        shc:   (B,25,T)
        image: (B,3,H,W)
        """
        mag_a, pha_a = stft_mag_pha(audio, n_fft=self.n_fft, hop_length=self.hop, win_length=self.win)
        mag_s, _ = stft_mag_pha(shc, n_fft=self.n_fft, hop_length=self.hop, win_length=self.win)

        mag_a = F.interpolate(mag_a, size=(self.out_freq, self.out_time), mode="bilinear", align_corners=False)
        mag_s = F.interpolate(mag_s, size=(self.out_freq, self.out_time), mode="bilinear", align_corners=False)

        fused_mag_4 = self.sh_fusion(mag_a, mag_s)
        fused_mag_4_pos = F.softplus(fused_mag_4) + 1e-8

        pha_ref = pha_a[:, 0, :, :]
        pha_ref = F.interpolate(
            pha_ref.unsqueeze(1),
            size=(self.out_freq, self.out_time),
            mode="bilinear",
            align_corners=False
        ).squeeze(1)

        scp_list = []
        with torch.autocast(device_type=audio.device.type, enabled=False):
            for c in range(4):
                scp_c = compute_scp_features(
                    fused_mag_4_pos[:, c].float(),
                    num_bands=self.scp_num_bands,
                    n_mfcc=self.scp_n_mfcc
                )
                scp_list.append(scp_c.to(fused_mag_4.dtype))
        scp_4 = torch.stack(scp_list, dim=1)

        x = torch.cat(
            [
                fused_mag_4,
                scp_4,
                pha_ref.unsqueeze(1),
            ],
            dim=1
        ).permute(0, 1, 3, 2).contiguous()  # (B,9,16,224)

        encoded = self.encoder(x)

        block_out = encoded
        for blk in self.blocks:
            encoded, block_out = blk(encoded, return_intermediate=True)

        audio_map = F.adaptive_avg_pool2d(block_out, output_size=self.img_out_hw)
        audio_tokens = self._map_to_tokens(audio_map)        # (B, 14*14, C)

        image_tokens = self.image_encoder(image)             # (B, 14*14, C)
        image_tokens = self.image_conditioner(image_tokens, audio_tokens)
        image_tokens = image_tokens + self.image_ffn(image_tokens)
        
        fused_tokens = self.av_attention(audio_tokens, image_tokens)
        fused_tokens = fused_tokens + self.av_ffn(fused_tokens)

        feat_audio = audio_tokens.mean(dim=1)
        feat_img = image_tokens.mean(dim=1)
        feat_fused = fused_tokens.mean(dim=1)

        feat_audio_tan, feat_audio_hyp = self.hyperbolic_projector(feat_audio)
        feat_img_tan, feat_img_hyp = self.hyperbolic_projector(feat_img)
        hyp_dist = self.hyperbolic_projector.dist(feat_audio_hyp, feat_img_hyp)
        similarity = torch.exp(-F.softplus(self.similarity_scale) * hyp_dist)

        feat = torch.cat([feat_audio_tan, feat_img_tan, feat_fused, similarity], dim=1)

        cls = F.relu(self.cls_head1(feat))
        cls = self.cls_head2(cls)

        pos = F.relu(self.pos_head1(feat))
        pos = self.pos_head2(pos)

        return cls, pos
