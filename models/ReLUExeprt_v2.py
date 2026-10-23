import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


from models.moe_runtime import (
    relu_switch_lb_loss,           # 计算配对/负载均衡损失（单卡）
    relu_switch_cardinality_loss,
    MoEAuxLossAutoScaler,          # 把 aux loss 放入全局收集器（可微）
)

class ReLURouter(nn.Module):
    def __init__(self,
                 d_model: int,
                 num_experts: int,
                 topk: int = 2,                 # 目标平均激活数（仅用于LB归一化，不做硬裁剪）
                 l1_coeff: float = 1e-3,         # >0 启用负载均衡损失
                 input_jitter_eps: Optional[float] = None,
                 layer_number: int = 1,
                 num_layers: int = 1):
        super().__init__()
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.num_experts = num_experts
        self.topk = int(topk)
        self.l1_coeff = float(l1_coeff)
        self.input_jitter_eps = input_jitter_eps
        self._jitter_sampler = None

        # 仅用于调试打印
        self.layer_number = layer_number
        self.num_layers = num_layers
        self.relu_sparsity_accum = 0.0

    def _apply_input_jitter(self, x: torch.Tensor) -> torch.Tensor:
        eps = self.input_jitter_eps
        if eps is None or eps <= 0:
            return x
        if self._jitter_sampler is None:
            self._jitter_sampler = torch.distributions.uniform.Uniform(
                torch.tensor(1.0 - eps, device=x.device),
                torch.tensor(1.0 + eps, device=x.device),
            ).rsample
        return x * self._jitter_sampler(x.shape)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        参数
        ----
        x:
            (..., d_model) 的任意张量。除最后一维之外会被展平，以便路由操作在
            `(N, d_model)` 的二维上完成。

        返回
        ----
        scores:
            `(N, E)` 的非负实数，代表每个 token 与专家的相对得分。
        routing_map:
            `(N, E)` 的布尔掩码，标记哪一个专家被激活。
        """
        x = self._apply_input_jitter(x)
        logits = self.gate(x).view(-1, self.num_experts)
        scores = F.relu(logits)
        routing_map = scores > 0

        if self.training and self.l1_coeff > 0.0:
            aux = self._compute_auxiliary_losses(scores, routing_map, logits)
            scores = MoEAuxLossAutoScaler.apply(scores, aux)

        self.last_router_probs = scores
        return scores, routing_map

    def _compute_auxiliary_losses(
        self,
        scores: torch.Tensor,
        routing_map: torch.Tensor,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        组合负载均衡与基数约束惩罚，使路由保持稳定。
        """
        lb_coeff = self.l1_coeff
        card_coeff = self.l1_coeff * 0.1

        lb_loss = relu_switch_lb_loss(scores, routing_map, topk=self.topk, coeff=lb_coeff)
        card_loss = relu_switch_cardinality_loss(logits, topk=self.topk, tau=0.8, coeff=card_coeff)
        aux = lb_loss + card_loss

        if hasattr(self, "_debug_step") and self._debug_step % 100 == 0:
            self._log_routing_statistics(lb_loss, card_loss, aux, routing_map)

        return aux

    def _log_routing_statistics(
        self,
        lb_loss: torch.Tensor,
        card_loss: torch.Tensor,
        aux: torch.Tensor,
        routing_map: torch.Tensor,
    ) -> None:
        usage_rates = routing_map.float().mean(dim=0)
        print(
            "Layer {layer}/{total}: LB_loss={lb:.6f}, "
            "Card_loss={card:.6f}, Total_aux={aux:.6f}, Usage={usage}".format(
                layer=self.layer_number,
                total=self.num_layers,
                lb=lb_loss.item(),
                card=card_loss.item(),
                aux=aux.item(),
                usage=usage_rates.cpu().numpy(),
            )
        )

class CausalDWConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5, dilation: int = 1, dropout: float = 0.0):
        super().__init__()
        assert kernel_size >= 1 and dilation >= 1
        self.pad = (kernel_size - 1) * dilation  # 仅左填充，严格因果
        self.dw = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            groups=channels,
            dilation=dilation,
            bias=True
        )
        self.pw = nn.Conv1d(channels, channels, kernel_size=1, bias=True)  # 轻量 pointwise（可选，保留表达力）
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)

        # 轻微缩放，避免放大波动（可选）
        nn.init.normal_(self.dw.weight, std=0.02)
        nn.init.normal_(self.pw.weight, std=0.02)

    def forward(self, y_win: torch.Tensor) -> torch.Tensor:
        """
        y_win: (BF, L, C)
        return: (BF, L, C)
        """
        x = y_win.transpose(1, 2)               # (BF, C, L)
        x = F.pad(x, (self.pad, 0))             # 仅左填充 = 因果
        x = self.dw(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.pw(x)
        return x.transpose(1, 2)                 # 回到 (BF, L, C)

class ReFTExpert(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, rank: int, dropout_rate: float):
        super().__init__()
        # A, B, C 为低秩矩阵
        self.A = nn.Linear(d_model, rank)
        self.B = nn.Linear(rank, hidden_dim)
        self.C = nn.Linear(hidden_dim, d_model)
        self.act = nn.SiLU()
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x):
       # ReFT变换，先A，后B和C
        return self.C(self.dropout(self.act(self.B(self.A(x)))))


# ---------------------------------------------------------------------
# Sparse execution optimized for ReLU routing (no per-expert tiny loops)
# ---------------------------------------------------------------------

def sparse_execute_relu(
    x: torch.Tensor,
    experts,
    scores: torch.Tensor,
    routing_map: torch.Tensor,
    normalize: str = "none",
    eps: float = 1e-9,
) -> torch.Tensor:
    """
    根据 ReLU 路由结果仅执行被激活专家，并将输出按指定策略融合。

    参数
    ----
    x:
        `(N, C)` 的输入特征。
    experts:
        `nn.ModuleList` 或任意可迭代容器，长度为 `E`。
    scores:
        `(N, E)` 的非负路由得分。
    routing_map:
        `(N, E)` 的布尔掩码，指示激活的专家。
    normalize:
        权重归一化策略：
        - `"none"`：使用原始 ReLU 得分；
        - `"count"`：均匀平均；
        - `"l1"`：对激活专家的得分作 L1 归一。
    eps:
        避免除零的数值稳定常数。
    """
    if scores.dim() != 2:
        raise ValueError("scores must be a 2D tensor of shape (N, E)")
    if routing_map.shape != scores.shape:
        raise ValueError("routing_map must have the same shape as scores")
    if x.shape[0] != scores.shape[0]:
        raise ValueError("x and scores must agree on the batch dimension")

    weights = _compute_moe_weights(scores, routing_map, normalize, eps)
    output = torch.zeros_like(x)

    for expert_index, expert in enumerate(experts):
        token_indices = torch.nonzero(routing_map[:, expert_index], as_tuple=False).squeeze(-1)
        if token_indices.numel() == 0:
            continue
        expert_input = x.index_select(0, token_indices)
        expert_output = expert(expert_input)
        expert_weight = weights.index_select(0, token_indices)[:, expert_index].unsqueeze(1)
        output.index_add_(0, token_indices, expert_output * expert_weight)

    return output


def _compute_moe_weights(
    scores: torch.Tensor,
    routing_map: torch.Tensor,
    normalize: str,
    eps: float,
) -> torch.Tensor:
    """
    按照指定策略计算专家融合权重。
    """
    active_mask = routing_map.float()
    masked_scores = scores * active_mask
    sum_scores = masked_scores.sum(dim=1, keepdim=True)

    if normalize == "none":
        return masked_scores
    if normalize == "count":
        expert_count = active_mask.sum(dim=1, keepdim=True)
        return active_mask / (expert_count + eps)
    if normalize == "l1":
        return masked_scores / (sum_scores + eps)

    raise ValueError("normalize must be one of {'none', 'count', 'l1'}")


class TimeMoE(nn.Module):
    """
    单 Router、按时间窗滑动执行 MoE（窗口重叠用累加计数归一）
    """
    def __init__(self, d_model, hidden_dim, rank, num_experts,
                 window_size, hop_size,
                 top_k=2,                 # 目标平均激活数（仅用于LB归一化）
                 l1_coeff=0.0,
                 router_ln=True,
                 expert_dropout=0.0, expert_activation="silu",
                 expert_residual_scale=1.0,
                 input_jitter_eps: Optional[float] = None):
        super().__init__()
        self.W = window_size
        self.H = hop_size
        self.num_experts = num_experts

        # 全局专家池（时间窗共享）
        self.experts = nn.ModuleList([
            ReFTExpert(d_model, hidden_dim, rank, expert_dropout)
            for _ in range(num_experts)
        ])

        # 单个 ReLURouter（可选 LN 预处理）
        self.pre_ln = nn.LayerNorm(d_model) if router_ln else nn.Identity()
        self.router = ReLURouter(d_model=d_model,
                                 num_experts=num_experts,
                                 topk=top_k,
                                 l1_coeff=float(l1_coeff),
                                 input_jitter_eps=input_jitter_eps,
                                 layer_number=1, num_layers=1)

        self.tconv = CausalDWConv1d(
                channels=d_model,
                kernel_size=5,
                dilation=1,
                dropout=0.0
            )

    def forward(self, x: torch.Tensor, return_stats: bool = False) -> torch.Tensor:
        """
        参数
        ----
        x:
            `(B*F, T, C)` 的序列张量，`B*F` 为批量与频带维展平后的结果。
        return_stats:
            保留接口以兼容旧代码，当前未使用。
        """
        if x.dim() != 3:
            raise ValueError("TimeMoE expects a 3D tensor of shape (B*F, T, C)")

        total_batch, total_frames, channels = x.shape
        aggregated = torch.zeros_like(x)
        counts = torch.zeros(total_batch, total_frames, 1, device=x.device, dtype=x.dtype)

        for start in range(0, total_frames, self.H):
            end = min(start + self.W, total_frames)
            if start >= end:
                break
            window_output = self._forward_window(x[:, start:end, :])
            aggregated[:, start:end, :] += window_output
            counts[:, start:end, :] += 1.0

        return aggregated / (counts + 1e-8)

    def _forward_window(self, window: torch.Tensor) -> torch.Tensor:
        batch_frames, length, channels = window.shape
        flattened = window.reshape(-1, channels)
        preprocessed = self.pre_ln(flattened)
        scores, routing_map = self.router(preprocessed)
        moe_output = sparse_execute_relu(flattened, self.experts, scores, routing_map, normalize="none")
        moe_output = moe_output.view(batch_frames, length, channels)
        return self.tconv(moe_output)
