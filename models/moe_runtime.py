# moe_runtime.py
import torch
from typing import Optional

def relu_switch_lb_loss(
    scores: torch.Tensor,        # [N, E] = relu(logits)
    routing_map: torch.Tensor,   # [N, E] = (scores>0)
    topk: int,                   # 目标激活数（仅用于归一化，不做裁剪）
    coeff: float,                # 系数
    group: Optional[object] = None,  # 可传 torch.distributed 进程组；单卡填 None
    eps: float = 1e-9
) -> torch.Tensor:
    """
    改进的负载均衡损失 - 自适应缩放版本：
    1. 使用相对损失而不是绝对损失
    2. 自适应调整损失幅度
    3. 防止对主损失造成过大影响
    """
    assert scores.dim() == 2 and routing_map.shape == scores.shape
    N, E = scores.shape
    device, dtype = scores.device, scores.dtype

    tokens_per_exp = routing_map.sum(dim=0).to(dtype)   # [E]
    n_tokens = torch.tensor(float(N), device=device, dtype=dtype)

    # 可选的分布式聚合
    if group is not None:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(tokens_per_exp, op=dist.ReduceOp.SUM, group=group)
            dist.all_reduce(n_tokens, op=dist.ReduceOp.SUM, group=group)

    # 计算每个专家的使用率
    usage_rates = tokens_per_exp / (n_tokens + eps)  # [E]
    target_usage = 1.0 / E
    
    # 方法1: 使用相对方差损失（相对于目标使用率）
    relative_variance = torch.var(usage_rates / (target_usage + eps))
    
    # 方法2: 最小使用率约束（使用相对惩罚）
    min_usage_ratio = usage_rates.min() / (target_usage + eps)
    min_usage_penalty = torch.relu(0.1 - min_usage_ratio)  # 如果使用率低于目标的10%
    
    # 方法3: 使用相对熵损失（更稳定的KL散度）
    # 避免log(0)的问题，使用平滑的分布
    smooth_usage = usage_rates + eps
    smooth_target = torch.full_like(usage_rates, target_usage) + eps
    
    # 计算相对熵
    relative_entropy = (smooth_usage * torch.log(smooth_usage / smooth_target)).sum()
    
    # 组合损失 - 增强负载均衡效果
    # 学习率调小后，增加各项损失的权重
    loss_raw = 1.5 * relative_variance + 2.0 * min_usage_penalty + 0.2 * relative_entropy
    
    # 自适应缩放：根据专家数量调整损失幅度
    adaptive_scale = 1.0 / (E ** 0.5)  # 专家越多，单次损失越小
    
    return torch.tensor(coeff * adaptive_scale, device=device, dtype=dtype) * loss_raw
    
def relu_switch_cardinality_loss(
    logits: torch.Tensor,   # [N,E]  —— ReLU 前的 logits
    topk: int,
    tau: float = 1.0,
    theta: float = 0.0,
    alpha_token: float = 0.5,   # token级 vs 批均值
    use_l2: bool = True,
    coeff: float = 1.0,         # <<< 新增：系数
    normalize_by_k: bool = True # 让标度对 K 不敏感
) -> torch.Tensor:
    q = torch.sigmoid((logits - theta) / tau)  # 软开关
    c = q.sum(dim=-1)                          # 每 token 的软激活数

    if use_l2:
        Lk_token = ((c - topk) ** 2).mean()
        Lk_batch = (c.mean() - topk).pow(2)
    else:
        Lk_token = (c - topk).abs().mean()
        Lk_batch = (c.mean() - topk).abs()

    Lk = alpha_token * Lk_token + (1 - alpha_token) * Lk_batch
    if normalize_by_k:
        Lk = Lk / max(float(topk), 1.0)        # 归一，便于和 L_bal 同级
    return coeff * Lk

class MoEAuxLossAutoScaler(torch.autograd.Function):
    """
    与原版接口一致的简化实现：
    - forward: 原样返回 activation，并把 aux_loss 注册到全局 moe_aux_graph（参与反传）
    - backward: 对 activation 的梯度直通；aux_loss 的梯度由 total_loss += moe_aux_graph.total() 提供
    """
    @staticmethod
    def forward(ctx, activation: torch.Tensor, aux_loss: torch.Tensor) -> torch.Tensor:
        if isinstance(aux_loss, torch.Tensor):
            moe_aux_graph.add(aux_loss)
        return activation

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # activation 梯度不变；aux 的梯度通过 total_loss += moe_aux_graph.total() 注入
        return grad_output, None


from contextlib import contextmanager


class _MoeAuxGraph:
    """收集各层的可微 aux loss（参与反传）"""
    def __init__(self):
        self.enabled = False
        self._terms = []

    def reset(self):
        self._terms = []

    def add(self, t: torch.Tensor):
        # t 必须是参与反传的张量（不要 detach）
        if self.enabled:
            self._terms.append(t)

    def total(self, device=None, reduction="sum"):
        if not self._terms:
            # 返回一个可微的 0 张量，避免 loss += 0 报错
            return torch.zeros((), device=device, dtype=torch.float32)
        terms = self._terms
        return torch.stack(terms).sum() if reduction == "sum" else torch.stack(terms).mean()

moe_aux_graph = _MoeAuxGraph()

@contextmanager
def collect_moe_aux(enabled: bool = True):
    """在 train 端用 with collect_moe_aux(): 包住一次前向，最后 aux 可一次性取 total()"""
    prev = moe_aux_graph.enabled
    moe_aux_graph.enabled = bool(enabled)
    moe_aux_graph.reset()
    try:
        yield moe_aux_graph
    finally:
        moe_aux_graph.enabled = prev
