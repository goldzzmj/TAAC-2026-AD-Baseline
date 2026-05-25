"""Muon 优化器: Momentum + Orthogonalization (Newton-Schulz)。

纯 PyTorch 实现，无外部依赖。
参考: https://github.com/KellerJordan/Muon
"""

import torch


def _newton_schulz_ortho(M: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Newton-Schulz 迭代正交化。

    将矩阵 M 逼近其最近的正交矩阵。必须在 float32 下计算以保证数值稳定性。

    Args:
        M: 输入矩阵，形状 (m, n)，m <= n。
        steps: Newton-Schulz 迭代步数，默认 5。

    Returns:
        正交化后的矩阵，形状与 M 相同。
    """
    # 强制 float32 计算
    orig_dtype = M.dtype
    M = M.float()

    # 归一化: M = G / ||G||_F，防止数值溢出
    norm = M.norm()
    if norm == 0:
        return M.to(orig_dtype)
    M = M / norm

    # Newton-Schulz 迭代: 三阶收敛到正交矩阵
    for _ in range(steps):
        A = M @ M.T
        B = A @ M
        M = (3.0 * M - B) / 2.0

    return M.to(orig_dtype)


class Muon(torch.optim.Optimizer):
    """Muon 优化器: Momentum + Orthogonalization，解耦 weight decay。

    对 2D 参数（Linear 权重）执行 Newton-Schulz 正交化 + momentum 更新；
    对非 2D 参数（1D bias / LayerNorm）使用标准 momentum SGD。

    Args:
        params: 待优化参数。
        lr: 学习率，默认 0.02。
        momentum: 动量系数，默认 0.95。
        nesterov: 是否使用 Nesterov 动量，默认 True。
        weight_decay: 解耦权重衰减系数，默认 0.01。
        ns_steps: Newton-Schulz 迭代步数，默认 5。
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.01,
        ns_steps: int = 5,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """执行一步参数更新。"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            wd = group["weight_decay"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # 初始化 momentum buffer
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)

                buf = state["momentum_buffer"]

                if p.dim() == 2:
                    # --- 2D 参数: Newton-Schulz 正交化 + momentum ---
                    ortho_grad = _newton_schulz_ortho(grad, steps=ns_steps)

                    # Momentum: buf = momentum * buf + ortho_grad
                    buf.mul_(momentum).add_(ortho_grad)

                    # Nesterov 修正
                    if nesterov:
                        update = buf + momentum * ortho_grad
                    else:
                        update = buf
                else:
                    # --- 非 2D 参数: 标准 momentum SGD（不做正交化）---
                    buf.mul_(momentum).add_(grad)

                    if nesterov:
                        update = buf + momentum * grad
                    else:
                        update = buf

                # 参数更新: param -= lr * update
                p.add_(update, alpha=-lr)

                # 解耦 weight decay: param -= lr * wd * param
                if wd > 0.0:
                    p.add_(p, alpha=-lr * wd)

        return loss
