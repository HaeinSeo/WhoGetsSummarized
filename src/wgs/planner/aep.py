"""AEP: Aspect Evidence Planner (B4 local, B5 context, B6 full).

r_k (local head): is there sufficient verified evidence for candidate k (own features only).
q_k (budget head): is k worth including under the fixed budget; context variants attend over
all candidates of the product. s_k = r_k * q_k.
Generators, BGE, and judges stay frozen; only this small network is trained.
"""
from __future__ import annotations

import torch
import torch.nn as nn

VARIANTS = ("local", "context", "full")


class AEP(nn.Module):
    def __init__(self, d_in: int, hidden: int = 256, heads: int = 4, layers: int = 1,
                 ffn: int = 512, dropout: float = 0.1, variant: str = "full"):
        super().__init__()
        assert variant in VARIANTS
        self.variant = variant
        self.proj = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, hidden), nn.GELU(),
                                  nn.Dropout(dropout))
        self.r_head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))
        if variant != "local":
            layer = nn.TransformerEncoderLayer(hidden, heads, ffn, dropout, batch_first=True)
            self.ctx = nn.TransformerEncoder(layer, layers)
        self.q_head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))

    def forward(self, x: torch.Tensor, n_ctx: int | None = None):
        """x: [n_cand, d_in] for one product. If n_ctx is given, only the first n_ctx rows act as
        context (later rows are frozen probes that see the context but are not attended to)."""
        h = self.proj(x)
        r = torch.sigmoid(self.r_head(h)).squeeze(-1)
        if self.variant == "local":
            hq = h
        else:
            n = n_ctx or h.shape[0]
            mask = torch.zeros(h.shape[0], h.shape[0], dtype=torch.bool, device=h.device)
            mask[:, n:] = True                       # nobody attends to probe rows
            mask[torch.arange(h.shape[0]), torch.arange(h.shape[0])] = False
            hq = self.ctx(h.unsqueeze(0), mask=mask).squeeze(0)
        q = torch.sigmoid(self.q_head(hq)).squeeze(-1)
        return r, q, r * q


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def losses(model: AEP, prod: dict, lam_inv: float, lam_remove: float) -> dict:
    """prod: {'x','v','y','label_mask', optional 'inv': [(x_other, k, k_other)], 'remove': [x_rm_with_probe]}"""
    bce = nn.functional.binary_cross_entropy
    r, q, s = model(prod["x"])
    m = prod["label_mask"]
    out = {"sup": bce(r[m], prod["v"][m]) + bce(q[m], prod["y"][m]) if m.any() else r.sum() * 0}
    if lam_inv and model.variant == "full" and prod.get("inv"):
        terms = []
        for x_other, k, k_other in prod["inv"]:
            _, _, s_o = model(x_other)
            terms.append((s[k] - s_o[k_other]).abs())
        out["inv"] = torch.stack(terms).mean()
    if lam_remove and model.variant == "full" and prod.get("remove"):
        terms = []
        for x_rm, n_ctx in prod["remove"]:
            r_rm, _, _ = model(x_rm, n_ctx=n_ctx)
            terms.append(-torch.log(1 - r_rm[-1] + 1e-6))   # probe is the last row
        out["remove"] = torch.stack(terms).mean()
    out["total"] = out["sup"] + lam_inv * out.get("inv", 0) + lam_remove * out.get("remove", 0)
    return out
