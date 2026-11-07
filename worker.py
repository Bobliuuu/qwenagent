"""Internal stateful workers injected into a frozen decoder layer's residual stream.
Each worker slot has private persistent state. Slots share weights (count-agnostic).
Router: token -> slot assignment by attention between token query and slot key (slot keys derived
from random per-sequence slot identity + evolving state, so binding is dynamic, not learned per-index).
Return: sum_k phi(s_k) added to residual at every position BEFORE remaining layers."""
import torch, torch.nn as nn, torch.nn.functional as F

class Workers(nn.Module):
    def __init__(s, d_model, K=4, d_s=64, mode="multi", d_k=64, n_adapter=None):
        super().__init__()
        s.mode, s.K, s.d_s = mode, K, d_s
        if mode == "single": s.K_eff, s.ds_eff = 1, d_s * K      # one big shared state, same total state size
        else: s.K_eff, s.ds_eff = K, d_s
        ds = s.ds_eff
        s.inp = nn.Linear(d_model, ds)
        s.q = nn.Linear(d_model, d_k); s.k = nn.Linear(ds, d_k)
        s.wg = nn.Linear(d_model, 1)
        s.cell = nn.GRUCell(ds, ds)
        s.ln = nn.LayerNorm(ds)
        s.phi = nn.Sequential(nn.Linear(ds, 4 * ds), nn.GELU(), nn.Linear(4 * ds, d_model))
        nn.init.zeros_(s.phi[2].weight); nn.init.zeros_(s.phi[2].bias)
        if mode == "adapter":   # no persistent state: per-token MLP, param-matched-ish
            s.ad = nn.Sequential(nn.Linear(d_model, 4 * ds), nn.GELU(), nn.Linear(4 * ds, d_model))
            nn.init.zeros_(s.ad[2].weight); nn.init.zeros_(s.ad[2].bias)
        s.ablate = None   # None|'zero'|'shuffle'|'drop:<k>'
        s.stats = None

    def forward(s, h, mask):
        """h: [B,T,D] float; mask: [B,T] 1=real. returns delta [B,T,D]."""
        B, T, D = h.shape
        if s.mode == "adapter":
            return s.ad(h.float()) * mask[..., None]
        hf = h.float(); K, ds = s.K_eff, s.ds_eff
        st = torch.randn(B, K, ds, device=h.device) * 0.5          # fresh slot identities per sequence
        x = s.inp(hf); q = s.q(hf); wg = torch.sigmoid(s.wg(hf)).squeeze(-1)
        outs = []; assigns = []
        for t in range(T):
            m = mask[:, t]
            key = s.k(st)                                           # [B,K,dk]
            logit = (key * q[:, t, None]).sum(-1) / key.shape[-1] ** 0.5
            a = F.softmax(logit * 3.0, dim=-1) if K > 1 else torch.ones(B, 1, device=h.device)
            if s.ablate and s.ablate.startswith("drop:"):
                a = a.clone(); a[:, int(s.ablate[5:])] = 0
            g = (a * wg[:, t, None] * m[:, None]).unsqueeze(-1)    # [B,K,1]
            new = s.cell(x[:, t, None].expand(B, K, ds).reshape(B * K, ds), st.reshape(B * K, ds)).view(B, K, ds)
            st = (1 - g) * st + g * new
            outs.append(s.phi(s.ln(st)).sum(1))                     # [B,D] order-invariant over slots
            assigns.append(a.detach())
        d = torch.stack(outs, 1) * mask[..., None]
        s.stats = torch.stack(assigns, 1)
        if s.ablate == "zero": d = d * 0
        if s.ablate == "shuffle": d = d.roll(1, 0)
        return d
