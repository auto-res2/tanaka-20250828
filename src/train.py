import os
import math
import time
import copy
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------ Utilities ------------------------------

def set_seed(seed: int = 42):
    try:
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_auto():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def causal_mask(sz: int, device: torch.device):
    # returns additive mask (float) with -inf above diagonal
    mask = torch.full((sz, sz), float("-inf"), device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask


# ------------------------------ Model Components ------------------------------

class LayerNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.bias = nn.Parameter(torch.zeros(d))

    def forward(self, x, eps=1e-5):
        mu = x.mean(-1, keepdim=True)
        var = (x - mu).pow(2).mean(-1, keepdim=True)
        xhat = (x - mu) / torch.sqrt(var + eps)
        return xhat * self.weight + self.bias


class LoRAAdapter(nn.Module):
    """Simple LoRA-like adapter for Linear layers; supports merge() to fold into base weight."""
    def __init__(self, linear: nn.Linear, rank=4, alpha=8.0):
        super().__init__()
        self.linear = linear
        self.rank = rank
        self.alpha = alpha
        self.merged = False
        if rank > 0:
            # ensure adapter params are on same device/dtype as the wrapped linear
            w = self.linear.weight
            dev = w.device
            dtype = w.dtype
            self.A = nn.Parameter(torch.zeros(linear.out_features, rank, device=dev, dtype=dtype))
            self.B = nn.Parameter(torch.zeros(rank, linear.in_features, device=dev, dtype=dtype))
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))
            self.scaling = alpha / rank
        else:
            self.register_parameter('A', None)
            self.register_parameter('B', None)
            self.scaling = 0.0

    def forward(self, x):
        base = self.linear(x)
        if self.rank <= 0:
            return base
        update = F.linear(x, self.B)
        update = F.linear(update, self.A) * self.scaling
        return base + update

    @torch.no_grad()
    def merge(self):
        if self.merged or self.rank <= 0:
            return
        W = self.linear.weight
        delta = (self.A @ self.B) * self.scaling
        self.linear.weight.copy_(W + delta)
        self.merged = True


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads=1, bias=False):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.Wq = nn.Linear(d_model, d_model, bias=bias)
        self.Wk = nn.Linear(d_model, self.head_dim * n_kv_heads, bias=bias)
        self.Wv = nn.Linear(d_model, self.head_dim * n_kv_heads, bias=bias)
        self.Wo = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, x, attn_mask=None):
        B, T, C = x.shape
        q = self.Wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # B, h, T, d
        k = self.Wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)  # B, kv, T, d
        v = self.Wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        if self.n_kv_heads == 1:
            k = k.expand(B, self.n_heads, T, self.head_dim)
            v = v.expand(B, self.n_heads, T, self.head_dim)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            att = att + attn_mask  # additive mask with -inf
        p = F.softmax(att, dim=-1)
        y = (p @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.Wo(y)


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, d_ff, dropout=0.0):
        super().__init__()
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, n_kv_heads)
        self.ln2 = LayerNorm(d_model)
        self.mlp = SwiGLU(d_model, d_ff)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        x = x + self.drop(self.attn(self.ln1(x), attn_mask))
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x


class AdaptedBlock(nn.Module):
    """Wraps a TransformerBlock by replacing Linear layers with LoRAAdapter for distillation."""
    def __init__(self, base_block: TransformerBlock, rank=4, alpha=8.0):
        super().__init__()
        self.base = base_block
        # Replace submodules with adapter-wrapped versions
        self.base.attn.Wq = LoRAAdapter(self.base.attn.Wq, rank=rank, alpha=alpha)
        self.base.attn.Wk = LoRAAdapter(self.base.attn.Wk, rank=rank, alpha=alpha)
        self.base.attn.Wv = LoRAAdapter(self.base.attn.Wv, rank=rank, alpha=alpha)
        self.base.attn.Wo = LoRAAdapter(self.base.attn.Wo, rank=rank, alpha=alpha)
        self.base.mlp.w1 = LoRAAdapter(self.base.mlp.w1, rank=rank, alpha=alpha)
        self.base.mlp.w2 = LoRAAdapter(self.base.mlp.w2, rank=rank, alpha=alpha)
        self.base.mlp.w3 = LoRAAdapter(self.base.mlp.w3, rank=rank, alpha=alpha)

    def forward(self, x, attn_mask=None):
        return self.base(x, attn_mask)

    @torch.no_grad()
    def merge_all(self):
        for mod in [self.base.attn.Wq, self.base.attn.Wk, self.base.attn.Wv, self.base.attn.Wo, self.base.mlp.w1, self.base.mlp.w2, self.base.mlp.w3]:
            if isinstance(mod, LoRAAdapter):
                mod.merge()


@dataclass
class PLADConfig:
    d_model: int = 128
    n_heads: int = 4
    n_kv_heads: int = 1
    d_ff: int = 256
    n_pairs: int = 3
    vocab_size: int = 1000
    max_seq: int = 64
    dropout: float = 0.0


class PLADTransformer(nn.Module):
    """Transformer with odd/even paired blocks. Even blocks start as Identity; can be activated progressively.
    LS baseline can be created by aliasing even=odd (weight sharing) from step 1.
    """
    def __init__(self, cfg: PLADConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.max_seq, cfg.d_model))
        self.blocks_odd = nn.ModuleList()
        self.blocks_even = nn.ModuleList()
        for _ in range(cfg.n_pairs):
            odd = TransformerBlock(cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.d_ff, cfg.dropout)
            self.blocks_odd.append(odd)
            self.blocks_even.append(nn.Identity())  # inactive at start
        self.ln_f = LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        device = idx.device
        x = self.embed(idx) + self.pos_emb[:, :T, :]
        mask = causal_mask(T, device)
        for odd, even in zip(self.blocks_odd, self.blocks_even):
            x = odd(x, attn_mask=mask)
            if isinstance(even, nn.Identity):
                pass
            else:
                x = even(x, attn_mask=mask)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits


class Activator:
    def __init__(self, model: PLADTransformer, lora_rank=4, lora_alpha=8.0):
        self.m = model
        self.rank = lora_rank
        self.alpha = lora_alpha

    def is_active(self, pair_idx: int) -> bool:
        return not isinstance(self.m.blocks_even[pair_idx], nn.Identity)

    def activate_pair(self, pair_idx: int, use_adapters: bool = True):
        if self.is_active(pair_idx):
            return
        odd = self.m.blocks_odd[pair_idx]
        even_base = copy.deepcopy(odd)
        if use_adapters:
            even = AdaptedBlock(even_base, rank=self.rank, alpha=self.alpha)
        else:
            even = even_base
        self.m.blocks_even[pair_idx] = even

    @torch.no_grad()
    def merge_pair_adapters(self, pair_idx: int):
        even = self.m.blocks_even[pair_idx]
        if isinstance(even, AdaptedBlock):
            even.merge_all()

    @torch.no_grad()
    def merge_all(self):
        for i in range(self.m.cfg.n_pairs):
            self.merge_pair_adapters(i)


# ------------------------------ Distillation ------------------------------

@torch.no_grad()
def get_post_odd_hidden(model: PLADTransformer, x: torch.Tensor, pair_idx: int) -> torch.Tensor:
    # Returns hidden state after applying odd block at pair_idx (inclusive), passing through any activated evens before it
    B, T = x.shape[:2]
    device = x.device
    h = model.embed(x) + model.pos_emb[:, :T, :]
    mask = causal_mask(T, device)
    for i in range(pair_idx + 1):
        h = model.blocks_odd[i](h, attn_mask=mask)
        if i < pair_idx:
            even = model.blocks_even[i]
            if not isinstance(even, nn.Identity):
                h = even(h, attn_mask=mask)
    return h


def distill_even_block(model: PLADTransformer,
                       pair_idx: int,
                       data_iter,
                       steps: int = 50,
                       lr: float = 5e-4,
                       device: Optional[torch.device] = None,
                       dataset: Optional[torch.utils.data.Dataset] = None,
                       batch_size: Optional[int] = None):
    if device is None:
        device = device_auto()
    model.train()
    odd_teacher = copy.deepcopy(model.blocks_odd[pair_idx]).to(device).eval()
    for p in odd_teacher.parameters():
        p.requires_grad_(False)
    even_mod = model.blocks_even[pair_idx]
    assert not isinstance(even_mod, nn.Identity), "Even block must be active for distillation"
    params = [p for p in even_mod.parameters() if p.requires_grad]
    if len(params) == 0:
        return
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    moving = None
    for t in range(steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            if dataset is None or batch_size is None:
                # If we cannot rebuild the iterator, re-raise to surface the issue
                raise
            data_iter = iter(torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True))
            batch = next(data_iter)
        x = batch["input_ids"].to(device)
        with torch.no_grad():
            teacher_in = get_post_odd_hidden(model, x, pair_idx - 1) if pair_idx > 0 else (model.embed(x) + model.pos_emb[:, :x.size(1), :])
            mask = causal_mask(x.size(1), device)
            h_teacher = odd_teacher(teacher_in, attn_mask=mask)
        student_in = get_post_odd_hidden(model, x, pair_idx - 1) if pair_idx > 0 else (model.embed(x) + model.pos_emb[:, :x.size(1), :])
        mask = causal_mask(x.size(1), device)
        h_student = even_mod(student_in, attn_mask=mask)
        ht = F.layer_norm(h_teacher, (h_teacher.size(-1),))
        hs = F.layer_norm(h_student, (h_student.size(-1),))
        loss = F.mse_loss(hs, ht)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        moving = 0.9 * moving + 0.1 * loss.item() if moving is not None else loss.item()
        if (t + 1) % max(1, steps // 5) == 0:
            print(f"  Distill pair {pair_idx} step {t+1}/{steps} | loss={moving:.4f}")


# ------------------------------ FLOPs Estimation ------------------------------

def estimate_flops_per_token(d_model: int, d_ff: int, n_heads: int, seq_len: int, active_layers: int) -> float:
    # Rough forward FLOPs per layer per token; multiply by 2 for backward
    attn = 2 * d_model * d_model + 2 * seq_len * (d_model * d_model // n_heads)
    mlp = 8 * d_model * d_ff
    per_layer_fwd = attn + mlp
    return active_layers * per_layer_fwd * 2.0


# ------------------------------ Training / Evaluation ------------------------------

def compute_loss_on_batch(model: PLADTransformer, x: torch.Tensor) -> torch.Tensor:
    logits = model(x)
    loss = F.cross_entropy(
        logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
        x[:, 1:].contiguous().view(-1)
    )
    return loss


@dataclass
class TrainLog:
    train_losses: List[float]
    val_steps: List[int]
    val_losses: List[float]
    step_times: List[float]
    tokens_per_sec: List[float]
    flops_accum: List[float]
    active_layers_hist: List[int]


def train_plad(model: PLADTransformer,
               train_loader,
               val_loader,
               total_steps: int = 200,
               phase1_frac: float = 0.6,
               activate_batch_size: int = 1,
               activate_every: int = 50,
               distill_steps_per_pair: int = 20,
               lr: float = 3e-4,
               device: Optional[torch.device] = None,
               use_adapters: bool = True,
               do_distill: bool = True,
               val_interval: int = 20) -> TrainLog:
    from .evaluate import evaluate_loss  # relative import as required
    if device is None:
        device = device_auto()
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    activator = Activator(model, lora_rank=4, lora_alpha=8.0)
    data_iter = iter(torch.utils.data.DataLoader(train_loader.dataset, batch_size=train_loader.batch_size, shuffle=True, drop_last=True))

    phase1_steps = int(total_steps * phase1_frac)
    next_activation = phase1_steps

    train_losses: List[float] = []
    val_steps: List[int] = []
    val_losses: List[float] = []
    step_times: List[float] = []
    tokens_per_sec: List[float] = []
    flops_accum: List[float] = []
    active_layers_hist: List[int] = []

    seq_len = train_loader.dataset.seq_len
    d = model.cfg.d_model
    dff = model.cfg.d_ff
    H = model.cfg.n_heads

    # Precompute odd layer count
    always_active_layers = model.cfg.n_pairs  # only odd layers guaranteed active in phase 1

    steps_done = 0
    pairs = list(range(model.cfg.n_pairs))
    activated_ptr = 0

    print(f"[PLAD] Start training: total_steps={total_steps}, phase1_steps={phase1_steps}, activate_every={activate_every}, distill_steps={distill_steps_per_pair}")

    while steps_done < total_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(torch.utils.data.DataLoader(train_loader.dataset, batch_size=train_loader.batch_size, shuffle=True, drop_last=True))
            batch = next(data_iter)
        x = batch["input_ids"].to(device)
        t0 = time.time()
        model.train()
        loss = compute_loss_on_batch(model, x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        dt = time.time() - t0

        train_losses.append(loss.item())
        step_times.append(dt)
        tokens_per_sec.append((x.numel()) / dt)

        # Active layers count
        active_evens = sum([0 if isinstance(model.blocks_even[i], nn.Identity) else 1 for i in range(model.cfg.n_pairs)])
        active_layers = always_active_layers + active_evens
        active_layers_hist.append(active_layers)
        flops_per_tok = estimate_flops_per_token(d, dff, H, seq_len, active_layers)
        flops_accum.append(flops_per_tok * x.numel())

        steps_done += 1

        # Validation
        if (steps_done % val_interval) == 0 or steps_done == total_steps:
            vloss = evaluate_loss(model, val_loader, device, max_batches=5)
            val_steps.append(steps_done)
            val_losses.append(vloss)
            print(f"  Step {steps_done:4d} | train_loss={loss.item():.4f} | val_loss={vloss:.4f} | active_layers={active_layers} | tok/s={tokens_per_sec[-1]:.1f}")

        # Progressive activation schedule
        if steps_done == next_activation:
            # Activate up to activate_batch_size pairs per activation event
            for _ in range(activate_batch_size):
                if activated_ptr >= len(pairs):
                    break
                pidx = pairs[activated_ptr]
                activator.activate_pair(pidx, use_adapters=use_adapters)
                print(f"  Activated even block for pair {pidx} (use_adapters={use_adapters})")
                if do_distill and use_adapters and distill_steps_per_pair > 0:
                    distill_even_block(model, pidx, data_iter, steps=distill_steps_per_pair, lr=5e-4, device=device,
                                       dataset=train_loader.dataset, batch_size=train_loader.batch_size)
                    activator.merge_pair_adapters(pidx)
                    print(f"  Merged adapters for pair {pidx} after distillation")
                activated_ptr += 1
            next_activation += activate_every

    return TrainLog(train_losses, val_steps, val_losses, step_times, tokens_per_sec, flops_accum, active_layers_hist)


def make_ls_baseline(cfg: PLADConfig) -> PLADTransformer:
    m = PLADTransformer(cfg)
    # weight sharing: even points to the same module as odd within each pair
    for i in range(cfg.n_pairs):
        m.blocks_even[i] = m.blocks_odd[i]
    return m


def train_baseline_ls(model: PLADTransformer,
                      train_loader,
                      val_loader,
                      total_steps: int = 200,
                      lr: float = 3e-4,
                      device: Optional[torch.device] = None,
                      val_interval: int = 20) -> TrainLog:
    from .evaluate import evaluate_loss  # relative import
    if device is None:
        device = device_auto()
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    data_iter = iter(torch.utils.data.DataLoader(val_loader.dataset if train_loader is None else train_loader.dataset,
                                                batch_size=train_loader.batch_size if train_loader is not None else 16,
                                                shuffle=True, drop_last=True))

    train_losses: List[float] = []
    val_steps: List[int] = []
    val_losses: List[float] = []
    step_times: List[float] = []
    tokens_per_sec: List[float] = []
    flops_accum: List[float] = []
    active_layers_hist: List[float] = []

    seq_len = (train_loader.dataset.seq_len if train_loader is not None else 64)
    d = model.cfg.d_model
    dff = model.cfg.d_ff
    H = model.cfg.n_heads

    always_active_layers = 2 * model.cfg.n_pairs

    print(f"[LS] Start training baseline: total_steps={total_steps}")

    for step in range(1, total_steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(torch.utils.data.DataLoader(train_loader.dataset, batch_size=train_loader.batch_size, shuffle=True, drop_last=True))
            batch = next(data_iter)
        x = batch["input_ids"].to(device)
        t0 = time.time()
        model.train()
        loss = compute_loss_on_batch(model, x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        dt = time.time() - t0

        train_losses.append(loss.item())
        step_times.append(dt)
        tokens_per_sec.append((x.numel()) / dt)

        active_layers_hist.append(always_active_layers)
        flops_per_tok = estimate_flops_per_token(d, dff, H, seq_len, always_active_layers)
        flops_accum.append(flops_per_tok * x.numel())

        if (step % val_interval) == 0 or step == total_steps:
            vloss = evaluate_loss(model, val_loader, device, max_batches=5)
            val_steps.append(step)
            val_losses.append(vloss)
            print(f"  Step {step:4d} | train_loss={loss.item():.4f} | val_loss={vloss:.4f} | active_layers={always_active_layers} | tok/s={tokens_per_sec[-1]:.1f}")

    return TrainLog(train_losses, val_steps, val_losses, step_times, tokens_per_sec, flops_accum, active_layers_hist)


def benchmark_inference_latency(model: PLADTransformer, cfg: PLADConfig, batch_sizes=[1, 4], seq_len=64, device: Optional[torch.device] = None) -> Dict[str, float]:
    if device is None:
        device = device_auto()
    model.eval().to(device)
    lat_ms: Dict[str, float] = {}
    for b in batch_sizes:
        x = torch.randint(0, cfg.vocab_size, (b, seq_len), device=device)
        with torch.no_grad():
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(10):
                _ = model(x)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            dt = time.time() - t0
        lat = (dt / 10.0) * 1000.0
        lat_ms[f"bs{b}"] = lat
        print(f"  Inference latency bs={b}: {lat:.3f} ms")
    return lat_ms


def simple_finetune(model: PLADTransformer, train_loader, val_loader, steps: int = 50, lr: float = 5e-4, device: Optional[torch.device] = None) -> Tuple[List[float], List[float]]:
    # Build a synthetic classification target: label = parity(sum(tokens)) in {0,1}
    if device is None:
        device = device_auto()
    model.to(device)
    clf_head = nn.Linear(model.cfg.d_model, 2).to(device)
    opt = torch.optim.AdamW(list(model.parameters()) + list(clf_head.parameters()), lr=lr)
    accs: List[float] = []
    val_accs: List[float] = []
    model.train()
    for step, batch in enumerate(train_loader):
        if step >= steps:
            break
        x = batch["input_ids"].to(device)
        y = (x.sum(dim=1) % 2).long()  # parity label
        logits = model(x)
        h_last = logits[:, -1, :].detach()  # use last-step logits as features (toy)
        pred = clf_head(h_last)
        loss = F.cross_entropy(pred, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            acc = (pred.argmax(dim=-1) == y).float().mean().item()
            accs.append(acc)
    # val accuracy
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= 10:
                break
            x = batch["input_ids"].to(device)
            y = (x.sum(dim=1) % 2).long()
            logits = model(x)
            h_last = logits[:, -1, :]
            pred = clf_head(h_last)
            correct += (pred.argmax(dim=-1) == y).sum().item()
            total += y.numel()
    val_acc = correct / max(1, total)
    val_accs.append(val_acc)
    print(f"  Finetune end | val_acc={val_acc:.3f}")
    return accs, val_accs
