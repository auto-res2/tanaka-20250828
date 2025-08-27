import os
import copy
from typing import Optional, Dict, List, Tuple

import torch
import torch.nn.functional as F
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from .train import (
    PLADConfig,
    PLADTransformer,
    make_ls_baseline,
    train_plad,
    train_baseline_ls,
    benchmark_inference_latency,
    simple_finetune,
    Activator,
)
from .preprocess import build_dataloaders, set_seed

sns.set_theme(style="whitegrid")

# ------------------------------ Core evaluation utils ------------------------------

def evaluate_loss(model: PLADTransformer, val_loader, device: Optional[torch.device] = None, max_batches: Optional[int] = None) -> float:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    losses: List[float] = []
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            x = batch["input_ids"].to(device)
            logits = model(x)
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
                x[:, 1:].contiguous().view(-1)
            )
            losses.append(loss.item())
            if max_batches is not None and (i + 1) >= max_batches:
                break
    return float(np.mean(losses)) if losses else float('nan')


# ------------------------------ Plotting ------------------------------

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_pdf(fig, filename: str):
    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight", format="pdf")
    plt.close(fig)
    print(f"Saved figure: {filename}")


def plot_training_losses(log_plad, log_ls, title: str, filename: str):
    fig = plt.figure(figsize=(6, 4))
    sns.lineplot(x=np.arange(len(log_plad.train_losses)), y=log_plad.train_losses, label="PLAD-T")
    sns.lineplot(x=np.arange(len(log_ls.train_losses)), y=log_ls.train_losses, label="LS baseline")
    plt.xlabel("Step")
    plt.ylabel("Train loss")
    plt.title(title)
    plt.legend()
    save_pdf(fig, filename)


def plot_validation_losses(log_plad, log_ls, title: str, filename: str):
    fig = plt.figure(figsize=(6, 4))
    sns.lineplot(x=log_plad.val_steps, y=log_plad.val_losses, marker="o", label="PLAD-T")
    sns.lineplot(x=log_ls.val_steps, y=log_ls.val_losses, marker="o", label="LS baseline")
    plt.xlabel("Step")
    plt.ylabel("Val loss")
    plt.title(title)
    plt.legend()
    save_pdf(fig, filename)


def plot_flops_bar(flops_plad: float, flops_ls: float, title: str, filename: str):
    fig = plt.figure(figsize=(5, 4))
    vals = [flops_plad / 1e9, flops_ls / 1e9]
    sns.barplot(x=["PLAD-T", "LS"], y=vals)
    plt.ylabel("Theoretical FLOPs (billions)")
    plt.title(title)
    save_pdf(fig, filename)


def plot_throughput_bar(tps_plad: float, tps_ls: float, title: str, filename: str):
    fig = plt.figure(figsize=(5, 4))
    sns.barplot(x=["PLAD-T", "LS"], y=[tps_plad, tps_ls])
    plt.ylabel("Tokens/sec (train)")
    plt.title(title)
    save_pdf(fig, filename)


def plot_ablation_losses(log_a, log_b, label_a: str, label_b: str, title: str, filename: str):
    fig = plt.figure(figsize=(6, 4))
    sns.lineplot(x=np.arange(len(log_a.train_losses)), y=log_a.train_losses, label=label_a)
    sns.lineplot(x=np.arange(len(log_b.train_losses)), y=log_b.train_losses, label=label_b)
    plt.xlabel("Step")
    plt.ylabel("Train loss")
    plt.title(title)
    plt.legend()
    save_pdf(fig, filename)


def plot_phase1_tradeoff(f_vals: List[float], final_val_losses: List[float], title: str, filename: str):
    fig = plt.figure(figsize=(6, 4))
    sns.lineplot(x=f_vals, y=final_val_losses, marker="o")
    plt.xlabel("Phase-1 fraction f")
    plt.ylabel("Final val loss")
    plt.title(title)
    save_pdf(fig, filename)


def plot_latency(latencies: Dict[str, float], title: str, filename: str):
    fig = plt.figure(figsize=(6, 4))
    labels = list(latencies.keys())
    vals = [latencies[k] for k in labels]
    sns.barplot(x=labels, y=vals)
    plt.ylabel("Latency per forward (ms)")
    plt.title(title)
    save_pdf(fig, filename)


def plot_finetune_accuracy(acc_dict: Dict[str, List[float]], title: str, filename: str):
    fig = plt.figure(figsize=(6, 4))
    for label, accs in acc_dict.items():
        sns.lineplot(x=np.arange(len(accs)), y=accs, label=label)
    plt.xlabel("Step")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.0)
    plt.title(title)
    plt.legend()
    save_pdf(fig, filename)


def plot_quantization(val_losses: Dict[str, float], title: str, filename: str):
    fig = plt.figure(figsize=(5, 4))
    labels = list(val_losses.keys())
    vals = [val_losses[k] for k in labels]
    sns.barplot(x=labels, y=vals)
    plt.ylabel("Validation loss")
    plt.title(title)
    save_pdf(fig, filename)


# ------------------------------ Quantization helpers ------------------------------

def dynamic_quantize_cpu(model: PLADTransformer) -> torch.nn.Module:
    from torch.ao.quantization import quantize_dynamic
    model_cpu = copy.deepcopy(model).cpu()
    model_cpu.eval()
    q_model = quantize_dynamic(model_cpu, {torch.nn.Linear}, dtype=torch.qint8)
    return q_model


def evaluate_val_loss_cpu(model: torch.nn.Module, val_loader) -> float:
    model.eval()
    losses: List[float] = []
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            x = batch["input_ids"].cpu()
            logits = model(x)
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
                x[:, 1:].contiguous().view(-1)
            )
            losses.append(loss.item())
            if i >= 4:
                break
    return float(np.mean(losses)) if losses else float('nan')


# ------------------------------ Experiments ------------------------------

def experiment1_compute_vs_accuracy(image_dir: str,
                                    pattern: str = "uniform",
                                    total_steps: int = 120,
                                    phase1_frac: float = 0.6,
                                    activate_every: int = 40,
                                    distill_steps_per_pair: int = 10,
                                    seed: int = 0):
    _ensure_dir(image_dir)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Config and data
    cfg = PLADConfig(d_model=96, n_heads=4, n_kv_heads=1, d_ff=192, n_pairs=3, vocab_size=1000, max_seq=64, dropout=0.0)
    train_loader, val_loader = build_dataloaders(vocab_size=cfg.vocab_size, seq_len=cfg.max_seq, n_train=600, n_val=200, pattern=pattern, batch_size=16, seed=seed)

    # PLAD-T
    plad = PLADTransformer(cfg)
    log_plad = train_plad(plad, train_loader, val_loader, total_steps=total_steps,
                          phase1_frac=phase1_frac, activate_batch_size=1, activate_every=activate_every,
                          distill_steps_per_pair=distill_steps_per_pair, lr=3e-4, device=device,
                          use_adapters=True, do_distill=True, val_interval=20)

    # LS baseline
    ls = make_ls_baseline(cfg)
    log_ls = train_baseline_ls(ls, train_loader, val_loader, total_steps=total_steps, lr=3e-4, device=device, val_interval=20)

    # Summaries
    total_flops_plad = float(np.sum(log_plad.flops_accum))
    total_flops_ls = float(np.sum(log_ls.flops_accum))
    avg_tps_plad = float(np.mean(log_plad.tokens_per_sec))
    avg_tps_ls = float(np.mean(log_ls.tokens_per_sec))

    print("Experiment 1 Summary (pattern: {}):".format(pattern))
    print(f"  Total theoretical FLOPs (PLAD-T): {total_flops_plad/1e9:.2f} B  | (LS): {total_flops_ls/1e9:.2f} B")
    print(f"  FLOPs reduction: {100.0*(1 - total_flops_plad/total_flops_ls):.1f}%")
    print(f"  Avg tokens/sec train (PLAD-T): {avg_tps_plad:.1f} | (LS): {avg_tps_ls:.1f}")
    print(f"  Final val loss (PLAD-T): {log_plad.val_losses[-1]:.3f} | (LS): {log_ls.val_losses[-1]:.3f}")

    # Plots
    plot_training_losses(log_plad, log_ls, title=f"Training loss ({pattern})", filename=os.path.join(image_dir, f"training_loss_pladt_vs_ls_{pattern}.pdf"))
    plot_validation_losses(log_plad, log_ls, title=f"Validation loss ({pattern})", filename=os.path.join(image_dir, f"validation_loss_pladt_vs_ls_{pattern}.pdf"))
    plot_flops_bar(total_flops_plad, total_flops_ls, title=f"Theoretical FLOPs ({pattern})", filename=os.path.join(image_dir, f"theoretical_flops_pladt_vs_ls_{pattern}.pdf"))
    plot_throughput_bar(avg_tps_plad, avg_tps_ls, title=f"Throughput ({pattern})", filename=os.path.join(image_dir, f"throughput_pladt_vs_ls_{pattern}.pdf"))

    # Save models
    os.makedirs("models", exist_ok=True)
    torch.save(plad.state_dict(), os.path.join("models", f"pladt_{pattern}.pt"))
    torch.save(ls.state_dict(), os.path.join("models", f"ls_{pattern}.pt"))

    return {
        "log_plad": log_plad,
        "log_ls": log_ls,
        "plad_model": plad,
        "ls_model": ls,
        "cfg": cfg,
        "val_loader": val_loader,
    }


def experiment2_ablations(image_dir: str,
                           pattern: str = "uniform",
                           total_steps: int = 100,
                           seed: int = 0):
    _ensure_dir(image_dir)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = PLADConfig(d_model=96, n_heads=4, n_kv_heads=1, d_ff=192, n_pairs=3, vocab_size=1000, max_seq=64, dropout=0.0)
    train_loader, val_loader = build_dataloaders(vocab_size=cfg.vocab_size, seq_len=cfg.max_seq, n_train=600, n_val=200, pattern=pattern, batch_size=16, seed=seed)

    # Copy-only (no adapters, no distill)
    from .train import train_plad as _train_plad
    plad_copy = PLADTransformer(cfg)
    log_copy = _train_plad(plad_copy, train_loader, val_loader, total_steps=total_steps,
                           phase1_frac=0.6, activate_batch_size=1, activate_every=40,
                           distill_steps_per_pair=0, lr=3e-4, device=device,
                           use_adapters=False, do_distill=False, val_interval=20)

    # Distill (adapters + MSE)
    plad_distill = PLADTransformer(cfg)
    log_distill = _train_plad(plad_distill, train_loader, val_loader, total_steps=total_steps,
                              phase1_frac=0.6, activate_batch_size=1, activate_every=40,
                              distill_steps_per_pair=10, lr=3e-4, device=device,
                              use_adapters=True, do_distill=True, val_interval=20)

    print("Experiment 2A (Copy-only vs Distill)")
    print(f"  Final val loss (copy-only): {log_copy.val_losses[-1]:.3f} | (distill): {log_distill.val_losses[-1]:.3f}")

    plot_ablation_losses(log_copy, log_distill, label_a="Copy-only", label_b="Distill (MSE)",
                         title=f"Ablation loss ({pattern})", filename=os.path.join(image_dir, "ablation_loss_copy_vs_distill.pdf"))

    # Phase-1 fraction sweep
    f_vals = [0.4, 0.6, 0.8]
    final_losses: List[float] = []
    for f in f_vals:
        model = PLADTransformer(cfg)
        logf = _train_plad(model, train_loader, val_loader, total_steps=total_steps,
                           phase1_frac=f, activate_batch_size=1, activate_every=max(10, int((1.0-f)*total_steps/3)),
                           distill_steps_per_pair=10, lr=3e-4, device=device,
                           use_adapters=True, do_distill=True, val_interval=25)
        final_losses.append(logf.val_losses[-1])
        print(f"  Phase-1 f={f:.1f} | final val loss={logf.val_losses[-1]:.3f}")

    plot_phase1_tradeoff(f_vals, final_losses, title=f"Phase-1 fraction trade-off ({pattern})", filename=os.path.join(image_dir, "phase1_tradeoff_pladt.pdf"))

    return {
        "copy_log": log_copy,
        "distill_log": log_distill,
        "phase1_f": f_vals,
        "phase1_val_losses": final_losses,
    }


def experiment3_hardware_quant_finetune(image_dir: str,
                                        pretrained_plad: PLADTransformer,
                                        pretrained_ls: PLADTransformer,
                                        cfg: PLADConfig,
                                        val_loader,
                                        pattern: str = "uniform"):
    _ensure_dir(image_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Ensure adapters are merged (PLAD)
    Activator(pretrained_plad).merge_all()

    print("Experiment 3: Inference latency and profiling (simple timers)")
    lat_plad = benchmark_inference_latency(pretrained_plad, cfg, batch_sizes=[1, 4], seq_len=cfg.max_seq, device=device)
    lat_ls = benchmark_inference_latency(pretrained_ls, cfg, batch_sizes=[1, 4], seq_len=cfg.max_seq, device=device)

    # Plot latency for bs=1 and bs=4
    plot_latency({"PLAD bs1": lat_plad["bs1"], "LS bs1": lat_ls["bs1"], "PLAD bs4": lat_plad["bs4"], "LS bs4": lat_ls["bs4"]},
                 title=f"Inference latency ({pattern})", filename=os.path.join(image_dir, "inference_latency_pladt_vs_ls.pdf"))

    print("Experiment 3: Dynamic quantization (CPU)")
    q_plad = dynamic_quantize_cpu(pretrained_plad)
    vloss_float = evaluate_val_loss_cpu(copy.deepcopy(pretrained_plad).cpu(), val_loader)
    vloss_int8 = evaluate_val_loss_cpu(q_plad, val_loader)
    print(f"  Val loss float: {vloss_float:.3f} | int8: {vloss_int8:.3f}")
    plot_quantization({"PLAD float": vloss_float, "PLAD int8": vloss_int8}, title=f"Quantization (val loss, {pattern})", filename=os.path.join(image_dir, "quantization_pladt.pdf"))

    print("Experiment 3: Fine-tuning robustness (synthetic parity classification)")
    # Build small loaders for finetune
    ft_train_loader, ft_val_loader = build_dataloaders(vocab_size=cfg.vocab_size, seq_len=cfg.max_seq, n_train=400, n_val=200, pattern=pattern, batch_size=16, seed=123)
    accs_plad, valacc_plad = simple_finetune(copy.deepcopy(pretrained_plad), ft_train_loader, ft_val_loader, steps=50, lr=5e-4, device=device)
    accs_ls, valacc_ls = simple_finetune(copy.deepcopy(pretrained_ls), ft_train_loader, ft_val_loader, steps=50, lr=5e-4, device=device)
    plot_finetune_accuracy({"PLAD-T": accs_plad, "LS": accs_ls}, title=f"Finetune accuracy ({pattern})", filename=os.path.join(image_dir, "finetune_accuracy_pladt_vs_ls.pdf"))
    print(f"  Final finetune val acc | PLAD: {valacc_plad[-1]:.3f} | LS: {valacc_ls[-1]:.3f}")


# ------------------------------ High-level orchestrators ------------------------------

def run_experiment_suite(image_dir: str):
    # Experiment 1 on multiple patterns for robustness
    patterns = ["uniform", "bigram", "copyshift"]
    exp1_results = {}
    for p in patterns:
        res = experiment1_compute_vs_accuracy(image_dir=image_dir, pattern=p, total_steps=120, phase1_frac=0.6, activate_every=40, distill_steps_per_pair=10, seed=0)
        exp1_results[p] = res

    # Experiment 2 (ablations) on one representative pattern
    experiment2_ablations(image_dir=image_dir, pattern="uniform", total_steps=100, seed=1)

    # Experiment 3 using one of the trained models from Exp1 (uniform)
    res_uniform = exp1_results["uniform"]
    experiment3_hardware_quant_finetune(image_dir=image_dir,
                                        pretrained_plad=res_uniform["plad_model"],
                                        pretrained_ls=res_uniform["ls_model"],
                                        cfg=res_uniform["cfg"],
                                        val_loader=res_uniform["val_loader"],
                                        pattern="uniform")


def run_quick_test(image_dir: str):
    """Fast functional test to ensure the code runs and produces outputs quickly."""
    print("Running quick test...")
    set_seed(123)
    # Use fewer steps for speed
    res = experiment1_compute_vs_accuracy(image_dir=image_dir, pattern="uniform", total_steps=40, phase1_frac=0.6, activate_every=20, distill_steps_per_pair=5, seed=123)
    experiment2_ablations(image_dir=image_dir, pattern="uniform", total_steps=40, seed=321)
    experiment3_hardware_quant_finetune(image_dir=image_dir,
                                        pretrained_plad=res["plad_model"],
                                        pretrained_ls=res["ls_model"],
                                        cfg=res["cfg"],
                                        val_loader=res["val_loader"],
                                        pattern="uniform")
    print("Quick test completed.")
