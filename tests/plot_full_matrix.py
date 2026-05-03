"""Plot bench_full_matrix results.

Generates:
  - speedup_heatmap.png   — ratio(Nelux/tc) per (pix_fmt × target × threads). <1 = Nelux faster.
  - fps_pix_fmt.png       — FPS bar chart per pix_fmt (8 pix_fmt clips)
  - fps_codec_res.png     — FPS bar chart per codec/resolution
  - quality_scatter.png   — VMAF scatter (Nelux vs torchcodec)
  - winloss_summary.png   — count of wins/losses across categories
  - ratio_by_target.png   — ratio distribution per resize target
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "output" / "full_matrix"
RESULTS = json.loads((OUT / "results.json").read_text())

# ratio = nelux_t / tc_t. <1 means Nelux faster.
# Convert to "speedup of Nelux over tc": tc_t / nelux_t. >1 = Nelux faster.
for r in RESULTS:
    r["speedup"] = r["tc_t"] / r["nelux_t"]


# ============================================================
# 1) Speedup heatmap (pix_fmt clips, since group=="pix_fmt")
# ============================================================
def plot_speedup_heatmap_pix_fmt():
    rows = [r for r in RESULTS if r["group"] == "pix_fmt"]
    if not rows:
        return
    pix_fmts = sorted({r["pix_fmt"] for r in rows})
    targets = ["native", "half", "quarter"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax_i, threads in enumerate((0, 1)):
        ax = axes[ax_i]
        mat = np.full((len(pix_fmts), len(targets)), np.nan)
        for i, pf in enumerate(pix_fmts):
            for j, tg in enumerate(targets):
                cell = [r for r in rows if r["pix_fmt"] == pf and r["target"] == tg and r["threads"] == threads]
                if cell:
                    mat[i, j] = cell[0]["speedup"]
        # log scale color (centered at 1)
        log_mat = np.log2(mat)
        im = ax.imshow(log_mat, cmap="RdYlGn", aspect="auto", vmin=-3, vmax=3)
        ax.set_xticks(range(len(targets)), targets)
        ax.set_yticks(range(len(pix_fmts)), pix_fmts)
        ax.set_title(f"Nelux speedup over torchcodec (threads={threads})\n"
                     "green = Nelux faster, red = torchcodec faster")
        for i in range(len(pix_fmts)):
            for j in range(len(targets)):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.2f}x",
                            ha="center", va="center",
                            color="black" if abs(log_mat[i, j]) < 1.5 else "white",
                            fontsize=9)
        plt.colorbar(im, ax=ax, label="log2(speedup)")
    plt.tight_layout()
    out = OUT / "speedup_heatmap_pix_fmt.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  {out}")


# ============================================================
# 2) FPS bar per pix_fmt (threads=1, native+half+quarter)
# ============================================================
def plot_fps_pix_fmt():
    rows = [r for r in RESULTS if r["group"] == "pix_fmt" and r["threads"] == 1]
    pix_fmts = sorted({r["pix_fmt"] for r in rows})
    targets = ["native", "half", "quarter"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax_i, tg in enumerate(targets):
        ax = axes[ax_i]
        nelux_fps = []
        tc_fps = []
        for pf in pix_fmts:
            cell = [r for r in rows if r["pix_fmt"] == pf and r["target"] == tg]
            if cell:
                nelux_fps.append(cell[0]["nelux_fps"])
                tc_fps.append(cell[0]["tc_fps"])
            else:
                nelux_fps.append(0)
                tc_fps.append(0)
        x = np.arange(len(pix_fmts))
        w = 0.35
        ax.bar(x - w/2, nelux_fps, w, label="Nelux", color="#2ecc71")
        ax.bar(x + w/2, tc_fps, w, label="torchcodec", color="#3498db")
        ax.set_xticks(x, pix_fmts, rotation=45, ha="right")
        ax.set_ylabel("FPS")
        ax.set_title(f"{tg} (threads=1)")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
    plt.suptitle("Decode FPS by pix_fmt, 1080p source", fontsize=14)
    plt.tight_layout()
    out = OUT / "fps_pix_fmt.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  {out}")


# ============================================================
# 3) FPS bar per (codec, resolution) — yuv420p only for clarity
# ============================================================
def plot_fps_codec_res():
    rows = [r for r in RESULTS if r["group"] == "encoder" and r["threads"] == 1
            and r["pix_fmt"] == "yuv420p" and r["target"] == "native"]
    if not rows:
        return
    # Group by codec + source resolution
    rows.sort(key=lambda r: (r["clip"].split(":")[0], r["src_w"]))
    labels = []
    nelux_fps = []
    tc_fps = []
    for r in rows:
        codec = r["clip"].split(":")[0]
        labels.append(f"{codec}\n{r['src_w']}x{r['src_h']}")
        nelux_fps.append(r["nelux_fps"])
        tc_fps.append(r["tc_fps"])

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w/2, nelux_fps, w, label="Nelux", color="#2ecc71")
    ax.bar(x + w/2, tc_fps, w, label="torchcodec", color="#3498db")
    ax.set_xticks(x, labels)
    ax.set_ylabel("FPS (log scale)")
    ax.set_yscale("log")
    ax.set_title("Decode FPS — codec × resolution (yuv420p, native, threads=1)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3, which="both")
    plt.tight_layout()
    out = OUT / "fps_codec_res.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  {out}")


# ============================================================
# 4) Quality scatter: VMAF
# ============================================================
def plot_quality_scatter():
    nelux_vmaf = [r["nelux_quality"]["vmaf"] for r in RESULTS
                  if r["nelux_quality"]["vmaf"] == r["nelux_quality"]["vmaf"]]  # not nan
    tc_vmaf = [r["tc_quality"]["vmaf"] for r in RESULTS
               if r["tc_quality"]["vmaf"] == r["tc_quality"]["vmaf"]]
    targets = [r["target"] for r in RESULTS
               if r["nelux_quality"]["vmaf"] == r["nelux_quality"]["vmaf"]
               and r["tc_quality"]["vmaf"] == r["tc_quality"]["vmaf"]]

    if not nelux_vmaf:
        return

    color_map = {"native": "#3498db", "half": "#e74c3c",
                 "quarter": "#f39c12", "up1.5x": "#9b59b6"}
    fig, ax = plt.subplots(figsize=(8, 8))
    for tg in set(targets):
        idx = [i for i, t in enumerate(targets) if t == tg]
        ax.scatter([nelux_vmaf[i] for i in idx], [tc_vmaf[i] for i in idx],
                   c=color_map.get(tg, "gray"), label=tg, alpha=0.7, s=60)
    lim = (60, 102)
    ax.plot(lim, lim, "k--", alpha=0.4, label="parity")
    ax.set_xlabel("Nelux VMAF (vs zimg-lanczos ref)")
    ax.set_ylabel("torchcodec VMAF (vs zimg-lanczos ref)")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_title("Output quality: VMAF vs zimg-lanczos reference\n(closer to 100 = closer to gold-standard scaler)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = OUT / "quality_scatter.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  {out}")


# ============================================================
# 5) Win/loss summary
# ============================================================
def plot_winloss():
    # speedup>=1.05 = win (Nelux), <=0.95 = loss, else tie
    cats = ["pix_fmt native", "pix_fmt resize", "encoder native", "encoder resize"]
    wins = [0, 0, 0, 0]
    losses = [0, 0, 0, 0]
    ties = [0, 0, 0, 0]
    for r in RESULTS:
        if r["group"] == "pix_fmt":
            i = 0 if r["target"] == "native" else 1
        else:
            i = 2 if r["target"] == "native" else 3
        s = r["speedup"]
        if s >= 1.05:
            wins[i] += 1
        elif s <= 0.95:
            losses[i] += 1
        else:
            ties[i] += 1

    x = np.arange(len(cats))
    w = 0.27
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - w, wins, w, label="Nelux wins (>=1.05x)", color="#2ecc71")
    ax.bar(x, ties, w, label="tie (0.95-1.05x)", color="#95a5a6")
    ax.bar(x + w, losses, w, label="torchcodec wins (<=0.95x)", color="#e74c3c")
    ax.set_xticks(x, cats)
    ax.set_ylabel("count")
    ax.set_title("Performance head-to-head (threads=0 + threads=1 combined)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    for i in range(len(cats)):
        ax.text(i - w, wins[i] + 0.3, str(wins[i]), ha="center", fontsize=10)
        ax.text(i, ties[i] + 0.3, str(ties[i]), ha="center", fontsize=10)
        ax.text(i + w, losses[i] + 0.3, str(losses[i]), ha="center", fontsize=10)
    plt.tight_layout()
    out = OUT / "winloss_summary.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  {out}")


# ============================================================
# 6) Ratio distribution per target
# ============================================================
def plot_ratio_by_target():
    targets = ["native", "half", "quarter", "up1.5x"]
    data_by_target = {tg: [r["speedup"] for r in RESULTS if r["target"] == tg]
                      for tg in targets}
    data_by_target = {k: v for k, v in data_by_target.items() if v}

    fig, ax = plt.subplots(figsize=(10, 6))
    positions = list(range(1, len(data_by_target) + 1))
    bp = ax.boxplot(list(data_by_target.values()), positions=positions,
                    widths=0.6, patch_artist=True, showmeans=True,
                    meanprops={"marker": "D", "markerfacecolor": "yellow",
                               "markeredgecolor": "black", "markersize": 8})
    for patch, c in zip(bp["boxes"],
                        ["#3498db", "#e74c3c", "#f39c12", "#9b59b6"]):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.axhline(1.0, color="black", linestyle="--", alpha=0.5, label="parity")
    ax.set_xticks(positions, list(data_by_target.keys()))
    ax.set_ylabel("speedup (Nelux fps / torchcodec fps)")
    ax.set_title("Speedup distribution by resize target\n>1 = Nelux faster, <1 = torchcodec faster")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3, which="both")
    plt.tight_layout()
    out = OUT / "ratio_by_target.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  {out}")


def main():
    print(f"Plotting {len(RESULTS)} rows from {OUT/'results.json'}")
    plot_speedup_heatmap_pix_fmt()
    plot_fps_pix_fmt()
    plot_fps_codec_res()
    plot_quality_scatter()
    plot_winloss()
    plot_ratio_by_target()
    print("\nAll plots saved.")

    # Also print summary stats
    print("\n=== SUMMARY ===")
    by_target = {}
    for r in RESULTS:
        by_target.setdefault(r["target"], []).append(r["speedup"])
    for tg, sps in by_target.items():
        med = float(np.median(sps))
        win_count = sum(1 for s in sps if s >= 1.05)
        loss_count = sum(1 for s in sps if s <= 0.95)
        print(f"  {tg:8} median speedup {med:.2f}x  "
              f"Nelux wins {win_count}/{len(sps)}  losses {loss_count}/{len(sps)}")

    overall = [r["speedup"] for r in RESULTS]
    win_count = sum(1 for s in overall if s >= 1.05)
    loss_count = sum(1 for s in overall if s <= 0.95)
    tie_count = len(overall) - win_count - loss_count
    print(f"\n  OVERALL: {len(overall)} cases  Nelux wins {win_count}  ties {tie_count}  losses {loss_count}")
    print(f"  median speedup {float(np.median(overall)):.2f}x")
    print(f"  geo-mean speedup {float(np.exp(np.mean(np.log(overall)))):.2f}x")


if __name__ == "__main__":
    main()
