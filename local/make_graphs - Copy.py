"""Plotting script that aggregates multiple PEFT methods per model/language.

This version requires explicit model(s) and language(s) and will look for the
three PEFT method directories under the repository layout:
    ./{model}_{peft}/{lang}/

It expects train/eval CSV files inside each language folder (any filename ending
with _train.csv / _eval.csv). The default PEFT methods are: lora, vera, randlora.

It produces two multi-panel figures:
- loss grid: rows=models, cols=languages, each subplot shows train loss for all PEFTs
- wer grid: same layout, each subplot shows eval WER for all PEFTs

Usage:
    python3 local/make_graphs.py --root . --models qwen3-asr --langs irish --outdir plots

Dependencies:
    pip install pandas matplotlib seaborn
"""

import argparse
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set(style="whitegrid")


def safe_read_csv(path):
    df = pd.read_csv(path)
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors="ignore")
        except Exception:
            pass
    return df


def collect_specified(root, models, langs, pefts):
    """Collect available train/eval csvs for the specified models, languages and peft methods.

    Returns struct[model][lang][peft] = {"train": Path or None, "eval": Path or None}
    """
    root = Path(root)
    struct = {}
    for model in models:
        struct.setdefault(model, {})
        for lang in langs:
            struct[model].setdefault(lang, {})
            for peft in pefts:
                dirpath = root / f"{model}_{peft}" / lang
                train_path = None
                eval_path = None
                if dirpath.exists() and dirpath.is_dir():
                    # find any *_train.csv and *_eval.csv
                    trains = sorted(dirpath.glob("*_train.csv"))
                    evals = sorted(dirpath.glob("*_eval.csv"))
                    if trains:
                        train_path = trains[0]
                    if evals:
                        eval_path = evals[0]
                struct[model][lang][peft] = {"train": train_path, "eval": eval_path}
    return struct


def plot_grid(struct, outdir, metric, metric_col, y_label, smoothing=None):
    models = sorted(struct.keys())
    langs = sorted({l for m in struct.values() for l in m.keys()})
    pefts = sorted({p for m in struct.values() for l in m.values() for p in l.keys()})

    rows = len(models)
    cols = len(langs)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)

    palette = sns.color_palette(n_colors=max(len(pefts), 3))
    color_map = {p: palette[i % len(palette)] for i, p in enumerate(pefts)}

    for i, model in enumerate(models):
        for j, lang in enumerate(langs):
            ax = axes[i][j]
            cell = struct.get(model, {}).get(lang, {})
            if not cell:
                ax.text(0.5, 0.5, "no data", ha='center', va='center')
                ax.set_title(f"{model} / {lang}")
                ax.set_xticks([])
                continue

            any_plotted = False
            epoch_used = False
            for p in sorted(cell.keys()):
                paths = cell[p]
                path = paths.get(metric)
                if not path:
                    continue
                try:
                    df = safe_read_csv(path)
                except Exception:
                    continue

                if metric == 'train':
                    if 'loss' not in df.columns:
                        continue
                    y = pd.to_numeric(df['loss'], errors='coerce')
                else:
                    if metric_col not in df.columns:
                        continue
                    y = pd.to_numeric(df[metric_col], errors='coerce')

                epoch_used = False
                if 'epoch' in df.columns:
                    epoch_used = True
                    epochs = df['epoch'].astype(float)
                    xlabel = 'epoch'

                    # aggregate to a single series per integer epoch to avoid duplicate runs
                    if metric == 'eval':
                        # take last eval value for each epoch (sorted by appearance)
                        grouped = df.assign(epoch=epochs).groupby('epoch', sort=True).last().reset_index()
                        x = grouped['epoch'].astype(float)
                        y = pd.to_numeric(grouped[metric_col], errors='coerce')
                    else:  # train
                        # group fractional training steps into integer epoch bins and average loss
                        epoch_int = np.floor(epochs).astype(int)
                        grouped = df.assign(epoch_int=epoch_int).groupby('epoch_int', sort=True).mean(numeric_only=True).reset_index()
                        x = grouped['epoch_int'].astype(float)
                        y = pd.to_numeric(grouped['loss'], errors='coerce')
                else:
                    x = np.arange(len(df))
                    xlabel = 'step'

                if smoothing and len(y) >= smoothing:
                    # if y is a pandas Series after grouping, use rolling
                    try:
                        y = y.rolling(window=smoothing, min_periods=1, center=True).mean()
                    except Exception:
                        pass

                # prepare plotting kwargs
                linestyle = '-' if metric == 'train' else '--'
                plot_kwargs = dict(color=color_map.get(p), linewidth=1, linestyle=linestyle, label=p)
                if metric != 'train':
                    plot_kwargs['marker'] = 'o'
                    plot_kwargs['markersize'] = 3

                # convert to numpy arrays
                try:
                    x_arr = np.asarray(x, dtype=float)
                except Exception:
                    x_arr = np.array(list(x))
                y_arr = np.asarray(y, dtype=float)

                # plot (after aggregation x should be monotonic)
                ax.plot(x_arr, y_arr, **plot_kwargs)
                any_plotted = True

            if not any_plotted:
                ax.text(0.5, 0.5, "no data for metric", ha='center', va='center')

            # if epoch used as x-axis, enforce 0-12 limits
            if epoch_used:
                try:
                    ax.set_xlim(0, 12)
                    ax.set_xticks(list(range(0, 13)))
                except Exception:
                    pass

            ax.set_title(f"{model} / {lang}")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(y_label)
            ax.grid(True)

    # create a single legend placed above the subplots and leave extra top margin
    handles = []
    labels = []
    for p in pefts:
        handles.append(plt.Line2D([0], [0], color=color_map.get(p), marker='o'))
        labels.append(p)

    if handles:
        # place legend slightly above the axes area to avoid overlapping titles
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=min(len(labels), 6))

    # leave more room at the top for the legend (reduce top to 0.90)
    fig.tight_layout(rect=[0, 0, 1, 0.90])

    outpath_png = os.path.join(outdir, f"{model}_{lang}_{metric}_{metric_col}.png")
    #outpath_svg = os.path.join(outdir, f"grid_{metric}_{metric_col}.svg")
    fig.savefig(outpath_png)
    #fig.savefig(outpath_svg)
    plt.close(fig)
    print(f"Saved {outpath_png}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.', help='repo root containing model_peft directories')
    ap.add_argument('--models', required=True, help='comma-separated model names (e.g. qwen3-asr)')
    ap.add_argument('--langs', required=True, help='comma-separated language names (e.g. irish)')
    ap.add_argument('--pefts', default='lora,vera,randlora', help='comma-separated peft methods to include')
    ap.add_argument('--outdir', default='plots', help='output directory for plots')
    ap.add_argument('--smoothing', type=int, default=1, help='rolling window for smoothing (1 = no smoothing)')
    args = ap.parse_args()

    models = [s.strip() for s in args.models.split(',') if s.strip()]
    langs = [s.strip() for s in args.langs.split(',') if s.strip()]
    pefts = [s.strip() for s in args.pefts.split(',') if s.strip()]

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    struct = collect_specified(args.root, models, langs, pefts)

    plot_grid(struct, outdir, metric='train', metric_col='loss', y_label='loss', smoothing=args.smoothing)
    plot_grid(struct, outdir, metric='eval', metric_col='eval_wer', y_label='WER', smoothing=args.smoothing)

    print('Done')
