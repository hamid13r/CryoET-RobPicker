"""
Compare peak-picking strategies on a validation/annotated split.

Runs model inference ONCE, caches the reconstructed per-class probability
volumes, then applies each pick mode ("nms", "blur_nms", "cc") -- plus optional
parameter sweeps -- to the same cache. For every setting it reports:

  - overall weighted F-beta (from robpicker/metrics/metric.py)
  - per-class F-beta / precision / recall
  - mean picks per ground-truth particle
  - runtime (picking only; inference is shared)

Results are printed as a table and written to ``<out>/picker_compare.csv``.

Example:
    python scripts/compare_pickers.py \
        --config cfg_resnet34 \
        --checkpoint /path/to/checkpoint_best.pth \
        --split meta \
        --out output/picker_compare

Optional sweeps (each value produces its own row for that mode):
    --cc-thresh 0.3 0.5 0.7        # sweep cc probability threshold
    --nms-frac 0.75 1.0 1.25       # sweep blur_nms NMS-radius fraction
"""

import argparse
import copy
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robpicker.utils import load_config
from robpicker.evaluate import (
    EvalDataset,
    normalize_patch_overlap,
    load_model,
    run_inference,
    extract_points,
    load_annotations,
)
from robpicker.metrics import metric as metric_mod


def _per_class_stats(solution, submission, cfg):
    """Per-class tp/fp/fn/precision/recall using the metric's matching logic."""
    dist_mult = getattr(cfg, "metric_distance_multiplier", 0.5)
    beta = getattr(cfg, "metric_beta", 1)
    radii = {k: v * dist_mult for k, v in dict(cfg.particle_radi).items()}
    experiments = set(solution["experiment"].unique())

    stats = {}
    for ptype in cfg.classes:
        tp = fp = fn = 0
        for exp in experiments:
            ref = solution[(solution["experiment"] == exp) & (solution["particle_type"] == ptype)][["x", "y", "z"]].values
            cand = submission[(submission["experiment"] == exp) & (submission["particle_type"] == ptype)][["x", "y", "z"]].values
            r = radii.get(ptype, 1)
            if len(ref) == 0:
                ref = np.array([])
                r = 1
            if len(cand) == 0:
                cand = np.array([])
            t, f, n = metric_mod.compute_metrics(ref, r, cand)
            tp += t
            fp += f
            fn += n
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fbeta = ((1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)
                 if (precision + recall) > 0 else 0.0)
        stats[ptype] = dict(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, fbeta=fbeta)
    return stats


def _evaluate_setting(preds_by_exp, cfg, effective_spacing, gt_df, label):
    """Pick with the current cfg, score it, return a list of per-class + overall rows."""
    t0 = time.time()
    preds = extract_points(preds_by_exp, cfg, effective_spacing)
    runtime = time.time() - t0

    pred_df = (pd.DataFrame(preds) if preds else
               pd.DataFrame(columns=["experiment", "x", "y", "z", "particle_type", "conf"]))

    solution = gt_df.copy()
    solution["id"] = range(len(solution))
    submission = pred_df.copy()
    submission["id"] = range(len(submission))

    if len(solution) > 0 and len(submission) > 0:
        overall, per_class_fb = metric_mod.score(
            solution, submission, row_id_column_name="id",
            distance_multiplier=getattr(cfg, "metric_distance_multiplier", 0.5),
            beta=getattr(cfg, "metric_beta", 1), weighted=True, cfg=cfg,
        )
        stats = _per_class_stats(solution, submission, cfg)
    else:
        overall = 0.0
        per_class_fb = {}
        stats = {p: dict(tp=0, fp=0, fn=0, precision=0.0, recall=0.0, fbeta=0.0) for p in cfg.classes}

    rows = []
    for p in cfg.classes:
        n_gt = int(((gt_df["particle_type"] == p)).sum())
        n_pred = int((pred_df["particle_type"] == p).sum()) if len(pred_df) else 0
        s = stats.get(p, {})
        rows.append({
            "setting": label,
            "particle_type": p,
            "fbeta": s.get("fbeta", per_class_fb.get(p, 0.0)),
            "precision": s.get("precision", 0.0),
            "recall": s.get("recall", 0.0),
            "n_pred": n_pred,
            "n_gt": n_gt,
            "picks_per_gt": (n_pred / n_gt) if n_gt > 0 else float("nan"),
            "overall_fbeta": overall,
            "runtime_s": runtime,
        })
    total_gt = len(gt_df)
    total_pred = len(pred_df)
    rows.append({
        "setting": label,
        "particle_type": "OVERALL",
        "fbeta": overall,
        "precision": float("nan"),
        "recall": float("nan"),
        "n_pred": total_pred,
        "n_gt": total_gt,
        "picks_per_gt": (total_pred / total_gt) if total_gt > 0 else float("nan"),
        "overall_fbeta": overall,
        "runtime_s": runtime,
    })
    return rows


def _build_settings(args):
    """Return list of (label, overrides-dict) describing each cfg to evaluate."""
    settings = []
    # Base run of every mode with defaults.
    settings.append(("nms", {"pick_mode": "nms"}))
    settings.append(("blur_nms", {"pick_mode": "blur_nms"}))
    settings.append(("cc", {"pick_mode": "cc"}))
    # Optional sweeps.
    for t in (args.cc_thresh or []):
        settings.append((f"cc[thresh={t}]", {"pick_mode": "cc", "pick_cc_thresh": float(t)}))
    for f in (args.nms_frac or []):
        settings.append((f"blur_nms[nms_frac={f}]", {"pick_mode": "blur_nms", "pick_nms_frac": float(f)}))
    return settings


def main():
    parser = argparse.ArgumentParser(description="Compare peak-picking strategies")
    parser.add_argument("--config", default="cfg_resnet34", help="Config name or path")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint file")
    parser.add_argument("--split", default="meta",
                        help="Data split folder under cfg.data_dir (e.g. meta, test)")
    parser.add_argument("--data_dir", default=None,
                        help="Override data directory (default: <cfg.data_dir>/<split>)")
    parser.add_argument("--out", default="output/picker_compare", help="Output directory")
    parser.add_argument("--gt_csv", default=None, help="Optional ground-truth CSV")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--patch_overlap", default=0.3)
    parser.add_argument("--no_flip_tta", dest="flip_tta", action="store_false", default=True)
    parser.add_argument("--cc-thresh", nargs="*", type=float, default=None,
                        help="Sweep cc probability thresholds, e.g. --cc-thresh 0.3 0.5 0.7")
    parser.add_argument("--nms-frac", nargs="*", type=float, default=None,
                        help="Sweep blur_nms NMS-radius fractions, e.g. --nms-frac 0.75 1.0 1.25")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    cfg, _ = load_config(args.config)
    cfg.device = device
    cfg.batch_size_val = args.batch_size

    data_dir = args.data_dir or os.path.join(cfg.data_dir, args.split)
    print(f"Config: {args.config}   device: {device}")
    print(f"Data dir: {data_dir}")

    patch_overlap, patch_overlap_enabled = normalize_patch_overlap(args.patch_overlap, cfg.roi_size)

    dataset = EvalDataset(cfg, data_dir, patch_overlap=patch_overlap, preload_all=False)
    dataset.patch_overlap_enabled = patch_overlap_enabled

    # Ground truth.
    if args.gt_csv:
        gt_df = pd.read_csv(args.gt_csv)
        gt_df = gt_df[gt_df["experiment"].isin(dataset.experiments)].copy()
    else:
        gt_df = load_annotations(data_dir, cfg.classes, cfg.class_mapping,
                                 expected_spacing=cfg.voxel_spacing)
    print(f"Loaded {len(gt_df)} ground-truth annotations")

    model = load_model(cfg, args.checkpoint, device)

    voxel_spacing = getattr(dataset, "actual_voxel_spacing", cfg.voxel_spacing)
    effective_spacing = voxel_spacing * 2

    # Inference ONCE; cache probability volumes.
    print("Running inference once (cached for all pick modes)...")
    t_inf = time.time()
    preds_by_exp = run_inference(model, dataset, cfg, device, flip_tta=args.flip_tta)
    print(f"Inference done in {time.time() - t_inf:.1f}s "
          f"({len(preds_by_exp)} experiments cached)")

    all_rows = []
    for label, overrides in _build_settings(args):
        run_cfg = copy.copy(cfg)
        for k, v in overrides.items():
            setattr(run_cfg, k, v)
        print(f"  Picking: {label} ...")
        all_rows.extend(_evaluate_setting(preds_by_exp, run_cfg, effective_spacing, gt_df, label))

    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(args.out, "picker_compare.csv")
    df.to_csv(csv_path, index=False)

    # Pretty table.
    print("\n" + "=" * 92)
    print("PICKER COMPARISON")
    print("=" * 92)
    hdr = f"{'setting':<22}{'class':<16}{'F-beta':>8}{'prec':>8}{'rec':>8}{'n_pred':>8}{'picks/gt':>10}{'time(s)':>9}"
    print(hdr)
    print("-" * 92)
    for label, _ in _build_settings(args):
        sub = df[df["setting"] == label]
        for _, r in sub.iterrows():
            pg = "" if (isinstance(r["picks_per_gt"], float) and np.isnan(r["picks_per_gt"])) else f"{r['picks_per_gt']:.2f}"
            prec = "" if (isinstance(r["precision"], float) and np.isnan(r["precision"])) else f"{r['precision']:.3f}"
            rec = "" if (isinstance(r["recall"], float) and np.isnan(r["recall"])) else f"{r['recall']:.3f}"
            print(f"{r['setting']:<22}{r['particle_type']:<16}{r['fbeta']:>8.3f}{prec:>8}{rec:>8}"
                  f"{int(r['n_pred']):>8}{pg:>10}{r['runtime_s']:>9.2f}")
        print("-" * 92)
    print(f"\nSaved comparison table to: {csv_path}")


if __name__ == "__main__":
    main()
