# Inference and Evaluation Guide

This note explains the key arguments in `robpicker/evaluate.py` and a recommended thresholding workflow.

## Required arguments

- `--checkpoint`: Path to the trained model checkpoint (e.g., `.../checkpoint_best.pth`).
- `--data_dir`: Path to the dataset folder containing tomograms and XML files (e.g., `/path/to/my_dataset/meta` or `/path/to/my_dataset/test`).
- `--output_dir`: Directory to save predictions (and metrics if enabled).
- `--config`: Config name or module path. Use a config that matches your dataset classes and voxel spacing (e.g., `cfg_resnet34` or your custom config).

### Config
The inference tomograms should match the voxel spacing of the training tomograms and it should be set with `cfg.voxel_spacing` in the config file.

The tomogram dimension (in angstroms) should be set in the config file like this:
```
cfg.pp_x_max = 10500
cfg.pp_y_max = 10500
cfg.pp_z_max = 5500
```
The `pp_x_max`, `pp_y_max`, `pp_z_max` correspond to the X, Y, Z axis respectively.

For metric (like F-beta scores) calculation, the follow config can be set in the config file:
```
cfg.metric_beta = 1
cfg.metric_distance_multiplier = 0.5
cfg.metric_weights = {
    "ribosome80s": 1,
    "atp": 1,
}
```
The `metric_beta` denotes the beta in F-beta, so 1 means using F1 score. The `metric_distance_multiplier` means the multiplier for the `cfg.particle_radi` in config: the final distance threshold (for a prediction to be considered as true positive) is `particle_radi * metric_distance_multiplier`. And the `metric_weights` is the weight for the particle classes in calculating the overall metric.

## Common optional arguments

- `--batch_size`: Batch size for inference. Increase for faster inference if GPU memory allows.
- `--thresholds`: Comma-separated list of per-class thresholds in the order of `cfg.classes`.
- `--inference_only`: Skip metric calculation. This is useful when ground-truth annotations are not available.
- `--pick_mode`: Peak-picking strategy, `nms` | `blur_nms` | `cc` (overrides `cfg.pick_mode`). See [Peak picking](train.md#peak-picking-post-processing) for what each mode does and when to use it.
- `--no_flip_tta`: Disable flipping the tomograms as test time augmentation. It might be useful to detect particles with certain handedness (also need to disable the flip augmentation during training).
- `--threshold_range`: Change the threshold search range if you find any of the output thresholds is out of the default range [0.1, 0.6].

## Threshold workflow

1) **Calibrate thresholds on annotated tomograms (typically the meta set).**

Run evaluation on a set with annotations (e.g., `meta/`). If you do not pass `--thresholds`, the script performs a grid search and stores the best thresholds in `metrics.json`.

```bash
robpicker-eval \
  --config cfg_resnet34 \
  --checkpoint /path/to/checkpoint_best.pth \
  --data_dir /path/to/my_dataset/meta \
  --output_dir ./eval_meta
```

2) **Use the calibrated thresholds for new tomograms.**

Pass the thresholds to apply them during inference. If you have unlabeled data (for picking on tomograms without annotations), add `--inference_only`.

```bash
robpicker-eval \
  --config cfg_resnet34 \
  --checkpoint /path/to/checkpoint_best.pth \
  --data_dir /path/to/my_dataset/test \
  --output_dir ./eval_test \
  --thresholds 0.12,0.08 \
  --inference_only
```

The script will write:
- `predictions.csv` (raw predictions)
- `predictions_thresholded.csv` (if `--thresholds` is supplied in inference-only mode)

In the predictions, the unit of the coordinates (x,y,z) is Angstrom (Å), and the `conf` column indicates model confidence of the predictions.

## Peak picking

The evaluation post-processor turns each class probability map into picks using
`cfg.pick_mode` (`nms` | `blur_nms` | `cc`), the same code path used during
training-time validation, so results are identical. See
[Peak picking](train.md#peak-picking-post-processing) for the modes, their
config fields, and guidance:

- `cc` — sparse / medium density.
- `blur_nms` — crowded samples with touching particles.
- `nms` — legacy behaviour (default), useful for Gaussian-blob targets or
  reproducing prior results.

Override the config's mode from the CLI with `--pick_mode`:

```bash
robpicker-eval \
  --config cfg_resnet34 \
  --checkpoint /path/to/checkpoint_best.pth \
  --data_dir /path/to/my_dataset/meta \
  --output_dir ./eval_meta \
  --pick_mode blur_nms
```

## Comparing pick modes

To decide which mode fits your data, run `scripts/compare_pickers.py`. It runs
inference **once**, caches the reconstructed probability volumes, then applies
every mode (plus optional parameter sweeps) to the cache and reports the overall
weighted F-beta, per-class F-beta/precision/recall, mean picks per ground-truth
particle, and runtime. Results are printed as a table and written to
`<out>/picker_compare.csv`.

```bash
python scripts/compare_pickers.py \
  --config cfg_resnet34 \
  --checkpoint /path/to/checkpoint_best.pth \
  --split meta \
  --out output/picker_compare
```

Optional sweeps (each value adds a row for that mode):

```bash
python scripts/compare_pickers.py \
  --config cfg_resnet34 \
  --checkpoint /path/to/checkpoint_best.pth \
  --split meta \
  --out output/picker_compare \
  --cc-thresh 0.3 0.5 0.7 \
  --nms-frac 0.75 1.0 1.25
```

`--split` names a folder under `cfg.data_dir` (e.g. `meta`, `test`); ground
truth is read from that split's XML files, or pass `--gt_csv` to supply it.

## Notes

- Thresholds must match the order of `cfg.classes` in your config.
- If you omit `--inference_only` and annotations are present, the script computes metrics and saves `metrics.json`.
