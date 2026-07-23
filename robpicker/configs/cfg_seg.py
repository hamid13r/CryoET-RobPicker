"""
Example config demonstrating a COMBINED label space:

- Dense segmentation MRC classes (painted directly from ``{tomo_name}_seg.mrc``
  voxel labels) come first.
- Sparse XML point classes (painted as Gaussian blobs from
  ``{tomo_name}_objl.xml``) come after.

Convention (see robpicker/data/ds.py):
  * Segmentation MRC voxel labels use IDs 1, 2, 3, ...
  * XML particle ``class_label`` values use IDs 4, 5, 6, ...
  * ``cfg.class_mapping`` enumerates ALL of them (seg + point), one distinct
    class name / channel per label ID.
  * ``cfg.seg_classes`` lists which class names are dense segmentation, so the
    coordinate post-processing / F-beta metric can EXCLUDE them (point-picking
    local-maxima is not meaningful for dense regions).

Either source may be present per tomogram; a tomogram is valid if it has an XML
OR a segmentation MRC. When no seg MRC exists anywhere, behavior is identical to
the point-only configs (e.g. cfg_resnet34.py).
"""

from copy import copy
import os
import numpy as np

from robpicker.configs.meta_config import meta_cfg

cfg = copy(meta_cfg)

# Paths
cfg.name = os.path.basename(__file__).split(".")[0]
cfg.output_dir = f"./output/{os.path.basename(__file__).split('.')[0]}"

# Dataset-specific settings
cfg.data_dir = "./data"
cfg.train_folder = "train"
cfg.meta_folder = "meta"
cfg.test_folder = "test"
cfg.voxel_spacing = 7.84

# Segmentation volume suffix: {tomo_name}_seg.mrc (default; shown for clarity)
cfg.seg_suffix = "_seg"

# Combined label space (seg classes first, point classes after).
cfg.classes = ["membrane", "microtubule", "ribosome80s", "atp"]
cfg.n_classes = len(cfg.classes)
cfg.class_mapping = {
    1: "membrane",      # dense segmentation label ID
    2: "microtubule",   # dense segmentation label ID
    4: "ribosome80s",   # XML point class_label
    5: "atp",           # XML point class_label
}
# Which channels are dense segmentation (vs. point particles).
cfg.seg_classes = ["membrane", "microtubule"]

# Particle radii (Angstroms). Used by point-picking NMS and the F-beta metric.
# Seg classes are excluded from coordinate post-processing, but radii are kept
# here so cfg.particle_radi and cfg.metric_weights share identical keys.
cfg.particle_radi = {
    "membrane": 100,
    "microtubule": 120,
    "ribosome80s": 150,
    "atp": 80,
}

# Post-processing bounds for tomograms
cfg.pp_x_max = 10500
cfg.pp_y_max = 10500
cfg.pp_z_max = 5500
cfg.pp_conf_thresh = 0.01

cfg.metric_beta = 1
cfg.metric_distance_multiplier = 0.5
cfg.metric_weights = {
    "membrane": 1,
    "microtubule": 1,
    "ribosome80s": 1,
    "atp": 1,
}

# Model configuration tied to class count (head has out_channels = n_classes + 1
# internally, the +1 being the background channel).
cfg.backbone_args = dict(
    spatial_dims=3,
    in_channels=cfg.in_channels,
    out_channels=cfg.n_classes,
    backbone=cfg.backbone,
    pretrained=cfg.pretrained,
)

cfg.lr = 5e-4
cfg.unroll_steps = 5     # Number of inner loop steps before meta-update
cfg.warmup_steps = 50    # Warmup steps before meta-learning starts
cfg.train_iters = 1500   # Total training iterations
cfg.valid_steps = 500    # Validation frequency (steps)
cfg.log_step = 20        # Logging frequency

cfg.meta_lr = 2e-4

# Class weights have length n_classes + 1 (the trailing entry is background).
cfg.class_weights = np.array([64, 64, 64, 512, 1])
cfg.lvl_weights = np.array([0, 0, 0, 1])
cfg.meta_class_weights = np.array([64, 64, 64, 512, 1])

# Resampling settings for class-aware crops (one weight per class, no bg).
cfg.resample_weight = [1.0, 1.0, 1.0, 2.0]
cfg.resample_bg_weight = 0.1
cfg.resample_stats_batches = 30

default_cfg = cfg
