from copy import copy
import numpy as np

from robpicker.configs.common_config import basic_cfg

cfg = copy(basic_cfg)

# Model configuration (meta-learning enabled)
cfg.model = "robpicker.models.net_meta"
cfg.backbone = "resnet34"
cfg.backbone_args = None
cfg.class_weights = None
cfg.lvl_weights = np.array([0, 0, 0, 1])

# Meta-learning configuration
cfg.reweight = True
cfg.correct = True
cfg.loss_weight = True
cfg.meta_mixup = False

# Betty configuration
cfg.meta_type = "darts"
cfg.unroll_steps = 5
cfg.warmup_steps = 50
cfg.train_iters = 5000
cfg.valid_steps = 200
cfg.rollback = False
cfg.log_step = 100

# Meta module optimizer settings
cfg.meta_lr = 2e-4
cfg.meta_weight_decay = 0.0
cfg.meta_alpha = 0.0
cfg.meta_temperature = 1.0
cfg.meta_lambda = 0.0

# Label correction module type: "simple" or "feature"
cfg.correct_type = "simple"
cfg.correct_backbone = "resnet34"

# LossWeight module configuration
cfg.loss_weight_hidden_channels = 16
cfg.loss_weight_num_layers = 3

# MetaMixup module configuration
cfg.meta_mixup_hidden_channels = 16
cfg.meta_mixup_num_layers = 3
cfg.meta_mixup_use_targets = True

# Meta dataset settings
cfg.meta_class_weights = None

# Checkpoint resume path
cfg.resume_checkpoint = None

# Peak-picking / post-processing (see robpicker/postprocess/peaks.py).
# pick_mode selects how a class probability map is turned into picks:
#   "nms"      - legacy max-pool NMS (default; backward compatible)
#   "blur_nms" - Gaussian blur + max-pool NMS (best for crowded/touching particles)
#   "cc"       - threshold + connected components (best for sparse/medium density)
cfg.pick_mode = "nms"
cfg.pick_blur_sigma_frac = 0.5   # blur_nms: sigma = frac * radius_vox
cfg.pick_nms_frac = 1.0          # blur_nms: NMS radius = round(frac * radius_vox)
cfg.pick_cc_thresh = 0.5         # cc: probability threshold for components
cfg.pick_cc_min_frac = 0.1       # cc: drop components < frac * expected sphere volume
cfg.pick_cc_conf = "max"         # cc: component confidence, "max" or "mean"

meta_cfg = cfg
