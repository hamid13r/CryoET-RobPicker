# Training Guide

## Data format (EMPIAR-style)

Prepare a dataset with the following folder structure that includes at least a train folder and a meta folder (meta set is required; test set is optional):

```
/my_dataset/
  train/
    tomo0001.mrc
    tomo0001_objl.xml
    tomo0002.mrc
    tomo0002_objl.xml
  meta/
    tomo0003.mrc
    tomo0003_objl.xml
  test/                # optional; if missing, meta is used for test
    tomo0004.mrc
    tomo0004_objl.xml
```

Requirements:
- Each tomogram is an MRC file: `{tomo_name}.mrc`. The filename does not need to be `tomo00xx`, and it just needs to match the XML file.
- Each XML file is named `{tomo_name}_objl.xml`.
- XML entries must include **voxel** coordinates (not coordinates in angstroms) and class labels (from 1 to num_classes):
  - `<object tomo_name="tomo0001" class_label="1" x="592" y="772" z="287" ... />`

Class labels are mapped to class names by `cfg.class_mapping` in your config (see Configuration below). The labels in XML must match the keys in that mapping, and the mapped names must appear in `cfg.classes`.

## Training targets (per-class spheres)

Each annotated particle is painted into its class channel of the
`(n_classes, X, Y, Z)` target as a **solid binary sphere** with a **Gaussian
taper shell**:

- **Hard core** — every voxel within the core radius is set to `1.0`.
- **Gaussian taper** — voxels just outside the core fall off smoothly from
  ~`1.0` toward `0.0` over the taper width.
- Beyond core radius + taper, the target is `0.0`.

Overlapping particles combine with an element-wise maximum.

The **core radius is per class** and is derived from existing config (no new
radius field): for each class it is

```
r = max(round(cfg.particle_radi[class] / voxel_spacing), 1)   # in voxels
```

where `voxel_spacing` is each tomogram's own spacing (angstroms/voxel). So
`cfg.particle_radi` (radii in angstroms) now **controls the core sphere size**
as well as inference. Every class in `cfg.classes` must have an entry in
`cfg.particle_radi`, or target generation raises a clear error.

The Gaussian shell is controlled by two **optional** fields (no config changes
required to run — sensible defaults are used):

- `cfg.target_taper_vox` — taper width in voxels (default `3`).
- `cfg.target_taper_sigma_vox` — Gaussian sigma in voxels (default
  `target_taper_vox / 2`, i.e. `1.5` when the taper is `3`).

The falloff, measured from the core surface (distance `d` from the particle
center, core radius `r`):

```
value = 1.0                              if d <= r
value = exp(-(d - r)^2 / (2 * sigma^2))  if r < d <= r + taper
value = 0.0                              if d > r + taper
```

## Convert STAR to XML

Many cryo-ET annotations are in RELION `.star` files. Use the converter:

```bash
robpicker-star2xml \
  --input /path/to/classA.star /path/to/classB.star \
  --class-map classA:1 classB:2 \
  --tomo-name tomo0001 \
  --output /path/to/tomo0001_objl.xml
```

Notes:
- `--input` accepts files, globs, or directories; each file should contain items for one particle class in one tomogram.
- `--class-map` maps each input file's basename to a class label.
- Use `--no-angles` if the STAR file lacks angle columns; angle fields are omitted from XML (they are unnecessary in the training).
- `--tomo-name` is the filename of the corresponding MRC file.

## Configuration

Create a config by copying `robpicker/configs/cfg_resnet34.py` and editing:
- `cfg.data_dir` (dataset root)
- `cfg.classes` (class names)
- `cfg.class_mapping` (label ID to class name)
- `cfg.particle_radi` (radii in angstroms; controls both the training-target core sphere size and the inference process — see [Training targets](#training-targets-per-class-spheres))
- Optional: `cfg.target_taper_vox`, `cfg.target_taper_sigma_vox` (Gaussian taper shell around the core sphere; default to `3` and `taper/2`)
- Optional: `cfg.train_folder`, `cfg.meta_folder`, `cfg.test_folder`

Example (minimal):

```python
from copy import copy
from robpicker.configs.meta_config import meta_cfg

cfg = copy(meta_cfg)

cfg.name = "cfg_my_dataset"
cfg.output_dir = "/path/to/where/you/want/to/save/models"

cfg.data_dir = "/path/to/my_dataset"
cfg.train_folder = "train"
cfg.meta_folder = "meta"
cfg.test_folder = "test"  # optional

cfg.classes = ["class_a", "class_b"]
cfg.n_classes = len(cfg.classes)

cfg.class_mapping = {
    1: "class_a",
    2: "class_b",
}

cfg.particle_radi = {
    "class_a": 120,
    "class_b": 80,
}
```
For additional configurations related to inference and evaluation, please see [evaluate.md](evaluate.md). 
The model checkpoints are saved to the `cfg.output_dir` you specify in the config. 

Save your config file with a name like `cfg_my_dataset.py`.

## Training

Train with your configurations (supply the complete path to the config file, or the file name if it is under the configs folder):

```bash
robpicker-train -C cfg_my_dataset # can also be /path/to/cfg_my_dataset.py
```
