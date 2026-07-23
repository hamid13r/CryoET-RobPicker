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

### Optional: dense segmentation labels (MRC)

In addition to (or instead of) an XML point list, each tomogram may have a
**segmentation MRC** of the same shape as the tomogram, named
`{tomo_name}_seg.mrc` (the suffix is configurable via `cfg.seg_suffix`, default
`_seg`). Its voxel values are integer class label IDs; each labeled region is
painted directly (dense hard label) into its class channel, while XML points
remain Gaussian blobs. Both sources are merged into one
`(n_classes, X, Y, Z)` target.

Conventions for the combined label space:

- Segmentation MRC voxel labels use IDs `1, 2, 3, ...`
- XML particle `class_label` values use IDs `4, 5, 6, ...`
- `cfg.class_mapping` and `cfg.classes` must enumerate **all** of them (seg and
  point), one distinct class name / channel per label ID.
- `cfg.seg_classes` lists the dense-segmentation class names so post-processing
  and the F-beta metric can exclude them (point-picking is not meaningful for
  dense regions). Defaults to `[]` (point-only, unchanged behavior).

A tomogram is valid if it has an XML **or** a seg MRC (or both); it is skipped
only when both are missing. Voxel spacing is asserted on the tomogram MRC but
not on the seg MRC; the seg volume shape must equal the tomogram shape (Z,Y,X).

See `robpicker/configs/cfg_seg.py` for a complete combined-mapping example.
A minimal illustration:

```python
cfg.classes       = ["membrane", "microtubule", "ribosome80s", "atp"]  # seg first, points after
cfg.class_mapping = {1: "membrane", 2: "microtubule", 4: "ribosome80s", 5: "atp"}
cfg.seg_classes   = ["membrane", "microtubule"]
cfg.n_classes     = len(cfg.classes)
# class_weights / meta_class_weights have length n_classes + 1 (trailing = background)
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
- `cfg.particle_radi` (radii in angstroms; it affects the inference process)
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
