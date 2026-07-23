"""
Smoke test for combined XML-point + MRC-segmentation label loading.

Synthesizes tiny tomograms plus matching segmentation MRCs and XML point lists,
runs ``CustomDataset`` in ``train`` mode, and asserts the unified target tensor
has nonzero voxels in BOTH a dense segmentation channel and a Gaussian-blob
point channel. Also exercises the required edge cases:

  * seg-only tomogram (no XML)
  * XML-only tomogram (no seg)
  * seg label present that is not in class_mapping (ignored)
  * seg / tomogram shape mismatch (raises a clear error)

Run:  python tests/test_seg_labels.py       (no pytest required)
"""

import os
import sys
import tempfile
from copy import copy
from types import SimpleNamespace

import numpy as np
import mrcfile
from monai import transforms as mt

# Make the package importable when run directly from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robpicker.data import ds  # noqa: E402

VOXEL_SPACING = 7.84
SHAPE_ZYX = (48, 48, 48)  # (Z, Y, X)

# Combined label space: seg IDs 1,2 ; point IDs 4,5.
CLASSES = ["membrane", "microtubule", "ribosome80s", "atp"]
CLASS_MAPPING = {1: "membrane", 2: "microtubule", 4: "ribosome80s", 5: "atp"}
SEG_CLASSES = ["membrane", "microtubule"]


def write_tomogram(path, shape_zyx=SHAPE_ZYX):
    data = np.random.randn(*shape_zyx).astype(np.float32)
    with mrcfile.new(path, overwrite=True) as m:
        m.set_data(data)
        m.voxel_size = VOXEL_SPACING
    return data


def write_seg(path, shape_zyx, blocks):
    """blocks: list of (label_id, (z0,z1,y0,y1,x0,x1))."""
    seg = np.zeros(shape_zyx, dtype=np.int16)
    for label_id, (z0, z1, y0, y1, x0, x1) in blocks:
        seg[z0:z1, y0:y1, x0:x1] = label_id
    with mrcfile.new(path, overwrite=True) as m:
        m.set_data(seg)
        m.voxel_size = 1.0  # deliberately different; not enforced for seg
    return seg


def write_xml(path, tomo_name, points):
    """points: list of (class_label, x, y, z) in voxel coordinates."""
    objs = "\n".join(
        f'  <object tomo_name="{tomo_name}" class_label="{c}" x="{x}" y="{y}" z="{z}"/>'
        for (c, x, y, z) in points
    )
    with open(path, "w") as f:
        f.write(f"<objlist>\n{objs}\n</objlist>\n")


def make_cfg(data_dir):
    cfg = SimpleNamespace()
    cfg.data_dir = data_dir
    cfg.train_folder = "train"
    cfg.meta_folder = "meta"
    cfg.test_folder = "test"
    cfg.voxel_spacing = VOXEL_SPACING
    cfg.seg_suffix = "_seg"
    cfg.classes = CLASSES
    cfg.n_classes = len(CLASSES)
    cfg.class_mapping = CLASS_MAPPING
    cfg.seg_classes = SEG_CLASSES
    cfg.target_radius_vox = 6
    # Small crops so a patch comfortably fits the tiny tomogram.
    cfg.roi_size = [32, 32, 32]
    cfg.sub_batch_size = 2
    cfg.train_sub_epochs = 1
    # One class-aware weight per class (no background entry).
    cfg.resample_weight = [1.0, 1.0, 1.0, 1.0]
    cfg.resample_bg_weight = 0.1
    cfg.static_transforms = mt.Compose([
        mt.EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
        mt.NormalizeIntensityd(keys="image"),
    ])
    return cfg


def build_train_aug(cfg):
    return mt.Compose([
        mt.RandSpatialCropSamplesd(
            keys=["image", "label"],
            roi_size=cfg.roi_size,
            num_samples=cfg.sub_batch_size,
        ),
    ])


def channel_counts(label, classes):
    return {c: int((label[i] > 0).sum()) for i, c in enumerate(classes)}


def main():
    root = tempfile.mkdtemp(prefix="robpicker_segtest_")
    train_dir = os.path.join(root, "train")
    os.makedirs(train_dir, exist_ok=True)

    class2id = {c: i for i, c in enumerate(CLASSES)}

    # --- tomoA: BOTH seg (membrane, label 1) + xml (ribosome80s, label 4) ---
    #     plus a stray seg label 9 that is NOT in class_mapping (must be ignored).
    write_tomogram(os.path.join(train_dir, "tomoA.mrc"))
    write_seg(
        os.path.join(train_dir, "tomoA_seg.mrc"), SHAPE_ZYX,
        blocks=[
            (1, (8, 20, 8, 20, 8, 20)),    # membrane dense block
            (9, (40, 45, 40, 45, 40, 45)),  # unknown label -> ignored
        ],
    )
    write_xml(os.path.join(train_dir, "tomoA_objl.xml"), "tomoA",
              points=[(4, 30, 30, 30)])  # ribosome80s point

    # --- tomoB: seg-only (microtubule, label 2), no XML ---
    write_tomogram(os.path.join(train_dir, "tomoB.mrc"))
    write_seg(os.path.join(train_dir, "tomoB_seg.mrc"), SHAPE_ZYX,
              blocks=[(2, (10, 24, 10, 24, 10, 24))])

    # --- tomoC: xml-only (atp, label 5), no seg ---
    write_tomogram(os.path.join(train_dir, "tomoC.mrc"))
    write_xml(os.path.join(train_dir, "tomoC_objl.xml"), "tomoC",
              points=[(5, 24, 24, 24)])

    cfg = make_cfg(root)
    aug = build_train_aug(cfg)

    print(f"Dataset root: {root}")
    dataset = ds.CustomDataset(df=None, cfg=cfg, aug=aug, mode="train")

    # Discovery must find all three (seg-only kept, xml-only kept).
    names = sorted(t["tomo_name"] for t in dataset.tomograms)
    assert names == ["tomoA", "tomoB", "tomoC"], names
    by_name = {t["tomo_name"]: t for t in dataset.tomograms}
    assert by_name["tomoB"]["xml_path"] is None and by_name["tomoB"]["seg_path"]
    assert by_name["tomoC"]["seg_path"] is None and by_name["tomoC"]["xml_path"]
    print("OK: discovery kept seg-only and xml-only tomograms.")

    # --- Unified target for tomoA (full, uncropped) ---
    labA = dataset.load_one(by_name["tomoA"])["label"]
    cntA = channel_counts(labA, CLASSES)
    print(f"tomoA per-channel voxel counts: {cntA}")
    seg_c = cntA["membrane"]
    pt_c = cntA["ribosome80s"]
    assert seg_c > 0, "seg (membrane) channel must have nonzero voxels"
    assert pt_c > 0, "point (ribosome80s) channel must have nonzero voxels"
    # Dense block (12^3 = 1728) should dwarf the small Gaussian blob.
    assert seg_c == 12 * 12 * 12, f"dense block count {seg_c} != 1728"
    assert seg_c > pt_c, "dense seg region should exceed the point blob"
    # Unknown seg label 9 must NOT leak into any channel.
    assert cntA["microtubule"] == 0 and cntA["atp"] == 0, \
        f"unexpected leakage from unknown/other labels: {cntA}"
    print("OK: tomoA has dense seg channel + point blob; unknown label ignored.")

    # --- seg-only tomoB ---
    labB = dataset.load_one(by_name["tomoB"])["label"]
    cntB = channel_counts(labB, CLASSES)
    print(f"tomoB (seg-only) per-channel voxel counts: {cntB}")
    assert cntB["microtubule"] == 14 * 14 * 14 and cntB["ribosome80s"] == 0 \
        and cntB["atp"] == 0 and cntB["membrane"] == 0
    print("OK: seg-only tomogram paints only its seg channel.")

    # --- xml-only tomoC ---
    labC = dataset.load_one(by_name["tomoC"])["label"]
    cntC = channel_counts(labC, CLASSES)
    print(f"tomoC (xml-only) per-channel voxel counts: {cntC}")
    assert cntC["atp"] > 0 and cntC["membrane"] == 0 and cntC["microtubule"] == 0
    print("OK: xml-only tomogram paints only its point channel.")

    # --- Full train pipeline produces input/target tensors ---
    sample = dataset[0]
    tgt = sample["target"]
    print(f"train sample target shape: {tuple(tgt.shape)} "
          f"(sub_batch, n_classes, X, Y, Z)")
    assert tgt.shape[1] == cfg.n_classes
    assert float(tgt.sum()) > 0, "cropped training target should contain labels"
    print("OK: CustomDataset train pipeline runs and yields nonzero targets.")

    # --- edge case: seg / tomogram shape mismatch must raise ---
    bad_dir = os.path.join(root, "bad")
    bad_train = os.path.join(bad_dir, "train")
    os.makedirs(bad_train, exist_ok=True)
    write_tomogram(os.path.join(bad_train, "tomoX.mrc"), shape_zyx=(48, 48, 48))
    write_seg(os.path.join(bad_train, "tomoX_seg.mrc"), (48, 48, 40),
              blocks=[(1, (5, 15, 5, 15, 5, 15))])
    raised = False
    try:
        ds.discover_tomograms(bad_train, expected_spacing=VOXEL_SPACING,
                              seg_suffix="_seg")
    except ValueError as e:
        raised = True
        print(f"OK: shape mismatch raised ValueError: {e}")
    assert raised, "shape mismatch should raise ValueError"

    print("\nALL SMOKE-TEST ASSERTIONS PASSED.")


if __name__ == "__main__":
    main()
