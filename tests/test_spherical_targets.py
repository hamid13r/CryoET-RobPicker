"""
Smoke test for per-class SOLID spherical training targets with a Gaussian taper.

Synthesizes a tiny tomogram plus an XML point list with two particles of two
different classes whose ``particle_radi`` / ``voxel_spacing`` yield known
integer core radii, runs ``CustomDataset`` in ``train`` mode, and asserts:

  * the max target value is exactly 1.0;
  * the number of voxels equal to 1.0 in a channel equals the hard-core stencil
    sum (Euclidean distance <= r);
  * there exist voxels with values strictly between 0 and 1 (the Gaussian shell);
  * every voxel beyond distance r + taper is exactly 0;
  * two classes with different ``particle_radi`` produce cores of the expected
    differing sizes.

Also prints per-class derived voxel radii, taper width, sigma, and per-class
nonzero voxel counts.

Run:  python tests/test_spherical_targets.py       (no pytest required)
"""

import os
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import mrcfile
from monai import transforms as mt

# Make the package importable when run directly from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robpicker.data import ds  # noqa: E402

# voxel_spacing chosen with particle_radi so radii are exact integers:
#   ribosome80s: 50 / 10 = 5   ->  r = 5
#   atp:         30 / 10 = 3   ->  r = 3
VOXEL_SPACING = 10.0
SHAPE_ZYX = (48, 48, 48)  # (Z, Y, X)
TAPER = 3

CLASSES = ["ribosome80s", "atp"]
CLASS_MAPPING = {1: "ribosome80s", 2: "atp"}
PARTICLE_RADI = {"ribosome80s": 50, "atp": 30}
EXPECTED_R = {"ribosome80s": 5, "atp": 3}

# Particle centers (x, y, z) placed far apart so their spheres never overlap.
CENTERS = {"ribosome80s": (12, 12, 12), "atp": (36, 36, 36)}


def write_tomogram(path, shape_zyx=SHAPE_ZYX):
    data = np.random.randn(*shape_zyx).astype(np.float32)
    with mrcfile.new(path, overwrite=True) as m:
        m.set_data(data)
        m.voxel_size = VOXEL_SPACING
    return data


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
    cfg.classes = CLASSES
    cfg.n_classes = len(CLASSES)
    cfg.class_mapping = CLASS_MAPPING
    cfg.particle_radi = PARTICLE_RADI
    cfg.target_taper_vox = TAPER  # sigma defaults to taper / 2 = 1.5
    cfg.roi_size = [32, 32, 32]
    cfg.sub_batch_size = 2
    cfg.train_sub_epochs = 1
    cfg.resample_weight = [1.0, 1.0]
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


def reference_core_count(r):
    """Number of stencil voxels with Euclidean distance <= r (the hard core)."""
    R = r + TAPER
    offs = np.arange(-R, R + 1)
    gx, gy, gz = np.meshgrid(offs, offs, offs, indexing="ij")
    dist = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
    return int((dist <= r).sum())


def channel_dist_field(shape, center):
    """Euclidean distance from `center` for every voxel in an (X,Y,Z) volume."""
    ax = np.arange(shape[0]) - center[0]
    ay = np.arange(shape[1]) - center[1]
    az = np.arange(shape[2]) - center[2]
    gx, gy, gz = np.meshgrid(ax, ay, az, indexing="ij")
    return np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)


def main():
    root = tempfile.mkdtemp(prefix="robpicker_sphere_")
    train_dir = os.path.join(root, "train")
    os.makedirs(train_dir, exist_ok=True)

    write_tomogram(os.path.join(train_dir, "tomoA.mrc"))
    write_xml(
        os.path.join(train_dir, "tomoA_objl.xml"), "tomoA",
        points=[
            (1, *CENTERS["ribosome80s"]),  # ribosome80s, r = 5
            (2, *CENTERS["atp"]),          # atp,          r = 3
        ],
    )

    cfg = make_cfg(root)
    aug = build_train_aug(cfg)

    print(f"Dataset root: {root}")
    dataset = ds.CustomDataset(df=None, cfg=cfg, aug=aug, mode="train")

    tomo_info = dataset.tomograms[0]

    # --- Derived per-class geometry ---
    radii = dataset._class_radii_vox(float(tomo_info["voxel_spacing"]))
    taper = int(getattr(cfg, "target_taper_vox", 3))
    sigma = float(getattr(cfg, "target_taper_sigma_vox", taper / 2.0))
    print(f"Derived voxel radii: {radii}")
    print(f"Taper width (vox): {taper}   Sigma (vox): {sigma}")
    for c in CLASSES:
        assert radii[c] == EXPECTED_R[c], f"{c}: r={radii[c]} != {EXPECTED_R[c]}"
    print("OK: per-class derived radii match expected integers.")

    label = dataset.load_one(tomo_info)["label"]
    x_shape = label.shape[-3:]

    # --- Global max is exactly 1.0 ---
    assert np.isclose(label.max(), 1.0), f"max target {label.max()} != 1.0"
    print(f"OK: global max target value == {float(label.max())}")

    for c in CLASSES:
        cid = dataset.class2id[c]
        chan = label[cid]
        r = radii[c]
        R = r + taper
        center = CENTERS[c]

        core_voxels = int((chan == 1.0).sum())
        shell_voxels = int(((chan > 0.0) & (chan < 1.0)).sum())
        nonzero = int((chan > 0.0).sum())
        print(f"[{c}] r={r} R={R} core(==1.0)={core_voxels} "
              f"shell(0<v<1)={shell_voxels} nonzero={nonzero}")

        # Hard core count matches the analytic stencil core sum.
        expected_core = reference_core_count(r)
        assert core_voxels == expected_core, \
            f"{c}: core voxels {core_voxels} != stencil core {expected_core}"

        # There must be a Gaussian shell (values strictly between 0 and 1).
        assert shell_voxels > 0, f"{c}: expected a Gaussian taper shell"

        # Everything beyond distance r + taper must be exactly 0.
        dist = channel_dist_field(x_shape, center)
        assert np.all(chan[dist > R] == 0.0), \
            f"{c}: found nonzero target beyond r + taper"

        # Every hard-core voxel lies within distance r.
        assert np.all(chan[dist <= r] == 1.0), \
            f"{c}: core voxel(s) within r are not 1.0"

    print("OK: per-class core count, Gaussian shell, and taper cutoff verified.")

    # --- Two classes with different radii produce different core sizes ---
    core_rib = int((label[dataset.class2id["ribosome80s"]] == 1.0).sum())
    core_atp = int((label[dataset.class2id["atp"]] == 1.0).sum())
    assert core_rib > core_atp, \
        f"larger-radius class core ({core_rib}) should exceed smaller ({core_atp})"
    print(f"OK: differing radii -> differing cores (ribosome80s={core_rib} "
          f"> atp={core_atp}).")

    # --- Missing particle_radi entry must raise a clear error ---
    bad_cfg = make_cfg(root)
    bad_cfg.particle_radi = {"ribosome80s": 50}  # atp intentionally missing
    raised = False
    try:
        ds.CustomDataset(df=None, cfg=bad_cfg, aug=build_train_aug(bad_cfg), mode="train")
    except ValueError as e:
        raised = True
        print(f"OK: missing particle_radi raised ValueError: {e}")
    assert raised, "missing particle_radi entry should raise ValueError"

    # --- Full train pipeline still yields nonzero targets ---
    sample = dataset[0]
    tgt = sample["target"]
    print(f"train sample target shape: {tuple(tgt.shape)} "
          f"(sub_batch, n_classes, X, Y, Z)")
    assert tgt.shape[1] == cfg.n_classes
    assert float(tgt.sum()) > 0, "cropped training target should contain labels"
    print("OK: CustomDataset train pipeline runs and yields nonzero targets.")

    print("\nALL SMOKE-TEST ASSERTIONS PASSED.")


if __name__ == "__main__":
    main()
