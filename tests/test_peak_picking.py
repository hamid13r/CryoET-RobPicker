"""
CPU-only smoke test for the peak-picking strategies in
``robpicker/postprocess/peaks.py`` (no pytest required).

Synthesizes probability volumes built from the SAME solid-sphere-plus-Gaussian
taper stencil the trainer paints (see ``ds._get_sphere_stencil``), for two
classes, at noise levels 0 and 0.02, and checks:

  Case A (well-separated particles)
    - "blur_nms" and "cc" each return exactly one pick per particle, within
      1 voxel of the true center;
    - "nms" returns many picks per particle (reproduces the plateau bug).

  Case B (two touching same-class spheres, centers ~1.8*r apart)
    - report the pick count for every mode; assert "blur_nms" returns 2.
      (cc is expected to merge them into one -- a known limitation -- so its
      count is only printed.)

  Case C (Gaussian-blob targets, the main-branch scheme)
    - all three modes return one pick per particle (no regression).

Run:  python tests/test_peak_picking.py
"""

import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robpicker.postprocess.peaks import pick_peaks  # noqa: E402


TAPER = 3
SIGMA_TAPER = TAPER / 2.0
CONF_THRESH = 0.2  # above the 0.02 noise floor, below the ~1.0 particle peak


# ---------------------------------------------------------------------------
# synthetic volume builders
# ---------------------------------------------------------------------------

def sphere_stencil(r, taper=TAPER, sigma=SIGMA_TAPER):
    """Solid binary core (dist<=r) + Gaussian taper shell; ds._get_sphere_stencil."""
    R = r + taper
    offs = np.arange(-R, R + 1)
    gx, gy, gz = np.meshgrid(offs, offs, offs, indexing="ij")
    dist = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        stencil = np.exp(-((dist - r) ** 2) / (2 * sigma ** 2)).astype(np.float32)
    stencil[dist <= r] = 1.0
    stencil[dist > R] = 0.0
    return stencil


def gaussian_blob(shape, center, sigma):
    """A pure Gaussian blob peaking at 1.0 (the main-branch target scheme)."""
    ax = np.arange(shape[0]) - center[0]
    ay = np.arange(shape[1]) - center[1]
    az = np.arange(shape[2]) - center[2]
    gx, gy, gz = np.meshgrid(ax, ay, az, indexing="ij")
    dist2 = gx ** 2 + gy ** 2 + gz ** 2
    return np.exp(-dist2 / (2 * sigma ** 2)).astype(np.float32)


def paint_sphere(vol, center, r):
    """Paint a sphere stencil into ``vol`` (element-wise max), clipped to bounds."""
    R = r + TAPER
    st = sphere_stencil(r)
    cx, cy, cz = center
    x_lo, x_hi = max(0, cx - R), min(vol.shape[0], cx + R + 1)
    y_lo, y_hi = max(0, cy - R), min(vol.shape[1], cy + R + 1)
    z_lo, z_hi = max(0, cz - R), min(vol.shape[2], cz + R + 1)
    sx_lo, sx_hi = x_lo - (cx - R), x_hi - (cx - R)
    sy_lo, sy_hi = y_lo - (cy - R), y_hi - (cy - R)
    sz_lo, sz_hi = z_lo - (cz - R), z_hi - (cz - R)
    vol[x_lo:x_hi, y_lo:y_hi, z_lo:z_hi] = np.maximum(
        vol[x_lo:x_hi, y_lo:y_hi, z_lo:z_hi],
        st[sx_lo:sx_hi, sy_lo:sy_hi, sz_lo:sz_hi],
    )
    return vol


def add_noise(vol, noise):
    if noise <= 0:
        return vol
    rng = np.random.default_rng(0)
    out = vol + noise * rng.standard_normal(vol.shape).astype(np.float32)
    return np.clip(out, 0.0, 1.0)


# ---------------------------------------------------------------------------
# cfg + picking helpers
# ---------------------------------------------------------------------------

def make_cfg(mode):
    return SimpleNamespace(
        pick_mode=mode,
        pick_blur_sigma_frac=0.5,
        pick_nms_frac=1.0,
        pick_cc_thresh=0.5,
        pick_cc_min_frac=0.1,
        pick_cc_conf="max",
        pp_conf_thresh=CONF_THRESH,
        metric_distance_multiplier=0.5,
    )


def pick(vol_np, radius_vox, mode):
    """Run pick_peaks then apply the same conf filter as the eval pipeline."""
    prob = torch.from_numpy(vol_np)
    coords, conf = pick_peaks(prob, radius_vox, make_cfg(mode))
    if coords.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)
    keep = conf > CONF_THRESH
    return coords[keep].cpu().numpy()


def nearest_dist(coords, center):
    if len(coords) == 0:
        return float("inf")
    d = np.sqrt(((coords - np.asarray(center)) ** 2).sum(-1))
    return float(d.min())


# ---------------------------------------------------------------------------
# test cases
# ---------------------------------------------------------------------------

def case_a(noise):
    print(f"\n--- Case A (well-separated), noise={noise} ---")
    shape = (48, 48, 48)
    specs = [("ribosome80s", 6, (12, 12, 12)), ("atp", 4, (34, 34, 34))]
    for name, r, center in specs:
        vol = np.zeros(shape, dtype=np.float32)
        paint_sphere(vol, center, r)
        vol = add_noise(vol, noise)
        radius_vox = float(r)

        n_nms = len(pick(vol, radius_vox, "nms"))
        blur = pick(vol, radius_vox, "blur_nms")
        cc = pick(vol, radius_vox, "cc")
        print(f"  [{name} r={r}] nms={n_nms}  blur_nms={len(blur)}  cc={len(cc)}"
              f"  blur_dist={nearest_dist(blur, center):.2f}  cc_dist={nearest_dist(cc, center):.2f}")

        assert len(blur) == 1, f"{name}: blur_nms expected 1, got {len(blur)}"
        assert len(cc) == 1, f"{name}: cc expected 1, got {len(cc)}"
        assert nearest_dist(blur, center) <= 1.0, f"{name}: blur_nms pick too far"
        assert nearest_dist(cc, center) <= 1.0, f"{name}: cc pick too far"
        assert n_nms > 1, f"{name}: nms expected many picks (bug), got {n_nms}"
    print("  OK: blur_nms/cc -> 1 pick within 1 voxel; nms -> many.")


def case_b(noise):
    print(f"\n--- Case B (two touching same-class spheres), noise={noise} ---")
    shape = (64, 48, 48)
    r = 6
    sep = int(round(1.8 * r))  # ~1.8*r apart
    c1 = (24, 24, 24)
    c2 = (24 + sep, 24, 24)
    vol = np.zeros(shape, dtype=np.float32)
    paint_sphere(vol, c1, r)
    paint_sphere(vol, c2, r)
    vol = add_noise(vol, noise)
    radius_vox = float(r)

    n_nms = len(pick(vol, radius_vox, "nms"))
    n_blur = len(pick(vol, radius_vox, "blur_nms"))
    n_cc = len(pick(vol, radius_vox, "cc"))
    print(f"  centers {sep} voxels apart (1.8*r={1.8*r:.1f}):")
    print(f"  nms={n_nms}  blur_nms={n_blur}  cc={n_cc}")

    assert n_blur == 2, f"blur_nms expected 2, got {n_blur}"
    print("  OK: blur_nms separates the pair into 2 (cc merge is a known limitation).")


def case_c(noise):
    print(f"\n--- Case C (Gaussian-blob targets, main-branch scheme), noise={noise} ---")
    shape = (48, 48, 48)
    specs = [("ribosome80s", 6, (12, 12, 12)), ("atp", 4, (34, 34, 34))]
    for name, r, center in specs:
        vol = gaussian_blob(shape, center, sigma=0.6 * r)
        vol = add_noise(vol, noise)
        radius_vox = float(r)

        n_nms = len(pick(vol, radius_vox, "nms"))
        blur = pick(vol, radius_vox, "blur_nms")
        cc = pick(vol, radius_vox, "cc")
        print(f"  [{name} r={r}] nms={n_nms}  blur_nms={len(blur)}  cc={len(cc)}")

        assert n_nms == 1, f"{name}: nms expected 1 on Gaussian blob, got {n_nms}"
        assert len(blur) == 1, f"{name}: blur_nms expected 1, got {len(blur)}"
        assert len(cc) == 1, f"{name}: cc expected 1, got {len(cc)}"
    print("  OK: all modes -> 1 pick per Gaussian blob (no regression).")


def main():
    torch.manual_seed(0)
    for noise in (0.0, 0.02):
        case_a(noise)
        case_b(noise)
        case_c(noise)
    print("\nALL PEAK-PICKING ASSERTIONS PASSED.")


if __name__ == "__main__":
    main()
