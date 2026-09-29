"""
Peak-picking strategies for per-class probability maps.

This module exposes a single entry point, :func:`pick_peaks`, that converts one
class's probability volume into a set of particle picks. Three strategies are
selectable via ``cfg.pick_mode``:

- ``"nms"`` (default): the legacy ``simple_nms`` behaviour (max-pool
  non-maximum suppression on the raw probabilities), kept for backward
  compatibility.
- ``"blur_nms"``: Gaussian-blur the map, then run max-pool NMS with plateau
  tie-breaking. Robust to the flat ~1.0 plateaus that sphere-trained models
  produce.
- ``"cc"``: threshold + 3D connected components; each component becomes one
  pick at its probability-weighted center of mass.

Why this module exists
----------------------
Sphere targets (see ``robpicker/data/ds.py:_get_sphere_stencil``) paint a solid
binary core (value 1.0) into each class channel. A well-trained model therefore
predicts a *flat plateau* of probabilities near 1.0 across the whole particle
core. ``simple_nms`` keeps every voxel that equals the local max over its
window, so on a flat plateau **every core voxel is a "maximum"** and is returned
as a separate pick. On a single simulated ribosome (150 Å core, 7.84 Å/voxel)
this produced 4,022 picks with no noise and ~100 picks with 1% noise, versus a
single pick for the old Gaussian-blob targets. ``blur_nms`` and ``cc`` collapse
each plateau back to one pick.

All strategies finish with the same greedy same-class dedup so that no two picks
of a class are closer than ``cfg.metric_distance_multiplier * particle_radius``.
The code is device-agnostic; only the ``cc`` mode briefly moves data to the CPU
(scipy) and returns tensors on the input's device.
"""

import math

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _maxpool_nms_radius(vol4d: torch.Tensor, radius: int) -> torch.Tensor:
    """Max-pool over a cube of side ``2*radius+1`` (stride 1, same size)."""
    return F.max_pool3d(vol4d, kernel_size=radius * 2 + 1, stride=1, padding=radius)


def _gaussian_blur_3d(vol: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable 3D Gaussian blur of a ``[D, H, W]`` volume (edge-replicated)."""
    if sigma <= 0:
        return vol
    radius = max(1, int(round(3.0 * sigma)))
    coords = torch.arange(-radius, radius + 1, device=vol.device, dtype=vol.dtype)
    k1d = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    k1d = k1d / k1d.sum()

    x = vol[None, None]  # [1, 1, D, H, W]
    for dim in range(3):
        shape = [1, 1, 1, 1, 1]
        shape[2 + dim] = k1d.numel()
        kernel = k1d.view(shape)
        pad = [0, 0, 0, 0, 0, 0]  # F.pad order: (Wl, Wr, Hl, Hr, Dl, Dr)
        pad[(2 - dim) * 2] = radius
        pad[(2 - dim) * 2 + 1] = radius
        x = F.pad(x, pad, mode="replicate")
        x = F.conv3d(x, kernel)
    return x[0, 0]


def _greedy_dedup(coords: torch.Tensor, conf: torch.Tensor, radius: float):
    """Greedily keep the highest-confidence pick and drop any within ``radius``.

    ``coords`` is ``[N, 3]`` (float, voxel units), ``conf`` is ``[N]``. Distance
    is Euclidean in voxels. Returns the surviving ``(coords, conf)``.
    """
    if coords.shape[0] <= 1 or radius <= 0:
        return coords, conf

    order = torch.argsort(conf, descending=True)
    coords_s = coords[order].float()
    conf_s = conf[order]

    taken = torch.zeros(coords_s.shape[0], dtype=torch.bool, device=coords.device)
    keep = []
    r2 = float(radius) ** 2
    for i in range(coords_s.shape[0]):
        if taken[i]:
            continue
        keep.append(i)
        d2 = ((coords_s[i:i + 1] - coords_s) ** 2).sum(-1)
        taken |= d2 <= r2
    keep_idx = torch.tensor(keep, device=coords.device, dtype=torch.long)
    return coords_s[keep_idx], conf_s[keep_idx]


# ---------------------------------------------------------------------------
# per-mode pickers
# ---------------------------------------------------------------------------

def _pick_nms(prob: torch.Tensor, radius_vox: float, cfg):
    """Legacy ``simple_nms``: keep voxels equal to the local max (score > 0)."""
    nms_radius = int(0.5 * radius_vox)
    p = prob[None]  # [1, D, H, W]
    if nms_radius > 0:
        pooled = _maxpool_nms_radius(p, nms_radius)
    else:
        pooled = p
    y = torch.where(p == pooled, p, torch.zeros_like(p))
    kps = torch.where(y > 0)
    coords = torch.stack(kps[1:], dim=-1).float()
    conf = y[kps]
    return coords, conf


def _pick_blur_nms(prob: torch.Tensor, radius_vox: float, cfg):
    """Gaussian blur + max-pool NMS with plateau tie-breaking."""
    sigma = float(getattr(cfg, "pick_blur_sigma_frac", 0.5)) * radius_vox
    nms_radius = max(1, int(round(float(getattr(cfg, "pick_nms_frac", 1.0)) * radius_vox)))
    conf_thresh = float(getattr(cfg, "pp_conf_thresh", 0.01))

    blurred = _gaussian_blur_3d(prob, sigma)

    # Break plateau ties: add a tiny, strictly monotonic ramp so that equal
    # blurred values inside one pooling window resolve to a single winner
    # instead of the whole flat region qualifying as a maximum.
    n = blurred.numel()
    ramp = torch.arange(n, device=blurred.device, dtype=blurred.dtype).reshape(blurred.shape)
    tb = blurred + 1e-6 * (ramp / n)

    pooled = _maxpool_nms_radius(tb[None], nms_radius)[0]
    peak_mask = (tb == pooled) & (blurred > conf_thresh)

    coords = peak_mask.nonzero(as_tuple=False).float()
    if coords.shape[0] == 0:
        return coords, torch.zeros(0, device=prob.device, dtype=prob.dtype)
    idx = peak_mask.nonzero(as_tuple=True)
    conf = prob[idx]  # confidence = UNBLURRED probability at the peak
    return coords, conf


def _pick_cc(prob: torch.Tensor, radius_vox: float, cfg):
    """Threshold + 3D connected components (26-connectivity)."""
    import numpy as np
    import scipy.ndimage as ndi

    thresh = float(getattr(cfg, "pick_cc_thresh", 0.5))
    min_frac = float(getattr(cfg, "pick_cc_min_frac", 0.1))
    conf_mode = str(getattr(cfg, "pick_cc_conf", "max"))

    arr = prob.detach().cpu().numpy()
    mask = arr >= thresh

    structure = ndi.generate_binary_structure(3, 3)  # 26-connectivity
    labeled, n = ndi.label(mask, structure=structure)

    expected_vol = (4.0 / 3.0) * math.pi * (radius_vox ** 3)
    min_vol = min_frac * expected_vol

    coords_list = []
    conf_list = []
    if n > 0:
        # Component volumes in one pass.
        comp_sizes = np.bincount(labeled.ravel())
        for lbl in range(1, n + 1):
            if comp_sizes[lbl] < min_vol:
                continue
            com = ndi.center_of_mass(arr, labeled, lbl)  # probability-weighted
            comp_vals = arr[labeled == lbl]
            if conf_mode == "mean":
                c = float(comp_vals.mean())
            else:
                c = float(comp_vals.max())
            coords_list.append(com)
            conf_list.append(c)

    if not coords_list:
        empty = torch.zeros((0, 3), device=prob.device, dtype=torch.float32)
        return empty, torch.zeros(0, device=prob.device, dtype=torch.float32)

    coords = torch.tensor(np.asarray(coords_list), device=prob.device, dtype=torch.float32)
    conf = torch.tensor(np.asarray(conf_list), device=prob.device, dtype=torch.float32)
    return coords, conf


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

_PICKERS = {
    "nms": _pick_nms,
    "blur_nms": _pick_blur_nms,
    "cc": _pick_cc,
}


def pick_peaks(prob: torch.Tensor, radius_vox: float, cfg):
    """Extract particle picks from one class's probability map.

    Args:
        prob: ``[D, H, W]`` probability map for a single class (already
            downsampled x2 by the caller).
        radius_vox: particle radius in (downsampled) voxels, i.e.
            ``particle_radi / effective_spacing``.
        cfg: config; ``cfg.pick_mode`` selects the strategy
            (``"nms"`` | ``"blur_nms"`` | ``"cc"``).

    Returns:
        ``(coords_vox, conf)`` where ``coords_vox`` is an ``[N, 3]`` float
        tensor of voxel coordinates (same axis order as ``prob``) and ``conf``
        is an ``[N]`` tensor of confidences, both on ``prob.device``.
    """
    mode = str(getattr(cfg, "pick_mode", "nms"))
    picker = _PICKERS.get(mode)
    if picker is None:
        raise ValueError(
            f"Unknown cfg.pick_mode={mode!r}; expected one of {sorted(_PICKERS)}."
        )

    coords, conf = picker(prob, radius_vox, cfg)

    # Universal final same-class greedy dedup: merge picks closer than
    # metric_distance_multiplier * particle_radius (in voxels), keep highest conf.
    dedup_radius = float(getattr(cfg, "metric_distance_multiplier", 0.5)) * radius_vox
    coords, conf = _greedy_dedup(coords, conf, dedup_radius)
    return coords, conf
