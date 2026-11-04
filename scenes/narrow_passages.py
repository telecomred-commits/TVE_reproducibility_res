"""Synthetic maps with one narrow passage of controlled width (experiment E9, revision 1).

Three geometries, each with a narrow passage whose width in the configuration space
(after the dilation by the robot radius) is exactly ``gap_px`` pixels, and a wide
opening far from it, so that a path always exists and the narrow passage is the
short way:

* ``door``     - a wall joined to the frame (a single obstacle) with a short door;
* ``slit``     - a gap between two separate blocks (two different obstacles);
* ``corridor`` - a 3-m-long straight corridor through a thick block joined to the frame.

Random clutter (discs and boxes) is added on both sides. The maps are used directly
as detected maps (no camera), since the question is geometric.
"""
from __future__ import annotations

import cv2
import numpy as np

SIDE_M = 12.0
SIZE = 400


def _cspace_gap(gt: np.ndarray, r: int, row: int, x0: int, x1: int) -> int:
    from common.cspace import configuration_space
    occ = configuration_space(gt, r)
    return int((~occ[row, x0:x1]).sum())


def narrow_map(kind: str, gap_px: int, seed: int, robot_radius_px: int):
    """Returns (ground-truth mask, passage box (x0, y0, x1, y1), door centre x)."""
    rng = np.random.default_rng(10_000 * {"door": 1, "slit": 2, "corridor": 3}[kind] + 100 * gap_px + seed)
    mpp = SIDE_M / SIZE
    xd = int(rng.integers(170, 231))                      # centre of the passage
    half_t = int(round(0.15 / mpp))                         # wall half thickness (0.30 m)
    if kind == "corridor":
        half_t = int(round(1.5 / mpp))                      # 3-m-long corridor
    yw = SIZE // 2

    def build(w):
        g = np.zeros((SIZE, SIZE), np.uint8)
        a, b = xd - w // 2, xd - w // 2 + w                 # door [a, b)
        if kind in ("door", "corridor"):
            g[yw - half_t:yw + half_t + 1, :a] = 1
            g[yw - half_t:yw + half_t + 1, b:] = 1
            wide = int(round(1.6 / mpp))                    # wide opening near the left border
            g[yw - half_t:yw + half_t + 1, 20:20 + wide] = 0
        else:                                               # slit between two separate blocks
            m = int(round(1.6 / mpp))
            g[yw - half_t:yw + half_t + 1, m:a] = 1
            g[yw - half_t:yw + half_t + 1, b:SIZE - m] = 1
        return g, a, b

    # door width in the image such that the free gap of the configuration space is gap_px
    w = gap_px + 2 * robot_radius_px
    for _ in range(12):
        g, a, b = build(w)
        meas = _cspace_gap(g.astype(bool), robot_radius_px, yw, a - 3 * robot_radius_px, b + 3 * robot_radius_px)
        if meas == gap_px:
            break
        w += 1 if meas < gap_px else -1
    # clutter on both sides, away from the passage
    for _ in range(int(rng.integers(6, 11))):
        for _try in range(50):
            cx = int(rng.integers(30, SIZE - 30))
            cy = int(rng.choice([rng.integers(30, yw - half_t - 40), rng.integers(yw + half_t + 40, SIZE - 30)]))
            if abs(cx - xd) < 70 and abs(cy - yw) < half_t + 90:
                continue
            if rng.random() < 0.5:
                rad = int(rng.integers(int(0.2 / mpp), int(0.5 / mpp)))
                cv2.circle(g, (cx, cy), rad, 1, -1)
            else:
                hw, hh = int(rng.integers(8, 25)), int(rng.integers(8, 25))
                g[cy - hh:cy + hh, cx - hw:cx + hw] = 1
            break
    g[[0, -1], :] = 1
    g[:, [0, -1]] = 1
    box = (a - 2, yw - half_t - 2 * robot_radius_px, b + 2, yw + half_t + 2 * robot_radius_px)
    return g.astype(bool), box, xd
