"""Procedural flower garden — expansion of the original single-flower renderer.

The original petal geometry, glow layering, stalk/leaf aesthetic, breathing
style and overall visual quality are preserved verbatim.  This module only
generalises the renderer so many parametrised ``Flower`` instances can share
buffers inside a ``FlowerGarden``.
"""

import math
import random

import cv2
import numpy as np

from utils import (
    clamp,
    lerp,
    map_value,
    smoothstep,
    mix_color,
    rotate_point,
    cubic_bezier,
    draw_layer,
    over_alpha,
)


BREATH_SPEED = 1.35
BREATH_SCALE = 0.028
BREATH_GLOW = 0.10

# Original anchor palette (soft red / rose) — kept exactly as before.
COL_PETAL_RICH = (62, 50, 168)
COL_PETAL_MID = (98, 86, 212)
COL_PETAL_SOFT = (146, 128, 234)
COL_PETAL_TIP = (192, 176, 252)
COL_PETAL_DARK = (44, 38, 134)

COL_CENTER_DEEP = (48, 44, 150)
COL_CENTER_CORE = (152, 130, 244)
COL_CENTER_HOT = (214, 200, 254)

COL_STALK_DARK = (20, 56, 40)
COL_STALK_MID = (38, 106, 68)
COL_STALK_HI = (78, 168, 114)

COL_LEAF_DARK = (20, 62, 42)
COL_LEAF_MID = (40, 116, 74)
COL_LEAF_PALE = (82, 176, 110)

COL_SEPAL = (46, 76, 112)
COL_SEPAL_TIP = (86, 116, 156)

COL_SHADOW = (18, 14, 22)

# Dusty rose satin ribbon (BGR) — complements the bouquet palette.
RIB_BASE = (150, 130, 183)
RIB_LIGHT = (196, 178, 214)
RIB_SHEEN = (224, 210, 232)
RIB_DARK = (112, 92, 138)
RIB_DEEP = (86, 68, 110)
RIB_TAIL = (136, 114, 166)
RIB_GLOW = (150, 128, 172)

GLOW_INNER = (96, 84, 190)
GLOW_TIP = (124, 108, 214)
GLOW_CORE = (152, 130, 244)
GLOW_HALO = (54, 46, 112)


def _dim(color, factor):
    """Subtle depth dimming — scales a BGR color toward black."""
    f = clamp(factor, 0.0, 1.2)
    return tuple(int(max(0, min(255, c * f))) for c in color)


def make_palette(petal_rich, petal_mid, petal_soft, petal_tip, petal_dark):
    """Build a full flower palette from 5 petal tones.

    Centers and glows are derived so every flower keeps the same
    soft internal-light structure as the original, just re-tinted.
    All colors are BGR tuples for OpenCV.
    """
    center_deep = petal_dark
    center_core = mix_color(petal_mid, petal_tip, 0.55)
    center_hot = mix_color(petal_tip, (235, 232, 252), 0.45)
    glow_inner = mix_color(petal_mid, (0, 0, 0), 0.25)
    glow_tip = mix_color(petal_tip, (0, 0, 0), 0.35)
    glow_core = center_core
    glow_halo = mix_color(petal_dark, (0, 0, 0), 0.15)
    return {
        "rich": petal_rich,
        "mid": petal_mid,
        "soft": petal_soft,
        "tip": petal_tip,
        "dark": petal_dark,
        "center_deep": center_deep,
        "center_core": center_core,
        "center_hot": center_hot,
        "glow_inner": glow_inner,
        "glow_tip": glow_tip,
        "glow_core": glow_core,
        "glow_halo": glow_halo,
    }


# Curated pastel / dreamy palette (BGR).  All soft + luminous, no neons.
PALETTES = {
    "soft_red": make_palette(
        COL_PETAL_RICH, COL_PETAL_MID, COL_PETAL_SOFT, COL_PETAL_TIP, COL_PETAL_DARK
    ),
    "rose_pink": make_palette(
        (96, 64, 192), (122, 100, 228), (170, 144, 244), (208, 188, 254), (62, 46, 142)
    ),
    "blush": make_palette(
        (128, 108, 208), (156, 136, 230), (192, 176, 248), (222, 212, 254), (82, 70, 152)
    ),
    "peach": make_palette(
        (96, 118, 218), (126, 146, 234), (178, 188, 248), (216, 222, 255), (64, 82, 162)
    ),
    "coral": make_palette(
        (74, 102, 212), (108, 132, 230), (162, 172, 245), (202, 206, 255), (50, 72, 156)
    ),
    "lavender": make_palette(
        (152, 88, 178), (177, 118, 212), (206, 162, 236), (231, 202, 250), (107, 62, 132)
    ),
    "soft_blue": make_palette(
        (172, 112, 142), (196, 147, 177), (226, 187, 211), (246, 221, 236), (122, 82, 102)
    ),
    "warm_cream": make_palette(
        (142, 152, 196), (172, 182, 221), (206, 211, 240), (236, 236, 251), (97, 107, 147)
    ),
}


def _pt(point):
    return (int(point[0]), int(point[1]))


def _petal_spec(rng):
    return {
        "len": lerp(0.86, 1.14, rng.random()),
        "wid": lerp(0.84, 1.16, rng.random()),
        "tilt": rng.uniform(0.0, math.tau),
        "phase": rng.uniform(0.0, math.tau),
        "speed": lerp(0.85, 1.2, rng.random()),
        "bend": rng.uniform(-1.0, 1.0),
        "dark": rng.uniform(-0.08, 0.08),
        "alpha": lerp(0.82, 0.95, rng.random()),
    }


class Flower:
    """One parametrised flower.  Rendering math mirrors the original."""

    def __init__(self, cfg):
        # cfg fields: x_frac, ground_frac, size, height_mult, palette,
        # sway_amp, sway_speed, sway_phase, sway_amp2, sway_speed2,
        # sway_phase2, breath_speed, breath_phase, breath_scale,
        # rotation, depth_bright, glow_gain, bloom_offset, seed
        self.cfg = dict(cfg)
        self.pal = PALETTES.get(cfg.get("palette", "soft_red"), PALETTES["soft_red"])

        seed = int(cfg.get("seed", 2026))
        rng = random.Random(seed)
        self.outer = [_petal_spec(rng) for _ in range(11)]
        self.inner = [_petal_spec(rng) for _ in range(7)]

    # -- geometry helpers (unchanged) ------------------------------------

    def _petal_silhouette(self, length, width, bend, samples=24):
        left = []
        right = []
        for k in range(samples):
            s = k / (samples - 1)
            y = -length * s
            xoff = bend * width * 0.32 * (s ** 2)
            half = 0.5 * width * (math.sin(math.pi * s) ** 0.8) * (0.55 + 0.45 * s)
            left.append((xoff - half, y))
            right.append((xoff + half, y))
        return left + right[::-1]

    def _world_poly(self, pts_local, cx, cy, angle):
        ca = math.cos(angle)
        sa = math.sin(angle)
        return np.array(
            [
                [int(cx + x * ca - y * sa), int(cy + x * sa + y * ca)]
                for (x, y) in pts_local
            ],
            np.int32,
        )

    # -- petal / flower ----------------------------------------------------

    def _render_petal(self, body, glow, scratch, cx, cy, angle, length, width,
                      bend, base_c, mid_c, tip_c, alpha, glow_tip_col):
        pts = self._petal_silhouette(length, width, bend)
        poly = self._world_poly(pts, cx, cy, angle)

        mx, my = rotate_point(0.0, -length * 0.34, angle)
        mx += cx
        my += cy

        inner = np.array(
            [[int(mx + (px - mx) * 0.58), int(my + (py - my) * 0.58)] for px, py in poly],
            np.int32,
        )

        scratch.fill(0)

        cv2.fillPoly(scratch, [poly], base_c)
        cv2.fillPoly(scratch, [inner], mid_c)

        v0 = rotate_point(0.0, -length * 0.08, angle)
        v1 = rotate_point(0.0, -length * 0.72, angle)

        cv2.line(
            scratch,
            (int(cx + v0[0]), int(cy + v0[1])),
            (int(cx + v1[0]), int(cy + v1[1])),
            mix_color(base_c, (0, 0, 0), 0.28),
            1,
            cv2.LINE_AA,
        )

        over_alpha(
            body,
            scratch,
            alpha,
            (
                int(poly[:, 0].min()),
                int(poly[:, 1].min()),
                int(poly[:, 0].max()) + 1,
                int(poly[:, 1].max()) + 1,
            ),
        )

        cv2.fillPoly(glow, [inner], mix_color(tip_c, (0, 0, 0), 0.45))

        tip = rotate_point(0.0, -length * 0.98, angle)
        ex, ey = int(cx + tip[0]), int(cy + tip[1])

        cv2.ellipse(
            glow,
            (ex, ey),
            (max(2, int(width * 0.24)), max(3, int(length * 0.13))),
            math.degrees(angle),
            0,
            360,
            glow_tip_col,
            cv2.FILLED,
            cv2.LINE_AA,
        )

    def _draw_sepals(self, body, scratch2, cx, cy, openn, breathe, t, SZ, bright):
        sep_alpha = map_value(openn, 0.06, 0.55, 0.9, 0.0)
        if sep_alpha <= 0.01:
            return
        layer = scratch2
        layer.fill(0)

        # Calyx cup: at low bloom the sepals fold upward to cradle the bud
        # instead of splaying outward; at full bloom they rest as before.
        cup = 0.25 + 0.75 * openn
        length = (14 + 4 * (1.0 - openn)) * breathe * SZ
        width = length * 0.5

        xs_min = ys_min = 1 << 30
        xs_max = ys_max = -1

        for j in range(4):
            ang = (1.48 + (j - 1.5) * 0.62) * cup + 0.025 * math.sin(t * 1.2 + j * 2.1)
            pts = self._petal_silhouette(length, width, 0.6, samples=14)
            poly = self._world_poly(pts, cx, cy, ang)
            xs_min = min(xs_min, int(poly[:, 0].min()))
            ys_min = min(ys_min, int(poly[:, 1].min()))
            xs_max = max(xs_max, int(poly[:, 0].max()))
            ys_max = max(ys_max, int(poly[:, 1].max()))
            col = mix_color(COL_SEPAL, COL_SEPAL_TIP, 0.3 + 0.3 * openn)
            col = _dim(col, bright)
            cv2.fillPoly(layer, [poly], col)

        over_alpha(body, layer, sep_alpha, (xs_min, ys_min, xs_max + 1, ys_max + 1))

    def _draw_flower(self, body, glow, scratch, cx, cy, openn, breathe, t, SZ, bright):
        P = self.pal
        rich = _dim(P["rich"], bright)
        mid = _dim(P["mid"], bright)
        soft = _dim(P["soft"], bright)
        tip = _dim(P["tip"], bright)
        dark = _dim(P["dark"], bright)
        c_deep = _dim(P["center_deep"], bright)
        c_core = _dim(P["center_core"], bright)
        c_hot = _dim(P["center_hot"], bright)
        glow_tip_col = _dim(P["glow_tip"], bright)
        glow_core_col = _dim(P["glow_core"], bright)

        # Bud folding: at low bloom all petal directions collapse toward
        # "up" (local -Y) so the same procedural petals form a tight bud
        # instead of an open radial star.  spread=~0.15 (bud) -> 1.0 (open).
        spread = 0.15 + 0.85 * openn
        width_gain = 0.62 + 0.38 * openn

        def _fold(ang):
            dx = math.sin(ang) * spread
            dy = -math.cos(ang) * spread + (-1.0) * (1.0 - spread)
            return math.atan2(dx, -dy)

        rot = float(self.cfg.get("rotation", 0.0))
        petal_len = (11 + 36 * openn) * breathe * SZ
        n_outer = len(self.outer)

        for i, sp in enumerate(self.outer):
            flut = (0.018 + 0.028 * openn) * math.sin(t * sp["speed"] * 2.0 + sp["phase"])
            ang = rot + sp["tilt"] + i * math.tau / n_outer + flut
            ang = _fold(ang)
            lp = petal_len * sp["len"]
            wp = petal_len * 0.5 * sp["wid"] * width_gain
            depth = mix_color(dark, rich, openn)
            base_c = mix_color(depth, mid, 0.5 + 0.5 * openn)
            base_c = mix_color(base_c, rich, sp["dark"])
            mid_c = mix_color(base_c, soft, 0.34 + 0.5 * openn)
            tip_c = mix_color(tip, soft, 0.25)
            self._render_petal(body, glow, scratch, cx, cy, ang, lp, wp,
                               sp["bend"], base_c, mid_c, tip_c, sp["alpha"],
                               glow_tip_col)

        core_r = max(1.5, (3.0 + 4.0 * openn) * SZ)
        cv2.circle(body, _pt((cx, cy)), max(1, int(core_r)), c_deep, cv2.FILLED, cv2.LINE_AA)
        cv2.circle(body, _pt((cx, cy)), max(1, int(core_r * 0.55)), c_core, cv2.FILLED, cv2.LINE_AA)
        cv2.circle(glow, _pt((cx, cy)), max(2, int(core_r * 2.6)), glow_core_col, cv2.FILLED, cv2.LINE_AA)

        n_inner = len(self.inner)
        for i, sp in enumerate(self.inner):
            flut = (0.02 + 0.035 * openn) * math.sin(t * sp["speed"] * 2.3 + sp["phase"] + 1.7)
            ang = rot + 0.4 + sp["tilt"] + i * math.tau / n_inner + flut
            ang = _fold(ang)
            lp = petal_len * 0.8 * sp["len"]
            wp = petal_len * 0.46 * sp["wid"] * width_gain
            base_c = mix_color(rich, soft, 0.3 + 0.5 * openn)
            base_c = mix_color(base_c, mid, sp["dark"])
            mid_c = mix_color(base_c, tip, 0.3)
            tip_c = tip
            self._render_petal(body, glow, scratch, cx, cy, ang, lp, wp,
                               sp["bend"], base_c, mid_c, tip_c, sp["alpha"] * 0.95,
                               glow_tip_col)

        cv2.circle(body, _pt((cx, cy)), max(1, int(core_r * 0.3)), c_hot, cv2.FILLED, cv2.LINE_AA)

    # -- stalk / leaves ------------------------------------------------------

    def _draw_stalk(self, body, curve, t, SZ, bright, wg=1.0):
        n = len(curve)
        if n < 2:
            return
        w0 = 5.5 * SZ * wg
        w1 = 1.5 * SZ * wg
        for i in range(1, n):
            s = i / (n - 1)
            a = curve[i - 1]
            b = curve[i]
            ww = max(1.2, lerp(w0, w1, s))
            col = mix_color(COL_STALK_DARK, COL_STALK_MID, 0.25 + 0.55 * s)
            col = _dim(col, 0.6 + 0.4 * bright)
            cv2.line(body, _pt(a), _pt(b), col, int(ww + 1.0), cv2.LINE_AA)
            cv2.line(
                body, _pt(a), _pt(b),
                mix_color(col, (255, 255, 255), 0.10 + 0.16 * s),
                max(1, int(ww * 0.5)), cv2.LINE_AA,
            )
            cv2.line(
                body,
                (int(a[0] - 1.3 * SZ), int(a[1])),
                (int(b[0] - 1.3 * SZ), int(b[1])),
                _dim(COL_STALK_HI, 0.6 + 0.4 * bright),
                max(1, int(ww * 0.28)), cv2.LINE_AA,
            )

    def _draw_leaf(self, body, scratch, origin, angle, length, side, SZ, bright):
        width = length * 0.44
        pts = self._petal_silhouette(length, width, side * 0.5, samples=16)
        poly = self._world_poly(pts, origin[0], origin[1], angle)
        mx, my = origin[0], origin[1]
        inner = np.array(
            [[int(mx + (px - mx) * 0.6), int(my + (py - my) * 0.6)] for px, py in poly],
            np.int32,
        )
        scratch.fill(0)
        cv2.fillPoly(scratch, [poly], _dim(COL_LEAF_MID, 0.6 + 0.4 * bright))
        cv2.fillPoly(scratch, [inner], _dim(COL_LEAF_PALE, 0.6 + 0.4 * bright))
        v0 = _pt((origin[0], origin[1]))
        v1v = rotate_point(0.0, -length * 0.82, angle)
        v1 = _pt((origin[0] + v1v[0], origin[1] + v1v[1]))
        cv2.line(scratch, v0, v1, COL_LEAF_DARK, 1, cv2.LINE_AA)
        over_alpha(
            body, scratch, 0.96,
            (int(poly[:, 0].min()), int(poly[:, 1].min()),
             int(poly[:, 0].max()) + 1, int(poly[:, 1].max()) + 1),
        )

    def _draw_leaves(self, body, scratch, curve, stalk_eff, t, SZ, bright):
        reveal = float(stalk_eff)
        if reveal < 0.45:
            return
        n = len(curve)
        specs = [(0.38, 1.0, 0.0), (0.62, -1.0, 2.0), (0.84, 1.0, 4.0)]
        for frac, side, phase in specs:
            i = int(frac * (n - 1))
            if i == 0:
                continue
            p = curve[i]
            prev = curve[max(0, i - 1)]
            dx = p[0] - prev[0]
            dy = p[1] - prev[1]
            mag = math.hypot(dx, dy)
            if mag < 1e-3:
                heading = math.radians(90)
            else:
                heading = math.atan2(dy, dx)
            grow = smoothstep(0.45, 0.8, reveal)
            leaf_len = (12 + 18 * grow) * SZ
            sway = 0.06 * math.sin(t * 1.5 + phase) + 0.018 * math.sin(
                t * float(self.cfg.get("breath_speed", BREATH_SPEED)) + phase)
            self._draw_leaf(body, scratch, p, heading + side * 0.95 + sway,
                            leaf_len, side, SZ, bright)

    # -- organic stalk spine with wind -----------------------------------------

    @staticmethod
    def _catmull_rom(p0, p1, p2, p3, u):
        u2 = u * u
        u3 = u2 * u
        return (
            0.5 * (2.0 * p1[0] + (-p0[0] + p2[0]) * u
                   + (2.0 * p0[0] - 5.0 * p1[0] + 4.0 * p2[0] - p3[0]) * u2
                   + (-p0[0] + 3.0 * p1[0] - 3.0 * p2[0] + p3[0]) * u3),
            0.5 * (2.0 * p1[1] + (-p0[1] + p2[1]) * u
                   + (2.0 * p0[1] - 5.0 * p1[1] + 4.0 * p2[1] - p3[1]) * u2
                   + (-p0[1] + 3.0 * p1[1] - 3.0 * p2[1] + p3[1]) * u3),
        )

    def _sample_spine(self, spine, reveal, n=60):
        """Resample the first `reveal` fraction of the spine smoothly."""
        m = len(spine)
        if reveal <= 0.001:
            return [spine[0], spine[0]]
        pts = []
        for i in range(n):
            u = reveal * (i / (n - 1)) * (m - 1)
            seg = min(int(u), m - 2)
            local = u - seg
            p0 = spine[max(0, seg - 1)]
            p1 = spine[seg]
            p2 = spine[seg + 1]
            p3 = spine[min(m - 1, seg + 2)]
            pts.append(self._catmull_rom(p0, p1, p2, p3, local))
        return pts

    def _stalk_curve(self, baseX, baseY, stalk_eff, t, SZ, minH, maxH):
        c = self.cfg
        # Full organic length for this flower (persistent, deterministic).
        full_h = maxH * float(c.get("height_mult", 1.0))
        # Progressive reveal along the curve: 0 at minimum stalk, so the
        # bouquet collapses into a bud cluster with no visible stalks.
        delay = float(c.get("grow_delay", 0.0))
        reveal = smoothstep(0.0, 0.9, max(0.0, stalk_eff - delay))

        # Layered deterministic breeze (same character as before).
        sway = (math.sin(t * float(c.get("sway_speed", 0.6)) + float(c.get("sway_phase", 0.0)))
                * float(c.get("sway_amp", 8.0))
                + math.sin(t * float(c.get("sway_speed2", 0.23)) + float(c.get("sway_phase2", 0.0)))
                * float(c.get("sway_amp2", 3.0)))
        sway *= SZ

        # Organic 5-point spine: leaves the base at an individual angle,
        # bends mid-way (S-curves possible), relaxes toward vertical
        # near the head.  Built by integrating a direction profile so it
        # can never collapse into a straight line.
        a0 = float(c.get("stem_angle", 0.0))
        bend = float(c.get("bend", 0.0))          # mid lateral bow (signed)
        bow2 = float(c.get("bow2", 0.0))          # second bend (S-curves)
        tip_lean = float(c.get("tip_lean", 0.0))  # head drift
        relax = float(c.get("relax", 0.5))        # how vertical the tip gets

        NB = 5
        step = full_h / (NB - 1)
        spine = [(baseX, baseY)]
        # Sway grows along the stalk: base pinned, head moves most.
        bend_w = (0.0, 0.15, 0.40, 0.70, 1.0)
        for k in range(1, NB):
            s = k / (NB - 1)
            ang = (a0 * (1.0 - s) ** 1.6
                   + bend * math.sin(math.pi * s)
                   + bow2 * math.sin(2.0 * math.pi * s)
                   + tip_lean * s * relax)
            dx, dy = math.sin(ang), -math.cos(ang)
            px, py = spine[-1]
            spine.append((px + dx * step + sway * bend_w[k],
                          py + dy * step))

        curve = self._sample_spine(spine, reveal)
        return curve, curve[-1], reveal

    # -- per-flower draw into shared garden buffers ----------------------------

    def draw_into(self, body, glow, shadow, scratch, scratch2,
                  w, h, SZ, minH, maxH, stalk_shared, bloom_shared, t,
                  bouquet_base=None):
        c = self.cfg
        if bouquet_base is not None:
            bx, by = bouquet_base
            # Tiny gather jitter so stems feel held, not mathematically identical.
            jx = float(c.get("base_jitter_x", 0.0)) * SZ
            jy = (float(c.get("base_jitter_y", 0.0)) + float(c.get("stem_tail", 0.0))) * SZ
            baseX, baseY = bx + jx, by + jy
        else:
            baseX = float(c.get("x_frac", 0.5)) * w
            baseY = float(c.get("ground_frac", 0.94)) * h
        size = float(c.get("size", 1.0))
        fSZ = SZ * size
        bright = float(c.get("depth_bright", 1.0))

        stalk_eff = clamp(float(stalk_shared) * float(c.get("height_mult", 1.0)), 0.0, 1.0)
        bloom_eff = clamp(float(bloom_shared) * float(c.get("bloom_gain", 1.0))
                          + float(c.get("bloom_offset", 0.0)), 0.0, 1.0)

        # Ground shadow for this plant.
        cv2.ellipse(
            shadow, _pt((baseX, baseY + 7 * fSZ)),
            (max(2, int((16 + 10 * stalk_eff) * fSZ)), max(2, int((6 + 3 * stalk_eff) * fSZ))),
            0, 0, 360, COL_SHADOW, cv2.FILLED, cv2.LINE_AA,
        )

        curve, tip, reveal = self._stalk_curve(baseX, baseY, stalk_eff, t, fSZ, minH, maxH)

        wg = 0.35 + 0.65 * smoothstep(0.0, 0.35, reveal)
        self._draw_stalk(body, curve, t, fSZ, bright, wg)
        self._draw_leaves(body, scratch, curve, reveal, t, fSZ, bright)

        openn = smoothstep(0.05, 0.6, bloom_eff)
        b_speed = float(c.get("breath_speed", BREATH_SPEED))
        b_phase = float(c.get("breath_phase", 0.0))
        b_scale = float(c.get("breath_scale", BREATH_SCALE))
        breathe = 1.0 + b_scale * math.sin(t * b_speed + b_phase)

        # Bud mound: at zero reveal the heads rest in a gathered cluster
        # just above the ribbon tie; hands off to the curve tip as the
        # stalk grows out.
        mx = baseX + float(c.get("bud_mound_x", 0.0)) * SZ
        my = baseY - float(c.get("bud_mound_h", 20.0)) * SZ
        k = smoothstep(0.0, 0.15, reveal)
        hx = lerp(mx, tip[0], k)
        hy = lerp(my, tip[1], k)

        cx = hx + math.sin(t * 0.9 + b_phase) * 1.2
        cy = hy + math.cos(t * 0.7 + b_phase * 0.5) * 0.8

        cv2.ellipse(
            glow, _pt((cx, cy)),
            (max(2, int(3.5 * fSZ)), max(3, int(7.0 * fSZ))),
            0, 0, 360, _dim(self.pal["glow_core"], bright), cv2.FILLED, cv2.LINE_AA,
        )

        self._draw_sepals(body, scratch2, cx, cy, openn, breathe, t, fSZ, bright)
        self._draw_flower(body, glow, scratch, cx, cy, openn, breathe, t, fSZ, bright)

        return openn, breathe, reveal


def _garden_layout():
    """Fixed-seed bouquet arrangement: ONE shared base, fanned curved stems.

    stem_angle is radians from vertical (negative = leans left).
    All stems start at the bouquet grip; height_mult keeps individual
    proportions under the shared LEFT-hand height control.
    """
    rng = random.Random(77)
    angles_deg = [-14.0, -9.0, -4.0, 0.5, 5.5, 10.0, 15.0]
    specs = [
        # size, h_mult, palette, bright, glow_gain
        (0.80, 0.82, "soft_blue", 0.84, 0.78),
        (0.92, 1.00, "lavender", 0.89, 0.85),
        (0.85, 1.15, "warm_cream", 0.87, 0.80),
        (1.15, 0.90, "soft_red", 1.00, 1.00),
        (0.95, 1.25, "peach", 0.94, 0.90),
        (1.05, 1.05, "rose_pink", 0.97, 0.95),
        (0.88, 0.95, "blush", 0.90, 0.85),
    ]
    flowers = []
    for i, (size, hm, pal, bright, gg) in enumerate(specs):
        bend = rng.uniform(0.20, 0.55) * (1.0 if rng.random() < 0.5 else -1.0)
        tip_lean = rng.uniform(0.08, 0.30) * (1.0 if rng.random() < 0.5 else -1.0)
        flowers.append({
            "stem_angle": math.radians(angles_deg[i] + rng.uniform(-1.5, 1.5)),
            "bend": bend,
            "bow2": rng.uniform(-0.25, 0.25),
            "tip_lean": tip_lean,
            "relax": rng.uniform(0.35, 0.7),
            "grow_delay": rng.uniform(0.0, 0.08),
            "base_jitter_x": rng.uniform(-2.5, 2.5),
            "base_jitter_y": rng.uniform(-2.0, 2.0),
            "stem_tail": rng.uniform(0.0, 8.0),
            "bud_mound_x": rng.uniform(-10.0, 10.0),
            "bud_mound_h": rng.uniform(14.0, 30.0),
            "size": size,
            "height_mult": hm,
            "palette": pal,
            "sway_amp": rng.uniform(6.0, 11.0),
            "sway_speed": rng.uniform(0.45, 0.75),
            "sway_phase": i * 1.13 + rng.uniform(-0.3, 0.3),
            "sway_amp2": rng.uniform(1.5, 3.5),
            "sway_speed2": rng.uniform(0.16, 0.30),
            "sway_phase2": i * 2.07 + rng.uniform(0.0, 1.0),
            "breath_speed": rng.uniform(1.10, 1.60),
            "breath_phase": i * 0.9,
            "breath_scale": BREATH_SCALE * rng.uniform(0.85, 1.15),
            "rotation": rng.uniform(-0.25, 0.25),
            "depth_bright": bright,
            "glow_gain": gg,
            "bloom_gain": rng.uniform(0.92, 1.08),
            "bloom_offset": rng.uniform(-0.03, 0.03),
            "seed": 2026 + i * 131,
        })
    return flowers


class FlowerGarden:
    """Owns shared layers + bouquet environment; each Flower renders itself."""

    # Shared bouquet grip as fractions of frame size (one held point).
    BASE_X_FRAC = 0.5
    BASE_Y_FRAC = 0.965

    def __init__(self, configs=None):
        self.configs = configs if configs is not None else _garden_layout()
        self.flowers = [Flower(c) for c in self.configs]
        self.w = 0
        self.h = 0
        self.SZ = 1.0
        self.minH = 90.0
        self.maxH = 340.0
        self._body = None
        self._glow = None
        self._shadow = None
        self._scratch = None
        self._scratch2 = None

    def _ensure(self, w, h):
        if self._body is not None and self.w == w and self.h == h:
            return
        self.w, self.h = w, h
        self.SZ = min(w, h) / 480.0
        self.minH = h * 0.18
        self.maxH = h * 0.72
        self._body = np.zeros((h, w, 3), np.uint8)
        self._glow = np.zeros((h, w, 3), np.uint8)
        self._shadow = np.zeros((h, w, 3), np.uint8)
        self._scratch = np.zeros((h, w, 3), np.uint8)
        self._scratch2 = np.zeros((h, w, 3), np.uint8)

    def _draw_ground(self):
        # Bouquet: no garden bed.  Just one soft shadow + faint glow at the
        # shared grip so the stems feel held together.
        h, w = self.h, self.w
        bx, by = self.bouquet_base
        cv2.ellipse(
            self._shadow, _pt((bx, by + 8 * self.SZ)),
            (max(3, int(34 * self.SZ)), max(2, int(9 * self.SZ))),
            0, 0, 360, COL_SHADOW, cv2.FILLED, cv2.LINE_AA,
        )
        cv2.ellipse(
            self._glow, _pt((bx, by)),
            (int(46 * self.SZ), int(16 * self.SZ)),
            0, 0, 360, (40, 34, 66), cv2.FILLED, cv2.LINE_AA,
        )

    def _ribbon_tail(self, pts_center, w0, w1):
        """Build a tapered tail polygon with a fishtail notch from a centerline."""
        n = len(pts_center)
        left, right = [], []
        for k, (px, py) in enumerate(pts_center):
            s = k / (n - 1)
            q = pts_center[min(n - 1, k + 1)]
            p = pts_center[max(0, k - 1)]
            dx, dy = q[0] - p[0], q[1] - p[1]
            mag = math.hypot(dx, dy) or 1.0
            nx, ny = -dy / mag, dx / mag
            hw = lerp(w0, w1, s) * 0.5
            left.append((px + nx * hw, py + ny * hw))
            right.append((px - nx * hw, py - ny * hw))
        # Fishtail notch: pull the end middle back along the tangent.
        ex, ey = pts_center[-1]
        tx, ty = ex - pts_center[-2][0], ey - pts_center[-2][1]
        tmag = math.hypot(tx, ty) or 1.0
        notch = (ex - tx / tmag * w1 * 0.9, ey - ty / tmag * w1 * 0.9)
        return (np.array(left + [notch] + right[::-1], np.int32),
                np.array(left, np.int32), np.array(right, np.int32),
                np.array(pts_center, np.int32))

    def _draw_ribbon(self, t, growth=1.0):
        # Hand-tied satin ribbon around the gathered stalks, drawn AFTER
        # stems/leaves/heads so it sits IN FRONT of the lower stalk bundle.
        # Stems pass behind it (~5% of their lower length occluded) and
        # continue visibly below it.
        # `growth` (mean stalk reveal 0..1) keeps the ribbon believable at
        # minimum stalk: it nestles into the bud cluster instead of floating
        # above it.
        bx, by = self.bouquet_base
        SZ = self.SZ
        gg = smoothstep(0.0, 0.6, clamp(growth, 0.0, 1.0))

        # Soft contact shadow (blurred later with the shared shadow layer).
        cv2.ellipse(
            self._shadow, _pt((bx + 3 * SZ, by - lerp(7.0, 30.0, gg) * SZ + 7 * SZ)),
            (max(3, int(28 * SZ)), max(3, int(15 * SZ))),
            -6, 0, 360, COL_SHADOW, cv2.FILLED, cv2.LINE_AA,
        )

        # Ribbon center: pinned near the base (stable grip) with a tiny live
        # wobble; slightly off-center + tilted so nothing looks mechanical.
        wob = (math.sin(t * 0.55) * 1.1 + math.sin(t * 0.23 + 1.0) * 0.7) * SZ
        cx = bx + wob * 0.15 + 1.0 * SZ
        cy = by - lerp(7.0, 30.0, gg) * SZ
        tilt = -0.10 + 0.012 * math.sin(t * 0.4)
        HW, HH = 26.0 * SZ * (0.82 + 0.18 * gg), 10.5 * SZ
        ax_x, ax_y = math.cos(tilt), math.sin(tilt)
        px_x, px_y = -math.sin(tilt), math.cos(tilt)

        N = 24
        top, bot = [], []
        for k in range(N + 1):
            u = k / N * 2.0 - 1.0
            mx = cx + ax_x * u * HW
            my = cy + ax_y * u * HW
            wave_t = (2.0 * math.sin(u * 1.3 + 0.4) - 0.8 * math.sin(u * 2.6)) * SZ
            wave_b = (2.2 * math.sin(u * 1.1 - 0.6) + 0.6) * SZ
            top.append((mx + px_x * (-HH + wave_t * 0.35), my + px_y * (-HH + wave_t * 0.35)))
            bot.append((mx + px_x * (HH + wave_b * 0.35), my + px_y * (HH + wave_b * 0.35)))
        top_a = np.array(top)
        bot_a = np.array(bot)

        def _band(f0, f1):
            a = top_a + (bot_a - top_a) * f0
            b = top_a + (bot_a - top_a) * f1
            return np.array(list(a) + list(b[::-1]), np.int32)

        cv2.fillPoly(self._body, [_band(0.0, 1.0)], RIB_BASE, cv2.LINE_AA)
        cv2.fillPoly(self._body, [_band(0.0, 0.40)], RIB_LIGHT, cv2.LINE_AA)
        cv2.fillPoly(self._body, [_band(0.70, 1.0)], RIB_DARK, cv2.LINE_AA)
        cv2.fillPoly(self._body, [_band(0.16, 0.30)], RIB_SHEEN, cv2.LINE_AA)
        # Shadowed lower edge so the wrap reads as folded fabric.
        cv2.polylines(self._body, [np.array(bot, np.int32)], False,
                      RIB_DEEP, max(1, int(1.8 * SZ)), cv2.LINE_AA)

        # -- tails (behind the knot, in front of the stems) ------------------
        kx, ky = cx + 2.0 * SZ, cy + 1.0 * SZ
        sw_l = (math.sin(t * 0.60 + 0.7) * 2.5 + math.sin(t * 0.27 + 0.2) * 1.2) * SZ
        sw_r = (math.sin(t * 0.58 + 2.4) * 2.0 + math.sin(t * 0.25 + 1.4) * 1.0) * SZ

        def _bezier(p0, p1, p2, p3, m=20):
            return [cubic_bezier(p0, p1, p2, p3, i / (m - 1)) for i in range(m)]

        tails = [
            _bezier((kx - 3 * SZ, ky + 7 * SZ),
                    (kx - 8 * SZ, ky + 20 * SZ),
                    (kx - 12 * SZ + sw_l, ky + 32 * SZ),
                    (kx - 10 * SZ + sw_l * 1.4, ky + 42 * SZ)),
            _bezier((kx + 3 * SZ, ky + 7 * SZ),
                    (kx + 7 * SZ, ky + 18 * SZ),
                    (kx + 10 * SZ + sw_r, ky + 26 * SZ),
                    (kx + 9 * SZ + sw_r * 1.4, ky + 33 * SZ)),
        ]
        widths = [(4.4 * SZ, 3.0 * SZ), (4.0 * SZ, 2.8 * SZ)]
        for cl, (w0, w1) in zip(tails, widths):
            poly, le, ri, cl_a = self._ribbon_tail(cl, w0, w1)
            cv2.fillPoly(self._body, [poly], RIB_TAIL, cv2.LINE_AA)
            cv2.polylines(self._body, [cl_a], False, RIB_DEEP, 1, cv2.LINE_AA)
            cv2.polylines(self._body, [le], False, RIB_LIGHT,
                          max(1, int(1.1 * SZ)), cv2.LINE_AA)

        # -- knot (sits on top of band + tail roots) --------------------------
        kw, kh = 7.5 * SZ, 12.0 * SZ
        kdegs = math.degrees(tilt)
        cv2.ellipse(self._body, _pt((kx, ky)), (max(2, int(kw)), max(3, int(kh))),
                    kdegs, 0, 360, RIB_DARK, cv2.FILLED, cv2.LINE_AA)
        cv2.ellipse(self._body, _pt((kx - SZ, ky - SZ)),
                    (max(1, int(kw * 0.55)), max(2, int(kh * 0.62))),
                    kdegs, 0, 360, RIB_BASE, cv2.FILLED, cv2.LINE_AA)
        # Fold creases where the band enters the knot + a small sheen.
        cv2.line(self._body, _pt((kx - kw * 0.9, ky - kh * 0.35)),
                 _pt((kx - kw * 1.9, ky - kh * 0.75)), RIB_DEEP, 1, cv2.LINE_AA)
        cv2.line(self._body, _pt((kx + kw * 0.9, ky + kh * 0.3)),
                 _pt((kx + kw * 1.9, ky + kh * 0.7)), RIB_DEEP, 1, cv2.LINE_AA)
        cv2.line(self._body, _pt((kx - kw * 0.4, ky - kh * 0.55)),
                 _pt((kx + kw * 0.15, ky - kh * 0.62)), RIB_SHEEN,
                 max(1, int(1.2 * SZ)), cv2.LINE_AA)
        cv2.circle(self._glow, _pt((kx, ky)), max(2, int(8 * SZ)),
                   RIB_GLOW, cv2.FILLED, cv2.LINE_AA)

    @property
    def bouquet_base(self):
        return (self.w * self.BASE_X_FRAC, self.h * self.BASE_Y_FRAC)

    def render(self, frame, stalk, bloom, t):
        h, w = frame.shape[:2]
        self._ensure(w, h)
        self._body.fill(0)
        self._glow.fill(0)
        self._shadow.fill(0)

        self._draw_ground()

        base = self.bouquet_base

        # Back-to-front: flowers list is already ordered by depth.
        per_bloom = []
        reveals = []
        for f in self.flowers:
            openn, breathe, reveal = f.draw_into(
                self._body, self._glow, self._shadow,
                self._scratch, self._scratch2,
                w, h, self.SZ, self.minH, self.maxH, stalk, bloom, t,
                bouquet_base=base,
            )
            per_bloom.append((openn, breathe, f.cfg.get("glow_gain", 1.0)))
            reveals.append(reveal)

        growth = float(sum(reveals) / max(1, len(reveals)))
        self._draw_ribbon(t, growth)

        # One shadow composite (was per-flower blur before).
        shadow = cv2.GaussianBlur(self._shadow, (21, 21), 0)
        over_alpha(frame, shadow, 0.32)

        # Shared glow blurs — 2 blurs total instead of 2 per flower.
        glow_a = cv2.GaussianBlur(self._glow, (13, 13), 0)
        glow_b = cv2.GaussianBlur(self._glow, (33, 33), 0)

        g = 1.0 + BREATH_GLOW * math.sin(t * BREATH_SPEED + 0.8)
        mean_open = float(sum(p[0] for p in per_bloom) / max(1, len(per_bloom)))
        mean_gain = float(sum(p[2] for p in per_bloom) / max(1, len(per_bloom)))
        draw_layer(frame, glow_a, clamp((0.30 + 0.24 * mean_open) * g * mean_gain, 0.0, 0.9))
        draw_layer(frame, glow_b, clamp((0.10 + 0.08 * mean_open) * mean_gain, 0.0, 0.6))

        over_alpha(frame, self._body, 1.0)


# Backwards compatibility: single-flower alias used by older main.
FlowerSingle = Flower
