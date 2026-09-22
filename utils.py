import cv2
import math

import numpy as np


def clamp(value, low, high):
    return max(low, min(value, high))


def lerp(a, b, t):
    return a + (b - a) * t


def map_value(value, in_lo, in_hi, out_lo, out_hi):

    t = 0.0 if in_hi <= in_lo else (clamp(value, in_lo, in_hi) - in_lo) / (in_hi - in_lo)

    return lerp(out_lo, out_hi, t)


def smoothstep(edge0, edge1, value):

    t = clamp((value - edge0) / (edge1 - edge0), 0.0, 1.0)

    return t * t * (3.0 - 2.0 * t)


def exp_smooth(current, target, tau, dt):

    if tau <= 0 or dt <= 0:
        return target

    alpha = 1.0 - math.exp(-dt / tau)

    return lerp(current, target, alpha)


def mix_color(color_a, color_b, t):

    t = clamp(t, 0.0, 1.0)

    return tuple(
        int(color_a[i] + (color_b[i] - color_a[i]) * t)
        for i in range(3)
    )


def rotate_point(x, y, angle):

    c = math.cos(angle)
    s = math.sin(angle)

    return (x * c - y * s, x * s + y * c)


def bezier2(p0, p1, p2, t):

    u = 1.0 - t

    return (
        u * u * p0[0] + 2.0 * u * t * p1[0] + t * t * p2[0],
        u * u * p0[1] + 2.0 * u * t * p1[1] + t * t * p2[1],
    )


def cubic_bezier(p0, p1, p2, p3, t):

    u = 1.0 - t

    return (
        u ** 3 * p0[0] + 3.0 * u * u * t * p1[0] + 3.0 * u * t * t * p2[0] + t ** 3 * p3[0],
        u ** 3 * p0[1] + 3.0 * u * u * t * p1[1] + 3.0 * u * t * t * p2[1] + t ** 3 * p3[1],
    )


def draw_layer(dst, layer, alpha):

    alpha = clamp(alpha, 0.0, 1.0)

    cv2.addWeighted(dst, 1.0, layer, alpha, 0, dst)


def over_alpha(dst, layer, alpha, bbox=None):

    alpha = clamp(alpha, 0.0, 1.0)
    alpha = float(alpha)

    if alpha <= 0.0:
        return

    if bbox is None:
        ys, xs = np.nonzero(layer[..., 0])
        if ys.size == 0:
            return
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)

    x0, y0, x1, y1 = bbox

    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(dst.shape[1], x1)
    y1 = min(dst.shape[0], y1)

    if x1 <= x0 or y1 <= y0:
        return

    sr = layer[y0:y1, x0:x1]
    br = dst[y0:y1, x0:x1]

    idx = np.nonzero(np.any(sr, axis=2))

    if idx[0].size == 0:
        return

    if alpha >= 1.0:
        br[idx] = sr[idx]
    else:
        br[idx] = (
            br[idx].astype(np.float32) * (1.0 - alpha)
            + sr[idx].astype(np.float32) * alpha
        ).astype(np.uint8)


class SmoothControl:

    def __init__(self, initial=0.0, tau=0.16, hold_time=0.6, decay_tau=0.45):

        self.value = float(initial)
        self.target = float(initial)
        self.tau = tau
        self.hold_time = hold_time
        self.decay_tau = decay_tau
        self.last_seen = None
        self.frozen = False

    def update(self, goal, now, dt):

        if goal is not None:
            self.target = float(clamp(goal, 0.0, 1.0))
            self.last_seen = now
            self.frozen = False
        elif self.last_seen is not None and not self.frozen:
            if (now - self.last_seen) > self.hold_time:
                self.target = exp_smooth(self.target, 0.0, self.decay_tau, dt)

        self.value = exp_smooth(self.value, self.target, self.tau, dt)

        return self.value

    def set_target(self, value, now):

        self.target = float(clamp(value, 0.0, 1.0))
        self.last_seen = now
        self.frozen = True