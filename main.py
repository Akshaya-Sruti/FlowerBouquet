import time

import cv2
import numpy as np

from hand_tracker import HandTracker
from flower import FlowerGarden
from utils import SmoothControl


PINCH_MIN = 18.0
PINCH_MAX = 190.0

SMOOTH_TAU = 0.16
HOLD_TIME = 0.6

RESET_DURATION = 1.4

FONT = cv2.FONT_HERSHEY_SIMPLEX


def build_bg_pack(w, h):

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    cx = w * 0.5
    cy = h * 0.44

    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    dmax = np.sqrt(cx * cx + cy * cy)

    v = np.clip((d / dmax) * 1.25, 0.0, 1.0) ** 1.8

    factor = (1.0 - v * 0.46) * (
        1.0 - np.clip((yy / h - 0.58) * 2.4, 0.0, 1.0) * 0.18
    )
    factor = np.clip(factor, 0.30, 1.0)

    tint = np.zeros((h, w, 3), np.float32)
    tint[..., 0] = 20.0
    tint[..., 1] = 14.0
    tint[..., 2] = 30.0
    tint = tint * (1.0 - v[..., None] * 0.25)

    return factor[..., None], tint


def apply_bg(frame, pack):

    factor, tint = pack

    out = frame.astype(np.float32) * factor + tint

    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def draw_text(frame, text, pos, size=0.45, color=(222, 206, 236), dark=(12, 8, 22)):

    cv2.putText(frame, text, pos, FONT, size, dark, 3, cv2.LINE_AA)
    cv2.putText(frame, text, pos, FONT, size, color, 1, cv2.LINE_AA)


def main():

    tracker = HandTracker()
    garden = FlowerGarden()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    stalk_ctrl = SmoothControl(0.0, tau=SMOOTH_TAU, hold_time=HOLD_TIME)
    bloom_ctrl = SmoothControl(0.06, tau=SMOOTH_TAU, hold_time=HOLD_TIME)

    reset_timer = 0.0
    bg_pack = None

    last_time = time.time()

    window_name = "Flower Bouquet"

    while True:

        success, frame = cap.read()

        if not success:
            continue

        frame = cv2.flip(frame, 1)

        now = time.time()
        dt = min(now - last_time, 0.1)
        last_time = now

        hands = tracker.process(frame)

        left_goal = None
        right_goal = None

        for hand in hands:
            pinch = tracker.pinch_amount(hand, PINCH_MIN, PINCH_MAX)
            if hand["type"] == "Right":
                right_goal = pinch
            else:
                left_goal = pinch

        if reset_timer > 0:
            reset_timer = max(0.0, reset_timer - dt)
            stalk_ctrl.set_target(0.0, now)
            bloom_ctrl.set_target(0.06, now)
        else:
            stalk_ctrl.update(left_goal, now, dt)
            bloom_ctrl.update(right_goal, now, dt)

        stalk = stalk_ctrl.value
        bloom = bloom_ctrl.value

        if bg_pack is None or bg_pack[0].shape[0] != frame.shape[0] or bg_pack[0].shape[1] != frame.shape[1]:
            bg_pack = build_bg_pack(frame.shape[1], frame.shape[0])

        frame = apply_bg(frame, bg_pack)

        garden.render(frame, stalk, bloom, now)

        for hand in hands:
            tracker.draw_pinch(frame, hand)

        h, w = frame.shape[:2]

        draw_text(frame, "LEFT  ·  STALK   (open hand = tall)", (14, h - 18))
        draw_text(frame, f"stalk  {stalk:0.2f}", (14, h - 40), 0.4, (168, 156, 198))

        right_label = "RIGHT  ·  BLOOM  (open hand = bloom)"
        size = cv2.getTextSize(right_label, FONT, 0.45, 1)[0]
        draw_text(frame, right_label, (w - 14 - size[0], h - 18))

        ri_label = f"bloom  {bloom:0.2f}"
        size = cv2.getTextSize(ri_label, FONT, 0.4, 1)[0]
        draw_text(frame, ri_label, (w - 14 - size[0], h - 40), 0.4, (168, 156, 198))

        hint = "R reset   ·   Q / Esc quit"
        size = cv2.getTextSize(hint, FONT, 0.4, 1)[0]
        draw_text(frame, hint, ((w - size[0]) // 2, h - 40), 0.4, (150, 138, 180))

        if reset_timer > 0:
            rtext = "resetting…"
            size = cv2.getTextSize(rtext, FONT, 0.42, 1)[0]
            draw_text(frame, rtext, ((w - size[0]) // 2, 30), 0.42, (210, 196, 232))

        gtext = "✦ BOUQUET"
        size = cv2.getTextSize(gtext, FONT, 0.5, 1)[0]
        draw_text(frame, gtext, ((w - size[0]) // 2, 24), 0.5, (222, 206, 236))

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            break

        if key == ord("r"):
            reset_timer = RESET_DURATION

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()