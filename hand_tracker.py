import math

import cv2
import mediapipe as mp


class HandTracker:

    THUMB_TIP = 4
    INDEX_TIP = 8

    def __init__(self, min_detection_confidence=0.6, min_tracking_confidence=0.6, max_hands=2):

        self.mp_hands = mp.solutions.hands

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame):

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.hands.process(rgb)

        detected = []

        if results.multi_hand_landmarks is None:
            return detected

        for landmarks, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness,
        ):

            width, height = frame.shape[1], frame.shape[0]

            points = [
                (lm.x * width, lm.y * height)
                for lm in landmarks.landmark
            ]

            label = handedness.classification[0].label

            detected.append({
                "type": label,
                "points": points,
            })

        return detected

    def pinch_distance(self, hand):

        a = hand["points"][self.THUMB_TIP]
        b = hand["points"][self.INDEX_TIP]

        return math.hypot(a[0] - b[0], a[1] - b[1])

    def pinch_amount(self, hand, in_min=18.0, in_max=190.0):

        distance = self.pinch_distance(hand)

        return clamp((distance - in_min) / (in_max - in_min), 0.0, 1.0)

    def draw_pinch(self, frame, hand, color=(235, 214, 242)):

        a = hand["points"][self.THUMB_TIP]
        b = hand["points"][self.INDEX_TIP]

        pa = (int(a[0]), int(a[1]))
        pb = (int(b[0]), int(b[1]))

        cv2.line(frame, pa, pb, color, 1, cv2.LINE_AA)
        cv2.circle(frame, pa, 3, color, 1, cv2.LINE_AA)
        cv2.circle(frame, pb, 3, color, 1, cv2.LINE_AA)


def clamp(value, low, high):
    return max(low, min(value, high))