import cv2
import numpy as np
from core.config import VIS_RANGES


def _draw_section(img, offset_x, label, coords):
    """ Draw bars for one hand. Handles coords=None for lost tracking. """

    # Header
    cv2.putText(img, label, (offset_x + 20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

    # State: LOST TRACKING
    if coords is None:
        cv2.putText(img, "SEARCHING...", (offset_x + 40, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50, 50, 200), 2)
        return

    # State: TRACKING
    x, y, z = coords
    bars = [
        {'L': 'X', 'val': x, 'min': VIS_RANGES['x_min'], 'max': VIS_RANGES['x_max'], 'c': (0, 0, 255)},
        {'L': 'Y', 'val': y, 'min': VIS_RANGES['y_min'], 'max': VIS_RANGES['y_max'], 'c': (0, 255, 0)},
        {'L': 'Z', 'val': z, 'min': VIS_RANGES['z_min'], 'max': VIS_RANGES['z_max'], 'c': (255, 0, 0)}
    ]

    base_y, max_h = 250, 200
    start_x = offset_x + 40

    for i, b in enumerate(bars):
        norm = (b['val'] - b['min']) / (b['max'] - b['min'] or 1)
        bar_h = int(np.clip(norm, 0, 1) * max_h)
        bx = start_x + i * 120  # Spacing

        # Draw Bar
        cv2.rectangle(img, (bx, base_y - max_h), (bx + 80, base_y), (50, 50, 50), -1)  # BG
        cv2.rectangle(img, (bx, base_y - bar_h), (bx + 80, base_y), b['c'], -1)  # Fill
        cv2.rectangle(img, (bx, base_y - max_h), (bx + 80, base_y), (255, 255, 255), 2)  # Border

        # Text
        cv2.putText(img, b['L'], (bx + 30, base_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, f"{b['val']:.2f}", (bx + 10, base_y - max_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)


def draw_visualizer(pos_left, pos_right):
    vis_img = np.zeros((300, 800, 3), dtype=np.uint8)
    _draw_section(vis_img, 0, "LEFT HAND", pos_left)
    _draw_section(vis_img, 400, "RIGHT HAND", pos_right)
    cv2.line(vis_img, (400, 20), (400, 280), (100, 100, 100), 2)
    return vis_img