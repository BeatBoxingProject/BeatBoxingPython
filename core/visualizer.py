import cv2
import numpy as np
from core.config import VIS_RANGES

def draw_visualizer(x, y, z):
    width, height = 400, 300
    vis_img = np.zeros((height, width, 3), dtype=np.uint8)

    bar_w, spacing, start_x = 80, 40, 40
    max_bar_h, base_y = 200, 250

    bars = [
        {'label': 'X', 'val': x, 'min': VIS_RANGES['x_min'], 'max': VIS_RANGES['x_max'], 'color': (0, 0, 255)},
        {'label': 'Y', 'val': y, 'min': VIS_RANGES['y_min'], 'max': VIS_RANGES['y_max'], 'color': (0, 255, 0)},
        {'label': 'Z', 'val': z, 'min': VIS_RANGES['z_min'], 'max': VIS_RANGES['z_max'], 'color': (255, 0, 0)}
    ]

    for i, b in enumerate(bars):
        denom = b['max'] - b['min']
        if denom == 0: denom = 1
        
        norm_val = (b['val'] - b['min']) / denom
        norm_val = np.clip(norm_val, 0.0, 1.0)
        bar_h = int(norm_val * max_bar_h)

        bx = start_x + i * (bar_w + spacing)
        by_top = base_y - bar_h

        # Draw UI Elements
        cv2.rectangle(vis_img, (bx, base_y - max_bar_h), (bx + bar_w, base_y), (50, 50, 50), -1)
        cv2.rectangle(vis_img, (bx, by_top), (bx + bar_w, base_y), b['color'], -1)
        cv2.rectangle(vis_img, (bx, base_y - max_bar_h), (bx + bar_w, base_y), (255, 255, 255), 2)
        cv2.putText(vis_img, b['label'], (bx + 30, base_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(vis_img, f"{b['val']:.2f}", (bx + 10, base_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return vis_img