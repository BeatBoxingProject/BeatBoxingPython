import cv2
import numpy as np
from core.config import VIS_RANGES

def _draw_section(img, start_offset, label, coords):
    """
    Helper function to draw the bars for a single hand at a specific x-offset.
    """
    x, y, z = coords
    
    # Layout settings
    bar_w, spacing = 80, 40
    max_bar_h, base_y = 200, 250
    section_start_x = start_offset + 40  # Margin within the section

    # Draw Hand Label
    cv2.putText(img, label, (start_offset + 20, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

    # Define bars properties
    bars = [
        {'label': 'X', 'val': x, 'min': VIS_RANGES['x_min'], 'max': VIS_RANGES['x_max'], 'color': (0, 0, 255)},
        {'label': 'Y', 'val': y, 'min': VIS_RANGES['y_min'], 'max': VIS_RANGES['y_max'], 'color': (0, 255, 0)},
        {'label': 'Z', 'val': z, 'min': VIS_RANGES['z_min'], 'max': VIS_RANGES['z_max'], 'color': (255, 0, 0)}
    ]

    for i, b in enumerate(bars):
        # Normalize value 0.0 to 1.0
        denom = b['max'] - b['min']
        if denom == 0: denom = 1
        
        norm_val = (b['val'] - b['min']) / denom
        norm_val = np.clip(norm_val, 0.0, 1.0)
        bar_h = int(norm_val * max_bar_h)

        # Calculate coordinates
        bx = section_start_x + i * (bar_w + spacing)
        by_top = base_y - bar_h

        # 1. Draw Background (Empty Slot)
        cv2.rectangle(img, (bx, base_y - max_bar_h), (bx + bar_w, base_y), (50, 50, 50), -1)
        
        # 2. Draw Filled Value
        cv2.rectangle(img, (bx, by_top), (bx + bar_w, base_y), b['color'], -1)
        
        # 3. Draw Border
        cv2.rectangle(img, (bx, base_y - max_bar_h), (bx + bar_w, base_y), (255, 255, 255), 2)
        
        # 4. Draw Axis Label (Bottom)
        cv2.putText(img, b['label'], (bx + 30, base_y + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # 5. Draw Value Label (Top)
        # Position slightly above the bar
        text_y = base_y - max_bar_h - 10
        cv2.putText(img, f"{b['val']:.2f}", (bx + 10, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def draw_visualizer(pos_left, pos_right):
    """
    Main entry point. Creates an image comparing both hands side-by-side.
    pos_left/pos_right are tuples: (x, y, z)
    """
    # Double the width (400 -> 800) to fit two hands
    width, height = 800, 300
    vis_img = np.zeros((height, width, 3), dtype=np.uint8)

    # Draw Left Hand (Pixels 0-400)
    _draw_section(vis_img, 0, "LEFT HAND", pos_left)

    # Draw Right Hand (Pixels 400-800)
    _draw_section(vis_img, 400, "RIGHT HAND", pos_right)

    # Draw a vertical separator line
    cv2.line(vis_img, (400, 20), (400, 280), (100, 100, 100), 2)

    return vis_img