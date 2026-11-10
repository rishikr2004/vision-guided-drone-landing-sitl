"""Create 10 by 7 squares: OpenCV detects 9 by 6 INNER corners."""
from pathlib import Path
import cv2
import numpy as np

SQUARE_PIXELS = 120
MARGIN = 120
image = np.full((7 * SQUARE_PIXELS + 2 * MARGIN,
                 10 * SQUARE_PIXELS + 2 * MARGIN), 255, dtype=np.uint8)
for row in range(7):
    for col in range(10):
        if (row + col) % 2 == 0:
            y = MARGIN + row * SQUARE_PIXELS
            x = MARGIN + col * SQUARE_PIXELS
            image[y:y + SQUARE_PIXELS, x:x + SQUARE_PIXELS] = 0
folder = Path(__file__).resolve().parent / "calibration"
folder.mkdir(exist_ok=True)
path = folder / "checkerboard_10x7.png"
if not cv2.imwrite(str(path), image):
    raise RuntimeError("Could not save checkerboard.")
found, corners = cv2.findChessboardCornersSB(image, (9, 6))
if not found:
    raise RuntimeError("Generated checkerboard failed its detection check.")
print("Checkerboard created: 10 x 7 squares, 9 x 6 inner corners.")
print(path)
