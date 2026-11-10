"""Generate the landing marker and check that OpenCV can read it."""
from pathlib import Path
import cv2

# Use this same dictionary and ID in the webcam script.
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
marker = cv2.aruco.generateImageMarker(dictionary, 0, 600)
# White space outside the black marker border makes it easier to detect.
image = cv2.copyMakeBorder(marker, 100, 100, 100, 100,
                           cv2.BORDER_CONSTANT, value=255)
folder = Path(__file__).resolve().parent / "markers"
folder.mkdir(exist_ok=True)
path = folder / "landing_marker_0.png"
if not cv2.imwrite(str(path), image):
    raise RuntimeError("Could not save the marker image.")
detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
corners, ids, rejected = detector.detectMarkers(image)
if ids is None or 0 not in ids.flatten():
    raise RuntimeError("Generated marker did not pass the detection check.")
print("Marker ID 0 generated and detection check passed.")
print(f"Image saved to: {path}")
