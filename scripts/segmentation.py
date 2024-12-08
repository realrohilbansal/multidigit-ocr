import cv2
import numpy as np

def segment_digits(binary_image):
    """Segment individual digits from binary image"""
    # Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours left-to-right
    sorted_contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])
    
    digits = []
    for contour in sorted_contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Filter out noise by size
        if w * h > 100:  # Minimum area threshold
            digit = binary_image[y:y+h, x:x+w]
            # Resize to MNIST size (28x28)
            digit = cv2.resize(digit, (28, 28))
            digits.append(digit)
    
    return digits