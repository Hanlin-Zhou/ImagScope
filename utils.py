import cv2
import numpy as np
from PIL import Image

def center_align_images(img1, img2):
    """
    Resize both images to the same height and center-align them horizontally within a common canvas.
    This ensures that the center of both images are aligned when displayed together.
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    target_height = max(h1, h2)
    
    def resize_and_pad(img, target_height):
        h, w = img.shape[:2]
        scale = target_height / h
        new_w = int(w * scale)
        resized = cv2.resize(img, (new_w, target_height))
        
        max_width = max(w1 * target_height // h1, w2 * target_height // h2)
        total_pad = max_width - new_w
        pad_left = total_pad // 2
        pad_right = total_pad - pad_left
        
        padded = cv2.copyMakeBorder(resized, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return padded

    aligned1 = resize_and_pad(img1, target_height)
    aligned2 = resize_and_pad(img2, target_height)
    
    return aligned1, aligned2

# Example usage with dummy images for demonstration
img1 = np.ones((300, 200, 3), dtype=np.uint8) * 255  # White image
img2 = np.ones((150, 400, 3), dtype=np.uint8) * 100  # Gray image

aligned1, aligned2 = center_align_images(img1, img2)

# Convert to PIL for display preview
Image.fromarray(aligned1).show()
Image.fromarray(aligned2).show()
