from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

# Globals to store the rectangle's corners
crop_coords = {"x1": None, "y1": None, "x2": None, "y2": None}

def line_select_callback(eclick, erelease):
    """Callback for rectangle selection."""
    x1, y1 = int(eclick.xdata), int(eclick.ydata)
    x2, y2 = int(erelease.xdata), int(erelease.ydata)
    crop_coords.update({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
    plt.close()  # Close figure to continue

def toggle_selector(event):
    """Toggle selector with 'q' key."""
    if event.key in ['Q', 'q'] and rect_selector.active:
        rect_selector.set_active(False)

def crop_object(image_path):
    # Load image (BGR to RGB)
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Setup figure
    fig, ax = plt.subplots()
    ax.imshow(rgb)
    ax.set_title("Draw a bounding box around the target object")

    global rect_selector
    # Create RectangleSelector with minimal arguments
    rect_selector = RectangleSelector(
        ax, 
        line_select_callback,
        useblit=True,
        button=[1],  # Left mouse button
        minspanx=5, 
        minspany=5,
        spancoords='pixels',
        interactive=True
    )
    
    # Set rectangle style - Correct method for Matplotlib 3.10.3
    rect_selector.artists[0].set_edgecolor('red')
    rect_selector.artists[0].set_linewidth(2)
    rect_selector.artists[0].set_fill(False)

    plt.connect('key_press_event', toggle_selector)  # Keypress handler
    plt.show()  # Display & wait for user

    # Extract coordinates
    x1, y1 = crop_coords["x1"], crop_coords["y1"]
    x2, y2 = crop_coords["x2"], crop_coords["y2"]
    if None in [x1, y1, x2, y2]:
        raise RuntimeError("No region selected!")

    # Crop and convert to RGB PIL Image
    x0, x1 = sorted([x1, x2])
    y0, y1 = sorted([y1, y2])
    roi = bgr[y0:y1, x0:x1]  # BGR region
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    return Image.fromarray(roi_rgb)

# def main():
#     path = input("Enter path to source image: ").strip()
#     try:
#         cropped_bgr = crop_object(path)
#     except Exception as e:
#         print(f"Error: {e}")
#         return

#     # Show the cropped ROI in a Matplotlib window
#     crop_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
#     plt.figure()
#     plt.imshow(crop_rgb)
#     plt.axis('off')
#     plt.title("Cropped Object")
#     plt.show()

#     # 'cropped_bgr' is your NumPy array ROI, ready for your detection pipeline

# ========================================================================================
def load_image(image_path):
    """Load image in RGB format from a file path"""
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image