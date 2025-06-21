import matplotlib.pyplot as plt
from PIL import ImageDraw, ImageFont
import numpy as np
from datetime import datetime
import os

def plot_detection(image, boxes, scores, labels=None, show_labels=True):
    """
    Annotate image with bounding boxes and optional labels using PIL.

    Args:
        image: PIL Image
        boxes: Detected bounding boxes (list of [xmin, ymin, xmax, ymax])
        scores: Confidence scores
        labels: Text labels or None
        show_labels: Whether to show text annotations

    Returns:
        Annotated PIL Image
    """
    
    print("---| PLOTING DETECTION STARTED |---")
    
    # Ensure image is in RGB mode
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.copy()
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for idx, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = map(int, box.tolist())

        # Random color
        color = tuple(np.random.randint(0, 256, size=3).tolist())

        # Draw bounding box
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=2)

        # Draw text if requested
        if show_labels:
            score = scores[idx]
            label = labels[idx] if labels is not None else None
            # Compose text: label and score if label exists, else score only
            if label:
                text = f"{label}: {score:.2f}"
            else:
                text = f"{score:.2f}"

            # Measure text size
            bbox = font.getbbox(text)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # Center of the bounding box
            x_center = xmin + (xmax - xmin) // 2
            y_center = ymin + (ymax - ymin) // 2

            # Top-left corner for text box
            text_x = x_center - text_width // 2
            text_y = y_center - text_height // 2

            # Draw background rectangle for text
            draw.rectangle(
                [text_x, text_y, text_x + text_width, text_y + text_height],
                fill=color
                )
            # Draw the text
            draw.text((text_x, text_y), text, fill='white', font=font)

    return image


def display_result(image, save=True, save_dir="results"):
    """
    Display and optionally save annotated PIL image with timestamped filename.

    Args:
        image: PIL Image
        save: Whether to save the image
        save_dir: Directory to save the image
    Returns:
        str: Filename of the saved image (or None if not saved)
    """
    print("---| DISPLAYING DETECTION RESULT |---")

    filename = None
    if save:
        # Create results directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        now = datetime.now()
        
        # Format: detection-HHMM-DD-MM.jpg
        filename = now.strftime("detection-%H%M-%d-%m.jpg")
        save_path = os.path.join(save_dir, filename)
        image.save(save_path)
        print(f"Image saved to {save_path}")

    # Display image
    plt.imshow(image)
    plt.axis('off')
    plt.show()