import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def plot(image, boxes, scores, labels, show_labels=True, save=True):
    """
    Plot detections on image
    
    Args:
        image: PIL Image
        boxes: Detected bounding boxes
        scores: Confidence scores
        labels: Text labels
    """

    width, height = image.size

    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    ax.imshow(image)
    ax.set_axis_off()

    for box, score in zip(boxes, scores):
        xmin, ymin, xmax, ymax = box.tolist()
        color = (np.array(np.random.randint(0, 100, size=3))/100).tolist()

        x_center = xmin + (xmax - xmin) / 2
        y_center = ymin + (ymax - ymin) / 2

        # Bounding box
        rect = patches.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            linewidth=2,
            edgecolor=color,
            facecolor='none'
        )
        ax.add_patch(rect)
        
        # Label
        if show_labels:
            ax.text(
                x_center, y_center,
                f"{score:.2f}",
                color='white',
                fontsize=12,
                ha='center',
                va='center',
                bbox=dict(facecolor=color, edgecolor=color, boxstyle='round', alpha=0.5)
            )

    plt.subplots_adjust(0, 0, 1, 1)  # Remove padding/margin
    plt.axis('off')
    ax.set_xlim([0, width])
    ax.set_ylim([height, 0])  # Invert y-axis for PIL image format

    if save:
        plt.savefig("detection_by_image_results.jpg", bbox_inches='tight', pad_inches=0)
        print("Image saved to detection_by_image_results.jpg")
    plt.show()
    
# --------------------------------------------------------------------------------------

def annotate_image(image, boxes, scores, labels, show_labels=False):
    """
    Annotate image with bounding boxes and labels using PIL.

    Args:
        image: PIL Image
        boxes: Detected bounding boxes (list of [xmin, ymin, xmax, ymax])
        scores: Confidence scores
        labels: Text labels
        show_labels: Whether to show labels and scores

    Returns:
        Annotated PIL Image
    """
    # Ensure image is in RGB mode
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image = image.copy()
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for box, score, label in zip(boxes, scores, labels):
        xmin, ymin, xmax, ymax = map(int, box.tolist())

        # Random color
        color = tuple(np.random.randint(0, 256, size=3).tolist())

        # Draw rectangle
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=2)

        # Draw label if required
        if show_labels:
            text = f"{label}: {score:.2f}"

            # Get text size
            bbox = font.getbbox(text)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # Center of the box
            x_center = xmin + (xmax - xmin) // 2
            y_center = ymin + (ymax - ymin) // 2

            # Top-left of text box so that text is centered
            text_x = x_center - text_width // 2
            text_y = y_center - text_height // 2

            # Background for text
            draw.rectangle(
                [text_x, text_y, text_x + text_width, text_y + text_height],
                fill=color
            )

            # Draw the text
            draw.text((text_x, text_y), text, fill='white', font=font)

    return image


def display_result(image, save=True, save_path="detection_by_text_results.jpg"):
    """
    Display and optionally save annotated PIL image.

    Args:
        image: PIL Image
        save: Whether to save the image
        save_path: File path to save
    """
    if save:
        image.save(save_path)
        print(f"Image saved to {save_path}")
    
    plt.axis('off')
    plt.imshow(image)