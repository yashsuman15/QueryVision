import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_detections(image, boxes, scores, labels, save=True):
    """
    Plot detections on image
    
    Args:
        image: PIL Image
        boxes: Detected bounding boxes
        scores: Confidence scores
        labels: Text labels
    """
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)
    
    for box, score, label in zip(boxes, scores, labels):
        xmin, ymin, xmax, ymax = box.tolist()
        
        # Create bounding box rectangle
        rect = patches.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            linewidth=2,
            edgecolor='r',
            facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add label text
        ax.text(
            xmin, ymin - 10,
            f"{label}: {score:.2f}",
            color='white',
            fontsize=12,
            bbox=dict(facecolor='red', alpha=0.5)
        )
    
    plt.axis('off')
    if save:
        plt.savefig("detection_results.png", bbox_inches='tight')
    plt.show()