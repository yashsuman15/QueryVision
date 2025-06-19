import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from PIL import Image

class InteractiveCropper:
    """
    use in cropping image of object out of the provided image
    
    """
    def __init__(self, image):
        # Load image
        self.image = Image.open(image) if isinstance(image, str) else image
        self.coords = None

        # Display image
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.ax.imshow(self.image)
        self.ax.set_title("Draw a bounding box around the refernce object, then close this window.")
        self.ax.axis('off')

        # Initialize RectangleSelector
        self.selector = RectangleSelector(
            self.ax,
            self._on_select,
            useblit=True,
            button=[1],  # left click only
            minspanx=5,
            minspany=5,
            spancoords='pixels',
            interactive=True,
            props=dict(edgecolor='red', linestyle='--', linewidth=2, fill=False)
        )
        plt.connect('key_press_event', self._on_key)
        plt.show()

    def _on_select(self, eclick, erelease):
        # Get coordinates and round to nearest integers
        x1, y1 = round(eclick.xdata), round(eclick.ydata)
        x2, y2 = round(erelease.xdata), round(erelease.ydata)
        
        # Ensure coordinates are within image bounds
        self.coords = (
            max(0, min(x1, x2)),
            max(0, min(y1, y2)),
            min(self.image.width, max(x1, x2)),
            min(self.image.height, max(y1, y2)),
        )

    def _on_key(self, event):
        # Close window when Enter is pressed
        if event.key == 'enter':
            plt.close(self.fig)

    def crop_image(self):
        """Return cropped PIL.Image or raise if none selected."""
        if not self.coords:
            raise RuntimeError('No crop area selected')
        return self.image.crop(self.coords)


def target_object_cropper(image):
    """
    Open image and let user select ROI; return cropped PIL.Image.
    """
    cropper = InteractiveCropper(image)
    
    target_object = cropper.crop_image()
    print(type(target_object))
    return target_object


# if __name__ == "__main__":
#     cropped = target_object_cropper('sample/box.jpg')
#     cropped.show()
    