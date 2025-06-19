from PIL import Image

def load_image_by_path(image_path):
    """Load image in RGB format from a file path"""
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    # print(type(image))
    return image

