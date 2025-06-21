from models.modelpredictor import ModelPredictor
from utils.image_utils import load_image_by_path
from utils.object_cropper import target_object_cropper
import easygui
from utils.user_input import choose_model, get_queries

def detection_by_text():
    # Initialize system
    
    print("---| STARTING DETECTION BY TEXT |---")
    
    loader = choose_model()

    model, processor = loader.get_components()
    predictor = ModelPredictor(model, processor)
    
    # Configuration
    print("Select an image from the folder")
    IMAGE_PATH = easygui.fileopenbox(title="Select an Image", filetypes=["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.gif"])
    TEXT_QUERIES = get_queries("Enter object names to detect (comma-separated): ")
    
    # Run detection pipeline
    image = load_image_by_path(IMAGE_PATH)
    
    boxes, scores, labels = predictor.text_based_detection(image, TEXT_QUERIES)
    print("---| TEXT BASED DETECTION PIPELINE COMPLETED |---")
    
    if len(boxes)==0:
        print("No detections found.")
        return
    print(f"Detected {len(boxes)} objects.")
    
    print("---|>Detection completed<|---")
    return image, boxes, scores, labels
   
def detection_by_image():
    
    print("---| STARTING DETECTION BY IMAGE |---")
    
    loader = choose_model()
    model, processor = loader.get_components()
    predictor = ModelPredictor(model, processor)
    
    print("Select an Source/reference Image for feature extraction")
    SOURCE_IMAGE = target_object_cropper(easygui.fileopenbox(title="Select an Source/reference Image for feature extraction", filetypes=["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.gif"]))
    
    print("Select an Image in which object need to be detected")
    TARGET_IMAGE = easygui.fileopenbox(title="Select an Image in which object need to be detected", filetypes=["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.gif"])
    
    target_image = load_image_by_path(TARGET_IMAGE)
    source_image = SOURCE_IMAGE
    
    boxes, scores, labels = predictor.image_based_detection(target_image, source_image)
    print("---| IMAGE BASED DETECTION PIPELINE COMPLETED |---")
    
    if len(boxes)==0:
        print("No detections found.")
        return
    print(f"Detected {len(boxes)} objects.")
    
    return target_image, boxes, scores, labels