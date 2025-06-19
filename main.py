from models.model_loader import Owlv2ModelLoader, OwlvitModelLoader
from models.modelpredictor import ModelPredictor
from utils.image_utils import load_image_by_path
from utils.object_cropper import target_object_cropper
from utils.visualization import plot, annotate_image, display_result
import easygui

def choose_model():
    while True:
        try:
            user_input = int(input("""Choose a model:\n1. OWLv2\n2. OWLvit\nEnter 1 or 2: """))

            match user_input:
                case 1:
                    return Owlv2ModelLoader()
                case 2:
                    return OwlvitModelLoader()
                case _:
                    print("Invalid choice. Please select 1 or 2.")
        except ValueError:
            print("Invalid input. Please enter a valid integer.")
            
def chose_task():
    while True:
        try:
            user_input = int(input("""Choose a task:\n1. Text-based detection\n2. Image-based detection\nEnter 1 or 2: """))

            match user_input:
                case 1:
                    return detection_by_text()
                case 2:
                    return detection_by_image()
                case _:
                    print("Invalid choice. Please select 1 or 2.")
        except ValueError:
            print("Invalid input. Please enter a valid integer.")


def detection_by_text(IMAGE_PATH = None, TEXT_QUERIES = None):
    # Initialize system
    loader = choose_model()

    model, processor = loader.get_components()
    predictor = ModelPredictor(model, processor)
    
    # Configuration
    IMAGE_PATH = easygui.fileopenbox(title="Select an Image", filetypes=["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.gif"])
    TEXT_QUERIES = ["coin"]
    
    # Run detection pipeline
    image = load_image_by_path(IMAGE_PATH)
    
    boxes, scores, labels = predictor.text_based_detection(image, TEXT_QUERIES)
    print("pipeline run completed!")
    
    # Visualize results
    print("Visualizing results...")
    if len(boxes)==0:
        print("No detections found.")
        return
    print(f"Detected {len(boxes)} objects.")
    print(f"Boxes: {boxes.tolist()}")
    print(f"Scores: {scores.tolist()}")
    print(f"Labels: {labels}")
    # Visualize detections
    print("Visualizing detections...")
    image = annotate_image(image, boxes, scores, labels, show_labels=True)
    display_result(image, save=True)
    print("Detection completed") 
   
def detection_by_image(TARGET_IMAGE = None, SOURCE_IMAGE = None):
    # loader = OwlvitModelLoader()
    loader = choose_model()
    model, processor = loader.get_components()
    predictor = ModelPredictor(model, processor)
    
    SOURCE_IMAGE = target_object_cropper(easygui.fileopenbox(title="Select an Source/reference Image for feature extraction", filetypes=["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.gif"]))
    TARGET_IMAGE = easygui.fileopenbox(title="Select an Image in which object need to be detected", filetypes=["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.gif"])
    
    target_image = load_image_by_path(TARGET_IMAGE)
    source_image = SOURCE_IMAGE
    
    boxes, scores, labels = predictor.image_based_detection(target_image, source_image)
    print("pipeline run completed!")
    
    # Visualize results
    print("Visualizing results...")
    if len(boxes)==0:
        print("No detections found.")
        return
    print(f"Detected {len(boxes)} objects.")
    print(f"Boxes: {boxes.tolist()}")
    print(f"Scores: {scores.tolist()}")
    print(f"Labels: {labels}")
    
    # Visualize detections
    print("Visualizing detections...")
    plot(target_image, boxes, scores, labels, show_labels=True)

def main():
    chose_task()

if __name__ == "__main__":
    main()