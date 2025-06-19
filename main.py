from models.model_loader import Owlv2ModelLoader, OwlvitModelLoader
from models.predictor import ModelPredictor
from utils.image_utils import load_image
from utils.visualization import visualize_detections

def main():
    # Initialize system
    # loader = Owlv2ModelLoader()
    loader = OwlvitModelLoader()
    model, processor = loader.get_components()
    predictor = ModelPredictor(model, processor)
    
    # Configuration
    IMAGE_PATH = "sample/coin.jpg"
    TEXT_QUERIES = ["coin"]
    
    # Run detection pipeline
    image = load_image(IMAGE_PATH)
    
    boxes, scores, labels = predictor.run_pipeline(image, TEXT_QUERIES)
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
    visualize_detections(image, boxes, scores, labels)

if __name__ == "__main__":
    main()