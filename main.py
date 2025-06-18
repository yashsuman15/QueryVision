from models.model_loader import ModelLoader
from models.predictor import Owlv2Predictor
from utils.image_utils import load_image
from utils.visualization import visualize_detections

def main():
    # Initialize system
    loader = ModelLoader()
    model, processor = loader.get_components()
    predictor = Owlv2Predictor(model, processor)
    
    # Configuration
    IMAGE_PATH = "sample/data1.jpg"
    TEXT_QUERIES = ["bike", "person"]
    
    # Run detection pipeline
    image = load_image(IMAGE_PATH)
    boxes, scores, labels = predictor.run_pipeline(image, TEXT_QUERIES)
    
    # Visualize results
    visualize_detections(image, boxes, scores, labels)

if __name__ == "__main__":
    main()