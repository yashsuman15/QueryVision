import torch
from config.constants import SCORE_THRESHOLD, DEVICE

class Owlv2Predictor:
    def __init__(self, model, processor):
        """
        Initialize predictor with loaded model components
        
        Args:
            model: Preloaded Owlv2 model
            processor: Owlv2 processor for input handling
        """
        self.model = model
        self.processor = processor
        self.model.eval()
        print("OWLv2 model and processor initialized successfully.")
        
    def preprocess(self, image, texts):
        """
        Prepare inputs for the model
        
        Args:
            image: PIL Image or image path
            texts: List of text queries (e.g., ["a car", "traffic light"])
        
        Returns:
            Processed inputs and original image size
        """
        
        inputs = self.processor(
            images=image, 
            text=texts, 
            return_tensors="pt"
        ).to(DEVICE)
        print("preprocess completed!")
        return inputs, image.size[::-1]  # (height, width)
    
    def predict(self, inputs):
        """Run model inference on preprocessed inputs"""
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        print("predict completed!")
        return outputs
    
    def postprocess(self, outputs, orig_size, texts):
        """
        Convert raw outputs to detection results
        
        Args:
            outputs: Raw model outputs
            orig_size: Original (height, width) of image
            texts: Text queries used for detection
        
        Returns:
            boxes: Final bounding boxes [xmin, ymin, xmax, ymax]
            scores: Confidence scores
            labels: Text labels
        """
        # Convert outputs to COCO format
        target_sizes = torch.tensor([orig_size])
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs, 
            threshold=SCORE_THRESHOLD,
            target_sizes=target_sizes
        )[0]
        
        # Extract predictions
        boxes = results["boxes"]
        scores = results["scores"]
        labels = [texts[i] for i in results["labels"]]
        
        print("postprocess completed!")
        return boxes, scores, labels
    
    def run_pipeline(self, image, texts):
        """Complete end-to-end detection pipeline"""
        print("Running end-to-end detection pipeline...")
        
        inputs, orig_size = self.preprocess(image, texts)
        outputs = self.predict(inputs)
        
        return self.postprocess(outputs, orig_size, texts)
        