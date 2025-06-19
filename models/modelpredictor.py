import torch
from config.constants import TEXT_BASED_SCORE_THRESHOLD, IMAGE_BASED_SCORE_THRESHOLD,NMS_THRESHOLD, DEVICE

class ModelPredictor:
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
        
        print("model and processor initialized successfully.")
        
    
    def text_based_detection(self, image, texts):
        """
        Run text-based detection pipeline
        
        Args:
            image: PIL Image or image path
            texts: List of text queries (e.g., ["a car", "traffic light"])
        
        Returns:
            boxes: Detected bounding boxes
            scores: Confidence scores
            labels: Text labels
        """
        
        inputs = self.processor(
            images=image, 
            text=texts, 
            return_tensors="pt"
        ).to(DEVICE)
        print("preprocess completed!")

        orig_size =  image.size[::-1]  # (height, width)

        with torch.no_grad():
            outputs = self.model(**inputs)
            
        print("predict completed!")

        target_sizes = torch.tensor([orig_size])
        print("SCORE_THRESHOLD:", TEXT_BASED_SCORE_THRESHOLD)
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs, 
            threshold=TEXT_BASED_SCORE_THRESHOLD,
            target_sizes=target_sizes
        )[0]
        
        # Extract predictions
        boxes = results["boxes"]
        scores = results["scores"]
        labels = [texts[i] for i in results["labels"]]
        
        print("postprocess completed!")
        return boxes, scores, labels
    
    def image_based_detection(self, target_image, source_image):
        """
        detects objets in target_image based on source_image
        
        Args:
            target_image: PIL Image or image path for target
            source_image: PIL Image or image path for source
        
        Returns:
            boxes: Detected bounding boxes
            scores: Confidence scores
            labels: Text labels
        """
        inputs = self.processor(
            images=target_image,
            query_images=source_image,
            return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            outputs = self.model.image_guided_detection(**inputs)

        target_sizes = torch.tensor([target_image.size[::-1]]).to(DEVICE)
        results = self.processor.post_process_image_guided_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=IMAGE_BASED_SCORE_THRESHOLD,
            nms_threshold=NMS_THRESHOLD
        )[0]
        
        # Extract predictions
        boxes = results["boxes"]
        scores = results["scores"]
        labels = results["labels"]
        
        return boxes, scores, labels
        