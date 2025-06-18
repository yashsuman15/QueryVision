from transformers import Owlv2ForObjectDetection, Owlv2Processor
from config.constants import MODEL_NAME, DEVICE

class ModelLoader:
    def __init__(self):
        """Initialize model and processor with pretrained weights"""
        self.model = Owlv2ForObjectDetection.from_pretrained(MODEL_NAME).to(DEVICE)
        self.processor = Owlv2Processor.from_pretrained(MODEL_NAME)

    def get_components(self):
        """Return the model and processor components"""
        return self.model, self.processor
    