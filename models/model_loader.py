from transformers import Owlv2ForObjectDetection, Owlv2Processor
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from config.constants import OWL_V2, OWL_VIT, DEVICE

class Owlv2ModelLoader:
    """Class to load OWLv2 model and processor"""
    def __init__(self):
        """Initialize model and processor with pretrained weights"""
        self.model = Owlv2ForObjectDetection.from_pretrained(OWL_V2).to(DEVICE)
        self.processor = Owlv2Processor.from_pretrained(OWL_V2)
        print("+++++++++++| MODEL:", OWL_V2, "loaded successfully! |+++++++++++")

    def get_components(self):
        """Return the model and processor components"""
        return self.model, self.processor
    
class OwlvitModelLoader:
    """Class to load OWLvit model and processor"""
    def __init__(self):
        """Initialize model and processor with pretrained weights"""
        self.model = OwlViTForObjectDetection.from_pretrained(OWL_VIT).to(DEVICE)
        self.processor = OwlViTProcessor.from_pretrained(OWL_VIT)
        print("+++++++++++| MODEL:", OWL_VIT, "loaded successfully! |+++++++++++")

    def get_components(self):
        """Return the model and processor components"""
        return self.model, self.processor
    