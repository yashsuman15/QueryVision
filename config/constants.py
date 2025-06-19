import torch

# MODEL NAME
OWL_V2 = "google/owlv2-base-patch16-ensemble"
OWL_VIT = "google/owlvit-base-patch32"

# Model configuration
TEXT_BASED_SCORE_THRESHOLD = 0.3  # Minimum confidence score
IMAGE_BASED_SCORE_THRESHOLD = 0.5  # Minimum confidence score
NMS_THRESHOLD = 0.5  # Non-Maximum Suppression threshold
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



