import torch
# Model configuration

SCORE_THRESHOLD = 0.3  # Minimum confidence score
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



# MODEL NAME
OWL_V2 = "google/owlv2-base-patch16-ensemble"
OWL_VIT = "google/owlvit-base-patch32"


