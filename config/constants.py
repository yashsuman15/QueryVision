import torch
# Model configuration
MODEL_NAME = "google/owlv2-base-patch16-ensemble"
SCORE_THRESHOLD = 0.5  # Minimum confidence score
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"