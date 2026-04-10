import os
from PIL import Image
import numpy as np

# Safe import for deployment
try:
    import torch
    import torch.nn as nn
    from torchvision import transforms
    import timm
except ImportError:
    torch = None


class PneumoniaPredictor:
    def __init__(self, model_path: str, threshold: float = 0.5):
        self.model_path = model_path
        self.threshold = threshold

        if torch is None:
            raise Exception("❌ PyTorch is not available in this environment")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        # Create model (ViT base)
        model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=1)

        # Load weights
        state_dict = torch.load(self.model_path, map_location=self.device)

        # Handle different save formats
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

        model.load_state_dict(state_dict)

        return model

    def predict(self, image: Image.Image):
        if torch is None:
            raise Exception("❌ PyTorch not available")

        # Convert to RGB (important for X-ray grayscale images)
        image = image.convert("RGB")

        # Transform
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(img_tensor)
            prob = torch.sigmoid(output).item()

        prediction = "Yes" if prob >= self.threshold else "No"

        return {
            "pneumonia": prediction,
            "probability": prob
        }