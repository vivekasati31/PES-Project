import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm


MODEL_NAME = "vit_base_patch16_224"
IMAGE_SIZE = (224, 224)


class PneumoniaPredictor:
    def __init__(self, model_path: str, threshold: float = 0.5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.transform = transforms.Compose(
            [
                transforms.Resize(IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.model = self._load_model(model_path)

    def _load_model(self, model_path: str) -> nn.Module:
        model = timm.create_model(MODEL_NAME, pretrained=False)
        model.head = nn.Linear(model.head.in_features, 1)

        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)

        model = model.to(self.device)
        model.eval()
        return model

    def predict(self, image: Image.Image) -> dict:
        image = image.convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            prob = torch.sigmoid(logits).item()

        label = "Yes" if prob >= self.threshold else "No"
        return {
            "pneumonia": label,
            "probability": prob,
            "threshold": self.threshold,
        }
