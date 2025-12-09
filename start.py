import torch
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

CLASSES = [
    "Agaricus",
    "Amanita",
    "Boletus",
    "Cortinarius",
    "Entoloma",
    "Hygrocybe",
    "Lactarius",
    "Russula",
    "Suillus"
]

idx_to_class = {i: c for i, c in enumerate(CLASSES)}

def load_model(model_path="models/mushrooms_resnet50.pt"):
    print("Loading model...")

    model = resnet50(weights=None)

    model.fc = torch.nn.Sequential(
        torch.nn.Linear(2048, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(512, 9)
    )

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)

    model = model.to(device)
    model.eval()
    print("Model loaded.")

    return model

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

import torch.nn.functional as F
from PIL import Image

def predict_image(model, image_path):
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)

        class_idx = probs.argmax(dim=1).item()
        confidence = probs[0, class_idx].item()

    class_name = idx_to_class[class_idx]
    return class_name, confidence


if __name__ == "__main__":
    model = load_model("models/mushrooms_resnet50.pt")

    img_path = r"123.jpg"
    cls, prob = predict_image(model, img_path)

    print(f"Predicted class: {cls}")
    print(f"Confidence: {prob * 100:.2f}%")
