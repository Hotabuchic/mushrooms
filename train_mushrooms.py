import os
import copy
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms, models, datasets
from PIL import ImageFile, Image

ImageFile.LOAD_TRUNCATED_IMAGES = True

from tqdm import tqdm


DATA_DIR = "D:/Mushrooms"
MODEL_DIR = "models"
MODEL_NAME = "mushrooms_resnet18.pt"
NUM_CLASSES = 9
BATCH_SIZE = 32
NUM_EPOCHS = 15
LR = 3e-4
WEIGHT_DECAY = 1e-4
IMAGE_SIZE = 224
TRAIN_VAL_SPLIT = 0.8
SEED = 42
NUM_WORKERS = 4

CLASS_NAMES = [
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


def ensure(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_dataloaders_single_folder(data_dir: str):
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    transform_val = transforms.Compose([
        transforms.Resize(int(IMAGE_SIZE * 1.14)),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = datasets.ImageFolder(data_dir)

    class_to_idx = full_dataset.class_to_idx

    total_len = len(full_dataset)
    train_len = int(total_len * TRAIN_VAL_SPLIT)
    val_len = total_len - train_len

    torch.manual_seed(SEED)
    train_subset, val_subset = random_split(full_dataset, [train_len, val_len])

    train_subset.dataset.transform = transform_train
    val_subset.dataset.transform = transform_val

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=NUM_WORKERS)

    dataloaders = {"train": train_loader, "val": val_loader}
    dataset_sizes = {"train": train_len, "val": val_len}

    return dataloaders, dataset_sizes, class_to_idx


def build_model(num_classes=NUM_CLASSES):
    model = models.resnet18(pretrained=True)
    in_f = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_f, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )
    return model


def train_model(model, dataloaders, dataset_sizes, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    best_acc = 0
    best_wts = copy.deepcopy(model.state_dict())

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        print("-" * 40)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0
            running_corrects = 0

            for x, y in tqdm(dataloaders[phase], desc=phase):
                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    out = model(x)
                    loss = criterion(out, y)
                    _, preds = torch.max(out, 1)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * x.size(0)
                running_corrects += torch.sum(preds == y).item()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print(f"{phase} loss: {epoch_loss:.4f}  acc: {epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_wts = copy.deepcopy(model.state_dict())
                torch.save(best_wts, os.path.join(MODEL_DIR, MODEL_NAME))
                print(f"✔ Saved best model (acc={best_acc:.4f})")

        scheduler.step()

    print("\nTraining finished. Best val acc = ", best_acc)
    model.load_state_dict(best_wts)
    return model


def load_model(model_path):
    device = get_device()
    model = build_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict_image(model, image_path, class_to_idx):
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    transform = transforms.Compose([
        transforms.Resize(int(IMAGE_SIZE * 1.14)),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(get_device())

    with torch.no_grad():
        out = model(x)
        p = torch.softmax(out, 1)
        prob, idx = torch.max(p, 1)

    return idx_to_class[idx.item()], float(prob.item())


def main():
    ensure(MODEL_DIR)
    device = get_device()
    print("Device:", device)

    dataloaders, dataset_sizes, class_to_idx = make_dataloaders_single_folder(DATA_DIR)
    print("Dataset sizes:", dataset_sizes)

    model = build_model()
    model.to(device)

    model = train_model(model, dataloaders, dataset_sizes, device)

    with open(os.path.join(MODEL_DIR, "class_mapping.json"), "w") as f:
        json.dump(class_to_idx, f, indent=2)
    print("Saved class mapping.")


if __name__ == "__main__":
    main()
