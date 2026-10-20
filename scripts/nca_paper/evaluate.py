
import os
import random
import numpy as np
import torch
import torch.nn.functional as F

from PIL import ImageFile
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import resnet12

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ======================================================
# Configuration
# ======================================================

BASE_PATH = "ds_FSL"
MODEL_PATH = "nca_model_best.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
N_WAY = 5
N_QUERY = 1
NUM_EPISODES = 1000
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ======================================================
# Load datasets
# ======================================================

transform = transforms.Compose([
    transforms.Resize((224, 204)),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(
    os.path.join(BASE_PATH, "train"),
    transform=transform
)

test_dataset = datasets.ImageFolder(
    os.path.join(BASE_PATH, "test"),
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ======================================================
# Load trained model
# ======================================================

model = resnet12().to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(checkpoint)
model.eval()

print(f"Loaded {MODEL_PATH}", flush=True)
print(f"Device: {DEVICE}", flush=True)

# ======================================================
# Extract embeddings
# ======================================================

@torch.no_grad()
def extract_embeddings(loader):
    model.eval()

    embeddings_list = []
    labels_list = []

    for images, labels in loader:
        images = images.to(DEVICE)

        # ResNet12 returns two outputs
        embeddings, _ = model(images)

        embeddings_list.append(embeddings.cpu())
        labels_list.append(labels.cpu())

    return (
        torch.cat(embeddings_list, dim=0),
        torch.cat(labels_list, dim=0)
    )

# ======================================================
# CL2N normalization
# ======================================================

def normalize_embeddings(embeddings, train_mean):
    embeddings = embeddings - train_mean
    return F.normalize(embeddings, p=2, dim=1)

print("Extracting training embeddings...", flush=True)
train_embeddings, _ = extract_embeddings(train_loader)

train_mean = train_embeddings.mean(dim=0, keepdim=True)

print("Extracting test embeddings...", flush=True)
test_embeddings, test_labels = extract_embeddings(test_loader)

test_embeddings = normalize_embeddings(
    test_embeddings,
    train_mean
)

# ======================================================
# Group test images by class
# ======================================================

class_to_indices = {}

for index, label in enumerate(test_labels.tolist()):
    class_to_indices.setdefault(label, []).append(index)

print(f"Test images: {len(test_dataset)}")
print(f"Test classes: {len(class_to_indices)}")

# ======================================================
# Few-shot evaluation
# ======================================================

@torch.no_grad()
def evaluate_few_shot(n_shot, eligible_classes):
    accuracies = []

    for episode in range(NUM_EPISODES):

        selected_classes = random.sample(
            eligible_classes, N_WAY
        )

        prototypes = []
        queries = []
        targets = []

        for local_label, class_id in enumerate(selected_classes):

            selected_indices = random.sample(
                class_to_indices[class_id],
                n_shot + N_QUERY
            )

            support_indices = selected_indices[:n_shot]
            query_indices = selected_indices[n_shot:]

            support_embeddings = test_embeddings[
                support_indices
            ]

            # Mean embedding = class prototype
            prototype = support_embeddings.mean(dim=0)
            prototypes.append(prototype)

            queries.append(
                test_embeddings[query_indices]
            )

            targets.extend(
                [local_label] * N_QUERY
            )

        prototypes = torch.stack(prototypes)
        queries = torch.cat(queries, dim=0)
        targets = torch.tensor(targets)

        # Nearest-prototype classification
        distances = torch.cdist(
            queries, prototypes
        ).pow(2)

        predictions = distances.argmin(dim=1)

        accuracy = (
            predictions == targets
        ).float().mean().item()

        accuracies.append(accuracy)

    mean_accuracy = np.mean(accuracies) * 100
    ci95 = (
        1.96 * np.std(accuracies, ddof=1)
        / np.sqrt(NUM_EPISODES)
    ) * 100

    print(
        f"{N_WAY}-way {n_shot}-shot: "
        f"{mean_accuracy:.2f}% +/- {ci95:.2f}%"
    )

    return mean_accuracy, ci95

# ======================================================
# Run evaluation
# ======================================================

# Use the same class pool for 1-shot and 5-shot,
# so the results are more directly comparable.
eligible_classes = [
    cls for cls, indices in class_to_indices.items()
    if len(indices) >= 6
]

print(f"Eligible classes: {len(eligible_classes)}")

if len(eligible_classes) < N_WAY:
    raise ValueError(
        "Not enough classes with at least 6 test images "
        "for 5-way 5-shot evaluation."
    )

print("\nStarting few-shot evaluation...", flush=True)

evaluate_few_shot(
    n_shot=1,
    eligible_classes=eligible_classes
)

evaluate_few_shot(
    n_shot=5,
    eligible_classes=eligible_classes
)

print("\nEvaluation complete.")
