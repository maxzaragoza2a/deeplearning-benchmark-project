import os
import torch

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import resnet12
from loss import FewShotNCALoss

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


# ======================================================
# Configuration
# ======================================================

BATCH_SIZE = 128
EPOCHS = 120

LEARNING_RATE = 0.1
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9

TEMPERATURE = 1.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ======================================================
# Dataset
# ======================================================

BASE_PATH = "ds_FSL"

TRAIN_PATH = os.path.join(BASE_PATH, "train")
VALID_PATH = os.path.join(BASE_PATH, "valid")


transform = transforms.Compose([
    transforms.Resize((224, 204)),
    transforms.ToTensor()
])


train_dataset = datasets.ImageFolder(
    root=TRAIN_PATH,
    transform=transform
)

valid_dataset = datasets.ImageFolder(
    root=VALID_PATH,
    transform=transform
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=False
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=False
)


print(f"Device: {DEVICE}")
print(f"Training images: {len(train_dataset)}")
print(f"Validation images: {len(valid_dataset)}")
print(f"Training classes: {len(train_dataset.classes)}")


# ======================================================
# Model
# ======================================================

# Start with a new randomly initialized ResNet12.
# Do NOT load nca_model_best.pt here.
model = resnet12().to(DEVICE)


# ======================================================
# NCA Loss
# ======================================================

criterion = FewShotNCALoss(
    classes=len(train_dataset.classes),
    temperature=TEMPERATURE,
    batch_size=BATCH_SIZE,
    frac_negative_samples=1,
    frac_positive_samples=1
)


# ======================================================
# Optimizer
# ======================================================

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=LEARNING_RATE,
    momentum=MOMENTUM,
    weight_decay=WEIGHT_DECAY,
    nesterov=True
)


# ======================================================
# Learning-rate scheduler
# ======================================================

# Reduce the learning rate by 10x after 70% of training.
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[int(EPOCHS * 0.7)],
    gamma=0.1
)


# ======================================================
# Validation
# ======================================================

@torch.no_grad()
def validate(model, valid_loader, criterion, device):

    model.eval()

    running_loss = 0.0

    for images, labels in valid_loader:

        images = images.to(device)
        labels = labels.to(device)

        # resnet12 returns:
        # embeddings, fc_output
        embeddings, _ = model(images)

        loss = criterion(
            embeddings,
            labels
        )

        running_loss += loss.item()

    average_loss = running_loss / len(valid_loader)

    return average_loss


# ======================================================
# Training
# ======================================================

best_val_loss = float("inf")


for epoch in range(EPOCHS):

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    model.train()

    running_loss = 0.0


    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)


        # Forward pass
        #
        # Your ResNet12 returns:
        # (embedding, fc_output)
        #
        # We only need the embedding for NCA.
        embeddings, _ = model(images)


        # NCA loss
        loss = criterion(
            embeddings,
            labels
        )


        # Backpropagation
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


        running_loss += loss.item()


    train_loss = running_loss / len(train_loader)


    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    val_loss = validate(
        model,
        valid_loader,
        criterion,
        DEVICE
    )


    # --------------------------------------------------
    # Save best model
    # --------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            "nca_model_best.pt"
        )

        saved = " <-- best model saved"

    else:

        saved = ""


    # --------------------------------------------------
    # Learning-rate scheduler
    # --------------------------------------------------

    scheduler.step()

    current_lr = optimizer.param_groups[0]["lr"]


    # --------------------------------------------------
    # Print epoch results
    # --------------------------------------------------

    print(
        f"Epoch {epoch + 1:03d}/{EPOCHS} | "
        f"Train loss: {train_loss:.4f} | "
        f"Val loss: {val_loss:.4f} | "
        f"LR: {current_lr:.6f}"
        f"{saved}"
    )


# ======================================================
# Save final model
# ======================================================

torch.save(
    model.state_dict(),
    "nca_model_last.pt"
)


print()
print("Training complete.")
print(f"Best validation loss: {best_val_loss:.4f}")
print("Best model: nca_model_best.pt")
print("Last model: nca_model_last.pt")