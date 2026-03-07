import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random
import numpy as np

def set_seed(seed=42):

    '''Set random seeds for reproducibility.'''

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_loaders(train_dir, valid_dir, test_dir, batch_size=32, seed=42):

    '''
    Create DataLoaders for training, validation, and test datasets.

    Args:
        train_dir (str): Directory containing training images.
        valid_dir (str): Directory containing validation images.
        test_dir (str): Directory containing test images.
        batch_size (int): Batch size for DataLoaders.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: (train_loader, valid_loader, test_loader)
    '''

    set_seed(seed)

    #normalization parameters for CINIC-10 dataset
    cinic_mean_RGB = [0.47889522, 0.47227842, 0.43047404]
    cinic_std_RGB = [0.24205776, 0.23828046, 0.25874835]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cinic_mean_RGB, std=cinic_std_RGB)
    ])

    train_ds = datasets.ImageFolder(
        root=train_dir,
        transform=transform,
    )

    valid_ds = datasets.ImageFolder(
        root=valid_dir,
        transform=transform,
    )

    test_ds = datasets.ImageFolder(
        root=test_dir,
        transform=transform,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader



def train_epoch(model, dataloader, criterion, optimizer, device):

    '''
    Trains the model for one epoch.

    Args:
        model (nn.Module): The model to be trained.
        dataloader (DataLoader): DataLoader for the training data.
        criterion: Loss function.
        optimizer: Optimization algorithm.
        device: Device to run the training on (e.g., 'cuda' or 'cpu').

    Returns:
        float: Average loss for the epoch.
    '''

    model.train()
    running_loss = 0.0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def evaluate_epoch(model, dataloader, criterion, device):

    '''
    Evaluate the model on a validation or test set.

    Returns:
        loss, accuracy
    '''

    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def train(model, train_loader, valid_loader, criterion, optimizer, device=None, \
          num_epochs=20, verbose=True, verbose_interval=1):
    
    '''
    Train the model over multiple epochs and track training/validation metrics.

    Returns:
        dict: metrics_history with keys 'train_loss', 'valid_loss', 'valid_acc'
    '''

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    metrics_history = {'train_loss': [], 'valid_loss': [], 'valid_acc': []}
    best_acc = 0.0

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    
        valid_loss, valid_acc = evaluate_epoch(model, valid_loader, criterion, device)
        
        metrics_history['train_loss'].append(train_loss)
        metrics_history['valid_loss'].append(valid_loss)
        metrics_history['valid_acc'].append(valid_acc)

        if verbose and (epoch + 1) % verbose_interval == 0:
            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Valid Loss: {valid_loss:.4f} | "
                  f"Valid Acc: {valid_acc:.4f}")

        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(model.state_dict(), 'best_cnn_model.pth')

    print(f"Best validation accuracy: {best_acc:.4f}")
    return metrics_history