import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

PALETTE = {
    'train':  '#2E86AB',
    'valid':  '#E84855',
    'bg':     '#F8F9FA',
    'grid':   '#E0E0E0',
}

plt.rcParams.update({
    'figure.facecolor':  PALETTE['bg'],
    'axes.facecolor':    PALETTE['bg'],
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.color':        PALETTE['grid'],
    'grid.linestyle':    '--',
    'grid.alpha':        0.7,
    'font.family':       'sans-serif',
    'axes.titlesize':    13,
    'axes.labelsize':    11,
    'legend.frameon':    False,
})

def set_seed(seed=42):

    '''Set random seeds for reproducibility.'''

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_loaders(train_dir, valid_dir, test_dir, batch_size=32, image_size=224, seed=42, augmentation=None):

    '''
    Create DataLoaders for training, validation, and test datasets.

    Args:
        train_dir (str): Directory containing training images.
        valid_dir (str): Directory containing validation images.
        test_dir (str): Directory containing test images.
        batch_size (int): Batch size for DataLoaders.
        image_size (int): Size of the input images (default: 224 for 224x224 images).
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: (train_loader, valid_loader, test_loader)
    '''

    set_seed(seed)

    #normalization parameters for CINIC-10 dataset
    cinic_mean_RGB = [0.47889522, 0.47227842, 0.43047404]
    cinic_std_RGB = [0.24205776, 0.23828046, 0.25874835]

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
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

    if augmentation is not None:
        if augmentation == "none":
            pass

        elif augmentation == "standard":
            augmentation_train_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomCrop(image_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean=cinic_mean_RGB, std=cinic_std_RGB)
            ])

            train_ds = datasets.ImageFolder(
                root=train_dir,
                transform=augmentation_train_transform,
            )

        else:
            raise ValueError(f"Unknown augmentation type: {augmentation}")


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


def train(model, train_loader, valid_loader, criterion, optimizer, scheduler=None, device=None, \
          num_epochs=20, verbose=True, verbose_interval=1, checkpoint_name=None):
    
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

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(valid_loss)
            else:
                scheduler.step()
        
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
            if checkpoint_name is None:
                checkpoint_name='best_checkpoint'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'metrics_history': metrics_history,
                'valid_acc': valid_acc
            }, f'{checkpoint_name}.pth')

    torch.save(metrics_history, f"{checkpoint_name}_training_history.pth")

    print(f"Best validation accuracy: {best_acc:.4f}")
    return metrics_history



def predict(model, dataloader, device):

    '''
    Run predictions on a dataloader.

    Args:
        model (torch.nn.Module): Trained PyTorch model.
        dataloader (DataLoader): DataLoader with input samples.
        device (torch.device): Device to run the model on.

    Returns:
        tuple: (predictions, true_labels, probabilities) as NumPy arrays
    '''

    model.eval()
    all_predictions, all_labels, all_probabilities = [], [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)

            all_predictions.append(preds.cpu())
            all_labels.append(labels)
            all_probabilities.append(probs.cpu())

    all_predictions = torch.cat(all_predictions).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_probabilities = torch.cat(all_probabilities).numpy()

    return all_predictions, all_labels, all_probabilities

def predict_batch(model, dataloader, device):

    '''
    Run predictions on a single batch.

    Args:
        model (torch.nn.Module): Trained PyTorch model.
        dataloader (DataLoader): DataLoader with input samples.
        device (torch.device): Device to run the model on.

    Returns:
        tuple: (predictions, true_labels, probabilities) as NumPy arrays

    '''

    model.eval()
    inputs, labels = next(iter(dataloader))
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model(inputs)
        probs   = torch.softmax(outputs, dim=1)
        preds   = probs.argmax(dim=1)

    return preds.cpu().numpy(), labels.numpy(), probs.cpu().numpy(), inputs.cpu()

def evaluate_model(model, dataloader, criterion, device):

    '''
    Compute accuracy and loss for given criterion for the dataloader (train/test/set)

    Args:
        model (torch.nn.Module): Trained PyTorch model.
        dataloader (DataLoader): DataLoader with input samples.
        criterion (torch.nn): Loss function
        device (torch.device): Device to run the model on.

    Returns:
        tuple: (accuracy, average loss) as floats

    '''

    model.eval()
    running_loss = 0.0
    all_predictions, all_labels = [], []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            predictions = outputs.argmax(dim=1)
            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())
        
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)
        accuracy = (all_predictions == all_labels).float().mean().item()
        avg_loss = running_loss / len(dataloader.dataset)

    return accuracy, avg_loss


def plot_training_history(metrics_history):

    '''
    Plot training and validation loss and accuracy over epochs.

    This function creates a side-by-side plot:
      - Left: Training and validation loss
      - Right: Validation accuracy with the best epoch highlighted

    Args:
        metrics_history (dict): Dictionary containing training metrics with keys:
            - 'train_loss' (list or np.ndarray): Training loss per epoch
            - 'valid_loss' (list or np.ndarray): Validation loss per epoch
            - 'valid_acc'  (list or np.ndarray): Validation accuracy per epoch
    '''

    epochs     = range(1, len(metrics_history['train_loss']) + 1)
    best_epoch = int(np.argmax(metrics_history['valid_acc'])) + 1
    best_acc   = max(metrics_history['valid_acc'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Training History', fontsize=15, fontweight='bold', y=1.02)

    ax1.plot(epochs, metrics_history['train_loss'],
             color=PALETTE['train'], linewidth=2, label='Training Loss')
    ax1.plot(epochs, metrics_history['valid_loss'],
             color=PALETTE['valid'], linewidth=2, linestyle='--', label='Validation Loss')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(epochs, metrics_history['valid_acc'],
             color=PALETTE['valid'], linewidth=2, label='Validation Accuracy')
    ax2.axvline(best_epoch, color='gray', linestyle=':', linewidth=1.5)
    ax2.scatter([best_epoch], [best_acc], color=PALETTE['valid'], zorder=5)
    ax2.annotate(f'best: {best_acc:.3f}',
                 xy=(best_epoch, best_acc),
                 xytext=(8, -15), textcoords='offset points',
                 fontsize=9, color='gray')
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.show()

def plot_sample_predictions(model, dataloader, device):

    '''
    Plot sample images from a dataloader with their predicted and true labels.

    This function selects the first batch from the dataloader, runs predictions
    using the model, and displays a grid of sample images annotated with true and predicted label.

    Args:
        model (torch.nn.Module): Trained PyTorch model.
        dataloader (torch.utils.data.DataLoader): DataLoader providing the images.
        device (torch.device): Device to run the model on.

    '''

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                    'dog', 'frog', 'horse', 'ship', 'truck']
    
    preds, labels, probs, inputs = predict_batch(model, dataloader, device)

    mean = torch.tensor([0.47889522, 0.47227842, 0.43047404]).view(3, 1, 1)
    std  = torch.tensor([0.24205776, 0.23828046, 0.25874835]).view(3, 1, 1)
    inputs = (inputs.cpu() * std + mean).clamp(0, 1)

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle('Sample Predictions', fontsize=15, fontweight='bold')

    for i, ax in enumerate(axes.flat):
        img  = inputs[i].permute(1, 2, 0).numpy()
        true = class_names[labels[i]]
        pred = class_names[preds[i]]
        prob = probs[i, preds[i]]

        ax.imshow(img)
        ax.set_title(f'T: {true}\nP: {pred} ({prob:.0%})',
                     color=PALETTE['train'] if true == pred else PALETTE['valid'],
                     fontsize=8)
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(model, dataloader, device):

    '''
    Compute and plot confusion matrix for the given model and dataloader.

    Args:
        model (torch.nn.Module): Trained PyTorch model.
        dataloader (torch.utils.data.DataLoader): DataLoader providing the images.
        device (torch.device): Device to run the model on.

    '''

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                    'dog', 'frog', 'horse', 'ship', 'truck']

    y_pred, y_true, _ = predict(model, dataloader, device)

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] #normalization

    plt.figure(figsize=(8, 6), facecolor=PALETTE['bg'])
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                linewidths=1, linecolor=PALETTE['grid'], cbar=True)

    plt.title('Normalized Confusion Matrix', fontsize=15, fontweight='bold')
    plt.xlabel('Predicted label', fontsize=12)
    plt.ylabel('True label', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.grid(False) 
    plt.tight_layout()
    plt.show()



def plot_cumulative_training_history(all_metrics_history):

    '''
    Plot cumulative training history for multiple models.
    '''

    sns.set_style("whitegrid")
    num_models = len(all_metrics_history)
    palette = sns.color_palette("Set2", n_colors=num_models)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Cumulative Training History', fontsize=18, fontweight='bold', y=1.05)

    for idx, (model_name, metrics_history) in enumerate(all_metrics_history.items()):
        color = palette[idx]
        epochs = range(1, len(metrics_history['train_loss']) + 1)

        ax1.plot(epochs, metrics_history['train_loss'], color=color, linewidth=2, label=f'{model_name} Train')
        ax1.plot(epochs, metrics_history['valid_loss'], color=color, linewidth=2, linestyle='--', label=f'{model_name} Valid')

    ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)

    for idx, (model_name, metrics_history) in enumerate(all_metrics_history.items()):
        color = palette[idx]
        epochs = range(1, len(metrics_history['valid_acc']) + 1)
        best_epoch = int(np.argmax(metrics_history['valid_acc'])) + 1
        best_acc = max(metrics_history['valid_acc'])

        ax2.plot(epochs, metrics_history['valid_acc'], color=color, linewidth=2, label=f'{model_name}')
        ax2.axvline(best_epoch, color=color, linestyle=':', linewidth=1.5, alpha=0.5)
        ax2.scatter([best_epoch], [best_acc], color=color, zorder=5)
        ax2.text(best_epoch + 0.5, best_acc, f'{best_acc:.2f}', color=color, fontsize=9)

    ax2.set_title('Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_ylim(0, 1)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)

    plt.tight_layout()
    plt.show()