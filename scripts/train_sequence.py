import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import glob
import os
from sequence_model import SequenceRecognizer, MultiDigitDataset
from torchvision import transforms
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup
def collate_fn(batch):
    # Separate images, labels and label lengths
    images, labels, label_lengths = zip(*batch)
    
    # Stack images
    images = torch.stack(images, 0)
    
    # Pad labels to same length
    labels = pad_sequence([label for label in labels], batch_first=True, padding_value=10)
    
    # Stack label lengths
    label_lengths = torch.stack(label_lengths, 0)
    
    return images, labels, label_lengths

def train_sequence_model(model, device, train_loader, optimizer, criterion, scheduler, epoch):
    model.train()
    total_loss = 0
    correct_digits = 0
    total_digits = 0
    
    for batch_idx, (data, targets, target_lengths) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)
        target_lengths = target_lengths.to(device)
        
        optimizer.zero_grad()
        logits, length_logits = model(data)
        
        # Temperature scaling for sharper predictions
        temperature = max(0.5, 1.0 - epoch * 0.02)  # Gradually decrease temperature
        scaled_logits = logits / temperature
        
        # Apply label smoothing
        log_probs = F.log_softmax(scaled_logits, dim=2)
        smooth_factor = 0.1
        smooth_loss = -log_probs.mean(dim=-1).mean() * smooth_factor
        
        # CTC loss with reduced blank weight
        input_lengths = torch.full(size=(data.size(0),), 
                                 fill_value=logits.size(1), 
                                 dtype=torch.long).to(device)
        ctc_loss = criterion(log_probs.transpose(0, 1), targets, 
                           input_lengths, target_lengths.squeeze())
        
        # Length prediction loss
        length_target = target_lengths.squeeze() - 1
        length_loss = F.cross_entropy(length_logits, length_target)
        
        # Dynamic loss weighting
        ctc_weight = min(1.0, epoch / 5)  # Increase CTC weight faster
        loss = ctc_weight * ctc_loss + (1 - ctc_weight) * length_loss + smooth_loss
        
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        # Calculate accuracy
        pred = log_probs.argmax(dim=2)
        
        for i in range(len(target_lengths)):
            true_seq = targets[i][:target_lengths[i]]
            pred_seq = pred[i][:target_lengths[i]]
            
            for j in range(len(true_seq)):
                total_digits += 1
                if j < len(pred_seq) and true_seq[j] == pred_seq[j]:
                    correct_digits += 1
        
        if batch_idx % 10 == 0:
            current_accuracy = correct_digits / total_digits if total_digits > 0 else 0
            print(f'Epoch: {epoch} [{batch_idx}]\t'
                  f'Loss: {loss.item():.4f}\t'
                  f'Accuracy: {current_accuracy:.4f}\t'
                  f'Temp: {temperature:.2f}')
    
    epoch_accuracy = correct_digits / total_digits if total_digits > 0 else 0
    avg_loss = total_loss / len(train_loader)
    
    return avg_loss, epoch_accuracy

def get_most_confused_pairs(confusion_matrix):
    pairs = []
    for i in range(10):
        for j in range(10):
            if i != j:  # Don't include correct predictions
                pairs.append(((i, j), confusion_matrix[i][j].item()))
    return sorted(pairs, key=lambda x: x[1], reverse=True)  # Sort by frequency

def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Improved hyperparameters
    num_epochs = 200
    batch_size = 64
    base_lr = 3e-4
    
    # Enhanced data augmentation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomApply([
            transforms.RandomAffine(degrees=2, translate=(0.02, 0.02), 
                                 scale=(0.98, 1.02), shear=1)
        ], p=0.5),
        transforms.RandomApply([
            transforms.GaussianBlur(3, sigma=(0.1, 0.2))
        ], p=0.2),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Create datasets and dataloaders
    train_dataset = MultiDigitDataset(
        image_paths=glob.glob('multi_digit_ocr/data/images/train/*.png'),
        labels='multi_digit_ocr/data/labels/train.txt',
        transform=transform
    )
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # Initialize model
    model = SequenceRecognizer().to(device)
    
    # Load existing model if it exists
    model_path = 'multi_digit_ocr/models/sequence_model_best.pth'
    start_epoch = 0
    best_accuracy = 0
    
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        best_accuracy = checkpoint['accuracy']
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch} with accuracy {best_accuracy:.4f}")
    
    # Improved optimizer setup
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.01)
    
    # If loading existing model, also load optimizer state
    if os.path.exists(model_path):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Ensure optimizer state is moved to correct device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    
    # Learning rate scheduler with warm-up and cosine decay
    num_steps = len(train_loader) * num_epochs
    warmup_steps = len(train_loader) * 5  # 5 epochs of warm-up
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_steps
    )
    
    # CTC Loss with blank weight
    criterion = nn.CTCLoss(blank=10, zero_infinity=True)
    
    # Training loop with early stopping
    patience = 25
    patience_counter = 0
    
    # Update training loop to start from loaded epoch
    for epoch in range(start_epoch, num_epochs):
        avg_loss, accuracy = train_sequence_model(
            model, device, train_loader, optimizer, criterion, scheduler, epoch
        )
        
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': accuracy
            }, 'multi_digit_ocr/models/sequence_model_best.pth')
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered. Best accuracy: {best_accuracy:.4f}")
            break

if __name__ == '__main__':
    main()
