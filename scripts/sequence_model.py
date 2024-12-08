import torch
import cv2
import torch.nn as nn
import torch.nn.functional as F
import os
import torchvision.models as models
import math

class SequenceRecognizer(nn.Module):
    def __init__(self, input_channels=1, hidden_size=256, num_classes=11):
        super(SequenceRecognizer, self).__init__()
        
        # Use ResNet18 as backbone with preserved residual connections
        resnet = models.resnet18(pretrained=True)
        resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Keep residual blocks from ResNet
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # Multi-head self-attention with residual connection
        self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=8, dropout=0.1)
        self.attention_norm = nn.LayerNorm(512)
        self.pos_encoder = PositionalEncoding(512, dropout=0.1)
        
        # Additional residual attention block
        self.residual_attention = ResidualAttentionBlock(512, 8)
        
        # Bidirectional GRU with residual connection
        self.rnn = nn.GRU(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.2
        )
        self.rnn_norm = nn.LayerNorm(hidden_size * 2)
        
        # Classifiers remain the same
        self.sequence_classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, num_classes)
        )
        
        self.length_classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 5)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize the weights of the non-pretrained layers"""
        for name, module in self.named_modules():
            # Skip the pretrained ResNet backbone
            if 'backbone' in name:
                continue
                
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        # Normalize input
        x = x.float() / 255.0 if x.max() > 1 else x
        
        # Extract features using ResNet backbone (includes residual connections)
        features = self.backbone(x)  # [batch, channels, height, width]
        
        # Reshape for sequence modeling
        b, c, h, w = features.shape
        features = features.permute(0, 2, 3, 1)  # [batch, height, width, channels]
        features = features.reshape(b, h * w, c)  # [batch, sequence_length, channels]
        
        # Add positional encoding
        features = self.pos_encoder(features)
        
        # Multi-head self-attention with residual connection
        attended_features = features.transpose(0, 1)  # [seq_len, batch, channels]
        attn_out, _ = self.attention(attended_features, attended_features, attended_features)
        attended_features = attended_features + attn_out  # Residual connection
        attended_features = self.attention_norm(attended_features)
        attended_features = attended_features.transpose(0, 1)  # [batch, seq_len, channels]
        
        # Additional residual attention
        attended_features = self.residual_attention(attended_features)
        
        # RNN with residual connection
        rnn_out, _ = self.rnn(attended_features)
        rnn_out = attended_features + rnn_out  # Residual connection
        rnn_out = self.rnn_norm(rnn_out)
        
        # Sequence prediction
        sequence_logits = self.sequence_classifier(rnn_out)
        
        # Length prediction
        global_features = torch.mean(rnn_out, dim=1)
        length_logits = self.length_classifier(global_features)
        
        return sequence_logits, length_logits

    def decode_predictions(self, log_probs):
        # Get the most likely prediction at each timestep
        pred_sequences = log_probs.argmax(dim=2)
        return pred_sequences

class ResidualAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=0.1)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 4, dim)
        )
        self.attention_norm = nn.LayerNorm(dim)
        self.feed_forward_norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        # Self attention with residual
        attn_out = self.attention(x.transpose(0, 1), x.transpose(0, 1), x.transpose(0, 1))[0]
        x = x + attn_out.transpose(0, 1)
        x = self.attention_norm(x)
        
        # Feed forward with residual
        ff_out = self.feed_forward(x)
        x = x + ff_out
        x = self.feed_forward_norm(x)
        
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

def train_sequence_model(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, targets, target_lengths) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)
        
        optimizer.zero_grad()
        sequence_logits, length_logits = model(data)
        
        # Use sequence_logits instead of output
        input_lengths = torch.full(size=(data.size(0),), 
                                 fill_value=sequence_logits.size(1), 
                                 dtype=torch.long)
        
        # Calculate loss
        loss = criterion(sequence_logits.transpose(0, 1), # (T, N, C)
                        targets,
                        input_lengths,
                        target_lengths)
        
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx}]\tLoss: {loss.item():.6f}')

# Custom dataset for multi-digit sequences
class MultiDigitDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        
        # Load labels from file
        if isinstance(labels, str):
            with open(labels, 'r') as f:
                # Split each line into filename and label
                label_dict = {}
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 2:
                        raise ValueError(f"Invalid label format: {line.strip()}")
                    filename, label = parts
                    if not label.isdigit():
                        raise ValueError(f"Invalid label number: {label}")
                    label_dict[filename] = label
                
                # Match labels with image paths
                self.labels = []
                for path in image_paths:
                    filename = os.path.basename(path)
                    if filename not in label_dict:
                        raise ValueError(f"No label found for image: {filename}")
                    self.labels.append(label_dict[filename])
        else:
            self.labels = labels
            
        # Validate dataset
        if len(self.image_paths) != len(self.labels):
            raise ValueError(f"Number of images ({len(self.image_paths)}) doesn't match number of labels ({len(self.labels)})")
            
        print(f"Dataset initialized with {len(self.image_paths)} images")
        print(f"First few image-label pairs:")
        for i in range(min(5, len(self.image_paths))):
            print(f"  {os.path.basename(self.image_paths[i])} -> {self.labels[i]}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"Failed to load image at path: {image_path}")
        
        if self.transform:
            image = self.transform(image)
        
        # Convert label string to tensor of integers
        label = torch.tensor([int(d) for d in self.labels[idx]], dtype=torch.long)
        label_length = torch.tensor([len(self.labels[idx])], dtype=torch.long)
        
        return image, label, label_length